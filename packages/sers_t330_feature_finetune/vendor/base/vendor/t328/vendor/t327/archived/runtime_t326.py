"""Complete forward training and validation. No feature cache or test prediction."""
from __future__ import annotations
import hashlib
import json
import random
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from Model_T3_17 import TransformerClassifyRegress_sep as LegacyModel, decode_ordinal_numpy
from Model_T3_26 import TransformerClassifyRegress_sep as NewModel
from Model_T3_17 import present_only_corn_ordinal_loss
from data_audit import audit_selection, measured_masks, metadata_rows
from Relations_T3_26 import VARIANTS
from MultiPeak_T3_25 import DEFAULT_CENTERS


class SpectrumView(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.original = dataset
        self.indices = list(indices)
        self.samples = [dataset.samples[i] for i in indices]
        self.repository = dataset.repository
        self.split = dataset.split

    def __len__(self):return len(self.indices)
    def __getitem__(self, i):return self.original[self.indices[i]]


def smoke_validation_view(dataset, maximum=32):
    from collections import defaultdict
    groups=defaultdict(list)
    for i,ref in enumerate(dataset.samples):
        info=dataset.repository.real_data[ref.condition]['condition_info']
        count=int(info.presence.sum())
        groups[(info.matrix_name,count)].append(i)
    indices=[]
    while len(indices)<min(maximum,len(dataset)):
        for key in sorted(groups):
            if groups[key] and len(indices)<maximum:indices.append(groups[key].pop(0))
    return SpectrumView(dataset,indices)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def save_json(path, payload):
    Path(path).write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


def configure_project(project):
    project=Path(project).resolve()
    expected=json.loads((Path(__file__).parent/'expected_project_sources.json').read_text())
    found={name:sha256(project/name) for name in expected}
    mismatch={name:dict(expected=expected[name],actual=found[name]) for name in expected if found[name]!=expected[name]}
    if mismatch: raise RuntimeError('Server source differs from archived D26+T24; do not patch blindly: '+json.dumps(mismatch))
    sys.path.insert(0,str(project))
    import Dataset
    if Path(Dataset.__file__).resolve()!=project/'Dataset.py':raise RuntimeError('Wrong Dataset.py imported')
    datasets=Dataset.build_datasets(include_generated_train=True,maximum_generated_per_condition=48,
                                   require_complete_generated=True,concentration_map=None)
    audit=audit_selection(datasets)
    # Inspect sample metadata only; never call test_dataset.__getitem__.
    return datasets,audit,found


def load_model(path,device):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    version=payload.get('checkpoint_version')
    if version=='T3.26_cross_combination_v1':
        model=NewModel(**payload['model_config'])
    elif version=='T3.25_multipeak_curriculum_v1':
        from Model_T3_25 import TransformerClassifyRegress_sep as PreviousModel
        model=PreviousModel(**payload['model_config'])
    elif version=='T3.24_concentration_update_v1':
        from Model_T3_24 import ConcentrationUpdateModel
        model=ConcentrationUpdateModel(**payload['model_config'])
    else:
        model=LegacyModel(**payload['model_config'])
    model.load_state_dict(payload['model_state_dict'],strict=True)
    return model.to(device).eval(),payload


def build_variant(baseline,variant,device):
    if variant not in VARIANTS:raise ValueError('Unknown variant')
    config=dict(baseline['model_config'])
    for key in ('use_local_quantitative_branch','use_anchored_local_calibration',
                'use_adaptive_mixture_gate','use_boundary_specific_mixture_gate'):
        if config.get(key,False):raise ValueError('Use the unmodified T3.18 anchor')
    config.update(pooling_stride=3,
                  use_multi_peak_evidence=variant in ('coarse_multipeak','cross_relation'),
                  relation_projection_dim=16 if variant=='cross_relation' else 0,
                  peak_window_centers=list(DEFAULT_CENTERS),peak_half_width_cm1=24,
                  multi_peak_strength=0.25)
    model=NewModel(**config)
    incompatible=model.load_state_dict(baseline['model_state_dict'],strict=False)
    if incompatible.unexpected_keys or any(not key.startswith(('multi_peak_evidence.','relation_heads.')) for key in incompatible.missing_keys):
        raise RuntimeError('Unexpected checkpoint incompatibility: '+str(incompatible))
    model.requires_grad_(True)
    return model.to(device),config


def make_loader(dataset,batch_size=16,workers=4,indices=None,seed=2026):
    view=dataset if indices is None else Subset(dataset,indices)
    return DataLoader(view,batch_size=batch_size,shuffle=False,num_workers=workers,
                      pin_memory=torch.cuda.is_available(),generator=torch.Generator().manual_seed(seed),
                      persistent_workers=False)


def model_inputs(dataset,batch,device):
    # True class, grade, matrix and condition identifiers NEVER enter the model.
    result={key:batch[key].to(device,non_blocking=True) for key in
            ('raw','percentile','smoothed','valid_mask','raw_intensity')}
    result['measured_mask']=measured_masks(dataset,batch).to(device,non_blocking=True)
    return result


def outputs(model,data):
    if hasattr(model,'base'):
        values=model(data)
        return values[0],values[2],values[3]
    values=model(data,return_ordinal=True,return_ordinal_logits=True)
    return values[0],values[2],values[3]


def joint_loss(classify,logits,classes,levels):
    class_loss=nn.functional.binary_cross_entropy(classify,classes)
    corn=present_only_corn_ordinal_loss(logits,levels,classes)
    return .3*class_loss+.7*corn,class_loss,corn


def summarize(frame):
    drugs=('DEL','CHL','TEB')
    truth=frame[[f'true_{d}' for d in drugs]].to_numpy(int)
    ordinal=frame[[f'ordinal_{d}' for d in drugs]].to_numpy(int)
    profile=frame[[f'pred_{d}' for d in drugs]].to_numpy(int)
    actual=truth>0; predicted=profile>0
    present_correct=int(((truth==ordinal)&actual).sum())
    mae=float(np.abs(truth-ordinal)[actual].mean())
    f1=[]
    for j in range(3):
        tp=int((actual[:,j]&predicted[:,j]).sum());fp=int((~actual[:,j]&predicted[:,j]).sum());fn=int((actual[:,j]&~predicted[:,j]).sum())
        f1.append(2*tp/max(1,2*tp+fp+fn))
    ternary=actual.sum(1)==3
    result=dict(samples=len(frame),present_n=int(actual.sum()),present_correct=present_correct,
                present_accuracy=present_correct/max(1,int(actual.sum())),present_mae=mae,
                full_profile_correct=int((truth==profile).all(1).sum()),
                classification_exact_correct=int((actual==predicted).all(1).sum()),
                classification_macro_f1=float(np.mean(f1)),
                ternary_n=int(ternary.sum()),ternary_all_grade_correct=int((truth[ternary]==ordinal[ternary]).all(1).sum()))
    result['by_matrix']={}
    for matrix in sorted(frame.matrix.unique()):
        select=frame.matrix.to_numpy()==matrix;tri=select&ternary
        result['by_matrix'][str(matrix)]=dict(n=int(select.sum()),full_profile_correct=int((truth[select]==profile[select]).all(1).sum()),
                    ternary_n=int(tri.sum()),ternary_all_grade_correct=int((truth[tri]==ordinal[tri]).all(1).sum()))
    result['DEL_M_by_TEB']={}
    for matrix in sorted(frame.matrix.unique()):
        for level in (1,2,3):
            select=ternary&(truth[:,0]==2)&(truth[:,2]==level)&(frame.matrix.to_numpy()==matrix)
            result['DEL_M_by_TEB'][f'{matrix}_TEB_{level}']=dict(n=int(select.sum()),DEL_correct=int((ordinal[select,0]==2).sum()))
    result['per_drug']={d:dict(present_n=int(actual[:,j].sum()),correct=int(((truth[:,j]==ordinal[:,j])&actual[:,j]).sum()),
                              level_code_mae=float(np.abs(truth[:,j]-ordinal[:,j])[actual[:,j]].mean()) if actual[:,j].any() else None,
                              classification_f1=f1[j]) for j,d in enumerate(drugs)}
    return result


@torch.no_grad()
def evaluate(model,dataset,device,batch_size=16,workers=4,max_batches=None):
    model.eval();rows=[];sums=np.zeros(3);n=0
    for step,batch in enumerate(make_loader(dataset,batch_size,workers)):
        if max_batches is not None and step>=max_batches:break
        cp,op,logits=outputs(model,model_inputs(dataset,batch,device))
        classes=batch['class_target'].to(device);levels=batch['concentration_target'].to(device)
        loss,cl,co=joint_loss(cp,logits,classes,levels)
        count=len(classes);sums+=np.array([loss.item(),cl.item(),co.item()])*count;n+=count
        grade=decode_ordinal_numpy(op.cpu().numpy(),mode='median',threshold=.5)
        cp=cp.cpu().numpy();op=op.cpu().numpy()
        meta=metadata_rows(dataset,batch)
        for i,row in enumerate(meta):
            for j,d in enumerate(('DEL','CHL','TEB')):
                row[f'ordinal_{d}']=int(grade[i,j]);row[f'pred_{d}']=int(grade[i,j]) if cp[i,j]>=.5 else 0
                row[f'presence_probability_{d}']=float(cp[i,j]);row[f'P_ge_M_{d}']=float(op[i,j,0]);row[f'P_ge_H_{d}']=float(op[i,j,1])
            rows.append(row)
    if not rows:raise RuntimeError('Empty evaluation')
    frame=pd.DataFrame(rows);metrics=summarize(frame)
    metrics.update(loss_total=float(sums[0]/n),loss_class=float(sums[1]/n),loss_corn=float(sums[2]/n))
    return metrics,frame


def fit_real_normalization(model,train,workers=4):
    if model.multi_peak_evidence is None:return None
    indices=[i for i,s in enumerate(train.samples) if s.source=='real']
    if len(indices)!=1512:raise RuntimeError('Normalization must use the fixed 1512 real training spectra')
    intensities=[];masks=[]
    for batch in make_loader(train,64,workers,indices):
        intensities.append(batch['raw_intensity']);masks.append(measured_masks(train,batch))
    device=model.multi_peak_evidence.centers.device
    return model.multi_peak_evidence.fit_normalization(torch.cat(intensities).to(device),torch.cat(masks).to(device))


def rank(metrics):
    return (metrics['ternary_all_grade_correct'],metrics['full_profile_correct'],
            metrics['present_correct'],-metrics['present_mae'])


def adoption(metrics,reference,baseline):
    rules={
        'classification_exact_not_lower':metrics['classification_exact_correct']>=baseline['classification_exact_correct'],
        'classification_macro_f1_not_lower':metrics['classification_macro_f1']+1e-12>=baseline['classification_macro_f1'],
        'ternary_better_than_current':metrics['ternary_all_grade_correct']>reference['ternary_all_grade_correct'],
        'full_profile_not_lower':metrics['full_profile_correct']>=reference['full_profile_correct'],
        'present_grade_not_lower':metrics['present_correct']>=reference['present_correct'],
    }
    return dict(accepted=all(rules.values()),rules=rules)


def save_checkpoint(path,model,config,epoch,metrics,training_config,optimizer=None,scheduler=None):
    payload=dict(checkpoint_version='T3.26_cross_combination_v1',model_config=config,
                 model_state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                 epoch=epoch,validation_metrics=metrics,training_config=training_config,
                 test_predictions=0)
    if optimizer is not None:payload['optimizer_state_dict']=optimizer.state_dict()
    if scheduler is not None:payload['scheduler_state_dict']=scheduler.state_dict()
    if optimizer is not None:
        payload['rng_state'] = capture_rng()
    temporary=Path(str(path)+'.tmp')
    torch.save(payload,temporary)
    temporary.replace(path)


def gradient_audit(model):
    groups=('cnn.','cnn2.','reg_feat.','cnn_fusion.','token_expansion.',
            'transformer_encoder.layers.','peak_guided_attention.','classification_projection.',
            'classification_heads.','concentration_projection.','mixture_aware_query_fusion.',
            'ordinal_heads.')
    if model.multi_peak_evidence is not None:groups+=('multi_peak_evidence.curve_encoder.',
               'multi_peak_evidence.window_encoder.','multi_peak_evidence.cross_attention.',
               'multi_peak_evidence.output_projection.')
    if getattr(model,'relation_heads',None) is not None:groups+=('relation_heads.',)
    found={}
    for prefix in groups:
        parameters=[p for name,p in model.named_parameters() if name.startswith(prefix)]
        if not parameters:continue
        valid=[p.grad for p in parameters if p.grad is not None]
        found[prefix]=dict(has_gradient=bool(valid),finite=all(bool(torch.isfinite(g).all()) for g in valid),
                           nonzero=any(bool((g!=0).any()) for g in valid))
    if not all(x['has_gradient'] and x['finite'] and x['nonzero'] for x in found.values()):
        raise RuntimeError('Full forward gradient audit failed: '+json.dumps(found))
    return found


def capture_rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['cpu'])
    if state['cuda'] and torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda'])
