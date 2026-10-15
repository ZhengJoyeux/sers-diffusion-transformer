"""Check, smoke, train, export, evaluate and compare the independent 2-D recognizer."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from PIL import Image
from bridge import build_data, sha256
from image_data import PlotScale, LineImageDataset, fit_training_scale
from image_model import ImageSERSFormer, VERSION, warm_start
from spectral_core import present_only_corn_ordinal_loss, decode_ordinal_numpy
from relations import ReferencePlan, relation_loss

DRUGS = ('DEL', 'CHL', 'TEB')
DEFAULT_ANCHOR = '/home/wqzheng/project_transformer/outputs/t3_30_feature_finetune_combined_seed2026/train/concentration/checkpoints/TEB_only/best_accepted.pt'


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def model_inputs(batch, device):
    return {k: batch[k].to(device, non_blocking=True) for k in
            ('image', 'image_valid_columns', 'raw_intensity', 'measured_mask')}


def loader(dataset, args, shuffle=False):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.num_workers, pin_memory=args.device.startswith('cuda'),
        persistent_workers=args.num_workers > 0,
        generator=torch.Generator().manual_seed(args.seed) if shuffle else None)


def supervision(output, classes, levels, args):
    class_loss = F.binary_cross_entropy(output['cp'], classes)
    corn = present_only_corn_ordinal_loss(output['logits'], levels, classes)
    # Preserve the optional ordinal boundary margin/ternary weighting design.
    p = output['op']; target = torch.stack((levels >= 2, levels >= 3), -1).float()
    weight = torch.where(classes.sum(1) >= 2.5, args.ternary_multiplier, 1.)[:, None, None]
    active = (classes > .5)[..., None].float()*weight
    violation = F.relu(args.boundary_margin-(2*target-1)*(p-.5))
    boundary = (violation*active).sum()/(active.sum()*2).clamp_min(1.)
    return args.classification_weight*class_loss+args.concentration_weight*(corn+args.boundary_weight*boundary), class_loss, corn


def metrics(frame):
    truth = frame[['true_'+d for d in DRUGS]].to_numpy(dtype=int)
    predicted = frame[['pred_'+d for d in DRUGS]].to_numpy(dtype=int)
    ordinal = frame[['ordinal_'+d for d in DRUGS]].to_numpy(dtype=int)
    actual = truth > 0; detected = predicted > 0
    present_correct = ((ordinal == truth) & actual).sum()
    profile = (predicted == truth).all(1)
    ternary = actual.all(1)
    def accuracy(num, den): return float(num/den) if den else None
    result = dict(samples=len(frame), present_n=int(actual.sum()), present_correct=int(present_correct),
        present_accuracy=accuracy(present_correct, actual.sum()),
        classification_exact_correct=int((actual == detected).all(1).sum()),
        classification_exact_accuracy=float((actual == detected).all(1).mean()),
        full_profile_correct=int(profile.sum()), full_profile_accuracy=float(profile.mean()),
        ternary_n=int(ternary.sum()), ternary_all_grade_correct=int(profile[ternary].sum()),
        ternary_full_profile_accuracy=accuracy(profile[ternary].sum(), ternary.sum()),
        present_level_code_mae=float(np.abs(ordinal-truth)[actual].mean()),
        per_drug={}, by_mixture={}, by_matrix={})
    for j, d in enumerate(DRUGS):
        tp = int((actual[:, j] & detected[:, j]).sum())
        fp = int((~actual[:, j] & detected[:, j]).sum())
        fn = int((actual[:, j] & ~detected[:, j]).sum())
        result['per_drug'][d] = dict(present_n=int(actual[:, j].sum()),
            correct=int(((ordinal[:, j] == truth[:, j]) & actual[:, j]).sum()),
            accuracy=accuracy(((ordinal[:, j] == truth[:, j]) & actual[:, j]).sum(), actual[:, j].sum()),
            classification_f1=2*tp/max(2*tp+fp+fn, 1),
            confusion_S_M_H=[[int(((truth[:, j] == a) & (ordinal[:, j] == b)).sum()) for b in (1,2,3)] for a in (1,2,3)])
    result['classification_macro_f1'] = float(np.mean([v['classification_f1'] for v in result['per_drug'].values()]))
    for k in (1,2,3):
        select = actual.sum(1) == k
        result['by_mixture'][str(k)] = dict(samples=int(select.sum()), full_profile_correct=int(profile[select].sum()),
            full_profile_accuracy=accuracy(profile[select].sum(), select.sum()))
    for matrix in sorted(frame['matrix'].unique()):
        select = frame['matrix'].to_numpy() == matrix
        result['by_matrix'][matrix] = dict(samples=int(select.sum()), full_profile_correct=int(profile[select].sum()),
            full_profile_accuracy=accuracy(profile[select].sum(), select.sum()))
    return result


@torch.no_grad()
def evaluate(model, dataset, args, max_batches=None):
    model.eval(); rows = []; sums = np.zeros(3); count = 0; max_clipped = 0.
    for step, batch in enumerate(loader(dataset, args)):
        if max_batches is not None and step >= max_batches: break
        output = model(model_inputs(batch, args.device))
        classes = batch['class_target'].to(args.device)
        levels = batch['concentration_level'].to(args.device)
        loss, cl, co = supervision(output, classes, levels, args)
        n = len(classes); sums += np.array([loss.item(), cl.item(), co.item()])*n; count += n
        max_clipped = max(max_clipped, float(batch['image_clipped_fraction'].max()))
        cp = output['cp'].cpu().numpy(); op = output['op'].cpu().numpy()
        grade = decode_ordinal_numpy(op, mode=args.ordinal_decoding)
        truth = batch['concentration_level'].numpy().astype(int)
        for i in range(n):
            row = dict(condition=batch['condition'][i], source=batch['source'][i],
                spectrum_index=int(batch['spectrum_index'][i]), matrix=batch['matrix_name'][i],
                split=dataset.split, mixture_count=int((truth[i] > 0).sum()))
            for j, d in enumerate(DRUGS):
                row.update({'true_'+d:int(truth[i,j]), 'ordinal_'+d:int(grade[i,j]),
                            'pred_'+d:int(grade[i,j]) if cp[i,j] >= .5 else 0,
                            'presence_probability_'+d:float(cp[i,j]),
                            'P_ge_M_'+d:float(op[i,j,0]), 'P_ge_H_'+d:float(op[i,j,1])})
            rows.append(row)
    if not rows: raise RuntimeError('No evaluation samples')
    frame = pd.DataFrame(rows); result = metrics(frame)
    result.update(loss_total=float(sums[0]/count), loss_class=float(sums[1]/count),
        loss_corn=float(sums[2]/count), max_image_clipped_fraction=max_clipped,
        evaluation_split=dataset.split, ordinal_decoding=args.ordinal_decoding)
    return result, frame


def save_evaluation(directory, result, frame):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    write_json(directory/'ordinal_metrics.json', result)
    frame.to_csv(directory/'predictions.csv', index=False)
    pd.DataFrame([dict(pesticide=d, **{k:v for k,v in m.items() if k != 'confusion_S_M_H'})
                  for d,m in result['per_drug'].items()]).to_csv(directory/'ordinal_per_pesticide.csv', index=False)
    for d,m in result['per_drug'].items():
        pd.DataFrame(m['confusion_S_M_H'], index=['S','M','H'], columns=['S','M','H']).to_csv(directory/(d+'_confusion.csv'))
    for key, filename in [('by_mixture','ordinal_by_mixture_complexity.csv'), ('by_matrix','ordinal_by_matrix.csv')]:
        pd.DataFrame([dict(group=k, **v) for k,v in result[key].items()]).to_csv(directory/filename, index=False)


def make_optimizer(model, args):
    groups = {}
    for name, p in model.named_parameters():
        if name.startswith('relation_heads.') and not args.relation_weight:
            p.requires_grad_(False)
        if not p.requires_grad: continue
        if name.startswith('image_frontend.'): family, maximum = 'image', args.image_lr
        elif 't327_adapter.' in name: family, maximum = 'adapter', args.image_lr
        elif name.startswith('multi_peak_evidence.'): family, maximum = 'numeric', args.numeric_lr
        elif name.startswith(('transformer_encoder.', 'peak_guided_attention.')):
            family, maximum = 'backbone', args.image_lr if args.from_scratch else args.backbone_lr
        else: family, maximum = 'task', args.image_lr if args.from_scratch else args.head_lr
        decay = args.weight_decay if p.ndim > 1 and 'norm' not in name else 0.
        group = groups.setdefault((family,maximum,decay), dict(params=[], family=family, lr=maximum,
                                  maximum_lr=maximum, minimum_lr=maximum*args.minimum_ratio, weight_decay=decay))
        group['params'].append(p)
    intended = {id(p) for p in model.parameters() if p.requires_grad}
    actual = [id(p) for g in groups.values() for p in g['params']]
    if len(actual) != len(set(actual)) or set(actual) != intended:
        raise RuntimeError('Optimizer does not cover all intended parameters exactly once')
    return torch.optim.AdamW(list(groups.values()))


def set_rates(optimizer, update, total, warmup):
    for group in optimizer.param_groups:
        if warmup and update <= warmup:
            rate = group['maximum_lr']*update/warmup
        else:
            progress = (update-warmup)/max(total-warmup,1)
            rate = group['minimum_lr']+(group['maximum_lr']-group['minimum_lr'])*(1+math.cos(math.pi*progress))/2
        group['lr'] = rate


def save_checkpoint(path, model, scale, args, epoch, validation, optimizer, completed, audit):
    state = dict(checkpoint_version=VERSION, model_config=model.config, plot_scale=asdict(scale),
        model_state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        epoch=epoch, validation_metrics=validation, ordinal_decoding=args.ordinal_decoding,
        experiment_config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        optimizer_state_dict=optimizer.state_dict(), completed_updates=completed,
        split_audit=audit, test_predictions=0,
        source_hashes={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')})
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); torch.save(state, tmp); tmp.replace(path)


def load_checkpoint(path, device):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('checkpoint_version') != VERSION: raise ValueError('Expected a T3.32 image checkpoint')
    model = ImageSERSFormer(**state['model_config'])
    model.load_state_dict(state['model_state_dict'], strict=True)
    return model.to(device).eval(), PlotScale(**state['plot_scale']), state


def prepare_model(args, module, train):
    scale, values, masks, fit_audit = fit_training_scale(train, args.width, args.height)
    model = ImageSERSFormer(numeric_evidence=not args.image_only)
    transfer = dict(mode='from_scratch') if args.from_scratch else warm_start(model, args.anchor_checkpoint)
    # This is a fresh fit using exactly the training identities, not a refit on held-out data.
    prior, prior_summary = module.build_chemistry_hybrid_peak_priors(train.repository, top_k=4)
    model.peak_guided_attention.set_peak_prior(prior)
    numeric = None
    if model.multi_peak_evidence is not None:
        numeric = model.multi_peak_evidence.fit_normalization(values, masks)
    return model.to(args.device), scale, dict(plot_scale_fit=fit_audit, transfer=transfer,
                                              numeric_normalization=numeric, peak_prior=prior_summary)


def train(args, module, datasets, audit, smoke=False):
    directory = args.output_directory/('smoke' if smoke else 'train')
    if (directory/'checkpoints').exists():
        raise FileExistsError('Existing run will not be overwritten: '+str(directory))
    directory.mkdir(parents=True, exist_ok=True)
    model, scale, details = prepare_model(args, module, datasets[0])
    write_json(directory/'plot_scale.json', asdict(scale)); write_json(directory/'initialization.json', details)
    write_json(directory/'split_audit.json', audit)
    wrapped = [LineImageDataset(d, scale, args.cache_directory) for d in datasets]
    optimizer = make_optimizer(model, args)
    write_json(directory/'training_config.json', dict(
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        model_config=model.config, parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        groups=[{k:v for k,v in g.items() if k != 'params'} for g in optimizer.param_groups]))
    epochs = 1 if smoke else args.epochs
    train_loader = loader(wrapped[0], args, True)
    steps = min(len(train_loader), 2) if smoke else len(train_loader)
    total = steps*epochs; warmup = min(args.warmup_epochs*steps, max(total-1,0))
    references = ReferencePlan(datasets[0]) if args.relation_weight > 0 else None
    if references is not None: write_json(directory/'relation_audit.json', references.audit())
    history=[]; completed=0; best=None; stale=0
    for epoch in range(1, epochs+1):
        freeze = not args.from_scratch and not smoke and epoch <= args.freeze_backbone_epochs
        for name,p in model.named_parameters():
            if name.startswith(('transformer_encoder.', 'peak_guided_attention.')) and 't327_adapter.' not in name:
                p.requires_grad_(not freeze)
        model.train(); sums=np.zeros(4); n=0; start=time.time(); clipped=0.
        triples = references.epoch(epoch, steps, triples_per_step=2, seed=args.seed) if references else None
        for step,batch in enumerate(train_loader):
            if step >= steps: break
            optimizer.zero_grad(set_to_none=True)
            output=model(model_inputs(batch,args.device))
            loss,cl,co=supervision(output,batch['class_target'].to(args.device),batch['concentration_level'].to(args.device),args)
            relation=loss*0.
            if triples is not None:
                chosen=triples[step]; references.validate(chosen)
                indices=[i for triple in chosen for i in triple['indices']]
                other=next(iter(DataLoader(Subset(wrapped[0], indices), batch_size=len(indices), num_workers=0)))
                relation_output=model(model_inputs(other,args.device))
                drugs=torch.tensor([triple['drug'] for triple in chosen],device=args.device)
                relation,_=relation_loss(relation_output['relation'],drugs)
                loss=loss+args.relation_weight*relation
            if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_norm)
            set_rates(optimizer,completed+1,total,warmup); optimizer.step(); completed+=1
            count=len(batch['class_target']); sums+=np.array([loss.item(),cl.item(),co.item(),relation.item()])*count;n+=count
            clipped=max(clipped,float(batch['image_clipped_fraction'].max()))
        validation,frame=evaluate(model,wrapped[1],args,max_batches=2 if smoke else None)
        rank=(validation['present_correct'],validation['full_profile_correct'],
              validation['ternary_all_grade_correct'],-validation['loss_total'])
        improved=best is None or rank > best
        record=dict(epoch=epoch,train_loss=float(sums[0]/n),train_class_loss=float(sums[1]/n),
            train_corn_loss=float(sums[2]/n),train_relation_loss=float(sums[3]/n),
            validation_present_accuracy=validation['present_accuracy'],validation_full_profile_accuracy=validation['full_profile_accuracy'],
            validation_ternary_accuracy=validation['ternary_full_profile_accuracy'],validation_loss=validation['loss_total'],
            max_train_clipped_fraction=clipped,backbone_frozen=freeze,seconds=time.time()-start,
            learning_rates={g['family']:g['lr'] for g in optimizer.param_groups})
        history.append(record);write_json(directory/'training_history.json',history)
        pd.DataFrame([{k:v for k,v in r.items() if k!='learning_rates'} for r in history]).to_csv(directory/'training_history.csv',index=False)
        save_checkpoint(directory/'checkpoints/latest.pt',model,scale,args,epoch,validation,optimizer,completed,audit)
        if improved:
            best=rank;stale=0
            save_checkpoint(directory/'checkpoints/best_ordinal.pt',model,scale,args,epoch,validation,optimizer,completed,audit)
            save_evaluation(directory/'validation',validation,frame)
        else: stale+=1
        print(json.dumps(record,ensure_ascii=False),flush=True)
        if not smoke and stale >= args.patience: break
    write_json(directory/'run_summary.json',dict(status='completed',smoke=smoke,
        best_checkpoint=str(directory/'checkpoints/best_ordinal.pt'),completed_updates=completed,
        test_predictions=0,selection_split='validation',selection_order='present_correct,full_profile,ternary,negative_loss'))


def check(args,module,datasets,audit):
    model,scale,details=prepare_model(args,module,datasets[0])
    wrapped=LineImageDataset(datasets[1],scale,args.cache_directory)
    batch=next(iter(loader(Subset(wrapped,range(min(args.batch_size,len(wrapped)))),args)))
    model.eval()
    with torch.no_grad():
        output=model(model_inputs(batch,args.device))
    if not all(torch.isfinite(v).all() for v in output.values()): raise RuntimeError('Non-finite forward output')
    if not (output['op'][...,1] <= output['op'][...,0]+1e-7).all(): raise RuntimeError('Ordinal ordering violated')
    directory=args.output_directory/'check';directory.mkdir(parents=True,exist_ok=True)
    Image.fromarray((batch['image'][0,0].numpy()*255).round().astype(np.uint8)).save(directory/'example_input.png')
    write_json(directory/'plot_scale.json',asdict(scale));write_json(directory/'split_audit.json',audit)
    details.update(status='PASS',image_shape=list(batch['image'].shape),output_shapes={k:list(v.shape) for k,v in output.items()},
                   parameters=sum(p.numel() for p in model.parameters()),test_predictions=0,
                   real_train=1512,real_validation=504,real_test=504)
    write_json(directory/'check_report.json',details)
    print(json.dumps({k:details[k] for k in ('status','image_shape','output_shapes','parameters','test_predictions')},ensure_ascii=False),flush=True)


def compare(args):
    if args.baseline_predictions is None or args.image_predictions is None:
        raise ValueError('Compare needs --baseline-predictions and --image-predictions')
    left=pd.read_csv(args.baseline_predictions);right=pd.read_csv(args.image_predictions)
    keys=['condition','spectrum_index','source','split']
    for name,frame in [('baseline',left),('image',right)]:
        if not set(keys).issubset(frame) or frame.duplicated(keys).any(): raise ValueError('Missing or duplicated identities in '+name)
        if set(frame['source'])!={'real'} or frame['split'].nunique()!=1: raise ValueError('Compare real rows from one split only')
    aligned=left.merge(right,on=keys,how='outer',suffixes=('_baseline','_image'),indicator=True,validate='one_to_one')
    if not aligned['_merge'].eq('both').all(): raise ValueError('Predictions cover different real spectrum identities')
    for field in ['matrix']+['true_'+d for d in DRUGS]:
        if not aligned[field+'_baseline'].equals(aligned[field+'_image']): raise ValueError('Truth or metadata differs: '+field)
    result=dict(baseline=metrics(left),image=metrics(right),split=str(left['split'].iloc[0]),samples=len(left))
    result['delta']={k:result['image'][k]-result['baseline'][k] for k in
        ('present_correct','full_profile_correct','ternary_all_grade_correct','classification_exact_correct')}
    write_json(args.output_directory/'comparison.json',result)
    print(json.dumps(result['delta'],ensure_ascii=False),flush=True)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('check','smoke','train','evaluate','export','compare'),default='check')
    p.add_argument('--project',type=Path,default=Path('/home/wqzheng/project_transformer'))
    p.add_argument('--real-root',type=Path);p.add_argument('--generated-root',type=Path)
    p.add_argument('--output-directory',type=Path,default=Path('/home/wqzheng/project_transformer/outputs/t3_32_line_image_seed2026'))
    p.add_argument('--cache-directory',type=Path)
    p.add_argument('--include-generated',action='store_true')
    p.add_argument('--maximum-generated-per-condition',type=int,default=48)
    p.add_argument('--anchor-checkpoint',type=Path,default=Path(DEFAULT_ANCHOR))
    p.add_argument('--from-scratch',action='store_true');p.add_argument('--image-only',action='store_true')
    p.add_argument('--checkpoint',type=Path);p.add_argument('--split',choices=('train','validation','test'),default='validation')
    p.add_argument('--width',type=int,default=2048);p.add_argument('--height',type=int,default=512)
    p.add_argument('--epochs',type=int,default=50);p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--num-workers',type=int,default=0);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--image-lr',type=float,default=1e-4);p.add_argument('--backbone-lr',type=float,default=1e-5)
    p.add_argument('--head-lr',type=float,default=2e-5);p.add_argument('--numeric-lr',type=float,default=1e-5)
    p.add_argument('--weight-decay',type=float,default=.01);p.add_argument('--minimum-ratio',type=float,default=.01)
    p.add_argument('--warmup-epochs',type=int,default=3);p.add_argument('--freeze-backbone-epochs',type=int,default=3)
    p.add_argument('--patience',type=int,default=12);p.add_argument('--clip-norm',type=float,default=1.)
    p.add_argument('--classification-weight',type=float,default=.5);p.add_argument('--concentration-weight',type=float,default=.5)
    p.add_argument('--ordinal-decoding',choices=('median','map'),default='median')
    p.add_argument('--boundary-weight',type=float,default=0.);p.add_argument('--boundary-margin',type=float,default=.1)
    p.add_argument('--ternary-multiplier',type=float,default=1.);p.add_argument('--relation-weight',type=float,default=0.)
    p.add_argument('--baseline-predictions',type=Path);p.add_argument('--image-predictions',type=Path)
    args=p.parse_args()
    if args.epochs<1 or args.batch_size<1 or args.num_workers<0 or args.threads<1 or args.patience<1:
        p.error('Invalid training budget')
    if args.width<1901 or args.height<64: p.error('Keep width>=1901 and height>=64 for full spectral detail')
    if args.maximum_generated_per_condition<1: p.error('Generated cap must be positive')
    if args.freeze_backbone_epochs<0 or args.warmup_epochs<0: p.error('Warmup/freeze epochs must be nonnegative')
    if any(v<=0 for v in (args.image_lr,args.backbone_lr,args.head_lr,args.numeric_lr,args.clip_norm)):
        p.error('Learning rates and clip norm must be positive')
    if not 0<args.minimum_ratio<=1 or args.weight_decay<0 or min(args.boundary_weight,args.relation_weight)<0:
        p.error('Invalid optimizer/loss settings')
    if args.device.startswith('cuda') and not torch.cuda.is_available() and args.phase not in ('export','compare'):
        p.error('CUDA requested but unavailable; use --device cpu for checks')
    return args


def main():
    args=parse_args();torch.set_num_threads(args.threads);seed_all(args.seed)
    if args.cache_directory is None: args.cache_directory=args.output_directory/'image_cache'
    if args.phase=='compare': compare(args);return
    module,datasets,audit=build_data(args)
    if args.phase=='check': check(args,module,datasets,audit)
    elif args.phase in ('smoke','train'): train(args,module,datasets,audit,smoke=args.phase=='smoke')
    elif args.phase=='evaluate':
        if args.checkpoint is None: raise ValueError('Evaluate requires --checkpoint')
        model,scale,state=load_checkpoint(args.checkpoint,args.device)
        args.ordinal_decoding=state['ordinal_decoding']
        for name in ('classification_weight','concentration_weight','boundary_weight','boundary_margin','ternary_multiplier'):
            setattr(args,name,state['experiment_config'][name])
        original=datasets[('train','validation','test').index(args.split)]
        result,frame=evaluate(model,LineImageDataset(original,scale,args.cache_directory),args)
        save_evaluation(args.output_directory/'evaluation'/args.split,result,frame)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    elif args.phase=='export':
        scale_path=args.output_directory/'check'/'plot_scale.json'
        if args.checkpoint:
            _,scale,_=load_checkpoint(args.checkpoint,'cpu')
        elif scale_path.exists(): scale=PlotScale(**json.loads(scale_path.read_text()))
        else: raise FileNotFoundError('Run --phase check first or pass --checkpoint')
        original=datasets[('train','validation','test').index(args.split)]
        wrapped=LineImageDataset(original,scale,args.cache_directory);rows=[]
        directory=args.output_directory/'images'/args.split;directory.mkdir(parents=True,exist_ok=True)
        for i in range(len(wrapped)):
            sample=wrapped[i];name=f'{i:06d}.png'
            Image.fromarray((sample['image'][0].numpy()*255).round().astype(np.uint8)).save(directory/name)
            rows.append(dict(image=name,condition=sample['condition'],source=sample['source'],
                spectrum_index=sample['spectrum_index'],split=args.split,
                image_clipped_fraction=sample['image_clipped_fraction'],
                **{'level_'+d:int(sample['concentration_level'][j]) for j,d in enumerate(DRUGS)}))
        pd.DataFrame(rows).to_csv(directory/'labels.csv',index=False)
        print(f'Exported {len(rows)} images to {directory}',flush=True)


if __name__=='__main__': main()
