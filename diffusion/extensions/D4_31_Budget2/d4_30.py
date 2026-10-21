"""Prepare and run isolated equal-budget D4.30 boundary-loss finetuning arms."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import yaml

HERE=Path(__file__).resolve().parent
VERSION='d4_31_boundary_budget2_v1'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def canonical(value):
    import numpy as np
    if isinstance(value,dict): return {str(k):canonical(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [canonical(v) for v in value]
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    return value

def verify_files(project, hashes):
    for name,digest in hashes.items():
        if sha(project/name)!=digest: raise ValueError('Source SHA256 mismatch: '+name)

def extension_hashes():
    return {name:sha(HERE/name) for name in ['d4_30.py','boundary_loss.py','bounded_prior.py','BASELINE_SHA256.json']}

def verify_extension(expected):
    if expected!=extension_hashes(): raise ValueError('D4.30 extension changed since preparation')

def prepare(a):
    import torch
    from boundary_loss import settings
    project,suite,out=[x.expanduser().resolve() for x in [a.project,a.suite,a.output]]
    if out.exists(): raise FileExistsError('Use a new experiment directory: '+str(out))
    if not a.epochs>=1 or not 0<a.learning_rate<=1e-3: raise ValueError('Invalid epochs/learning rate')
    expected=json.loads((HERE/'BASELINE_SHA256.json').read_text())
    verify_files(project,expected['files'])
    index=json.loads((suite/'run_index.json').read_text())
    verify_files(project,index.get('source_hashes',{}))
    sources={**index.get('source_hashes',{}),**expected['files']}
    config_path=Path(index['config']).expanduser()
    if not config_path.is_absolute(): config_path=suite/config_path
    if sha(config_path)!=index['config_sha256']: raise ValueError('Original config SHA256 mismatch')
    cfg=yaml.safe_load(config_path.read_text())
    cp=Path(cfg['output']['checkpoint_directory']).expanduser()
    if not cp.is_absolute(): cp=project/cp
    cp/=cfg['output'].get('best_checkpoint_name','best.pt')
    pointer=json.loads((suite/'curve2d/generation_result_n50_prior_bounds.json').read_text())
    if sha(cp)!=pointer['checkpoint_sha256']: raise ValueError('Original checkpoint SHA256 mismatch')
    checkpoint=torch.load(cp,map_location='cpu',weights_only=False)
    if checkpoint['configuration']['model']['architecture']!='dit_curve2d': raise ValueError('Expected D4.28 curve DiT')
    trained_curve=checkpoint['configuration']['curve_image']
    if cfg.get('curve_image',{}).get('state') is not None and cfg['curve_image']['state']!=trained_curve['state']:
        raise ValueError('Runtime/checkpoint curve coordinates differ')
    # Original runtime YAML predates training and may not contain the fitted state.
    # Preserve the checkpoint's fitted axes/bridge; the original trainer will independently verify it.
    cfg['curve_image']=canonical(copy.deepcopy(trained_curve))
    meta=checkpoint['metadata']
    ids=meta['relative_source_files']
    groups={name:[i for i,n in enumerate(ids) if n==name] for name in sorted(set(ids))}
    splits=[set(meta[k]) for k in ['training_indices','validation_indices','test_indices']]
    if any(splits[i]&splits[j] for i in range(3) for j in range(i)): raise ValueError('Split overlap')
    if any([sum(i in s for i in group) for s in splits]!=[12,4,4] for group in groups.values()):
        raise ValueError('Expected fixed 12/4/4 split for each condition')
    original_manifest=json.loads((Path(pointer['output_directory']).expanduser()/'generation_manifest.json').read_text())
    source=original_manifest['model_source']
    if source not in ['ema','raw']: raise ValueError('Invalid original model_source')
    loss=settings()
    real_root=Path(cfg['data']['input_directory']).expanduser()
    if not real_root.is_absolute(): real_root=project/real_root
    inputs={name:sha(real_root/name) for name in groups}
    plan=dict(version=VERSION,project=str(project),original_suite=str(suite),output=str(out),
              checkpoint=str(cp),checkpoint_sha256=sha(cp),initial_model_source=source,
              epochs=a.epochs,learning_rate=a.learning_rate,conditions=len(groups),
              extension_hashes=extension_hashes(),source_hashes=sources,
              real_root=str(real_root.resolve()),input_sha256=inputs,
              selection='latest.pt after identical epoch budget; best losses differ between arms',
              optimizer='fresh AdamW and fresh EMA in both arms; fixed learning rate; no resume')
    out.mkdir(parents=True)
    write(out/'plan.json',plan)
    for arm in ['control','boundary']:
        folder=out/arm;folder.mkdir()
        config=copy.deepcopy(cfg)
        t=config['training'];t.update(number_of_epochs=a.epochs,learning_rate=a.learning_rate,
            validate_every_epochs=1,save_every_epochs=1,maximum_validation_batches=0)
        t['learning_rate_scheduler']={'enabled':False}
        config.setdefault('project',{})['random_seed']=cfg.get('project',{}).get('random_seed',2026)
        config['generation']['number_of_spectra']=50
        # Both arms generate from EMA after the same number of updates.
        config['generation']['model_source']='ema'
        config['data']['input_directory']=str((project/Path(config['data']['input_directory']).expanduser()).resolve())
        o=config['output'];o.update(output_directory=str(folder),checkpoint_directory=str(folder/'checkpoints'),
            log_directory=str(folder/'logs'),training_log_file=str(folder/'logs/training.csv'),
            generated_spectrum_directory=str(folder/'generated'),preview_plot_directory=str(folder/'plots'))
        config['d4_30']={**plan,'arm':arm,'loss':{**loss,'weight':0. if arm=='control' else loss['weight']}}
        config['_paths']={'project_root':str(project)}
        path=folder/'runtime_config.yaml';path.write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False))
        # Compatible with the existing D4.29 diagnostic runner, while preserving original source hashes.
        write(folder/'run_index.json',dict(project=str(project),config=str(path),config_sha256=sha(path),
              source_hashes=sources,extension_hashes=plan['extension_hashes'],d4_30=plan['version']))
    print('Prepared equal-budget arms:',out)
    print('Initial model source:',source,'; epochs:',a.epochs,'; learning rate:',a.learning_rate)

def load_arm(root,arm):
    root=root.expanduser().resolve()
    plan=json.loads((root/'plan.json').read_text())
    if plan['version']!=VERSION: raise ValueError('Wrong experiment version')
    project=Path(plan['project']);verify_files(project,plan['source_hashes']);verify_extension(plan['extension_hashes'])
    for name,digest in plan['input_sha256'].items():
        if sha(Path(plan['real_root'])/name)!=digest: raise ValueError('Real input file changed: '+name)
    if sha(plan['checkpoint'])!=plan['checkpoint_sha256']: raise ValueError('Initializer SHA256 mismatch')
    folder=root/arm;index=json.loads((folder/'run_index.json').read_text());path=Path(index['config'])
    if sha(path)!=index['config_sha256']: raise ValueError('Arm configuration was changed')
    config=yaml.safe_load(path.read_text())
    for key,value in plan.items():
        if config['d4_30'].get(key)!=value:
            raise ValueError('Arm/plan provenance mismatch: '+key)
    if config['d4_30']['arm']!=arm or config['d4_30']['checkpoint_sha256']!=plan['checkpoint_sha256']:
        raise ValueError('Arm provenance mismatch')
    return plan,folder,path,config

def configure_project(project):
    sys.path.insert(0,str(project))
    loader=importlib.import_module('src.configuration_loader')
    original=loader.load_configuration
    def load(path):
        config=original(path)
        config['_paths']['project_root']=str(project)
        return config
    loader.load_configuration=load

def verify_metadata(actual,expected):
    keys=['relative_source_files','training_indices','validation_indices','test_indices',
          'normalization_state','conditioning_metadata','conditional_prior_residual_state','curve_image_state']
    keys=sorted(set(keys)|{k for k in actual if k.endswith('_state')}|{k for k in expected if k.endswith('_state')})
    for key in keys:
        if canonical(actual.get(key))!=canonical(expected.get(key)):
            raise ValueError('Fresh training preprocessing differs from initializer: '+key)

def initialized_trainer_class(original,checkpoint,experiment):
    initial=(checkpoint['diffusion_state'] if experiment['initial_model_source']=='raw'
             else checkpoint['ema_state']['ema_model'])
    class InitializedTrainer(original):
        def __init__(self,**kw):
            verify_metadata(kw['metadata'],checkpoint['metadata'])
            if kw['configuration']['curve_image']!=checkpoint['configuration']['curve_image']:
                raise ValueError('Curve configuration changed during fresh preprocessing')
            kw['diffusion'].load_state_dict(initial,strict=True)
            kw['metadata']['d4_30']={**experiment,'fresh_optimizer':True,'fresh_ema':True}
            super().__init__(**kw)
            print('D4.30: initialized '+experiment['initial_model_source']+' weights; fresh optimizer/EMA; '
                  'boundary loss weight='+str(experiment.get('loss',{}).get('weight','test')),flush=True)
    return InitializedTrainer

def train(a):
    import torch
    from boundary_loss import extension_class,FIELDS
    plan,folder,path,config=load_arm(a.experiment,a.arm)
    for p in [folder/'checkpoints',folder/'logs',folder/'training_status.json']:
        if p.exists(): raise FileExistsError('Training output already exists; use a new experiment: '+str(p))
    if not torch.cuda.is_available() and not a.cpu_test: raise RuntimeError('Training requires CUDA; no CPU fallback')
    configure_project(Path(plan['project']))
    module=importlib.import_module('src.curve_image_diffusion')
    module.CurveImageGaussianDiffusion2D=extension_class(module.CurveImageGaussianDiffusion2D,config['d4_30']['loss'])
    checkpoint=torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)
    original=importlib.import_module('src.ddpm_trainer').DdpmTrainer
    importlib.import_module('src.ddpm_trainer').DdpmTrainer=initialized_trainer_class(original,checkpoint,config['d4_30'])
    log=importlib.import_module('src.training_logger')
    log.COMPONENT_FIELDS=(*log.COMPONENT_FIELDS,*FIELDS)
    log.TrainingLogger.FIELD_NAMES=(*log.TrainingLogger.FIELD_NAMES,
        *(f'{kind}_{name}' for kind in ['training','validation'] for name in FIELDS))
    status=folder/'training_status.json';write(status,dict(status='running',arm=a.arm))
    try:
        script=importlib.import_module('scripts.start_ddpm_training')
        sys.argv=['start_ddpm_training','--config',str(path)]
        script.main()
        latest=folder/'checkpoints/latest.pt'
        if not latest.is_file(): raise RuntimeError('Training did not produce latest.pt')
        write(status,dict(status='complete',checkpoint=str(latest),checkpoint_sha256=sha(latest)))
    except BaseException as e:
        write(status,dict(status='failed',error=f'{type(e).__name__}: {e}'));raise

def generate(a):
    import torch
    plan,folder,path,config=load_arm(a.experiment,a.arm)
    status=json.loads((folder/'training_status.json').read_text())
    if status['status']!='complete': raise ValueError('Training did not complete')
    cp=folder/'checkpoints/latest.pt'
    if sha(cp)!=status['checkpoint_sha256']: raise ValueError('Latest checkpoint changed')
    if not torch.cuda.is_available() and not a.cpu_test: raise RuntimeError('Generation requires CUDA')
    if (folder/'generated').exists() or (folder/'generation_status.json').exists():
        raise FileExistsError('Generation output exists; refusing overwrite')
    configure_project(Path(plan['project']))
    state=folder/'generation_status.json';write(state,dict(status='running'))
    try:
        from bounded_prior import bounded_prior_context
        from src.conditional_prior_residual import ConditionalPriorResidualBank
        from src.spectrum_conditioning import source_named_conditions_from_metadata
        saved=torch.load(cp,map_location='cpu',weights_only=False)
        fixed=saved['configuration']['curve_image']['state']
        names={cid:name for name,cid,_ in source_named_conditions_from_metadata(saved['metadata']['conditioning_metadata'])}
        del saved
        script=importlib.import_module('scripts.generate_conditional_spectra')
        sys.argv=['generate_conditional_spectra','--config',str(path),'--checkpoint',str(cp),
                  '--condition','all','--number','50','--device','cpu' if a.cpu_test else 'cuda',
                  '--output-directory',str(folder/'generated'),'--save-stage-trace']
        generated=folder/'generated';generated.mkdir()
        with bounded_prior_context(ConditionalPriorResidualBank,fixed['y_min'],fixed['y_max'],
                max_redraws=100,names=names,callback=lambda x:write(generated/'prior_bounds_audit.json',x)) as sampler:
            try:
                script.main()
            finally:
                write(generated/'prior_bounds_audit.json',sampler.report())
        manifest=json.loads((generated/'generation_manifest.json').read_text())
        if (manifest.get('number_of_spectra_per_condition')!=50 or manifest.get('model_source')!='ema'
                or {x['internal_condition_id'] for x in manifest.get('conditions',[])}!=set(names)):
            raise ValueError('Generation manifest does not match the requested experiment')
        for row in manifest['conditions']:
            if not row.get('output_files') or not all(Path(p).is_file() and Path(p).stat().st_size>0 for p in row['output_files']):
                raise ValueError('Missing generated output files')
        (folder/'curve2d').mkdir(exist_ok=True)
        write(folder/'curve2d/generation_result_n50_prior_bounds.json',dict(
            output_directory=str(folder/'generated'),checkpoint_sha256=sha(cp)))
        # D4.29 resolves the diagnostic checkpoint from the runtime config's best name.
        # Use an explicit diagnostic-only config pointing at latest.pt; preserve training config.
        diag=copy.deepcopy(config);diag['output']['best_checkpoint_name']='latest.pt'
        dp=folder/'diagnostic_config.yaml';dp.write_text(yaml.safe_dump(diag,allow_unicode=True,sort_keys=False))
        index=json.loads((folder/'run_index.json').read_text())
        index.update(config=str(dp),config_sha256=sha(dp))
        write(folder/'diagnostic_run_index.json',index)
        audit=folder/'diagnostic_suite';audit.mkdir();(audit/'curve2d').mkdir()
        write(audit/'run_index.json',index)
        write(audit/'curve2d/generation_result_n50_prior_bounds.json',dict(
            output_directory=str(folder/'generated'),checkpoint_sha256=sha(cp)))
        write(state,dict(status='complete',checkpoint_sha256=sha(cp),number_per_condition=50))
    except BaseException as e:
        write(state,dict(status='failed',error=f'{type(e).__name__}: {e}'));raise

def compare(a):
    import pandas as pd
    import numpy as np
    root=a.experiment.expanduser().resolve();frames={};manifests={}
    for arm,audit in [('control',a.control_audit),('boundary',a.boundary_audit)]:
        plan,folder,_,_=load_arm(root,arm)
        audit=audit.expanduser().resolve()
        m=json.loads((audit/'run_manifest.json').read_text())
        status=json.loads((folder/'training_status.json').read_text())
        if (status['status']!='complete' or sha(folder/'checkpoints/latest.pt')!=status['checkpoint_sha256']
                or m['status']!='complete' or m['checkpoint_sha256']!=status['checkpoint_sha256']):
            raise ValueError('Audit/checkpoint mismatch: '+arm)
        if Path(m['generated_root']).resolve()!=(folder/'generated').resolve():
            raise ValueError('Wrong generated directory: '+arm)
        inventory=json.loads((audit/'inventory.json').read_text())
        if len(inventory)!=plan['conditions'] or any(x['delivered_count']!=50 for x in inventory):
            raise ValueError('Expected all conditions with exactly 50 spectra each')
        p=pd.read_csv(audit/'postprocessing_stages.csv')
        b=pd.read_csv(audit/'patch_boundaries.csv')
        p=p.merge(b[['condition','stage','seam_to_nonseam_ratio']],on=['condition','stage'],validate='one_to_one')
        frames[arm]=p.set_index(['condition','stage']).sort_index();manifests[arm]=m
    for key in ['reference_split','test_evaluation','real_input_sha256','fixed_coordinates']:
        if manifests['control'][key]!=manifests['boundary'][key]: raise ValueError('Audits differ: '+key)
    left,right=frames['control'],frames['boundary']
    if not left.index.equals(right.index): raise ValueError('Audit condition/stage sets differ')
    metrics=['nearest_derivative_pearson_mean','nearest_pearson_mean','nearest_model_normalized_mse_mean',
             'mean_spectrum_mse','pairwise_mse_diversity_ratio_to_train','seam_to_nonseam_ratio',
             'pooled_intensity_span_ratio_lower_01_05','pooled_intensity_span_ratio_upper_95_99']
    delta=right[metrics]-left[metrics]
    results={}
    for stage in ['01_ddpm_reconstruction','07_terminal']:
        l=left.xs(stage,level='stage');r=right.xs(stage,level='stage');d=delta.xs(stage,level='stage')
        results[stage]={k:dict(control_mean=float(l[k].mean()),boundary_mean=float(r[k].mean()),
            mean_change=float(d[k].mean()),positive_conditions=int((d[k]>1e-9).sum()),
            negative_conditions=int((d[k]< -1e-9).sum())) for k in metrics}
    out=root/'comparison'
    if out.exists(): raise FileExistsError('Comparison exists: '+str(out))
    out.mkdir();delta.to_csv(out/'condition_changes_boundary_minus_control.csv',encoding='utf-8-sig')
    report=dict(status='comparison_complete',conditions=plan['conditions'],spectra_per_condition=50,
                equal_budget=True,independent_training_seeds=1,automatic_winner=False,results=results,
                notes=['Same initial weights and epoch budget; validation metrics only.',
                       'Pooled intensity spans are descriptive, not the original QQ estimator.',
                       'Positive derivative changes are favorable; negative MSE/seam-ratio changes are favorable.',
                       'One training seed is an engineering comparison, not a significance test.'])
    write(out/'comparison.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print('Comparison saved:',out)

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--project',type=Path,required=True);q.add_argument('--suite',type=Path,required=True)
    q.add_argument('--output',type=Path,required=True);q.add_argument('--epochs',type=int,default=5)
    q.add_argument('--learning-rate',type=float,default=1e-5)
    for name in ['train','generate']:
        q=sub.add_parser(name);q.add_argument('--experiment',type=Path,required=True)
        q.add_argument('--arm',choices=['control','boundary'],required=True)
        q.add_argument('--cpu-test',action='store_true',help=argparse.SUPPRESS)
    q=sub.add_parser('compare');q.add_argument('--experiment',type=Path,required=True)
    q.add_argument('--control-audit',type=Path,required=True);q.add_argument('--boundary-audit',type=Path,required=True)
    a=p.parse_args();globals()[a.command](a)

if __name__=='__main__': main()
