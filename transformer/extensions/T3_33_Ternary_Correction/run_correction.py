"""T3.33: verify epoch-16 baseline, cache frozen features and train DEL/TEB correction."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from baseline_bridge import configure,load_baseline,state_digest,assert_reference,input_fingerprint,sha256
from correction_model import TernaryCorrector,VERSION,correction_loss,protection_check
from feature_cache import build_cache,CachedTraining,tensor_inputs,evaluate_cached,prediction_frame,KEYS

BASELINE=Path('/home/wqzheng/project_transformer/outputs/t3_32_line_image_seed2026/train/checkpoints/best_ordinal.pt')


def rank(metrics):
    return (metrics['ternary_all_grade_correct'],metrics['full_profile_correct'],
            metrics['present_correct'],-metrics['loss_total'])


def admissible(metrics,reference):
    # No regression in overall grades/profiles; class and CHL outputs are protected separately.
    return all(metrics[key]>=reference[key] for key in
        ('present_correct','full_profile_correct','classification_exact_correct','ternary_all_grade_correct'))


def configure_data(args,state):
    from bridge import build_data
    config=state['experiment_config']
    baseline_generated=bool(config.get('include_generated',False))
    baseline_cap=int(config.get('maximum_generated_per_condition',48))
    args.include_generated=baseline_generated
    args.maximum_generated_per_condition=baseline_cap
    _,datasets,audit=build_data(args)
    for split in ('train','validation','test'):
        expected=state['split_audit'][split]['identity_sha256']
        if audit[split]['identity_sha256']!=expected:
            raise RuntimeError('Dataset split/subset differs from T3.32: '+split)
    return datasets,audit


def baseline_summary(state,path):
    return dict(checkpoint_sha256=sha256(path),epoch=state['epoch'],checkpoint_version=state['checkpoint_version'],
        model_config=state['model_config'],plot_scale=state['plot_scale'],
        validation_metrics=state['validation_metrics'],ordinal_decoding=state['ordinal_decoding'])


def setup(args,runtime):
    baseline,scale,state=load_baseline(runtime,args.baseline,args.device)
    args.ordinal_decoding=state['ordinal_decoding']
    datasets,audit=configure_data(args,state)
    reference=baseline_summary(state,args.baseline)
    fingerprint,inputs=input_fingerprint(datasets,audit,reference['checkpoint_sha256'],args.t332_hashes)
    runtime.write_json(args.output_directory/'input_audit.json',inputs)
    digest=state_digest(baseline)
    train_cache=build_cache(baseline,datasets[0],scale,args,fingerprint,runtime)
    val_cache=build_cache(baseline,datasets[1],scale,args,fingerprint,runtime)
    frame=prediction_frame(val_cache,val_cache['cp'],val_cache['op'],args.ordinal_decoding)
    baseline_metrics=runtime.metrics(frame)
    assert_reference(baseline_metrics,state['validation_metrics'],'Epoch-16 validation reproduction failed')
    runtime.save_evaluation(args.output_directory/'baseline_validation',baseline_metrics,frame)
    model=TernaryCorrector(baseline,hidden=args.hidden,bound=args.bound,dropout=args.dropout).to(args.device)
    real=torch.tensor([row['source']=='real' and 0<=row['spectrum_index']<12 for row in train_cache['metadata']])
    fit=model.fit_normalization(train_cache['features'].to(args.device),real.to(args.device),
        (train_cache['cp']>=.5).all(1).to(args.device),split='train')
    result,frame=evaluate_cached(model,val_cache,args,runtime)
    assert_reference(result,baseline_metrics,'Zero-initialized correction changed baseline metrics')
    report=dict(status='PASS',baseline=reference,split_audit=audit,normalization=fit,
        baseline_digest=digest,cache_fingerprint=fingerprint,feature_dimensions=model.config['feature_dim'],
        correction_parameters=sum(p.numel() for p in model.correction_heads.parameters()),
        baseline_trainable_parameters=sum(p.numel() for p in model.baseline.parameters() if p.requires_grad),
        zero_correction_identity='PASS',test_features=0,test_predictions=0)
    runtime.write_json(args.output_directory/'check_report.json',report)
    if state_digest(model.baseline)!=digest:raise RuntimeError('Baseline changed during feature extraction')
    return model,scale,reference,train_cache,val_cache,report


def checkpoint(path,model,scale,reference,report,args,epoch,metrics,optimizer,update):
    state=dict(checkpoint_version=VERSION,correction_config=model.config,baseline_reference=reference,
        model_state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        plot_scale=asdict(scale),epoch=epoch,validation_metrics=metrics,ordinal_decoding=args.ordinal_decoding,
        source_hashes={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},t332_hashes=args.t332_hashes,
        experiment_config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        cache_fingerprint=report['cache_fingerprint'],split_audit=report['split_audit'],
        baseline_digest=report['baseline_digest'],selection_role=Path(path).stem,
        optimizer_state_dict=None if optimizer is None else optimizer.state_dict(),
        completed_updates=update,test_predictions=0,selection_scope='validation, already reused across prior experiments')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');torch.save(state,temporary);temporary.replace(path)


def load_correction(path,args):
    from image_model import ImageSERSFormer
    from image_data import PlotScale
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state.get('checkpoint_version')!=VERSION:raise ValueError('Expected T3.33 correction checkpoint')
    if state['t332_hashes']!=args.t332_hashes:raise RuntimeError('Installed T3.32 code differs from checkpoint dependency')
    baseline=ImageSERSFormer(**state['baseline_reference']['model_config'])
    model=TernaryCorrector(baseline,**state['correction_config'])
    model.load_state_dict(state['model_state_dict'],strict=True)
    model=model.to(args.device).eval()
    if state_digest(model.baseline)!=state['baseline_digest']:raise RuntimeError('Embedded frozen baseline digest differs')
    return model,PlotScale(**state['plot_scale']),state


def make_optimizer(model,args):
    parameters=[p for p in model.correction_heads.parameters() if p.requires_grad]
    intended={id(p) for p in model.parameters() if p.requires_grad}
    if intended!={id(p) for p in parameters}:raise RuntimeError('Only correction heads may be trainable')
    return torch.optim.AdamW([dict(params=parameters,lr=args.learning_rate,maximum_lr=args.learning_rate,
        minimum_lr=args.minimum_learning_rate,weight_decay=args.weight_decay)])


def training(args,runtime,prepared,smoke=False):
    model,scale,reference,cache,val_cache,report=prepared
    directory=args.output_directory/('smoke' if smoke else 'train')
    if (directory/'checkpoints').exists():raise FileExistsError('Do not overwrite a previous run: '+str(directory))
    directory.mkdir(parents=True,exist_ok=True)
    dataset=CachedTraining(cache)
    counts={source:sum(cache['metadata'][int(i)]['source']==source for i in dataset.indices) for source in ('real','generated')}
    runtime.write_json(directory/'training_config.json',dict(correction_config=model.config,
        selected_true_and_predicted_ternary_rows=len(dataset),source_counts=counts,
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},baseline=reference,
        freeze='all baseline parameters and buffers; baseline always eval'))
    loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True,num_workers=0,
        generator=torch.Generator().manual_seed(args.seed))
    optimizer=make_optimizer(model,args)
    epochs=1 if smoke else args.epochs
    steps=min(len(loader),2) if smoke else len(loader)
    total=epochs*steps;warmup=min(args.warmup_epochs*steps,max(total-1,0))
    baseline_metrics=reference['validation_metrics']
    initial,frame=evaluate_cached(model,val_cache,args,runtime)
    accepted_rank=rank(initial);ternary_rank=rank(initial);overall_rank=(initial['present_correct'],initial['full_profile_correct'],-initial['loss_total'])
    # Always provide an explicitly marked epoch-0 baseline fallback.
    checkpoint(directory/'checkpoints/best_accepted.pt',model,scale,reference,report,args,0,initial,optimizer,0)
    checkpoint(directory/'checkpoints/best_ternary.pt',model,scale,reference,report,args,0,initial,optimizer,0)
    checkpoint(directory/'checkpoints/best_overall.pt',model,scale,reference,report,args,0,initial,optimizer,0)
    runtime.save_evaluation(directory/'accepted_validation',initial,frame)
    history=[];completed=0;stale=0;accepted_epoch=0
    for epoch in range(1,epochs+1):
        model.train();start=time.time();sum_loss=0.;rows=0
        for step,batch in enumerate(loader):
            if step>=steps:break
            inputs=tensor_inputs(batch,args.device);optimizer.zero_grad(set_to_none=True)
            output=model.forward_cached(inputs);protection_check(output,inputs)
            loss,details=correction_loss(output,batch['levels'].to(args.device),batch['presence'].to(args.device),
                error_multiplier=args.error_multiplier,delta_weight=args.delta_weight,stability_weight=args.stability_weight)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite correction loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.correction_heads.parameters(),1.)
            runtime.set_rates(optimizer,completed+1,total,warmup);optimizer.step();completed+=1
            sum_loss+=float(loss.detach())*len(batch['levels']);rows+=len(batch['levels'])
            if any(p.grad is not None for p in model.baseline.parameters()):raise RuntimeError('Gradient reached frozen baseline')
        metrics,frame=evaluate_cached(model,val_cache,args,runtime)
        if state_digest(model.baseline)!=report['baseline_digest']:raise RuntimeError('Baseline weights/buffers changed')
        acceptable=admissible(metrics,baseline_metrics)
        improved=acceptable and metrics['ternary_all_grade_correct']>baseline_metrics['ternary_all_grade_correct'] and rank(metrics)>accepted_rank
        if improved:
            accepted_rank=rank(metrics);accepted_epoch=epoch;stale=0
            checkpoint(directory/'checkpoints/best_accepted.pt',model,scale,reference,report,args,epoch,metrics,optimizer,completed)
            runtime.save_evaluation(directory/'accepted_validation',metrics,frame)
        elif epoch>args.warmup_epochs:stale+=1
        if rank(metrics)>ternary_rank:
            ternary_rank=rank(metrics)
            checkpoint(directory/'checkpoints/best_ternary.pt',model,scale,reference,report,args,epoch,metrics,optimizer,completed)
            runtime.save_evaluation(directory/'ternary_validation',metrics,frame)
        overall=(metrics['present_correct'],metrics['full_profile_correct'],-metrics['loss_total'])
        if overall>overall_rank:
            overall_rank=overall
            checkpoint(directory/'checkpoints/best_overall.pt',model,scale,reference,report,args,epoch,metrics,optimizer,completed)
        checkpoint(directory/'checkpoints/latest.pt',model,scale,reference,report,args,epoch,metrics,optimizer,completed)
        row=dict(epoch=epoch,train_loss=sum_loss/rows,validation_loss=metrics['loss_total'],
            present_correct=metrics['present_correct'],full_profile_correct=metrics['full_profile_correct'],
            ternary_correct=metrics['ternary_all_grade_correct'],classification_exact_correct=metrics['classification_exact_correct'],
            accepted=improved,passes_no_regression_guard=acceptable,learning_rate=optimizer.param_groups[0]['lr'],
            seconds=time.time()-start,output_protection='PASS',baseline_unchanged=True)
        history.append(row);runtime.write_json(directory/'training_history.json',history)
        import pandas as pd
        pd.DataFrame(history).to_csv(directory/'training_history.csv',index=False)
        print(json.dumps(row,ensure_ascii=False),flush=True)
        if not smoke and stale>=args.patience:break
    accepted,_,saved=load_correction(directory/'checkpoints/best_accepted.pt',args)
    metrics,frame=evaluate_cached(accepted,val_cache,args,runtime)
    runtime.save_evaluation(directory/'accepted_validation',metrics,frame)
    delta={key:metrics[key]-baseline_metrics[key] for key in
        ('present_correct','full_profile_correct','ternary_all_grade_correct','classification_exact_correct')}
    runtime.write_json(directory/'comparison.json',dict(baseline=baseline_metrics,accepted=metrics,delta=delta,
        accepted_epoch=saved['epoch'],status='improved' if delta['ternary_all_grade_correct']>0 else 'baseline retained',
        split='validation',test_predictions=0))
    runtime.write_json(directory/'run_summary.json',dict(status='completed',smoke=smoke,completed_updates=completed,
        accepted_epoch=saved['epoch'],best_accepted=str(directory/'checkpoints/best_accepted.pt'),
        baseline_unchanged=True,test_predictions=0))
    print('Accepted checkpoint comparison: '+json.dumps(delta),flush=True)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('check','smoke','train','evaluate','pack'),default='check')
    p.add_argument('--image-package',type=Path,default=Path('/home/wqzheng/T3_32_Image_Recognition'))
    p.add_argument('--baseline',type=Path,default=BASELINE)
    p.add_argument('--project',type=Path,default=Path('/home/wqzheng/project_transformer'))
    p.add_argument('--real-root',type=Path);p.add_argument('--generated-root',type=Path)
    p.add_argument('--output-directory',type=Path,default=Path('/home/wqzheng/project_transformer/outputs/t3_33_ternary_correction_seed2026'))
    p.add_argument('--image-cache',type=Path,default=Path('/home/wqzheng/project_transformer/outputs/t3_32_line_image_seed2026/image_cache'))
    p.add_argument('--checkpoint',type=Path);p.add_argument('--split',choices=('train','validation','test'),default='validation')
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--cache-batch-size',type=int,default=16);p.add_argument('--num-workers',type=int,default=2)
    p.add_argument('--device',default='cuda');p.add_argument('--threads',type=int,default=2)
    p.add_argument('--seed',type=int,default=2026);p.add_argument('--hidden',type=int,default=32)
    p.add_argument('--bound',type=float,default=.5);p.add_argument('--dropout',type=float,default=.1)
    p.add_argument('--learning-rate',type=float,default=1e-4);p.add_argument('--minimum-learning-rate',type=float,default=1e-6)
    p.add_argument('--weight-decay',type=float,default=.01);p.add_argument('--warmup-epochs',type=int,default=3)
    p.add_argument('--patience',type=int,default=5);p.add_argument('--error-multiplier',type=float,default=2.)
    p.add_argument('--delta-weight',type=float,default=.01);p.add_argument('--stability-weight',type=float,default=.05)
    args=p.parse_args()
    if min(args.epochs,args.batch_size,args.cache_batch_size,args.threads,args.patience)<1 or args.num_workers<0:
        p.error('Invalid training/loader budget')
    if not 0<args.minimum_learning_rate<=args.learning_rate or args.weight_decay<0 or args.warmup_epochs<0:
        p.error('Invalid learning-rate schedule')
    if not 0<args.bound<=1 or args.hidden<4 or not 0<=args.dropout<1 or args.error_multiplier<1 or min(args.delta_weight,args.stability_weight)<0:
        p.error('Invalid correction settings')
    if args.device.startswith('cuda') and not torch.cuda.is_available() and args.phase!='pack':
        p.error('CUDA unavailable; use --device cpu for checks')
    return args


def main():
    args=parse_args()
    if args.phase=='pack':
        import zipfile
        source=args.output_directory;target=source.parent/(source.name+'_reports.zip')
        if not source.exists():raise FileNotFoundError(source)
        with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for path in source.rglob('*'):
                if path.is_file() and path.suffix in ('.json','.csv','.log','.png') and not any(x in path.parts for x in ('feature_cache','image_cache')):
                    z.write(path,arcname=str(path.relative_to(source)))
        print(target);return
    runtime,hashes=configure(args.image_package);args.t332_hashes=hashes
    torch.set_num_threads(args.threads);runtime.seed_all(args.seed)
    args.output_directory.mkdir(parents=True,exist_ok=True)
    if args.phase=='evaluate':
        if args.checkpoint is None:raise ValueError('Evaluation requires --checkpoint')
        model,scale,state=load_correction(args.checkpoint,args)
        args.ordinal_decoding=state['ordinal_decoding']
        # Use embedded baseline metadata rather than requiring the original .pt for deployment.
        baseline_state=dict(experiment_config=state['experiment_config'],split_audit=state['split_audit'])
        datasets,audit=configure_data(args,baseline_state)
        fingerprint,_=input_fingerprint(datasets,audit,state['baseline_reference']['checkpoint_sha256'],hashes)
        original=datasets[('train','validation','test').index(args.split)]
        cache=build_cache(model.baseline,original,scale,args,fingerprint,runtime)
        metrics,frame=evaluate_cached(model,cache,args,runtime)
        runtime.save_evaluation(args.output_directory/'evaluation'/args.split,metrics,frame)
        print(json.dumps(metrics,ensure_ascii=False));return
    prepared=setup(args,runtime)
    if args.phase=='check':
        model,_,reference,train_cache,_,report=prepared
        dataset=CachedTraining(train_cache)
        optimizer=make_optimizer(model,args)
        sample=next(iter(DataLoader(dataset,batch_size=min(args.batch_size,len(dataset)))))
        before=state_digest(model.baseline);model.train()
        output=model.forward_cached(tensor_inputs(sample,args.device))
        loss,_=correction_loss(output,sample['levels'].to(args.device),sample['presence'].to(args.device))
        loss.backward()
        if not any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.correction_heads.parameters()):
            raise RuntimeError('Correction head received no gradient')
        if any(p.grad is not None for p in model.baseline.parameters()):raise RuntimeError('Baseline received gradient')
        optimizer.zero_grad(set_to_none=True)
        if state_digest(model.baseline)!=before:raise RuntimeError('Baseline changed during gradient check')
        print(json.dumps(dict(status='PASS',baseline_epoch=16,reference_counts={k:reference['validation_metrics'][k] for k in
            ('present_correct','full_profile_correct','ternary_all_grade_correct')},
            correction_parameters=report['correction_parameters'],frozen_gradients=0,test_predictions=0),ensure_ascii=False),flush=True)
    else:training(args,runtime,prepared,smoke=args.phase=='smoke')


if __name__=='__main__':main()
