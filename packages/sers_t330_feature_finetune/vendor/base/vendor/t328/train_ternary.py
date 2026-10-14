"""T3.28: train a ternary joint head from T3.27 Adapter epoch 6."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from base_bridge import ROOT, DEFAULT_ANCHOR, EXPECTED, load_adapter, previous, rt
from ternary_model import TernaryModel, ARMS, CONTROLS, training_loss
from selection_policy import acceptance as protected_acceptance
from schedule import UpdateSchedule
from sam_update import backward_sam

VERSION = 'T3.28_ternary_joint_v1'
MAIN_TRAIN_COUNT = 7560
TERNARY_TRAIN_COUNT = 3240


def verify_package():
    manifest = json.loads((ROOT/'MANIFEST.json').read_text())
    for name, digest in manifest.items():
        if rt.sha256(ROOT/name) != digest:
            raise RuntimeError('Package integrity failed: '+name)
    return manifest


def snapshot(project, checkpoint):
    files = []
    for source in ('real','generated'):
        root = (project/'data'/source).resolve(strict=True)
        selected = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in ('.csv','.xls','.xlsx'))
        if not selected:raise FileNotFoundError(f'No spectrum files in {root}')
        files.extend(selected)
    sources = json.loads((ROOT/'vendor/t327/archived/expected_project_sources.json').read_text())
    files.extend(project/name for name in sources)
    files.append(checkpoint)
    return {str(p.resolve()):rt.sha256(p) for p in files}


def options(args, arm):
    return dict(arm=arm, hidden=args.hidden, dropout=args.dropout, bound=args.residual_bound)


def construct(payload, opts, device):
    return TernaryModel(payload['model_config'], payload['optimization_spec'], opts).initialize(payload).to(device)


def rank(metrics):
    return (metrics['ternary_all_grade_correct'], metrics['by_mixture']['3']['present_correct'],
            metrics['full_profile_correct'], metrics['present_correct'], -metrics['present_mae'])


def adoption(metrics, anchor):
    result = protected_acceptance(metrics, anchor)
    rules = dict(result['rules'])
    rules['ternary_strict_improvement'] = metrics['ternary_all_grade_correct'] > anchor['ternary_all_grade_correct']
    for matrix, value in anchor['by_matrix'].items():
        rules[f'{matrix}_ternary_not_lower'] = (metrics['by_matrix'][matrix]['ternary_all_grade_correct'] >=
                                               value['ternary_all_grade_correct'])
    return dict(accepted=all(rules.values()), rules=rules,
                failed_rules=[name for name,value in rules.items() if not value])


def optimizer(model, args):
    groups={}
    weight_ids=set()
    for module in model.modules():
        if isinstance(module,torch.nn.Linear) and module._parameters.get('weight') is not None:
            weight_ids.add(id(module.weight))
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:continue
        head=name.startswith('joint_head.')
        maximum=args.head_lr if head else args.adapter_lr
        # Existing Adapter weights receive an anchored L2-SP penalty only in its arm,
        # rather than simultaneous decay towards zero.
        decay=args.weight_decay if head and id(parameter) in weight_ids else 0.
        key=(maximum,decay,'head' if head else 'adapter')
        group=groups.setdefault(key,dict(params=[],names=[],lr=maximum,maximum_lr=maximum,
                minimum_lr=maximum*args.minimum_ratio,weight_decay=decay,family=key[2]))
        group['params'].append(parameter);group['names'].append(name)
    values=list(groups.values())
    identities=[id(p) for g in values for p in g['params']]
    expected={id(p) for p in model.parameters() if p.requires_grad}
    if len(identities)!=len(set(identities)) or set(identities)!=expected:
        raise RuntimeError('Optimizer parameter coverage is invalid')
    return torch.optim.AdamW(values,betas=(.9,.999),eps=1e-8)


@torch.no_grad()
def evaluate(model, dataset, args):
    model.eval();rows=[];loss_total=0.;ternary_n=0
    for batch in rt.make_loader(dataset,args.batch_size,args.num_workers):
        output=model(rt.model_inputs(dataset,batch,args.device))
        cp,grade,op=model.predict(output)
        classes=batch['class_target'].to(args.device)
        levels=batch['concentration_target'].to(args.device)
        loss,parts=training_loss(model,output,classes,levels,args.marginal_weight,
                                 0.,0.)
        loss_total+=float(loss)*parts['ternary_samples'];ternary_n+=parts['ternary_samples']
        cp,grade,op=cp.cpu().numpy(),grade.cpu().numpy(),op.cpu().numpy()
        route=output['route'].cpu().numpy()
        for index,row in enumerate(rt.metadata_rows(dataset,batch)):
            row['ternary_route']=bool(route[index])
            row['decoder']='anchor_median' if model.arm=='anchor' or not route[index] else 'joint_MAP'
            for j,drug in enumerate(('DEL','CHL','TEB')):
                row['ordinal_'+drug]=int(grade[index,j])
                row['pred_'+drug]=int(grade[index,j]) if cp[index,j]>=.5 else 0
                row['presence_probability_'+drug]=float(cp[index,j])
                row['P_ge_M_'+drug]=float(op[index,j,0]);row['P_ge_H_'+drug]=float(op[index,j,1])
            rows.append(row)
    frame=pd.DataFrame(rows)
    metrics=previous.enrich(rt.summarize(frame),frame)
    metrics['validation_joint_loss']=loss_total/max(1,ternary_n)
    metrics['routing']={str(k):dict(samples=int((frame.mixture_count==k).sum()),
        routed=int(frame.loc[frame.mixture_count==k,'ternary_route'].sum())) for k in (1,2,3)}
    return metrics,frame


def save_checkpoint(path, model, epoch, metrics, config, optimizer_value=None, schedule=None, selection=None):
    state=dict(checkpoint_version=VERSION,model_config=model.model_config,
        adapter_spec=model.adapter_spec,ternary_options=model.options,
        model_state_dict={name:value.detach().cpu().clone() for name,value in model.state_dict().items()},
        epoch=epoch,validation_metrics=metrics,training_config=config,selection=selection,test_predictions=0)
    if optimizer_value is not None:
        state.update(optimizer_state_dict=optimizer_value.state_dict(),schedule_state=schedule.state_dict(),
                     rng_state=rt.capture_rng())
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(str(path)+'.tmp');torch.save(state,temporary);temporary.replace(path)


def load_checkpoint(path, device='cpu'):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state.get('checkpoint_version')!=VERSION:raise ValueError('Expected a T3.28 checkpoint')
    model=TernaryModel(state['model_config'],state['adapter_spec'],state['ternary_options'])
    model.load_state_dict(state['model_state_dict'],strict=True)
    return model.to(device).eval(),state


def run_check(args, datasets, audit, sources, package):
    folder=args.suite/'check';folder.mkdir(exist_ok=False)
    _,payload=load_adapter(args.checkpoint)
    anchor=construct(payload,options(args,'anchor'),args.device)
    baseline,frame=evaluate(anchor,datasets[1],args)
    for key,value in EXPECTED.items():
        if baseline[key]!=value:raise RuntimeError(f'Adapter reproduction failed: {key}={baseline[key]} expected {value}')
    if baseline['by_mixture']['3']['present_correct']!=512:
        raise RuntimeError('Adapter ternary component grades do not reproduce')
    frame.to_csv(folder/'anchor_validation.csv',index=False)
    models={}
    for arm in ('independent_map',)+ARMS:
        rt.seed_everything(args.seed)
        model=construct(payload,options(args,arm),args.device)
        metrics,predictions=evaluate(model,datasets[1],args)
        predictions.to_csv(folder/f'initial_{arm}_validation.csv',index=False)
        if not np.array_equal(frame[[f'presence_probability_{d}' for d in ('DEL','CHL','TEB')]].values,
                              predictions[[f'presence_probability_{d}' for d in ('DEL','CHL','TEB')]].values):
            raise RuntimeError('Frozen classification predictions changed')
        nonroute=~predictions.ternary_route
        columns=[f'pred_{d}' for d in ('DEL','CHL','TEB')]
        if not frame.loc[nonroute,columns].equals(predictions.loc[nonroute,columns]):
            raise RuntimeError('Non-routed predictions changed')
        if arm!='independent_map':
            initial_map=pd.read_csv(folder/'initial_independent_map_validation.csv')
            if not predictions[columns].equals(initial_map[columns]):
                raise RuntimeError('Zero residual must match the independent MAP control')
        models[arm]=dict(options=options(args,arm),initial_metrics=metrics,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            adoption=adoption(metrics,baseline))
        if arm=='independent_map':
            save_checkpoint(folder/'independent_map.pt',model,0,metrics,dict(control=True))
        del model
    result=dict(status='PASS',baseline=baseline,models=models,dataset_audit=audit,
        source_sha256=sources,package_sha256=package,anchor_sha256=rt.sha256(args.checkpoint),
        anchor_path=str(args.checkpoint),anchor_epoch=6,test_predictions=0)
    rt.save_json(folder/'check_summary.json',result)
    print('CHECK PASS: Adapter epoch 6, grades=981 full=376 ternary=121',flush=True)
    print('Independent MAP is a decoder control; zero residual changes no probabilities but can change median labels.',flush=True)


def train_config(args,arm,smoke,check):
    return dict(arm=arm,options=options(args,arm),epochs=2 if smoke else args.epochs,
        warmup_epochs=1 if smoke else args.warmup_epochs,batch_size=args.batch_size,
        seed=args.seed,num_workers=args.num_workers,device=args.device,head_lr=args.head_lr,
        adapter_lr=args.adapter_lr,weight_decay=args.weight_decay,minimum_ratio=args.minimum_ratio,
        clip_norm=args.clip_norm,marginal_weight=args.marginal_weight,
        rdrop_weight=args.rdrop_weight if arm=='joint_rdrop' else 0.,
        l2sp_weight=args.l2sp_weight if arm=='joint_adapter_l2sp' else 0.,
        sam_rho=args.sam_rho if arm=='joint_sam' else 0.,
        precision='fp32',smoke=smoke,max_batches=4 if smoke else None,
        anchor_sha256=check['anchor_sha256'],dataset_audit=check['dataset_audit'],
        package_sha256=check['package_sha256'],source_sha256=check['source_sha256'])


def run_train(args,datasets,audit,sources,package,smoke=False):
    check=json.loads((args.suite/'check/check_summary.json').read_text())
    if check['status']!='PASS' or check['package_sha256']!=package or check['dataset_audit']!=audit or check['source_sha256']!=sources or check['anchor_sha256']!=rt.sha256(args.checkpoint):
        raise RuntimeError('Check does not match this package/dataset/anchor')
    _,payload=load_adapter(args.checkpoint)
    train,val,_=datasets
    if smoke:val=rt.smoke_validation_view(val,32)
    selected=ARMS if args.arm=='all' else (args.arm,)
    for arm in selected:
        if options(args,arm)!=check['models'][arm]['options']:
            raise ValueError('Head options differ from check; use a new suite')
        folder=args.suite/('smoke' if smoke else 'train')/arm
        last=folder/'checkpoints/last.pt'
        if folder.exists() and not args.resume:raise FileExistsError(f'{folder}; use --resume or a new suite')
        if args.resume and folder.exists() and not last.exists():raise FileNotFoundError(f'No epoch checkpoint: {last}')
        folder.mkdir(parents=True,exist_ok=True)
        rt.seed_everything(args.seed)
        model=construct(payload,options(args,arm),args.device)
        rt.seed_everything(args.seed)
        opt=optimizer(model,args)
        config=train_config(args,arm,smoke,check)
        steps=math.ceil(len(train)/args.batch_size)
        if smoke:steps=min(steps,4)
        total=config['epochs']*steps
        schedule=UpdateSchedule(opt,total,config['warmup_epochs']*steps)
        frozen=previous.frozen_digest(model)
        start,best_candidate,best_accepted=1,None,None
        if args.resume and last.exists():
            state=torch.load(last,map_location='cpu',weights_only=False)
            if state['training_config']!=config:raise ValueError('Resume configuration changed')
            model.load_state_dict(state['model_state_dict'],strict=True)
            opt.load_state_dict(state['optimizer_state_dict']);schedule.load_state_dict(state['schedule_state'])
            rt.restore_rng(state['rng_state'])
            start=state['epoch']+1
            best_candidate=state['selection']['best_candidate'];best_accepted=state['selection']['best_accepted']
            if previous.frozen_digest(model)!=frozen:raise RuntimeError('Frozen state changed in checkpoint')
        rt.save_json(folder/'training_config.json',config)
        rt.save_json(folder/'optimizer_groups.json',[{k:v for k,v in group.items() if k!='params'} for group in opt.param_groups])
        for epoch in range(start,config['epochs']+1):
            model.train();seen=0;ternary_seen=0;loss_sum=0.;last_norm=0.;parts_sums=dict(joint_ce=0.,marginal_ce=0.,rdrop_kl=0.,l2sp=0.)
            loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,
                generator=torch.Generator().manual_seed(args.seed+1009*epoch),
                pin_memory=args.device=='cuda',persistent_workers=False)
            for index,batch in enumerate(loader):
                if index>=steps:break
                opt.zero_grad(set_to_none=True)
                data=rt.model_inputs(train,batch,args.device)
                classes=batch['class_target'].to(args.device);levels=batch['concentration_target'].to(args.device)
                if arm=='joint_sam':
                    loss,parts=backward_sam(model,data,classes,levels,opt,args.sam_rho,args.marginal_weight)
                else:
                    output=model(data)
                    loss,parts=training_loss(model,output,classes,levels,args.marginal_weight,args.rdrop_weight,args.l2sp_weight)
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite loss')
                    loss.backward()
                parameters=[p for p in model.parameters() if p.requires_grad]
                norm=torch.nn.utils.clip_grad_norm_(parameters,args.clip_norm,error_if_nonfinite=True)
                last_norm=float(norm)
                if index<2 and epoch==start:
                    rt.save_json(folder/f'gradient_epoch_{epoch:03d}_batch_{index:02d}.json',
                        {name:dict(has_gradient=p.grad is not None,nonzero=p.grad is not None and bool((p.grad!=0).any()))
                         for name,p in model.named_parameters() if p.requires_grad})
                schedule.prepare();opt.step();schedule.finish()
                seen+=len(batch['class_target']);ternary_seen+=parts['ternary_samples']
                loss_sum+=float(loss.detach())*parts['ternary_samples']
                for key in parts_sums:parts_sums[key]+=parts[key]*parts['ternary_samples']
            if not smoke and (seen!=MAIN_TRAIN_COUNT or ternary_seen!=TERNARY_TRAIN_COUNT):
                raise RuntimeError(f'Training exposure changed: all={seen}, ternary={ternary_seen}')
            metrics,frame=evaluate(model,val,args)
            if previous.frozen_digest(model)!=frozen:raise RuntimeError('Frozen anchor/parameters/buffers changed')
            candidate=dict(epoch=epoch,metrics=metrics)
            if best_candidate is None or rank(metrics)>rank(best_candidate['metrics']):
                best_candidate=candidate
                save_checkpoint(folder/'checkpoints/best_candidate.pt',model,epoch,metrics,config)
                frame.to_csv(folder/'candidate_validation.csv',index=False)
            decision=adoption(metrics,check['baseline']) if not smoke else dict(accepted=False,failed_rules=['smoke'])
            if decision['accepted'] and (best_accepted is None or rank(metrics)>rank(best_accepted['metrics'])):
                best_accepted=candidate
                save_checkpoint(folder/'checkpoints/best_accepted.pt',model,epoch,metrics,config)
                frame.to_csv(folder/'accepted_validation.csv',index=False)
            record=dict(epoch=epoch,validation=metrics,adoption=decision,completed_updates=schedule.completed,
                all_training_samples=seen,ternary_training_samples=ternary_seen,
                loss=loss_sum/max(1,ternary_seen),loss_parts={k:v/max(1,ternary_seen) for k,v in parts_sums.items()},
                last_gradient_norm=last_norm,learning_rates=[g['lr'] for g in opt.param_groups])
            rt.save_json(folder/f'epoch_{epoch:03d}.json',record)
            save_checkpoint(last,model,epoch,metrics,config,opt,schedule,
                dict(best_candidate=best_candidate,best_accepted=best_accepted))
            print(f'{arm} epoch={epoch}/{config["epochs"]} tri={metrics["ternary_all_grade_correct"]} '
                  f'tri_grade={metrics["by_mixture"]["3"]["present_correct"]} '
                  f'full={metrics["full_profile_correct"]} accepted={decision["accepted"]}',flush=True)
        if schedule.completed!=total:raise RuntimeError('Optimizer update count changed')
        rt.save_json(folder/'run_summary.json',dict(status='COMPLETE',arm=arm,smoke=smoke,
            best_candidate=best_candidate,best_accepted=best_accepted,anchor=check['baseline'],
            completed_updates=schedule.completed,frozen_state_unchanged=True,test_predictions=0,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
        del model,opt
        if args.device=='cuda':torch.cuda.empty_cache()


def compare(args):
    check=json.loads((args.suite/'check/check_summary.json').read_text());rows=[];eligible=[];missing=[]
    def row(arm,origin,metrics):
        result=dict(model=arm,origin=origin,**{k:metrics[k] for k in ('classification_exact_correct',
                    'full_profile_correct','present_correct','ternary_all_grade_correct')})
        result['ternary_present_correct']=metrics['by_mixture']['3']['present_correct']
        result.update({f'ternary_{d}_correct':metrics['by_mixture']['3']['per_drug'][d]['correct'] for d in ('DEL','CHL','TEB')})
        return result
    rows.append(row('Adapter_anchor','epoch_6',check['baseline']))
    control=check['models']['independent_map'];rows.append(row('independent_map','no_training',control['initial_metrics']))
    if control['adoption']['accepted']:
        eligible.append((rank(control['initial_metrics']),'independent_map',str(args.suite/'check/independent_map.pt')))
    for arm in ARMS:
        path=args.suite/'train'/arm/'run_summary.json'
        if not path.exists():missing.append(arm);continue
        summary=json.loads(path.read_text())
        if summary['status']!='COMPLETE' or summary['smoke']:raise RuntimeError('Only completed formal runs can be compared')
        candidate=summary['best_candidate'];accepted=summary['best_accepted']
        rows.append(row(arm+'_candidate',f'epoch_{candidate["epoch"]}',candidate['metrics']))
        rows.append(row(arm,'accepted' if accepted else 'Adapter_fallback',accepted['metrics'] if accepted else check['baseline']))
        if accepted:eligible.append((rank(accepted['metrics']),arm,str(args.suite/'train'/arm/'checkpoints/best_accepted.pt')))
    frame=pd.DataFrame(rows);frame.to_csv(args.suite/'comparison.csv',index=False)
    best=max(eligible,key=lambda item:item[0]) if eligible else None
    rt.save_json(args.suite/'comparison_summary.json',dict(status='PARTIAL' if missing else 'COMPLETE',
        missing=missing,recommended=best[1] if best else 'Adapter_epoch_6',
        recommendation_path=best[2] if best else str(args.checkpoint),
        selection_scope='exploratory reused validation; test predictions=0',test_predictions=0))
    print(frame.to_string(index=False));print('RECOMMENDED:',best[1] if best else 'Adapter_epoch_6',flush=True)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=Path('/home/wqzheng/project_transformer'))
    p.add_argument('--checkpoint',type=Path,default=Path(DEFAULT_ANCHOR))
    p.add_argument('--suite',type=Path,required=True)
    p.add_argument('--phase',choices=('check','smoke','train','compare','evaluate'),required=True)
    p.add_argument('--arm',choices=ARMS+('all',),default='all')
    p.add_argument('--resume',action='store_true');p.add_argument('--optimized-checkpoint',type=Path)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--num-workers',type=int,default=0)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--warmup-epochs',type=int,default=3);p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--hidden',type=int,default=64);p.add_argument('--dropout',type=float,default=.1)
    p.add_argument('--residual-bound',type=float,default=4.);p.add_argument('--head-lr',type=float,default=1e-4)
    p.add_argument('--adapter-lr',type=float,default=1e-5);p.add_argument('--weight-decay',type=float,default=.01)
    p.add_argument('--minimum-ratio',type=float,default=.01);p.add_argument('--clip-norm',type=float,default=1.)
    p.add_argument('--marginal-weight',type=float,default=.3);p.add_argument('--rdrop-weight',type=float,default=.1)
    p.add_argument('--l2sp-weight',type=float,default=.01)
    p.add_argument('--sam-rho',type=float,default=.05)
    args=p.parse_args()
    if min(args.batch_size,args.threads,args.hidden)<1 or args.num_workers<0 or args.epochs<2 or not 0<=args.warmup_epochs<args.epochs:
        p.error('Invalid integer settings')
    if not 0<=args.dropout<1 or not 0<args.minimum_ratio<=1:
        p.error('Invalid dropout/minimum ratio')
    positive=(args.residual_bound,args.head_lr,args.adapter_lr,args.clip_norm,args.sam_rho)
    nonnegative=(args.weight_decay,args.marginal_weight,args.rdrop_weight,args.l2sp_weight)
    if not all(math.isfinite(v) and v>0 for v in positive) or not all(math.isfinite(v) and v>=0 for v in nonnegative):
        p.error('Invalid floating-point settings')
    return args


def main():
    args=parse_args();package=verify_package()
    args.project,args.checkpoint,args.suite=args.project.resolve(),args.checkpoint.resolve(),args.suite.resolve()
    protected=[ROOT,args.project/'data']+[(args.project/'data'/s).resolve() for s in ('real','generated')]
    if any(args.suite.is_relative_to(p) for p in protected):raise ValueError('Suite must be outside package/input data')
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(args.threads);torch.backends.mha.set_fastpath_enabled(False);rt.seed_everything(args.seed)
    args.suite.mkdir(parents=True,exist_ok=True)
    marker=args.suite/'T328_SUITE.json'
    identity=dict(version=VERSION,project=str(args.project),anchor_path=str(args.checkpoint))
    if marker.exists():
        if json.loads(marker.read_text())!=identity:raise ValueError('Suite identity mismatch')
    else:
        if args.phase!='check' or any(p.suffix!='.log' for p in args.suite.iterdir()):raise ValueError('First check needs a new empty suite')
        rt.save_json(marker,identity)
    if args.phase=='compare':compare(args);return
    print(f'T3.28 START phase={args.phase} arm={args.arm}; base=Adapter epoch 6',flush=True)
    before=snapshot(args.project,args.checkpoint);completed=False
    rt.save_json(args.suite/f'input_hashes_before_{args.phase}.json',before)
    try:
        os.chdir(args.project)
        datasets,audit,sources=rt.configure_project(args.project)
        with previous.guard_test_dataset(datasets[2]):
            if args.phase=='check':run_check(args,datasets,audit,sources,package)
            elif args.phase in ('smoke','train'):run_train(args,datasets,audit,sources,package,args.phase=='smoke')
            else:
                if args.optimized_checkpoint is None:raise ValueError('--optimized-checkpoint required')
                model,_=load_checkpoint(args.optimized_checkpoint,args.device)
                metrics,frame=evaluate(model,datasets[1],args)
                rt.save_json(args.suite/'optimized_validation_metrics.json',metrics)
                frame.to_csv(args.suite/'optimized_validation.csv',index=False)
                print(json.dumps(metrics,ensure_ascii=False,indent=2))
        completed=True
    finally:
        unchanged=before==snapshot(args.project,args.checkpoint)
        rt.save_json(args.suite/f'input_audit_{args.phase}.json',dict(status='PASS' if unchanged else 'FAIL',
            phase_completed=completed,original_files_unchanged=unchanged,hashed_files=len(before),test_predictions=0))
        if not unchanged:raise RuntimeError('Protected input changed')


if __name__=='__main__':main()
