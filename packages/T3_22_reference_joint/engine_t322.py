"""Cache frozen inputs, train only the small correction, evaluate validation."""
from __future__ import annotations
import copy
import json
import math
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,Subset,TensorDataset
from Model_T3_22 import ReferenceJointModel,DRUGS,digest,corn_probabilities
from data_t322 import measured_masks,metadata_rows
from Model_T3_17 import present_only_corn_ordinal_loss
from T3_17_LR import warmup_cosine_factor,validate_schedule


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def make_loader(dataset,batch,workers,device):
    return DataLoader(dataset,batch_size=batch,shuffle=False,num_workers=workers,
                      pin_memory=device.type=="cuda",persistent_workers=workers>0)


def fit_bank(model,train_dataset,batch_size,workers,device):
    selected=Subset(train_dataset,[i for i,r in enumerate(train_dataset.samples) if r.source=="real"])
    intensities=[];masks=[];levels=[];metadata=[]
    for batch in make_loader(selected,batch_size,workers,device):
        intensities.append(batch["raw_intensity"]);masks.append(measured_masks(train_dataset,batch))
        levels.append(batch["concentration_target"]);metadata+=metadata_rows(train_dataset,batch)
    # Fit in CPU memory, independently of GPU precision and inference.
    audit=model.bank.cpu().fit(torch.cat(intensities),torch.cat(masks),torch.cat(levels),metadata)
    model.bank.to(device)
    return audit


@torch.no_grad()
def cache_dataset(model,dataset,batch_size,workers,device,smoke=False):
    selected=dataset
    if smoke:
        indices=[]
        for source in ("real","generated"):
            bucket=np.array([i for i,r in enumerate(dataset.samples) if r.source==source])
            if len(bucket):indices+=bucket[np.linspace(0,len(bucket)-1,min(48,len(bucket)),dtype=int)].tolist()
        selected=Subset(dataset,sorted(indices))
    model.eval()
    blocks={key:[] for key in ("features","classify","base_logits","base_ordinal","levels","presence")};metadata=[]
    for batch in make_loader(selected,batch_size,workers,device):
        data={k:batch[k].to(device) for k in ("raw","percentile","smoothed","valid_mask","raw_intensity")}
        data["measured_mask"]=measured_masks(dataset,batch).to(device)
        features,classify,logits,ordinal=model.frozen_features(data)
        for key,value in (("features",features),("classify",classify),("base_logits",logits),("base_ordinal",ordinal),
                          ("levels",batch["concentration_target"]),("presence",batch["class_target"])):
            blocks[key].append(value.detach().cpu())
        metadata+=metadata_rows(dataset,batch)
        if len(metadata)%512<batch_size or len(metadata)==len(selected):
            print(f"Frozen cache {dataset.split}: {len(metadata)}/{len(selected)}",flush=True)
    result={key:torch.cat(value) for key,value in blocks.items()};result["metadata"]=metadata
    return result


def make_frame(cache,ordinal=None,delta=None):
    p=cache["base_ordinal"].numpy() if ordinal is None else np.asarray(ordinal)
    classification=cache["classify"].numpy()
    grades=1+(p>=.5).sum(-1);classes=classification>=.5
    final=np.where(classes,grades,0);score=1+p.sum(-1)
    frame=pd.DataFrame(cache["metadata"])
    for j,d in enumerate(DRUGS):
        frame[f"class_{d}"]=classes[:,j].astype(int);frame[f"class_probability_{d}"]=classification[:,j]
        frame[f"head_{d}"]=grades[:,j];frame[f"final_{d}"]=final[:,j];frame[f"score_{d}"]=score[:,j]
        frame[f"p_ge_M_{d}"]=p[:,j,0];frame[f"p_ge_H_{d}"]=p[:,j,1]
        if delta is not None:
            for k in (0,1):frame[f"delta_{d}_{k}"]=delta[:,j,k]
    return frame


def metric_counts(frame):
    truth=frame[["true_"+d for d in DRUGS]].to_numpy(int)
    head=frame[["head_"+d for d in DRUGS]].to_numpy(int)
    final=frame[["final_"+d for d in DRUGS]].to_numpy(int)
    classes=frame[["class_"+d for d in DRUGS]].to_numpy(int)
    result={}
    groups={"all":np.ones(len(frame),bool),"single":frame.mixture_count.to_numpy()==1,
            "binary":frame.mixture_count.to_numpy()==2,"ternary":frame.mixture_count.to_numpy()==3}
    for matrix in ("water","soil"):
        for n,name in ((1,"single"),(2,"binary"),(3,"ternary")):
            groups[matrix+"_"+name]=(frame.matrix.to_numpy()==matrix)&(frame.mixture_count.to_numpy()==n)
    for name,take in groups.items():
        if not take.any():continue
        active=truth[take]>0;diff=np.abs(head[take][active]-truth[take][active])
        row=dict(n=int(take.sum()),present_n=int(active.sum()),head_correct=int((diff==0).sum()),
                 head_accuracy=float((diff==0).mean()),profile_correct=int((final[take]==truth[take]).all(1).sum()),
                 profile_accuracy=float((final[take]==truth[take]).all(1).mean()),
                 class_correct=int((classes[take]==active).all(1).sum()),adjacent_errors=int((diff==1).sum()),severe_errors=int((diff>=2).sum()))
        for j,d in enumerate(DRUGS):
            a=truth[take,j]>0
            row[f"accuracy_{d}"]=float((head[take,j][a]==truth[take,j][a]).mean()) if a.any() else None
        result[name]=row
    return result


def condition_report(frame,model_name):
    rows=[]
    for condition,g in frame.groupby("condition",sort=True):
        row=dict(model=model_name,condition=condition,matrix=g.matrix.iloc[0],rows=len(g),
                 mixture_count=int(g.mixture_count.iloc[0]))
        for d in DRUGS:
            row["true_"+d]=int(g["true_"+d].iloc[0])
            for level,name in ((1,"S"),(2,"M"),(3,"H")):
                row[f"pred_{d}_{name}"]=int((g["head_"+d]==level).sum()) if row["true_"+d]>0 else None
            row[f"correct_{d}"]=int((g["head_"+d]==g["true_"+d]).sum()) if row["true_"+d]>0 else None
            for boundary in (0,1):
                key=f"delta_{d}_{boundary}"
                row[key+"_mean"]=float(g[key].mean()) if key in g else 0.
        row["profile_correct"]=metric_counts(g)["all"]["profile_correct"]
        rows.append(row)
    return pd.DataFrame(rows)


def verify_original_validation(frame):
    metrics=metric_counts(frame)
    if (metrics["all"]["n"],metrics["all"]["head_correct"],metrics["all"]["profile_correct"],
        metrics["ternary"]["head_correct"],metrics["ternary"]["profile_correct"],metrics["soil_ternary"]["profile_correct"])!=(504,973,366,510,118,53):
        raise RuntimeError("Original validation predictions did not reproduce; no correction training allowed")
    print("ORIGINAL VALIDATION REPRODUCTION: PASS (973/1152, 366/504; ternary 118/216)",flush=True)


def adoption_guard(metrics,anchor):
    """A conservative adoption rule; zero correction remains fallback."""
    reasons=[]
    checks=(("all","head_correct"),("all","profile_correct"),("binary","profile_correct"),
            ("single","profile_correct"),("ternary","head_correct"),("soil_ternary","profile_correct"))
    for group,key in checks:
        if group in anchor and (group not in metrics or metrics[group][key]<anchor[group][key]):reasons.append(group+"_"+key+"_decreased")
    if "ternary" not in metrics or metrics["ternary"]["profile_correct"]<=anchor.get("ternary",{}).get("profile_correct",0):
        reasons.append("ternary_profile_not_improved")
    if metrics["all"]["class_correct"]!=anchor["all"]["class_correct"]:reasons.append("classification_changed")
    return not reasons,reasons


def selection_score(metrics,loss):
    ternary=metrics.get("ternary",metrics["all"])
    return (ternary["profile_correct"],ternary["head_correct"],metrics["all"]["head_correct"],-float(loss))


@torch.no_grad()
def evaluate_cache(model,cache,device,batch_size=128):
    model.eval();probabilities=[];deltas=[];loss_sum=0.;weight_sum=0
    for start in range(0,len(cache["features"]),batch_size):
        stop=start+batch_size
        x=cache["features"][start:stop].to(device);c=cache["classify"][start:stop].to(device);b=cache["base_logits"][start:stop].to(device)
        _,_,p,logits,delta=model.forward_cached(x,c,b)
        y=cache["levels"][start:stop].to(device);present=cache["presence"][start:stop].to(device)
        eligible=int(((present>.5).sum()+((present>.5)&(y>=2)).sum()).item())
        loss=present_only_corn_ordinal_loss(logits,y,present)
        loss_sum+=float(loss)*eligible;weight_sum+=eligible
        probabilities.append(p.cpu().numpy());deltas.append(delta.cpu().numpy())
    frame=make_frame(cache,np.concatenate(probabilities),np.concatenate(deltas))
    return frame,metric_counts(frame),loss_sum/max(1,weight_sum)


def checkpoint_payload(model,config,epoch,metrics,accepted,anchor_identity,optimizer=None,scheduler=None):
    result=dict(checkpoint_version="T3.22_reference_joint_v1",model_config=model.config,
                model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                training_config=config,epoch=epoch,validation_metrics=metrics,accepted=accepted,
                anchor_identity=anchor_identity,ordinal_decoding="median",classification_threshold=.5)
    if optimizer is not None:result["optimizer_state_dict"]=optimizer.state_dict()
    if scheduler is not None:result["scheduler_state_dict"]=scheduler.state_dict()
    return result


def load_checkpoint(path,device):
    payload=torch.load(path,map_location="cpu",weights_only=False)
    if payload.get("checkpoint_version")!="T3.22_reference_joint_v1":raise ValueError("Not a T3.22 checkpoint")
    model=ReferenceJointModel(**payload["model_config"])
    model.load_state_dict(payload["model_state_dict"],strict=True)
    return model.to(device).eval(),payload


def train_variant(initial,train_cache,val_cache,out,config,identity,device,smoke=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);ckpts=out/"checkpoints";ckpts.mkdir()
    seed_everything(config["seed"])
    model=ReferenceJointModel(**config["model_config"])
    model.load_state_dict(initial,strict=True);model.to(device)
    before=digest(model.base)
    parameters=list(model.correction.network.parameters())
    if {id(p) for p in model.parameters() if p.requires_grad}!={id(p) for p in parameters}:
        raise RuntimeError("Only the correction network may train")
    optimizer=torch.optim.Adam(parameters,lr=config["learning_rate"],weight_decay=0.)
    generator=torch.Generator().manual_seed(config["seed"])
    loader=DataLoader(TensorDataset(train_cache["features"],train_cache["classify"],train_cache["base_logits"],train_cache["levels"],train_cache["presence"]),
                      batch_size=config["batch_size"],shuffle=True,generator=generator,num_workers=0)
    total_steps=len(loader)*config["epochs"];warmup_steps=len(loader)*config["warmup_epochs"]
    validate_schedule(config["epochs"],config["warmup_epochs"],config["learning_rate"],config["minimum_learning_rate"],.1)
    schedule=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:warmup_cosine_factor(step,total_steps=total_steps,warmup_steps=warmup_steps,
        minimum_ratio=config["minimum_learning_rate"]/config["learning_rate"],start_ratio=.1))
    config=dict(config,planned_optimizer_updates=total_steps,warmup_optimizer_updates=warmup_steps,
                trainable_parameters=sum(p.numel() for p in parameters),frozen_parameters=sum(p.numel() for p in model.base.parameters()))
    anchor_frame=make_frame(val_cache);anchor=metric_counts(anchor_frame)
    anchor_frame.to_csv(out/"anchor_validation_predictions.csv",index=False)
    torch.save(checkpoint_payload(model,config,0,anchor,False,identity),ckpts/"original_fallback.pt")
    best_candidate=None;best_accepted=None;accepted_epoch=None;history=[]
    print(f"Variant={model.correction.variant}; trainable={config['trainable_parameters']}; frozen={config['frozen_parameters']}; planned_updates={total_steps}",flush=True)
    for epoch in range(1,config["epochs"]+1):
        model.train();sum_loss=0.;seen=0;max_delta=0.
        for x,c,b,y,present in loader:
            x,c,b,y,present=[v.to(device) for v in (x,c,b,y,present)]
            optimizer.zero_grad(set_to_none=True)
            _,_,_,logits,delta=model.forward_cached(x,c,b)
            corn=present_only_corn_ordinal_loss(logits,y,present)
            active=present>.5
            penalty=delta[active].square().mean() if active.any() else delta.sum()*0.
            loss=.7*corn+config["correction_penalty"]*penalty
            if not torch.isfinite(loss):raise RuntimeError("Non-finite correction loss")
            loss.backward();optimizer.step();schedule.step()
            sum_loss+=float(loss.detach())*len(x);seen+=len(x);max_delta=max(max_delta,float(delta.detach().abs().max()))
        frame,metrics,val_loss=evaluate_cache(model,val_cache,device)
        # Classification probabilities remain exactly the cached frozen output.
        if not np.array_equal(frame[[f"class_probability_{d}" for d in DRUGS]].to_numpy(),anchor_frame[[f"class_probability_{d}" for d in DRUGS]].to_numpy()):
            raise RuntimeError("Classification probabilities changed")
        if digest(model.base)!=before:raise RuntimeError("Frozen anchor parameters/buffers changed")
        accepted,reasons=adoption_guard(metrics,anchor)
        if smoke:accepted=False;reasons=["smoke_not_eligible_for_adoption"]
        score=selection_score(metrics,val_loss)
        payload=checkpoint_payload(model,config,epoch,metrics,accepted,identity,optimizer,schedule)
        torch.save(payload,ckpts/"latest.pt")
        if best_candidate is None or score>best_candidate:
            best_candidate=score;torch.save(payload,ckpts/"best_candidate.pt")
        if accepted and (best_accepted is None or score>best_accepted):
            best_accepted=score;accepted_epoch=epoch;torch.save(payload,ckpts/"best_accepted.pt")
        row=dict(epoch=epoch,train_loss=sum_loss/seen,validation_corn_loss=val_loss,
                 validation_head_accuracy=metrics["all"]["head_accuracy"],validation_profile_accuracy=metrics["all"]["profile_accuracy"],
                 validation_ternary_head_accuracy=metrics.get("ternary",metrics["all"])["head_accuracy"],
                 validation_ternary_profile_accuracy=metrics.get("ternary",metrics["all"])["profile_accuracy"],
                 learning_rate=optimizer.param_groups[0]["lr"],max_abs_training_delta=max_delta,
                 passes_adoption_guard=accepted,guard_reasons=";".join(reasons),optimizer_updates=schedule.last_epoch)
        history.append(row);pd.DataFrame(history).to_csv(out/"training_history.csv",index=False)
        print(f"epoch={epoch:03d} train={row['train_loss']:.6f} val={val_loss:.6f} val_head={row['validation_head_accuracy']:.4f} ternary_exact={row['validation_ternary_profile_accuracy']:.4f} accepted={accepted} lr={row['learning_rate']:.3e}",flush=True)
    candidate,candidate_payload=load_checkpoint(ckpts/"best_candidate.pt",device)
    candidate_frame,candidate_metrics,_=evaluate_cache(candidate,val_cache,device)
    candidate_frame.to_csv(out/"candidate_validation_predictions.csv",index=False)
    if best_accepted is not None:
        selected,selected_payload=load_checkpoint(ckpts/"best_accepted.pt",device)
        selected_kind="accepted_correction"
    else:
        selected,selected_payload=load_checkpoint(ckpts/"original_fallback.pt",device)
        selected_kind="original_zero_correction_fallback"
    torch.save(selected_payload,ckpts/"recommended.pt")
    selected_frame,selected_metrics,_=evaluate_cache(selected,val_cache,device)
    selected_frame.to_csv(out/"recommended_validation_predictions.csv",index=False)
    conditions=pd.concat([condition_report(f,name) for f,name in ((anchor_frame,"anchor"),
        (candidate_frame,"candidate"),(selected_frame,"recommended"))],ignore_index=True)
    conditions.to_csv(out/"validation_metrics_by_condition.csv",index=False)
    conditions[(conditions.matrix=="soil")&(conditions.true_DEL==2)&(conditions.mixture_count==3)].to_csv(
        out/"soil_fixed_DEL_M_nine_conditions.csv",index=False)
    rows=[]
    for name,cache in (("train",train_cache),("validation",val_cache)):
        f,_,_=evaluate_cache(candidate,cache,device)
        f.to_csv(out/("candidate_"+name+"_predictions.csv"),index=False)
        for scope,g in f.groupby("scope"):
            for group,m in metric_counts(g).items():rows.append(dict(model="candidate",scope=scope,group=group,**m))
    pd.DataFrame(rows).to_csv(out/"candidate_metrics_by_source.csv",index=False)
    summary=dict(status="PASS",variant=config["model_config"]["variant"],training_config=config,
                 anchor_validation_metrics=anchor,candidate_validation_metrics=candidate_metrics,
                 candidate_epoch=candidate_payload["epoch"],recommended_validation_metrics=selected_metrics,
                 selected_kind=selected_kind,accepted_epoch=accepted_epoch,
                 candidate_passes_guard=adoption_guard(candidate_metrics,anchor)[0] and not smoke,
                 anchor_state_before=before,anchor_state_after=digest(model.base),
                 classification_probabilities_unchanged=True,test_predictions=0,
                 normalization_and_references_fitted_on="real_train_only",
                 cache_training="Frozen deterministic anchor features; only correction network has optimizer updates",
                 correction_loss="0.7*CORN + correction_penalty*mean_squared_present_logit_correction")
    (out/"run_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(f"Recommended checkpoint: {ckpts/'recommended.pt'}; selection={selected_kind}",flush=True)
    return summary
