"""Two matched frozen-anchor experiments; no test predictions or refits on val."""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd().resolve()))
import torch
from Paths_T3_22 import build_datasets
from Model_T3_22 import ReferenceJointModel,DRUGS,CORE,digest
from data_t322 import audit_selection
from engine_t322 import (seed_everything,fit_bank,cache_dataset,make_frame,verify_original_validation,
                         train_variant,metric_counts)

PACKAGE=Path(__file__).resolve().parent


def build_cache(datasets,model,out,batch_size,workers,device,smoke=False):
    before=digest(model.base)
    audit=fit_bank(model,datasets[0],batch_size,workers,device)
    (out/"reference_fit_audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    train=cache_dataset(model,datasets[0],batch_size,workers,device,smoke)
    val=cache_dataset(model,datasets[1],batch_size,workers,device,smoke)
    real=[row["source"]=="real" for row in train["metadata"]]
    if smoke:
        # Fit normalization on all 1512 real training spectra even in smoke,
        # so smoke only reduces optimizer/evaluation work, not fitting policy.
        from torch.utils.data import Subset
        subset=Subset(datasets[0],[i for i,r in enumerate(datasets[0].samples) if r.source=="real"])
        subset.split=datasets[0].split;subset.samples=[datasets[0].samples[i] for i in subset.indices]
        subset.repository=datasets[0].repository
        complete=cache_dataset(model,subset,batch_size,workers,device,False)
        real_features=complete["features"]
    else:real_features=train["features"][torch.tensor(real)]
    model.correction.cpu().fit_normalization(real_features,["real"]*len(real_features),["train"]*len(real_features))
    model.correction.to(device)
    if digest(model.base)!=before:raise RuntimeError("Anchor changed during caching")
    model.eval()
    # Initialization must preserve the anchor outputs exactly.
    from engine_t322 import evaluate_cache
    zero_frame,_,_=evaluate_cache(model,val,device)
    anchor_frame=make_frame(val)
    for key in ["head_"+d for d in DRUGS]+["p_ge_M_"+d for d in DRUGS]+["p_ge_H_"+d for d in DRUGS]:
        if not (zero_frame[key].to_numpy()==anchor_frame[key].to_numpy()).all():raise RuntimeError("Zero correction changed predictions")
    if not smoke:verify_original_validation(anchor_frame)
    else:print("SMOKE: original full-validation reproduction is deferred to formal run",flush=True)
    anchor_frame.to_csv(out/"original_validation_predictions.csv",index=False)
    cache=dict(train=train,validation=val,initial_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
               model_config=model.config,anchor_state_sha256=before,normalization_fit_rows=len(real_features))
    torch.save(cache,out/"frozen_feature_cache.pt")
    (out/"cache_audit.json").write_text(json.dumps(dict(status="PASS",anchor_before=before,anchor_after=digest(model.base),
        optimizer_steps_during_cache=0,normalization_scope="real_train",normalization_fit_rows=len(real_features),
        train_rows=len(train["features"]),validation_rows=len(val["features"]),test_rows=0,
        sample_order="matches audited Dataset sample references; smoke uses a recorded deterministic subset"),indent=2))
    return cache


def suite_report(out,summaries,smoke):
    lines=["# T3.22 单药参考辅助联合浓度修正", "", "仅使用训练谱拟合参考与归一化；原模型冻结；验证用于选择，测试集未评价。", "",
           "| 实验 | 记录 | 总浓度准确率 | 三元浓度准确率 | 三元完整组合 | 土壤三元完整组合 |", "|---|---|---:|---:|---:|---:|"]
    for variant,summary in summaries.items():
        for label,key in (("原模型","anchor_validation_metrics"),("最佳训练候选","candidate_validation_metrics"),("推荐结果","recommended_validation_metrics")):
            m=summary[key];t=m.get("ternary",m["all"]);s=m.get("soil_ternary",m["all"])
            lines.append(f"| {variant} | {label} | {100*m['all']['head_accuracy']:.2f}% | {100*t['head_accuracy']:.2f}% | {100*t['profile_accuracy']:.2f}% | {100*s['profile_accuracy']:.2f}% |")
        lines += ["", f"{variant}: recommended.pt = {summary['selected_kind']}; accepted_epoch={summary['accepted_epoch']}。"]
    lines += ["", "推荐规则：三元完整组合必须增加；总体浓度/完整组合、单元/二元完整组合、三元浓度、土壤三元完整组合均不得下降；分类保持不变。未通过时推荐零修正原模型。",
              "", "候选和推荐结果分别保存，不能把候选文件当成已通过推荐规则的模型。两组比较控制结构、参数量、初始化、数据、训练轮数和损失，仅屏蔽参考匹配特征。",
              "", "此实验检查辅助参考是否有用，不证明匹配系数等于浓度，也不要求混合峰保持不变或严格线性叠加。"]
    if smoke:lines += ["", "SMOKE结果仅验证流程，不代表完整性能，不允许采用其训练候选。"]
    (out/"comparison.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (out/"suite_summary.json").write_text(json.dumps(dict(mode="smoke" if smoke else "train",variants=summaries,test_predictions=0),ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output-directory",type=Path,required=True)
    p.add_argument("--variant",choices=("both","reference","no_reference"),default="both")
    p.add_argument("--device",default="cuda")
    p.add_argument("--num-workers",type=int,default=4)
    p.add_argument("--cache-batch-size",type=int,default=32)
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    if args.num_workers<0 or args.cache_batch_size<1:raise ValueError("Invalid loader settings")
    device=torch.device(args.device)
    if device.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
    out=args.output_directory.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/"frozen_feature_cache.pt").exists():raise FileExistsError("Choose a fresh suite directory")
    state=json.loads((PACKAGE/"prepared_state.json").read_text())
    datasets=build_datasets(include_generated_train=True,maximum_generated_per_condition=48)
    (out/"selection_audit.json").write_text(json.dumps(audit_selection(datasets),indent=2))
    payload=torch.load(state["baseline_checkpoint"],map_location="cpu",weights_only=False)
    expected=json.loads((PACKAGE/"reference_random48.json").read_text())["model_config"]
    if payload["model_config"]!=expected:raise RuntimeError("Unexpected anchor architecture")
    prior=json.loads((Path(state["baseline_run"])/"training_peak_prior_summary.json").read_text())
    centers=sorted(set(CORE)|{float(c) for d in DRUGS for c in prior["pesticides"][d]["auxiliary_peak_centers_cm-1"]})
    known=json.loads((PACKAGE/"expected_reference_state.json").read_text())
    if centers!=known["peak_centers"]:raise RuntimeError("Anchor peak prior changed")
    seed_everything(2026)
    model=ReferenceJointModel(expected,centers)
    model.base.load_state_dict(payload["model_state_dict"],strict=True)
    if digest(model.base)!=known["baseline_model_state_sha256"]:raise RuntimeError("Anchor model state differs from diagnosis")
    model.to(device).eval()
    cache=build_cache(datasets,model,out,args.cache_batch_size,args.num_workers,device,args.smoke)
    identity=dict(checkpoint=state["baseline_checkpoint"],checkpoint_sha256=state["baseline_checkpoint_sha256"],
                  anchor_state_sha256=cache["anchor_state_sha256"])
    summaries={}
    for variant in (("reference","no_reference") if args.variant=="both" else (args.variant,)):
        mc=dict(cache["model_config"],variant=variant)
        config=dict(model_config=mc,epochs=2 if args.smoke else 50,batch_size=16,learning_rate=1e-4,
                    minimum_learning_rate=1e-6,warmup_epochs=1 if args.smoke else 3,seed=2026,
                    correction_penalty=.02,real_ordinal_weight=1.,generated_ordinal_weight=1.,
                    original_model_frozen=True,lr_schedule="warmup_cosine",ordinal_loss="corn",test_evaluation=False,
                    training_sources={"real":1512,"generated":6048},maximum_generated_per_condition=48,
                    actual_training_sources=dict(Counter(r["source"] for r in cache["train"]["metadata"])))
        summaries[variant]=train_variant(cache["initial_state"],cache["train"],cache["validation"],out/variant,config,identity,device,args.smoke)
    suite_report(out,summaries,args.smoke)
    print(f"T3.22 complete: {out/'comparison.md'}; test not evaluated",flush=True)


if __name__=="__main__":main()
