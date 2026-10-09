"""Matched concentration-update/residual arms; no test evaluation or val refits."""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd().resolve()))
import torch
from Paths_T3_24 import build_datasets
from Model_T3_24 import ConcentrationUpdateModel,DRUGS,CORE,digest
from data_t324 import audit_selection
from engine_t324 import (seed_everything,fit_bank,cache_dataset,make_frame,verify_original_validation,
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
        real_features=complete["features"][:,:model.residual_feature_dimension]
    else:real_features=train["features"][torch.tensor(real),:model.residual_feature_dimension]
    model.correction.cpu().fit_normalization(real_features,["real"]*len(real_features),["train"]*len(real_features))
    model.correction.to(device)
    if digest(model.base)!=before:raise RuntimeError("Anchor changed during caching")
    model.eval()
    from zero_t324 import verify_zero_reproduction
    anchor_frame,zero_audit=verify_zero_reproduction(model,val,device,out)
    if not smoke:verify_original_validation(anchor_frame)
    else:print("SMOKE: original full-validation reproduction is deferred to formal run",flush=True)
    anchor_frame.to_csv(out/"original_validation_predictions.csv",index=False)
    cache=dict(train=train,validation=val,initial_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
               model_config=model.config,anchor_state_sha256=before,normalization_fit_rows=len(real_features))
    torch.save(cache,out/"frozen_feature_cache.pt")
    (out/"cache_audit.json").write_text(json.dumps(dict(status="PASS",anchor_before=before,anchor_after=digest(model.base),
        optimizer_steps_during_cache=0,normalization_scope="real_train",zero_reproduction=zero_audit,
        cached_query_scope="PRE mixture_aware_query_fusion [B,3,128]",
        feature_dimensions=dict(residual=model.residual_feature_dimension,query=model.query_dimension,total=train["features"].shape[1]),normalization_fit_rows=len(real_features),
        train_rows=len(train["features"]),validation_rows=len(val["features"]),test_rows=0,
        sample_order="matches audited Dataset sample references; smoke uses a recorded deterministic subset"),indent=2))
    return cache


def suite_report(out,summaries,smoke):
    lines=["# T3.24 浓度分支更新与残差修正对照", "", "共享CNN、Encoder、峰引导Query、分类和浓度投影冻结。两组均训练相同残差修正，实验组额外更新原权重初始化的浓度融合及等级头；参考匹配均屏蔽；推理不使用标签；测试集未评价。", "",
           "| 实验 | 记录 | 总浓度准确率 | 三元完整组合 | 水三元完整组合 | 土壤三元完整组合 | 土壤DEL-M正确数 |", "|---|---|---:|---:|---:|---:|---:|"]
    def table_row(variant,label,m):
        t=m.get("ternary",m["all"]);s=m.get("soil_ternary",m["all"]);w=m.get("water_ternary",m["all"])
        focus=m.get("soil_ternary_DEL_M",{})
        return f"| {variant} | {label} | {100*m['all']['head_accuracy']:.2f}% | {100*t['profile_accuracy']:.2f}% | {100*w['profile_accuracy']:.2f}% | {100*s['profile_accuracy']:.2f}% | {focus.get('head_correct_DEL',0)}/{focus.get('present_n_DEL',0)} |"
    if not smoke:
        historical=json.loads((PACKAGE/"previous_validation_reference.json").read_text())
        for variant,m in historical["validation_metrics"].items():
            lines.append(table_row("T3.22-"+variant,"历史验证记录",m))
        previous=json.loads((PACKAGE/"previous_t323_reference.json").read_text())
        for arm,info in previous["validation_metrics"].items():
            lines.append(table_row("T3.23-"+arm,"历史最佳候选，lr=1e-4",info["candidate_validation_metrics"]))
    for variant,summary in summaries.items():
        for label,key in (("原模型","anchor_validation_metrics"),("最佳训练候选","candidate_validation_metrics"),("推荐结果","recommended_validation_metrics")):
            lines.append(table_row(variant,label,summary[key]))
    for variant,summary in summaries.items():
        lines += ["",f"{variant}: recommended.pt = {summary['selected_kind']}; accepted_epoch={summary['accepted_epoch']}。"]
    lines += ["", "推荐规则：三元完整组合及土壤三元DEL-M正确数必须增加；总体/单元/二元/三元及水、土壤三元的浓度和完整组合均不得下降；总体及三元各药物正确数不得下降；DEL-M的三种TEB分层不得下降；分类保持不变。",
              "", "推荐相对T3.18零修正起点判定，未通过则回到原模型。T3.22历史结果用于比较，不自动替换已有推荐。候选和推荐分别保存。",
              "", "本轮两组共享起点、特征缓存、数据、批次顺序、Adam、学习率和损失。差别是实验组额外学习浓度融合和等级头，因此可训练参数量不同。峰引导Query及Transformer仍被冻结。本轮lr=1e-5/min=1e-7；旧lr=1e-4记录只作历史比较。验证多轮选择属于探索，需要之后独立测试。"]
    if smoke:lines += ["", "SMOKE结果仅验证流程，不代表完整性能，不允许采用其训练候选。"]
    (out/"comparison.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (out/"suite_summary.json").write_text(json.dumps(dict(mode="smoke" if smoke else "train",variants=summaries,test_predictions=0),ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output-directory",type=Path,required=True)
    p.add_argument("--variant",choices=("both","concentration_update","residual_control"),default="both")
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
    model=ConcentrationUpdateModel(expected,centers,experiment_variant="concentration_update")
    model.initialize_anchor(payload["model_state_dict"])
    if digest(model.base)!=known["baseline_model_state_sha256"]:raise RuntimeError("Anchor model state differs from diagnosis")
    model.to(device).eval()
    cache=build_cache(datasets,model,out,args.cache_batch_size,args.num_workers,device,args.smoke)
    identity=dict(checkpoint=state["baseline_checkpoint"],checkpoint_sha256=state["baseline_checkpoint_sha256"],
                  anchor_state_sha256=cache["anchor_state_sha256"])
    summaries={}
    for variant in (("concentration_update","residual_control") if args.variant=="both" else (args.variant,)):
        mc=dict(cache["model_config"],variant="no_reference",experiment_variant=variant)
        config=dict(model_config=mc,epochs=2 if args.smoke else 50,batch_size=16,learning_rate=1e-5,
                    minimum_learning_rate=1e-7,warmup_epochs=1 if args.smoke else 3,seed=2026,
                    correction_penalty=.02,real_ordinal_weight=1.,generated_ordinal_weight=1.,
                    original_model_frozen=True,lr_schedule="warmup_cosine",ordinal_loss="corn",test_evaluation=False,
                    training_sources={"real":1512,"generated":6048},maximum_generated_per_condition=48,
                    actual_training_sources=dict(Counter(r["source"] for r in cache["train"]["metadata"])),
                    experiment_variant=variant,middle_weight=0.,
                    preservation_weight=0.,middle_margin=.35,gradient_clip_norm=1.,
                    preservation_margin_cap=.5)
        summaries[variant]=train_variant(cache["initial_state"],cache["train"],cache["validation"],out/variant,config,identity,device,args.smoke)
    suite_report(out,summaries,args.smoke)
    print(f"T3.24 complete: {out/'comparison.md'}; test not evaluated",flush=True)


if __name__=="__main__":main()
