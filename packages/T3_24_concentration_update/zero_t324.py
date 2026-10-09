"""Initialization audit: exact state/grade checks and small numeric tolerance."""
import json
import numpy as np
import torch
from engine_t324 import evaluate_cache,make_frame
from Model_T3_24 import DRUGS

PROBABILITY_ATOL=1e-6
BRANCH_ATOL=1e-6


def verify_initial_weights(model):
    final=model.correction.network[-1]
    if torch.count_nonzero(final.weight).item() or torch.count_nonzero(final.bias).item():
        raise RuntimeError("Initialization failed: residual output layer is not exactly zero")
    if model.experiment_variant=="concentration_update":
        for name,original,updated in (
            ("fusion",model.base.mixture_aware_query_fusion,model.updated_fusion),
            ("ordinal_heads",model.base.ordinal_heads,model.updated_ordinal_heads)):
            a,b=original.state_dict(),updated.state_dict()
            if a.keys()!=b.keys():raise RuntimeError("Initialization state keys differ: "+name)
            for key in a:
                if not torch.equal(a[key],b[key]):
                    raise RuntimeError("Initialization weights differ: "+name+"."+key)


def audit_zero_frames(anchor,zero):
    """Accept roundoff only; never accept a changed class or decoded grade."""
    reasons=[];diffs={};grade_changes={}
    identity=["condition","source","spectrum_index"]
    if not anchor[identity].equals(zero[identity]):reasons.append("sample_order_changed")
    for prefix in ("class_","head_","final_"):
        for drug in DRUGS:
            key=prefix+drug;n=int(np.count_nonzero(anchor[key].to_numpy()!=zero[key].to_numpy()))
            grade_changes[key]=n
            if n:reasons.append(key+"_changed")
    for drug in DRUGS:
        key="class_probability_"+drug
        if not np.array_equal(anchor[key].to_numpy(),zero[key].to_numpy()):reasons.append(key+"_changed")
    for prefix in ("p_ge_M_","p_ge_H_"):
        for drug in DRUGS:
            key=prefix+drug;a=anchor[key].to_numpy();b=zero[key].to_numpy()
            error=float(np.max(np.abs(a-b)));diffs[key]=error
            if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(a,b,atol=PROBABILITY_ATOL,rtol=0):
                reasons.append(key+"_probability_error")
    for prefix,tolerance in (("delta_",BRANCH_ATOL),("branch_delta_",BRANCH_ATOL),("residual_delta_",0.)):
        columns=[prefix+drug+"_"+str(k) for drug in DRUGS for k in (0,1)]
        values=zero[columns].to_numpy();error=float(np.max(np.abs(values)));diffs[prefix+"max_abs"]=error
        if not np.isfinite(values).all() or error>tolerance:reasons.append(prefix+"nonzero")
    return dict(status="PASS" if not reasons else "FAIL",fix_version="T3.24_zero_reproduction_v1",
                probability_atol=PROBABILITY_ATOL,probability_rtol=0,branch_atol=BRANCH_ATOL,
                probability_and_change_errors=diffs,decoded_grade_changes=grade_changes,reasons=reasons,
                policy="Identical initial weights; exact classification/grades; zero-change probabilities reuse anchor")


def verify_zero_reproduction(model,cache,device,out):
    verify_initial_weights(model)
    zero,_,_=evaluate_cache(model,cache,device)
    anchor=make_frame(cache)
    audit=audit_zero_frames(anchor,zero)
    (out/"zero_reproduction_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    if audit["status"]!="PASS":
        raise RuntimeError("Zero reproduction failed: "+";".join(audit["reasons"])+
                           "; measured_errors="+json.dumps(audit["probability_and_change_errors"])+
                           "; see "+str(out/"zero_reproduction_audit.json"))
    print("ZERO REPRODUCTION: PASS; class/grades unchanged; max_probability_error="+
          str(max(v for k,v in audit["probability_and_change_errors"].items() if k.startswith("p_ge_"))),flush=True)
    return anchor,audit
