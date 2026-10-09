"""Simulated roundoff regression; no claim of local CUDA execution."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from test_t324 import new_model
from engine_t324 import make_frame,evaluate_cache
from zero_t324 import audit_zero_frames,verify_initial_weights,verify_zero_reproduction
from Model_T3_17 import present_only_corn_ordinal_loss

torch.set_num_threads(2)


def fixture():
    model=new_model()
    model.correction.fit_normalization(torch.randn(4,1266),["real"]*4,["train"]*4)
    x=torch.randn(4,1650);c=torch.full((4,3),.9);b=torch.zeros(4,3,2)
    p=torch.empty_like(b);p[...,0]=torch.nextafter(torch.tensor(.5),torch.tensor(0.));p[...,1]=.25
    metadata=[dict(condition="synthetic_roundoff",source="real",spectrum_index=i,matrix="soil",split="validation",
                   scope="real_validation",mixture_count=3,true_DEL=2,true_CHL=2,true_TEB=2) for i in range(4)]
    cache=dict(features=x,classify=c,base_logits=b,base_ordinal=p,levels=torch.full((4,3),2.),
               presence=torch.ones(4,3),metadata=metadata)
    return model,cache


class ZeroTests(unittest.TestCase):
    def test_zero_cache_reuses_anchor_probability_next_to_threshold(self):
        model,cache=fixture()
        # Recalculation yields .5; simulate the original CUDA result one ULP
        # below .5. A zero-change model must retain the original S decision.
        raw=model.forward_cached(cache["features"],cache["classify"],cache["base_logits"])
        self.assertFalse(torch.equal(raw[2],cache["base_ordinal"]))
        frame,_,_=evaluate_cache(model,cache,torch.device("cpu"),batch_size=3)
        anchor=make_frame(cache)
        self.assertTrue(frame[["head_DEL","head_CHL","head_TEB"]].equals(anchor[["head_DEL","head_CHL","head_TEB"]]))
        self.assertTrue(np.array_equal(frame.p_ge_M_DEL.to_numpy(),anchor.p_ge_M_DEL.to_numpy()))
        self.assertEqual(audit_zero_frames(anchor,frame)["status"],"PASS")

    def test_nonzero_update_still_changes_probabilities(self):
        model,cache=fixture()
        with torch.no_grad():model.updated_ordinal_heads[0][-1].bias[0].add_(.1)
        outputs=model.forward_cached(cache["features"],cache["classify"],cache["base_logits"],cache["base_ordinal"])
        self.assertTrue((outputs[2][:,0,0]>.5).all())
        self.assertFalse(torch.equal(outputs[2][:,0],cache["base_ordinal"][:,0]))
        self.assertTrue(torch.equal(outputs[2][:,1:],cache["base_ordinal"][:,1:]))

    def test_small_probability_roundoff_without_grade_change_is_accepted(self):
        model,cache=fixture();frame,_,_=evaluate_cache(model,cache,torch.device("cpu"));anchor=make_frame(cache)
        frame["p_ge_H_DEL"]+=1e-7
        self.assertEqual(audit_zero_frames(anchor,frame)["status"],"PASS")

    def test_grade_change_is_rejected_even_inside_probability_tolerance(self):
        model,cache=fixture();frame,_,_=evaluate_cache(model,cache,torch.device("cpu"));anchor=make_frame(cache)
        frame.loc[0,"p_ge_M_DEL"]=.5;frame.loc[0,"head_DEL"]=2;frame.loc[0,"final_DEL"]=2
        audit=audit_zero_frames(anchor,frame)
        self.assertEqual(audit["status"],"FAIL");self.assertIn("head_DEL_changed",audit["reasons"])

    def test_large_probability_or_logit_error_is_rejected(self):
        model,cache=fixture();frame,_,_=evaluate_cache(model,cache,torch.device("cpu"));anchor=make_frame(cache)
        frame.loc[0,"p_ge_H_DEL"]+=1e-3
        self.assertIn("p_ge_H_DEL_probability_error",audit_zero_frames(anchor,frame)["reasons"])
        frame,_,_=evaluate_cache(model,cache,torch.device("cpu"));frame.loc[0,"branch_delta_DEL_0"]=1e-3
        self.assertIn("branch_delta_nonzero",audit_zero_frames(anchor,frame)["reasons"])

    def test_wrong_initial_weights_and_nonzero_residual_are_rejected(self):
        model,cache=fixture();verify_initial_weights(model)
        with torch.no_grad():model.updated_fusion.gate_logit[0].add_(1e-4)
        with self.assertRaisesRegex(RuntimeError,"weights differ"):verify_initial_weights(model)
        model,cache=fixture()
        with torch.no_grad():model.correction.network[-1].bias[0].fill_(1e-4)
        with self.assertRaisesRegex(RuntimeError,"not exactly zero"):verify_initial_weights(model)

    def test_zero_probability_reuse_does_not_block_corn_training_gradient(self):
        model,cache=fixture();model.train()
        outputs=model.forward_cached(cache["features"],cache["classify"],cache["base_logits"],cache["base_ordinal"])
        loss=present_only_corn_ordinal_loss(outputs[3],cache["levels"],cache["presence"]);loss.backward()
        self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in model.updated_ordinal_heads.parameters()))
        self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in model.updated_fusion.parameters()))
        self.assertTrue(all(p.grad is None for p in model.base.parameters()))

    def test_diagnostics_are_saved_for_rejected_zero_check(self):
        model,cache=fixture();frame,metrics,loss=evaluate_cache(model,cache,torch.device("cpu"))
        frame.loc[0,"p_ge_H_DEL"]+=.01
        with tempfile.TemporaryDirectory() as tmp,patch("zero_t324.evaluate_cache",return_value=(frame,metrics,loss)):
            with self.assertRaisesRegex(RuntimeError,"measured_errors"):
                verify_zero_reproduction(model,cache,torch.device("cpu"),Path(tmp))
            audit=json.loads((Path(tmp)/"zero_reproduction_audit.json").read_text())
            self.assertEqual(audit["status"],"FAIL")


if __name__=="__main__":unittest.main()
