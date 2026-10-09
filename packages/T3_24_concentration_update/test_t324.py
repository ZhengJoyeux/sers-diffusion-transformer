"""Meaningful artificial checks for cached concentration adaptation and isolation."""
import ast
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from Model_T3_24 import ConcentrationUpdateModel,ReferenceJointModel,DRUGS,digest,corn_probabilities
from engine_t324 import seed_everything,fit_bank,cache_dataset,evaluate_cache,load_checkpoint,checkpoint_payload,train_variant,adoption_guard,metric_counts,make_frame
from data_t324 import measured_masks
from T3_24_Training import build_cache,suite_report
from Loss_T3_24 import correction_objective
ROOT=Path(__file__).resolve().parent
torch.set_num_threads(2)
DEVICE=torch.device("cpu")

class ArtificialDataset(Dataset):
    def __init__(self,split):
        from itertools import product
        self.split=split;self.reads=0;self.samples=[];self.rows=[]
        records={}
        for matrix in ("water","soil"):
            if split=="train":
                for d,drug in enumerate(DRUGS):
                    for level in (1,2,3):
                        levels=[0.,0.,0.];levels[d]=float(level)
                        condition=f"{drug}-{level}_{matrix}"
                        for i in range(12):self._add("real",condition,i,matrix,levels)
                        records[condition]=self._record(drug=="DEL" and level==1)
            for levels in product((1.,2.,3.),repeat=3):
                condition=f"mix_{levels}_{matrix}";records[condition]=self._record(False)
                for i in (range(2) if split=="train" else [12]):self._add("real",condition,i,matrix,levels)
                if split=="train":self._add("generated",condition,100,matrix,levels)
        self.repository=SimpleNamespace(real_data=records,generated_data=records)

    def _record(self,short):
        axis=np.arange(600,2501,dtype=float)
        return dict(axis=axis,original_axis_points=1401 if short else 1901,
                    **{"original_axis_start_cm-1":600.,"original_axis_end_cm-1":2000. if short else 2500.})

    def _add(self,source,condition,index,matrix,levels):
        self.samples.append(SimpleNamespace(source=source,condition=condition,spectrum_index=index))
        self.rows.append((matrix,torch.tensor(levels),source,condition,index))

    def __len__(self):return len(self.samples)

    def __getitem__(self,i):
        if self.split=="test":raise AssertionError("Test spectra must never be evaluated")
        self.reads+=1;matrix,levels,source,condition,index=self.rows[i]
        generator=torch.Generator().manual_seed(2026+i)
        x=torch.arange(600,2501,dtype=torch.float32)
        signal=torch.randn(1901,generator=generator)*.3+(2 if matrix=="soil" else 1)
        for d,centers in enumerate(((1000.,1600.),(2230.,),(1090.,1597.))):
            for center in centers:signal+=levels[d]*20*torch.exp(-.5*((x-center)/8).square())
        # Runtime completion exists, but the reference bank must exclude it.
        if self.repository.real_data[condition]["original_axis_points"]==1401:signal[1401:]=777.
        raw=signal.sign()*torch.log1p(signal.abs());raw=(raw-raw.min())/(raw.max()-raw.min())
        return dict(raw=raw[None,:],smoothed=raw[None,:],percentile=raw[None,:].repeat(4,1),
                    raw_intensity=signal[None,:],valid_mask=torch.ones(1901,dtype=torch.bool),
                    concentration_target=levels,class_target=(levels>0).float(),source=source,
                    condition=condition,spectrum_index=index,matrix_name=matrix)




def new_model(arm="concentration_update"):
    seed_everything(2026)
    config=json.loads((ROOT/"reference_random48.json").read_text())["model_config"]
    centers=json.loads((ROOT/"expected_reference_state.json").read_text())["peak_centers"]
    return ConcentrationUpdateModel(config,centers,experiment_variant=arm).eval()


def config_for(model,epochs=2):
    return dict(model_config=model.config,experiment_variant=model.experiment_variant,epochs=epochs,batch_size=16,
        learning_rate=1e-5,minimum_learning_rate=1e-7,warmup_epochs=1 if epochs>1 else 0,seed=2026,
        correction_penalty=.02,middle_weight=0.,preservation_weight=0.,middle_margin=.35,
        preservation_margin_cap=.5,gradient_clip_norm=1.)


def slim(cache,n=64):
    return {k:v[:n] if isinstance(v,(torch.Tensor,list)) else v for k,v in cache.items()}


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.datasets=[ArtificialDataset(s) for s in ("train","validation","test")]
        cls.model=new_model()
        cls.audit=fit_bank(cls.model,cls.datasets[0],32,0,DEVICE)
        cls.cache=cache_dataset(cls.model,cls.datasets[0],32,0,DEVICE)
        real=torch.tensor([r["source"]=="real" for r in cls.cache["metadata"]])
        cls.model.correction.fit_normalization(cls.cache["features"][real,:cls.model.residual_feature_dimension],
            ["real"]*int(real.sum()),["train"]*int(real.sum()))
        cls.val=cache_dataset(cls.model,cls.datasets[1],32,0,DEVICE)

    def test_package_sources_compile_on_python_311_grammar(self):
        for file in ROOT.glob("*.py"):
            ast.parse(file.read_text(),feature_version=(3,11));compile(file.read_text(),str(file),"exec")

    def test_pre_fusion_cache_and_initial_exact_outputs(self):
        m=copy.deepcopy(self.model).eval()
        batch=next(iter(DataLoader(self.datasets[1],batch_size=8)))
        batch["measured_mask"]=measured_masks(self.datasets[1],batch)
        captured=[]
        h=m.base.mixture_aware_query_fusion.register_forward_pre_hook(lambda _,args:captured.append(args[0].detach().clone()))
        x,c,b,p=m.frozen_features(batch);h.remove()
        self.assertEqual(x.shape,(8,1650))
        self.assertTrue(torch.equal(x[:,-384:].reshape(8,3,128),captured[0]))
        outputs=m.forward_cached(x,c,b)
        self.assertTrue(torch.equal(outputs[0],c));self.assertTrue(torch.equal(outputs[2],p))
        self.assertTrue(torch.equal(outputs[3],b));self.assertEqual(int(torch.count_nonzero(outputs[4])),0)
        # Full online and cached forward must agree, not merely labels.
        for a,z in zip(m(batch),outputs):self.assertTrue(torch.equal(a,z))

    def test_only_allowed_modules_train_and_pretrained_dropout_stays_eval(self):
        m=copy.deepcopy(self.model).train()
        names=[n for n,p in m.trainable_named_parameters()]
        self.assertTrue(any(n.startswith("updated_fusion.") for n in names))
        self.assertTrue(any(n.startswith("updated_ordinal_heads.") for n in names))
        self.assertFalse(m.base.training);self.assertFalse(m.updated_fusion.training);self.assertFalse(m.updated_ordinal_heads.training)
        self.assertTrue(m.correction.network.training)
        self.assertTrue(all(not p.requires_grad for p in m.base.parameters()))
        m.base.classification_heads[0][0].weight.requires_grad_(True)
        with self.assertRaises(RuntimeError):m.trainable_named_parameters()

    def test_optimizer_really_updates_fusion_heads_and_residual_without_shared_change(self):
        m=copy.deepcopy(self.model).train();before=digest(m.base)
        states={k:v.detach().clone() for k,v in m.state_dict().items()}
        optimizer=torch.optim.Adam([p for _,p in m.trainable_named_parameters()],lr=1e-5)
        # Mixed samples include all three labels; real and generated masks match.
        data=slim({k:(v[-64:] if isinstance(v,(torch.Tensor,list)) else v) for k,v in self.cache.items()})
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            c,_,_,logits,delta=m.forward_cached(data["features"],data["classify"],data["base_logits"])
            loss=correction_objective(logits,data["base_logits"],delta,data["levels"],data["presence"],middle_weight=0,preservation_weight=0)["total"]
            loss.backward()
            self.assertTrue(all(p.grad is None for p in m.base.parameters()))
            for prefix in ("updated_fusion.","updated_ordinal_heads."):
                self.assertTrue(any(p.grad is not None and bool(torch.count_nonzero(p.grad)) for n,p in m.named_parameters() if n.startswith(prefix)))
            torch.nn.utils.clip_grad_norm_([p for _,p in m.trainable_named_parameters()],1.,error_if_nonfinite=True)
            optimizer.step()
        self.assertEqual(digest(m.base),before)
        for prefix in ("updated_fusion.","updated_ordinal_heads.","correction.network."):
            self.assertTrue(any(not torch.equal(v,states[k]) for k,v in m.state_dict().items() if k.startswith(prefix)))
        frame,_,_=evaluate_cache(m,self.val,DEVICE)
        anchor=make_frame(self.val)
        self.assertTrue(frame[["class_probability_"+d for d in DRUGS]].equals(anchor[["class_probability_"+d for d in DRUGS]]))

    def test_control_exactly_matches_existing_residual_computation(self):
        m=new_model("residual_control")
        state={k:v for k,v in self.model.state_dict().items() if k in m.state_dict()};m.load_state_dict(state,strict=True)
        legacy=ReferenceJointModel(**{k:v for k,v in m.config.items() if k!="experiment_variant"})
        legacy.load_state_dict(state,strict=True);legacy.eval()
        with torch.no_grad():m.correction.network[-1].bias.copy_(torch.linspace(-.2,.2,6));legacy.correction.network[-1].bias.copy_(m.correction.network[-1].bias)
        x=self.val["features"][:16];c=self.val["classify"][:16];b=self.val["base_logits"][:16]
        for a,z in zip(m.forward_cached(x,c,b),legacy.forward_cached(x[:,:1266],c,b)):
            self.assertTrue(torch.equal(a,z))
        self.assertFalse(any(n.startswith("updated_") for n,p in m.named_parameters()))

    def test_original_weight_initialization_and_freeze_parameter_counts(self):
        m=new_model();base=copy.deepcopy(m.base.state_dict())
        base["ordinal_heads.0.3.bias"]=torch.tensor([.123,.456])
        m.initialize_anchor(base)
        for k,v in m.base.ordinal_heads.state_dict().items():self.assertTrue(torch.equal(v,m.updated_ordinal_heads.state_dict()[k]))
        for k,v in m.base.mixture_aware_query_fusion.state_dict().items():self.assertTrue(torch.equal(v,m.updated_fusion.state_dict()[k]))
        self.assertEqual(sum(p.numel() for p in m.base.parameters()),1267823)
        control=new_model("residual_control")
        self.assertEqual(sum(p.numel() for _,p in control.trainable_named_parameters()),40742)
        self.assertGreater(sum(p.numel() for _,p in m.trainable_named_parameters()),40742)

    def test_learned_fusion_is_used_in_prediction_and_corn_order_is_preserved(self):
        m=copy.deepcopy(self.model).eval()
        x=self.val["features"][:32];c=self.val["classify"][:32];b=self.val["base_logits"][:32]
        with torch.no_grad():
            m.updated_fusion.gate_logit.add_(1.)
            m.updated_ordinal_heads[0][-1].bias.add_(torch.tensor([.2,-.2]))
        y=m.forward_cached(x,c,b)
        self.assertGreater(float(y[4].detach().abs().max()),0.)
        self.assertTrue(torch.equal(y[0],c));self.assertTrue((y[2][...,1]<=y[2][...,0]).all())
        self.assertTrue(torch.isfinite(y[3]).all())
        with self.assertRaises(ValueError):m.forward_cached(x[:,:1266],c,b)

    def test_forward_ignores_labels_matrix_and_source_metadata(self):
        m=copy.deepcopy(self.model).eval();batch=next(iter(DataLoader(self.datasets[1],batch_size=4)))
        batch["measured_mask"]=measured_masks(self.datasets[1],batch)
        first=m(batch);batch["concentration_target"].fill_(999);batch["class_target"].fill_(0)
        batch["matrix_name"]=["fake"]*4;batch["source"]=["fake"]*4
        for a,b in zip(first,m(batch)):self.assertTrue(torch.equal(a,b))

    def test_real_train_only_fit_and_unmeasured_tail_exclusion(self):
        self.assertEqual(self.audit["template_rows"],216)
        self.assertEqual(self.audit["validation_fit_rows"]+self.audit["test_fit_rows"]+self.audit["generated_fit_rows"],0)
        m=copy.deepcopy(self.model);raw=torch.randn(2,1,1901);mask=torch.arange(1901)[None,:].repeat(2,1)<1401
        a=m.bank(raw,mask);raw[:,:,1401:]=999999.;b=m.bank(raw,mask)
        for x,y in zip(a,b):self.assertTrue(torch.equal(x,y))
        with self.assertRaises(RuntimeError):m.correction.fit_normalization(torch.zeros(1,1266),["real"],["train"])
        m=new_model()
        with self.assertRaises(ValueError):m.correction.fit_normalization(torch.zeros(1,1266),["real"],["validation"])

    def test_checkpoint_roundtrip_preserves_updated_fusion_online_predictions(self):
        m=copy.deepcopy(self.model).eval()
        with torch.no_grad():m.updated_fusion.gate_logit.add_(.3);m.updated_ordinal_heads[1][-1].bias.add_(.1)
        batch=next(iter(DataLoader(self.datasets[1],batch_size=8)));batch["measured_mask"]=measured_masks(self.datasets[1],batch)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"model.pt";torch.save(checkpoint_payload(m,config_for(m),1,{},False,{}),path)
            restored,payload=load_checkpoint(path,DEVICE)
            self.assertEqual(payload["checkpoint_version"],"T3.24_concentration_update_v1")
            for a,b in zip(m(batch),restored(batch)):self.assertTrue(torch.equal(a,b))
            self.assertTrue(torch.equal(m.updated_fusion.gate_logit,restored.updated_fusion.gate_logit))

    def test_matched_loss_matches_plain_formula_and_masks_absent_drugs(self):
        logits=torch.tensor([[[.2,.1],[.5,-.2],[.1,-.3]]],requires_grad=True)
        base=logits.detach().clone();delta=logits-base
        levels=torch.tensor([[1.,2.,0.]]);presence=(levels>0).float()
        loss=correction_objective(logits,base,delta,levels,presence)
        from Model_T3_17 import present_only_corn_ordinal_loss
        self.assertTrue(torch.equal(loss["total"],.7*present_only_corn_ordinal_loss(logits,levels,presence)))
        loss["total"].backward()
        self.assertEqual(float(logits.grad[0,0,1]),0.)
        self.assertTrue(torch.equal(logits.grad[0,2],torch.zeros(2)))
        with self.assertRaises(ValueError):correction_objective(logits,base,delta,levels,presence,middle_weight=.1)

    def test_guard_keeps_actual_group_regression_and_no_gain_rejection(self):
        hist=json.loads((ROOT/"previous_t323_reference.json").read_text())["validation_metrics"]
        for arm in ("protected","plain"):
            passed,reasons=adoption_guard(hist[arm]["candidate_validation_metrics"],hist[arm]["anchor_validation_metrics"])
            self.assertFalse(passed);self.assertTrue(any("decreased" in x for x in reasons))
        anchor=metric_counts(make_frame(self.val));self.assertFalse(adoption_guard(anchor,anchor)[0])

    def test_two_arm_training_fallback_scheduler_reports_and_no_test_reads(self):
        initial=copy.deepcopy(self.model.state_dict());summaries={}
        with tempfile.TemporaryDirectory() as tmp:
            for arm in ("concentration_update","residual_control"):
                m=new_model(arm);cfg=config_for(m)
                out=Path(tmp)/arm
                summaries[arm]=train_variant(initial,slim(self.cache),self.val,out,cfg,{},DEVICE,smoke=True)
                history=pd.read_csv(out/"training_history.csv")
                self.assertEqual(int(history.optimizer_updates.iloc[-1]),8)
                self.assertAlmostEqual(float(history.learning_rate.iloc[-1]),1e-7)
                self.assertFalse(history.passes_adoption_guard.any())
                self.assertEqual(summaries[arm]["selected_kind"],"original_zero_correction_fallback")
                restored,_=load_checkpoint(out/"checkpoints/recommended.pt",DEVICE)
                frame,metrics,_=evaluate_cache(restored,self.val,DEVICE)
                self.assertEqual(metrics,metric_counts(make_frame(self.val)))
                self.assertTrue(all(summaries[arm]["latest_trainable_modules_changed"].values()))
                self.assertTrue((out/"latest_parameter_changes.csv").is_file())
                self.assertTrue((out/"new_validation_errors.csv").is_file())
            suite_report(Path(tmp),summaries,True)
            self.assertTrue((Path(tmp)/"comparison.md").is_file())
        self.assertEqual(self.datasets[2].reads,0)

    def test_accepted_checkpoint_branch_restores_learned_modules(self):
        # Mock selection only to cover accepted serialization, not performance.
        m=copy.deepcopy(self.model)
        with tempfile.TemporaryDirectory() as tmp,patch("engine_t324.adoption_guard",return_value=(True,[])):
            out=Path(tmp)/"accepted"
            summary=train_variant(copy.deepcopy(m.state_dict()),slim(self.cache,32),self.val,out,config_for(m,1),{},DEVICE,smoke=False)
            restored,payload=load_checkpoint(out/"checkpoints/recommended.pt",DEVICE)
            self.assertTrue(payload["accepted"]);self.assertEqual(summary["selected_kind"],"accepted_concentration_update")
            self.assertEqual(digest(m.base),digest(restored.base))
            self.assertTrue(any(not torch.equal(v,m.updated_ordinal_heads.state_dict()[k]) for k,v in restored.updated_ordinal_heads.state_dict().items()))

    def test_full_cache_builder_caches_queries_and_fits_complete_real_train(self):
        datasets=[ArtificialDataset(s) for s in ("train","validation","test")];m=new_model()
        with tempfile.TemporaryDirectory() as tmp:
            cache=build_cache(datasets,m,Path(tmp),32,0,DEVICE,smoke=True)
            self.assertEqual(cache["normalization_fit_rows"],324)
            self.assertEqual(cache["train"]["features"].shape,(96,1650))
            self.assertEqual(cache["validation"]["features"].shape,(48,1650))
            audit=json.loads((Path(tmp)/"cache_audit.json").read_text())
            self.assertEqual(audit["feature_dimensions"],dict(residual=1266,query=384,total=1650))
        self.assertEqual(datasets[2].reads,0)


if __name__=="__main__":unittest.main()
