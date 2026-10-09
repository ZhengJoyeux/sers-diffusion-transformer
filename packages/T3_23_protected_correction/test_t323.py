"""Artificial tests for train-only fitting, frozen anchor and saved inference."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from torch.utils.data import Dataset
from Model_T3_23 import (ReferenceBank,ReferenceJointModel,JointCorrection,DRUGS,digest,corn_probabilities)
from Model_T3_17 import present_only_corn_ordinal_loss
from data_t323 import measured_masks
from engine_t323 import (seed_everything,fit_bank,cache_dataset,evaluate_cache,load_checkpoint,
                         checkpoint_payload,train_variant,adoption_guard,metric_counts,make_frame)
from T3_23_Training import build_cache,suite_report
from Loss_T3_23 import cumulative_logits,correction_objective

ROOT=Path(__file__).resolve().parent
CENTERS=[1000.,1090.,1597.,1600.,2230.]
torch.set_num_threads(2)


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


def artificial_model():
    config=json.loads((ROOT/"reference_random48.json").read_text())["model_config"]
    seed_everything(2026)
    return ReferenceJointModel(config,CENTERS,variant="no_reference").eval()


def fitted_model():
    model=artificial_model();ds=ArtificialDataset("train")
    audit=fit_bank(model,ds,32,0,torch.device("cpu"))
    cache=cache_dataset(model,ds,32,0,torch.device("cpu"))
    take=torch.tensor([r["source"]=="real" for r in cache["metadata"]])
    model.correction.fit_normalization(cache["features"][take],["real"]*int(take.sum()),["train"]*int(take.sum()))
    return model,ds,cache,audit


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model,cls.dataset,cls.cache,cls.audit=fitted_model()

    def test_all_eighteen_templates_fit_only_twelve_real_training_spectra(self):
        self.assertEqual(self.audit["template_rows"],216)
        self.assertEqual(len(self.audit["templates"]),18)
        self.assertTrue(all(r["indices"]==list(range(12)) and r["source"]=="real" for r in self.audit["templates"]))
        self.assertEqual(self.audit["validation_fit_rows"]+self.audit["test_fit_rows"]+self.audit["generated_fit_rows"],0)

    def test_validation_or_generated_reference_fit_is_rejected(self):
        bank=ReferenceBank(CENTERS)
        for source,split in (("generated","train"),("real","validation")):
            with self.assertRaises(ValueError):
                bank.fit(torch.zeros(1,1,1901),torch.ones(1,1901,dtype=torch.bool),torch.tensor([[1.,0.,0.]]),
                         [dict(source=source,split=split)])

    def test_measured_tail_is_excluded_from_matching(self):
        model=copy.deepcopy(self.model)
        raw=torch.randn(2,1,1901);mask=torch.arange(1901)[None,:].repeat(2,1)<1401
        before=model.bank(raw,mask)
        raw[:,:,1401:]=999999.
        after=model.bank(raw,mask)
        for a,b in zip(before,after):self.assertTrue(torch.equal(a,b))
        self.assertFalse(model.bank.template_mask[0].reshape(5,49)[-1].any())

    def test_zero_correction_matches_anchor_probabilities_exactly(self):
        model=copy.deepcopy(self.model).eval()
        _,_,p,_,delta=model.forward_cached(self.cache["features"],self.cache["classify"],self.cache["base_logits"])
        self.assertTrue(torch.equal(p,self.cache["base_ordinal"]))
        self.assertEqual(float(delta.detach().abs().sum()),0.)

    def test_train_toggle_keeps_anchor_eval_and_only_new_network_trainable(self):
        model=copy.deepcopy(self.model).train()
        self.assertFalse(model.base.training);self.assertFalse(model.bank.training);self.assertTrue(model.correction.training)
        self.assertTrue(all(not p.requires_grad for p in model.base.parameters()))
        self.assertEqual({id(p) for p in model.parameters() if p.requires_grad},{id(p) for p in model.correction.network.parameters()})

    def test_optimizer_updates_correction_without_changing_anchor_or_classifier(self):
        model=copy.deepcopy(self.model).train();anchor=digest(model.base)
        original=copy.deepcopy(model.correction.state_dict())
        optimizer=torch.optim.Adam(model.correction.network.parameters(),lr=1e-3)
        for _ in range(3):
            _,_,_,logits,_=model.forward_cached(self.cache["features"][:16],self.cache["classify"][:16],self.cache["base_logits"][:16])
            loss=present_only_corn_ordinal_loss(logits,self.cache["levels"][:16],self.cache["presence"][:16])
            optimizer.zero_grad();loss.backward();optimizer.step()
        self.assertEqual(anchor,digest(model.base))
        self.assertTrue(any(not torch.equal(v,original[k]) for k,v in model.correction.state_dict().items()))
        classify=model.forward_cached(self.cache["features"][:16],self.cache["classify"][:16],self.cache["base_logits"][:16])[0]
        self.assertTrue(torch.equal(classify,self.cache["classify"][:16]))
        self.assertTrue(all(p.grad is None for p in model.base.parameters()))

    def test_reference_ablation_masks_after_normalization_and_has_equal_parameter_count(self):
        model=copy.deepcopy(self.model).eval()
        model.correction.variant="reference"
        torch.nn.init.normal_(model.correction.network[-1].weight)
        x=self.cache["features"][:3].clone();changed=x.clone()
        a,b=model.correction.reference_slice;changed[:,a:b]+=100.
        ref1=model.correction(x);ref2=model.correction(changed)
        self.assertFalse(torch.equal(ref1,ref2))
        ablation=copy.deepcopy(model);ablation.correction.variant="no_reference"
        self.assertTrue(torch.equal(ablation.correction(x),ablation.correction(changed)))
        self.assertEqual(sum(p.numel() for p in model.correction.parameters()),sum(p.numel() for p in ablation.correction.parameters()))

    def test_true_labels_and_matrix_metadata_do_not_enter_forward(self):
        model=copy.deepcopy(self.model).eval();sample=self.dataset[0]
        from torch.utils.data._utils.collate import default_collate
        batch=default_collate([sample]);batch["measured_mask"]=measured_masks(self.dataset,batch)
        first=model(batch)
        batch["concentration_target"].fill_(3.);batch["class_target"].zero_();batch["matrix_name"]=["soil"]
        batch["condition"]=["different"];batch["source"]=["generated"];batch["spectrum_index"].fill_(19)
        second=model(batch)
        for a,b in zip(first,second):self.assertTrue(torch.equal(a,b))

    def test_bound_and_corn_ordering(self):
        model=copy.deepcopy(self.model).eval()
        model.correction.network[-1].bias.data.fill_(100.)
        _,score,p,_,delta=model.forward_cached(self.cache["features"][:8],self.cache["classify"][:8],self.cache["base_logits"][:8])
        self.assertTrue((delta.abs()<=2.).all());self.assertTrue((p[...,1]<=p[...,0]).all())
        self.assertTrue(((score>=1)&(score<=3)).all())

    def test_unmeasured_windows_have_finite_zero_features(self):
        raw=torch.full((2,1,1901),-1e6);mask=torch.zeros(2,1901,dtype=torch.bool)
        features,reference=self.model.bank(raw,mask)
        self.assertTrue(torch.isfinite(features).all() and torch.isfinite(reference).all())
        self.assertEqual(float(features.abs().sum()+reference.abs().sum()),0.)

    def test_full_feature_model_matches_declared_parameter_count(self):
        centers=json.loads((ROOT/"expected_reference_state.json").read_text())["peak_centers"]
        config=json.loads((ROOT/"reference_random48.json").read_text())["model_config"]
        model=ReferenceJointModel(config,centers)
        self.assertEqual(len(model.correction.feature_mean),1266)
        self.assertEqual(sum(p.numel() for p in model.correction.parameters()),40742)
        self.assertEqual(sum(p.numel() for p in model.base.parameters()),1267823)

    def test_checkpoint_roundtrip_preserves_online_and_cached_prediction(self):
        model=copy.deepcopy(self.model).eval();batch=next(iter(torch.utils.data.DataLoader(self.dataset,batch_size=2)))
        batch["measured_mask"]=measured_masks(self.dataset,batch)
        before=model(batch)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"model.pt";torch.save(checkpoint_payload(model,{},0,{},False,{}),p)
            restored,payload=load_checkpoint(p,torch.device("cpu"));after=restored(batch)
            for a,b in zip(before,after):self.assertTrue(torch.equal(a,b))
            features,classify,logits,_=restored.frozen_features(batch)
            cached=restored.forward_cached(features,classify,logits)
            for a,b in zip(after,cached):self.assertTrue(torch.equal(a,b))

    def test_absent_and_s_second_boundary_have_no_loss_gradient(self):
        logits=torch.randn(1,3,2,requires_grad=True);levels=torch.tensor([[1.,2.,0.]])
        loss=present_only_corn_ordinal_loss(logits,levels,(levels>0).float());loss.backward()
        self.assertEqual(float(logits.grad[0,0,1]),0.)
        self.assertEqual(float(logits.grad[0,2].abs().sum()),0.)

    def test_normalization_refit_or_validation_fit_rejected(self):
        corrector=JointCorrection(4,(1,3))
        with self.assertRaises(ValueError):corrector.fit_normalization(torch.ones(2,4),["real"]*2,["validation"]*2)
        corrector.fit_normalization(torch.ones(2,4),["real"]*2,["train"]*2)
        with self.assertRaises(RuntimeError):corrector.fit_normalization(torch.ones(2,4),["real"]*2,["train"]*2)

    def test_adoption_guard_rejects_ternary_no_gain_and_group_regression(self):
        anchor=metric_counts(make_frame(self.cache));candidate=copy.deepcopy(anchor)
        self.assertFalse(adoption_guard(candidate,anchor)[0])
        candidate["ternary"]["profile_correct"]+=1
        candidate["soil_ternary_DEL_M"]["head_correct_DEL"]+=1
        self.assertTrue(adoption_guard(candidate,anchor)[0])
        candidate["all"]["head_correct"]-=1
        self.assertFalse(adoption_guard(candidate,anchor)[0])

    def test_new_guard_rejects_both_actual_t322_regressions(self):
        historical=json.loads((ROOT/"previous_validation_reference.json").read_text())
        for metrics in historical["validation_metrics"].values():
            accepted,reasons=adoption_guard(metrics,historical["anchor_metrics"])
            self.assertFalse(accepted)
            self.assertIn("water_ternary_profile_correct_decreased",reasons)
            self.assertIn("all_head_correct_DEL_decreased",reasons)
            self.assertIn("soil_DEL_M_TEB_H_head_correct_DEL_decreased",reasons)

    def test_cumulative_high_boundary_uses_product_not_conditional_probability(self):
        conditional=torch.logit(torch.tensor([[[.8,.6],[.6,.9],[.4,.9]]]))
        actual=torch.sigmoid(cumulative_logits(conditional))
        expected=corn_probabilities(conditional)
        self.assertTrue(torch.allclose(actual,expected,atol=1e-7))
        self.assertLess(float(cumulative_logits(conditional)[0,0,1]),0.)
        self.assertGreater(float(conditional[0,0,1]),0.)

    def test_cumulative_logits_and_loss_have_finite_extreme_gradients(self):
        logits=torch.tensor([[[1000.,1000.],[-1000.,1000.],[-1000.,-1000.]]],requires_grad=True)
        y=torch.tensor([[2.,2.,0.]]);base=logits.detach().clone()
        objective=correction_objective(logits,base,logits-base,y,(y>0).float())
        objective["total"].backward()
        self.assertTrue(torch.isfinite(objective["total"]))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_preservation_is_zero_at_anchor_and_penalizes_new_wrong_boundary(self):
        base=torch.logit(torch.tensor([[[.8,.3],[.2,.7],[.9,.9]]]))
        y=torch.tensor([[2.,1.,3.]]);presence=torch.ones_like(y)
        initial=correction_objective(base,base,torch.zeros_like(base),y,presence)
        self.assertEqual(float(initial["preservation"]),0.)
        changed=base.clone();changed[0,0,1]=3.;changed.requires_grad_(True)
        objective=correction_objective(changed,base,changed-base,y,presence)
        self.assertGreater(float(objective["preservation"].detach()),0.)
        objective["preservation"].backward()
        self.assertGreater(float(changed.grad[0,0,1]),0.)

    def test_anchor_errors_are_not_preserved_and_absent_entries_do_not_train(self):
        base=torch.zeros(1,3,2);y=torch.tensor([[3.,0.,0.]])
        changed=torch.ones(1,3,2,requires_grad=True)
        objective=correction_objective(changed,base,changed-base,y,(y>0).float())
        self.assertEqual(float(objective["preservation"].detach()),0.)
        objective["total"].backward()
        self.assertEqual(float(changed.grad[0,1:].abs().sum()),0.)

    def test_middle_margin_applies_only_to_mixed_middle_grades(self):
        base=torch.tensor([[[-.2,-1.],[0.,0.],[0.,0.]],[[.1,-.2],[1.,1.],[0.,0.]]])
        y=torch.tensor([[2.,0.,0.],[2.,3.,0.]])
        changed=base.clone().requires_grad_(True)
        objective=correction_objective(changed,base,changed-base,y,(y>0).float())
        objective["middle"].backward()
        self.assertEqual(float(changed.grad[0].abs().sum()),0.)
        self.assertLess(float(changed.grad[1,0,0]),0.)
        self.assertEqual(float(changed.grad[1,1:].abs().sum()),0.)

    def test_plain_objective_exactly_matches_t322_without_extra_terms(self):
        logits=torch.randn(4,3,2,requires_grad=True);base=torch.randn_like(logits);delta=logits-base
        y=torch.tensor([[1.,2.,3.],[0.,1.,2.],[3.,0.,0.],[2.,2.,0.]])
        active=y>0
        expected=.7*present_only_corn_ordinal_loss(logits,y,active.float())+.02*delta[active].square().mean()
        objective=correction_objective(logits,base,delta,y,active.float(),middle_weight=0.,preservation_weight=0.)
        self.assertTrue(torch.allclose(expected,objective["total"],atol=1e-7))
        a=torch.autograd.grad(expected,logits,retain_graph=True)[0]
        b=torch.autograd.grad(objective["total"],logits)[0]
        self.assertTrue(torch.equal(a,b))

    def test_two_epoch_training_report_and_fallback_never_evaluate_test(self):
        device=torch.device("cpu");val=ArtificialDataset("validation");test=ArtificialDataset("test")
        model=copy.deepcopy(self.model).eval()
        val_cache=cache_dataset(model,val,32,0,device)
        initial=copy.deepcopy(model.state_dict())
        config=dict(model_config=model.config,epochs=2,batch_size=16,learning_rate=1e-4,minimum_learning_rate=1e-6,
                    warmup_epochs=1,seed=2026,correction_penalty=.02,experiment_variant="protected",
                    middle_weight=.1,preservation_weight=.2,middle_margin=.35,preservation_margin_cap=.5)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"protected"
            summary=train_variant(initial,self.cache,val_cache,out,config,{},device,smoke=True)
            self.assertEqual(summary["selected_kind"],"original_zero_correction_fallback")
            self.assertEqual(summary["anchor_state_before"],summary["anchor_state_after"])
            history=__import__('pandas').read_csv(out/"training_history.csv")
            self.assertEqual(int(history.iloc[-1].optimizer_updates),2*int(np.ceil(len(self.cache["features"])/16)))
            self.assertAlmostEqual(float(history.iloc[-1].learning_rate),1e-6)
            self.assertTrue((out/"new_validation_errors.csv").is_file())
            self.assertTrue((out/"validation_transition_counts.csv").is_file())
            self.assertTrue({"train_corn_loss","train_preservation_loss","train_middle_loss"}.issubset(history.columns))
            restored,_=load_checkpoint(out/"checkpoints"/"recommended.pt",device)
            _,metrics,_=evaluate_cache(restored,val_cache,device)
            self.assertEqual(metrics,metric_counts(make_frame(val_cache)))
            suite_report(Path(tmp),{"protected":summary},True)
            self.assertTrue((Path(tmp)/"comparison.md").is_file())
        self.assertEqual(test.reads,0)

    def test_full_cache_pipeline_and_matched_two_variants(self):
        centers=json.loads((ROOT/"expected_reference_state.json").read_text())["peak_centers"]
        config=json.loads((ROOT/"reference_random48.json").read_text())["model_config"]
        seed_everything(2026);model=ReferenceJointModel(config,centers,variant="no_reference").eval()
        datasets=[ArtificialDataset(split) for split in ("train","validation","test")]
        device=torch.device("cpu")
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);cache=build_cache(datasets,model,out,32,0,device,smoke=True)
            self.assertEqual(cache["normalization_fit_rows"],324)
            self.assertEqual(len(cache["train"]["features"]),96)
            self.assertEqual(len(cache["validation"]["features"]),48)
            summaries={}
            for variant in ("protected","plain"):
                training=dict(model_config=dict(cache["model_config"],variant="no_reference"),epochs=2,
                    batch_size=16,learning_rate=1e-4,minimum_learning_rate=1e-6,warmup_epochs=1,
                    seed=2026,correction_penalty=.02,experiment_variant=variant,
                    middle_weight=.1 if variant=="protected" else 0.,preservation_weight=.2 if variant=="protected" else 0.,
                    middle_margin=.35,preservation_margin_cap=.5)
                summaries[variant]=train_variant(cache["initial_state"],cache["train"],cache["validation"],
                    out/variant,training,{},device,smoke=True)
                restored,_=load_checkpoint(out/variant/"checkpoints"/"best_candidate.pt",device)
                batch=next(iter(torch.utils.data.DataLoader(datasets[1],batch_size=8)))
                batch["measured_mask"]=measured_masks(datasets[1],batch)
                online=restored(batch)
                features,c,b,_=restored.frozen_features(batch)
                for a,z in zip(online,restored.forward_cached(features,c,b)):self.assertTrue(torch.equal(a,z))
                self.assertTrue((out/variant/"soil_fixed_DEL_M_nine_conditions.csv").is_file())
            suite_report(out,summaries,True)
            self.assertEqual(summaries["protected"]["training_config"]["trainable_parameters"],40742)
            self.assertEqual(summaries["protected"]["training_config"]["planned_optimizer_updates"],12)
            self.assertEqual(summaries["protected"]["anchor_state_before"],summaries["plain"]["anchor_state_before"])
        self.assertEqual(datasets[2].reads,0)

    def test_accepted_checkpoint_selection_and_reload_branch(self):
        # Selection is mocked only to exercise the accepted-file branch; this
        # test does not claim an artificial or real accuracy improvement.
        from unittest.mock import patch
        device=torch.device("cpu");model=copy.deepcopy(self.model).eval()
        val=cache_dataset(model,ArtificialDataset("validation"),32,0,device)
        small={k:(v[:32] if isinstance(v,(torch.Tensor,list)) else v) for k,v in self.cache.items()}
        config=dict(model_config=model.config,epochs=1,batch_size=16,learning_rate=1e-4,minimum_learning_rate=1e-6,
                    warmup_epochs=0,seed=2026,correction_penalty=.02,experiment_variant="protected",
                    middle_weight=.1,preservation_weight=.2,middle_margin=.35,preservation_margin_cap=.5)
        with tempfile.TemporaryDirectory() as tmp,patch("engine_t323.adoption_guard",return_value=(True,[])):
            out=Path(tmp)/"protected"
            summary=train_variant(copy.deepcopy(model.state_dict()),small,val,out,config,{},device,smoke=False)
            self.assertEqual(summary["selected_kind"],"accepted_correction")
            restored,payload=load_checkpoint(out/"checkpoints/recommended.pt",device)
            self.assertTrue(payload["accepted"])
            self.assertEqual(payload["epoch"],1)
            self.assertEqual(digest(restored.base),digest(model.base))
            self.assertTrue(any(not torch.equal(v,model.correction.state_dict()[k]) for k,v in restored.correction.state_dict().items()))


if __name__=="__main__":unittest.main()
