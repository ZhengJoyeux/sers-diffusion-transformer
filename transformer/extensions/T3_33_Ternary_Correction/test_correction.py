"""Verify zero initialization, frozen state, routing, gradients and checkpoints."""
import argparse
import copy
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import torch

p=argparse.ArgumentParser(add_help=False)
p.add_argument('--image-package',type=Path,default=Path(__file__).resolve().parent.parent/'T3_32_Image_Recognition')
cli,remaining=p.parse_known_args();sys.argv=[sys.argv[0]]+remaining
from baseline_bridge import configure,state_digest,load_baseline
runtime,hashes=configure(cli.image_package)
from image_model import ImageSERSFormer
from image_data import PlotScale
from correction_model import TernaryCorrector,extract_frozen,ordinal_probabilities,protection_check,correction_loss
from feature_cache import CachedTraining
from run_correction import make_optimizer,checkpoint,load_correction,admissible,rank
torch.set_num_threads(2)


def baseline():
    torch.manual_seed(2026)
    model=ImageSERSFormer(layers=2,tokens=24)
    model.multi_peak_evidence.fit_normalization(torch.randn(12,1,1901)+8,torch.ones(12,1901,dtype=torch.bool))
    return model.requires_grad_(False).eval()


def cache(b=20):
    torch.manual_seed(2026)
    cp=torch.full((b,3),.9);cp[b//2:,2]=.1
    logits=torch.randn(b,3,2)
    return dict(features=torch.randn(b,371),cp=cp,op=ordinal_probabilities(logits),logits=logits)


def initialized():
    model=TernaryCorrector(baseline())
    data=cache()
    model.fit_normalization(data['features'],torch.ones(20,dtype=torch.bool),(data['cp']>=.5).all(1),split='train')
    return model,data


def args():
    return SimpleNamespace(learning_rate=1e-4,minimum_learning_rate=1e-6,weight_decay=.01,
        device='cpu',ordinal_decoding='median',t332_hashes=hashes)


class CorrectionTests(unittest.TestCase):
    def test_zero_initialization_is_exact_identity(self):
        model,data=initialized();model.eval()
        with torch.no_grad():output=model.forward_cached(data)
        for key in ('cp','op','logits'):torch.testing.assert_close(output[key],data[key],rtol=0,atol=0)
        self.assertEqual(torch.count_nonzero(output['delta']),0)

    def test_exact_identity_with_cached_probability_roundoff(self):
        model,data=initialized();data['op']=data['op']+1e-7
        with torch.no_grad():output=model.forward_cached(data)
        torch.testing.assert_close(output['op'],data['op'],rtol=0,atol=0)

    def test_only_routed_DEL_TEB_can_change(self):
        model,data=initialized()
        with torch.no_grad():
            for head in model.correction_heads:head[-1].bias.fill_(1.)
            output=model.forward_cached(data)
        self.assertTrue(all(protection_check(output,data).values()))
        self.assertGreater(float((output['op'][:10,0]-data['op'][:10,0]).abs().sum()),0)
        self.assertGreater(float((output['op'][:10,2]-data['op'][:10,2]).abs().sum()),0)

    def test_logit_correction_bound(self):
        model,data=initialized()
        with torch.no_grad():
            for head in model.correction_heads:head[-1].bias.fill_(100.)
            output=model.forward_cached(data)
        self.assertLessEqual(float(output['delta'].abs().max()),.5)
        self.assertTrue((output['op'][...,1]<=output['op'][...,0]).all())

    def test_train_mode_keeps_entire_baseline_eval(self):
        model,_=initialized();model.train()
        self.assertFalse(any(m.training for m in model.baseline.modules()))
        self.assertTrue(model.correction_heads.training)
        self.assertFalse(any(p.requires_grad for p in model.baseline.parameters()))

    def test_backward_and_optimizer_cannot_change_baseline(self):
        model,data=initialized();model.train();before=state_digest(model.baseline)
        optimizer=make_optimizer(model,args())
        output=model.forward_cached(data)
        loss,_=correction_loss(output,torch.ones(20,3),torch.ones(20,3))
        loss.backward()
        self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in model.correction_heads.parameters()))
        self.assertFalse(any(p.grad is not None for p in model.baseline.parameters()))
        optimizer.step();self.assertEqual(state_digest(model.baseline),before)

    def test_normalization_never_fits_heldout(self):
        data=cache();model=TernaryCorrector(baseline())
        with self.assertRaises(ValueError):
            model.fit_normalization(data['features'],torch.ones(20,dtype=torch.bool),torch.ones(20,dtype=torch.bool),split='validation')

    def test_generated_values_cannot_change_normalization(self):
        data=cache();real=torch.arange(20)<10
        a=TernaryCorrector(baseline());b=TernaryCorrector(baseline())
        changed=data['features'].clone();changed[10:]=1e9
        a.fit_normalization(data['features'],real,torch.ones(20,dtype=torch.bool),split='train')
        b.fit_normalization(changed,real,torch.ones(20,dtype=torch.bool),split='train')
        torch.testing.assert_close(a.feature_mean,b.feature_mean,rtol=0,atol=0)
        torch.testing.assert_close(a.feature_std,b.feature_std,rtol=0,atol=0)

    def test_refit_rejected(self):
        model,data=initialized()
        with self.assertRaises(RuntimeError):
            model.fit_normalization(data['features'],torch.ones(20,dtype=torch.bool),torch.ones(20,dtype=torch.bool),split='train')

    def test_labels_and_metadata_rejected_by_forward(self):
        model,data=initialized();data['levels']=torch.ones(20,3)
        with self.assertRaises(ValueError):model.forward_cached(data)

    def test_prediction_route_is_independent_of_labels(self):
        model,data=initialized();output=model.forward_cached(data)
        self.assertEqual(int(output['route'].sum()),10)
        # Training labels can exclude rows from the loss, but cannot change the route.
        presence=torch.zeros(20,3)
        loss,detail=correction_loss(output,torch.zeros(20,3),presence)
        self.assertEqual(detail['supervised_rows'],0);self.assertEqual(float(loss.detach()),0.)

    def test_validation_cache_cannot_train(self):
        with self.assertRaises(ValueError):CachedTraining(dict(split='validation'))

    def test_feature_extraction_and_full_forward(self):
        base=baseline()
        with torch.no_grad():
            for head in base.classification_heads:
                head[-2].weight.zero_();head[-2].bias.fill_(4.)
        b=10
        inputs=dict(image=torch.ones(b,1,64,128),image_valid_columns=torch.ones(b,128,dtype=torch.bool),
            raw_intensity=torch.randn(b,1,1901)+8,measured_mask=torch.ones(b,1901,dtype=torch.bool))
        frozen=extract_frozen(base,inputs)
        self.assertEqual(tuple(frozen['features'].shape),(10,371))
        model=TernaryCorrector(base)
        model.fit_normalization(frozen['features'],torch.ones(b,dtype=torch.bool),torch.ones(b,dtype=torch.bool),split='train')
        with torch.no_grad():direct=model(inputs);cached=model.forward_cached(frozen)
        torch.testing.assert_close(direct['op'],cached['op'],rtol=0,atol=0)

    def test_embedded_checkpoint_roundtrip(self):
        model,data=initialized();model.eval();optimizer=make_optimizer(model,args())
        digest=state_digest(model.baseline)
        reference=dict(model_config=model.baseline.config,checkpoint_sha256='synthetic-test',epoch=16)
        report=dict(cache_fingerprint='test',split_audit={},baseline_digest=digest)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'test.pt'
            checkpoint(path,model,PlotScale(),reference,report,args(),0,{},optimizer,0)
            restored,scale,state=load_correction(path,args())
            with torch.no_grad():a=model.forward_cached(data);b=restored.forward_cached(data)
            torch.testing.assert_close(a['op'],b['op'],rtol=0,atol=0)
            self.assertEqual(state_digest(restored.baseline),digest)
            self.assertEqual(state['checkpoint_version'],'T3.33_frozen_image_ternary_DEL_TEB_v1')

    def test_wrong_epoch_baseline_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'wrong.pt'
            model=baseline()
            torch.save(dict(checkpoint_version='T3.32_line_image_recognition_v1',model_config=model.config,
                model_state_dict=model.state_dict(),plot_scale=asdict(PlotScale()),epoch=28),path)
            with self.assertRaises(ValueError):load_baseline(runtime,path)

    def test_unreported_epoch16_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'wrong_run.pt';model=baseline()
            torch.save(dict(checkpoint_version='T3.32_line_image_recognition_v1',model_config=model.config,
                model_state_dict=model.state_dict(),plot_scale=asdict(PlotScale()),epoch=16,
                validation_metrics=dict(samples=504,present_correct=1)),path)
            with self.assertRaises(ValueError):load_baseline(runtime,path)

    def test_guard_rejects_overall_regression(self):
        reference=dict(present_correct=1026,full_profile_correct=406,classification_exact_correct=504,ternary_all_grade_correct=131)
        candidate=dict(reference,ternary_all_grade_correct=140,present_correct=1025)
        self.assertFalse(admissible(candidate,reference))
        self.assertTrue(admissible(dict(reference,ternary_all_grade_correct=132,full_profile_correct=407),reference))


if __name__=='__main__':unittest.main(verbosity=2)
