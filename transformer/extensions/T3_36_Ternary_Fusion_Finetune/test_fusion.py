"""Small synthetic tests; no real data, weights, or test split are required."""
from pathlib import Path
import argparse
import copy
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

p = argparse.ArgumentParser(add_help=False)
p.add_argument('--image-package',type=Path,default=Path(__file__).parent.parent/'T3_32_Image_Recognition')
p.add_argument('--correction-package',type=Path,default=Path(__file__).parent.parent/'T3_33_Ternary_Correction')
cli, remaining = p.parse_known_args()
sys.argv = [sys.argv[0]] + remaining
sys.path.insert(0,str(cli.correction_package.resolve()))
from correction_model import TernaryCorrector, protection_check
sys.path.insert(0,str(cli.image_package.resolve()))
from spectral_core import MixtureAwareQueryFusion
from fusion_model import FusionFinetuner, ConcentrationBranch, head_loss, probabilities
from run_fusion import selected_indices, paired_batches, admissible, rank
torch.set_num_threads(2)


def fixture():
    torch.manual_seed(2026)
    teacher = TernaryCorrector(nn.Identity(),feature_dim=371)
    teacher.normalization_fitted.fill_(True)
    for head in teacher.correction_heads:
        with torch.no_grad():
            head[-1].bias.copy_(torch.tensor([.4,-.3]))
    heads = nn.ModuleList([nn.Sequential(nn.Linear(64,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,2)) for _ in range(2)])
    fusion=MixtureAwareQueryFusion(128,3,8,.1)
    branch=ConcentrationBranch(fusion,nn.Linear(128,64),heads)
    model = FusionFinetuner(teacher,branch).eval()
    features = torch.randn(18,371)
    cp = torch.full((18,3),.9)
    cp[9:,2] = .1
    logits = torch.randn(18,3,2)
    data = dict(features=features,cp=cp,op=probabilities(logits),logits=logits,queries=torch.randn(18,3,128))
    return model,data


class FusionTests(unittest.TestCase):
    def test_initial_output_exactly_matches_current_teacher(self):
        model,data=fixture()
        with torch.no_grad():
            old=model.teacher.forward_cached({k:data[k] for k in ('features','cp','op','logits')})
            new=model.forward_cached(data)
        for k in ('cp','op','logits'):
            torch.testing.assert_close(old[k],new[k],rtol=0,atol=0)

    def test_only_copied_concentration_branch_is_trainable(self):
        model,_=fixture()
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),148807)
        self.assertTrue(all(n.startswith('branch.') for n,p in model.named_parameters() if p.requires_grad))

    def test_training_keeps_teacher_and_reference_heads_eval(self):
        model,_=fixture();model.train()
        self.assertTrue(model.branch.heads.training)
        self.assertFalse(any(m.training for m in model.teacher.modules()))
        self.assertFalse(any(m.training for m in model.frozen_branch.modules()))

    def test_unbounded_head_change_and_output_protection(self):
        model,data=fixture()
        with torch.no_grad():
            for head in model.branch.heads:head[-1].bias.add_(3.)
            output=model.forward_cached(data)
        self.assertTrue(all(protection_check(output,data).values()))
        self.assertGreater(float(output['logit_change'][:9,0].abs().max()),.5)
        self.assertTrue((output['op'][...,1]<=output['op'][...,0]).all())

    def test_labels_are_rejected_by_forward(self):
        model,data=fixture()
        with self.assertRaises(ValueError):model.forward_cached(dict(data,levels=torch.ones(18,3)))

    def test_invalid_query_shape_is_rejected(self):
        model,data=fixture()
        data['queries']=torch.randn(18,3,64)
        with self.assertRaises(ValueError):model.forward_cached(data)

    def test_projection_changes_routed_concentration_without_changing_teacher(self):
        model,data=fixture()
        digest=model.frozen_digest()
        with torch.no_grad():
            before=model.forward_cached(data)
            model.branch.projection.bias.add_(.5)
            after=model.forward_cached(data)
        self.assertGreater(float((after['op'][:9,0]-before['op'][:9,0]).abs().max()),0)
        self.assertTrue(all(protection_check(after,data).values()))
        self.assertEqual(digest,model.frozen_digest())

    def test_query_cache_provenance_mismatch_is_rejected(self):
        from query_cache import augment_cache
        with tempfile.TemporaryDirectory() as folder:
            args=SimpleNamespace(output_directory=Path(folder))
            path=Path(folder)/'query_cache'/'expected'/'train.pt'
            path.parent.mkdir(parents=True)
            torch.save(dict(query_fingerprint='wrong',metadata=[],original_fingerprint='old'),path)
            baseline=SimpleNamespace(concentration_projection=nn.Linear(128,64))
            dataset=SimpleNamespace(split='train')
            original=dict(split='train',metadata=[],fingerprint='old')
            with self.assertRaises(RuntimeError):augment_cache(baseline,dataset,None,original,args,'expected',None)

    def test_gradient_and_optimizer_preserve_all_frozen_tensors(self):
        model,data=fixture();model.train()
        inputs={k:v[:9] for k,v in data.items()}
        before=model.frozen_digest()
        optimizer=torch.optim.AdamW(model.branch.parameters(),lr=1e-3)
        output=model.forward_cached(inputs)
        loss=head_loss(output,torch.ones(9,3,dtype=torch.long))+model.anchor_loss()
        loss.backward()
        for module in (model.branch.fusion,model.branch.projection,model.branch.heads):
            self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad) for p in module.parameters()))
        self.assertTrue(all(p.grad is None for n,p in model.named_parameters() if not n.startswith('branch.')))
        optimizer.step()
        self.assertEqual(before,model.frozen_digest())

    def test_absent_rows_cannot_enter_head_loss(self):
        model,data=fixture()
        with self.assertRaises(ValueError):head_loss(model.forward_cached(data),torch.ones(18,3))

    def test_conditional_corn_ignores_second_logit_for_small_grade(self):
        model,data=fixture()
        output=model.forward_cached({k:v[:9] for k,v in data.items()})
        output={k:v.detach() for k,v in output.items()}
        a=head_loss(output,torch.ones(9,3),stability_weight=0.)
        changed={k:v.clone() for k,v in output.items()}
        changed['logits'][...,1]+=100
        b=head_loss(changed,torch.ones(9,3),stability_weight=0.)
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_sampler_uses_every_real_row_once_and_equal_generated_count(self):
        real=torch.arange(13);generated=torch.arange(100,152)
        pairs=list(paired_batches(real,generated,8,2026))
        r=torch.cat([a for a,b in pairs]);g=torch.cat([b for a,b in pairs])
        self.assertEqual(sorted(r.tolist()),real.tolist())
        self.assertEqual(len(r),len(g))
        self.assertTrue(all(len(a)==len(b) for a,b in pairs))

    def test_heldout_cache_rejected_for_optimization(self):
        with self.assertRaises(ValueError):selected_indices(dict(split='validation'))

    def test_no_regression_guard_includes_both_drugs(self):
        reference=dict(present_correct=1027,full_profile_correct=408,classification_exact_correct=504,
                       ternary_all_grade_correct=133,loss_total=.08,per_drug={'DEL':{'correct':332},'TEB':{'correct':332}})
        candidate=copy.deepcopy(reference)
        candidate['ternary_all_grade_correct']+=2
        self.assertTrue(admissible(candidate,reference))
        candidate['per_drug']['DEL']['correct']-=1
        self.assertFalse(admissible(candidate,reference))

    def test_weight_roundtrip(self):
        model,data=fixture();other,_=fixture()
        with torch.no_grad():model.branch.heads[0][-1].bias.add_(1.)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'weights.pt';torch.save(model.state_dict(),path)
            other.load_state_dict(torch.load(path,weights_only=True))
        with torch.no_grad():
            a=model.forward_cached(data);b=other.forward_cached(data)
        for k in ('cp','op','logits'):torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
