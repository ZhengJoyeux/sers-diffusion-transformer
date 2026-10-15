"""Inject numeric branch differences; verify exact identity without hiding learning."""
from pathlib import Path
import argparse
import sys
from types import MethodType
import unittest
import torch
from torch import nn

p=argparse.ArgumentParser(add_help=False)
p.add_argument('--package-directory',type=Path,default=Path('/home/wqzheng/T3_36_Ternary_Fusion_Finetune'))
p.add_argument('--correction-package',type=Path,default=Path('/home/wqzheng/T3_33_Ternary_Correction'))
p.add_argument('--image-package',type=Path,default=Path('/home/wqzheng/T3_32_Image_Recognition'))
p.add_argument('--device',default='cpu')
cli,remaining=p.parse_known_args();sys.argv=[sys.argv[0]]+remaining
sys.path.insert(0,str(cli.correction_package.resolve()))
sys.path.insert(0,str(cli.image_package.resolve()))
sys.path.insert(0,str(cli.package_directory.resolve()))
from correction_model import TernaryCorrector,protection_check
from spectral_core import MixtureAwareQueryFusion
from fusion_model import FusionFinetuner,ConcentrationBranch,probabilities,head_loss
torch.set_num_threads(2)


def fixture():
    torch.manual_seed(2026)
    teacher=TernaryCorrector(nn.Identity(),feature_dim=371)
    teacher.normalization_fitted.fill_(True)
    heads=nn.ModuleList([nn.Sequential(nn.Linear(64,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,2)) for _ in range(2)])
    branch=ConcentrationBranch(MixtureAwareQueryFusion(128,3,8,.1),nn.Linear(128,64),heads)
    model=FusionFinetuner(teacher,branch).to(cli.device).eval()
    cp=torch.full((8,3),.9,device=cli.device);cp[4:,2]=.1
    logits=torch.randn(8,3,2,device=cli.device)
    data=dict(features=torch.randn(8,371,device=cli.device),queries=torch.randn(8,3,128,device=cli.device),
              cp=cp,logits=logits,op=probabilities(logits))
    return model,data


class IdentityTests(unittest.TestCase):
    def test_injected_kernel_roundoff_is_exact_zero_at_identical_weights(self):
        model,data=fixture()
        original=model.branch.forward
        model.branch.forward=MethodType(lambda self,q,cp: original(q,cp)+1e-4,model.branch)
        with torch.no_grad():
            teacher=model.teacher.forward_cached({k:data[k] for k in ('features','cp','op','logits')})
            output=model.forward_cached(data)
        for key in ('cp','op','logits'):torch.testing.assert_close(output[key],teacher[key],rtol=0,atol=0)
        self.assertEqual(int(torch.count_nonzero(output['logit_change'])),0)

    def test_genuine_small_parameter_update_is_not_suppressed(self):
        model,data=fixture()
        with torch.no_grad():
            model.branch.heads[0][-1].bias.add_(1e-4)
            output=model.forward_cached(data)
        self.assertGreater(float(output['logit_change'][:4,0].abs().max()),0.)
        self.assertTrue(all(protection_check(output,data).values()))

    def test_training_gradients_still_reach_all_trainable_modules(self):
        model,data=fixture();model.train()
        data={k:v[:4] for k,v in data.items()}
        frozen=model.frozen_digest()
        output=model.forward_cached(data)
        loss=head_loss(output,torch.ones(4,3,device=cli.device))
        loss.backward()
        for module in (model.branch.fusion,model.branch.projection,model.branch.heads):
            self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad) for p in module.parameters()))
        self.assertTrue(all(p.grad is None for n,p in model.named_parameters() if not n.startswith('branch.')))
        self.assertEqual(frozen,model.frozen_digest())


if __name__=='__main__':unittest.main()
