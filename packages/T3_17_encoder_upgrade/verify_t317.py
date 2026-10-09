"""Server-only artificial model tests; no real spectra or checkpoint updates."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import unittest

parser = argparse.ArgumentParser()
parser.add_argument('--project',type=Path,default=Path.cwd())
args = parser.parse_args()
project = args.project.expanduser().resolve()
package = Path(__file__).resolve().parent
manifest = json.loads((package/'source_manifest.json').read_text())
for name, expected in manifest.items():
    path = project/name
    if not path.is_file() or hashlib.sha256(path.read_text().encode()).hexdigest() != expected:
        raise SystemExit(f'Current source differs from uploaded snapshot: {name}; stop and report this output.')
print('SOURCE SNAPSHOT: PASS',flush=True)
os.chdir(project)
sys.path.insert(0,str(project))
sys.path.insert(1,str(package))

import torch
from Model_T3_17 import TransformerClassifyRegress_sep, _downsample_valid_mask
from Model_v2 import TransformerClassifyRegress_sep as LegacyModel
from T3_17_Training import run_epoch
from T3_17_LR import warmup_cosine_factor
from test_t317_static import StaticTests

torch.set_num_threads(4)


def config(expanded=True):
    return dict(dim_model=32,attn_head=8 if expanded else 4,dim_ff=256 if expanded else 64,
                drop=0.1,batch_f=True,encoder_layers=6 if expanded else 4,n_labels=3,
                model_length=1901,concentration_head_mode='ordinal',ordinal_head_hidden=64,
                use_mixture_aware_query_fusion=True,**({'attention_dim':128} if expanded else {}))


def artificial_batch():
    torch.manual_seed(2026)
    mask = torch.ones(2,1901,dtype=torch.bool)
    mask[1,1401:] = False
    return dict(raw=torch.rand(2,1,1901),smoothed=torch.rand(2,1,1901),
                percentile=torch.rand(2,4,1901),valid_mask=mask,
                class_target=torch.tensor([[1.,1.,0.],[1.,1.,1.]]),
                concentration_target=torch.tensor([[2.,1.,0.],[3.,2.,1.]]))


class RuntimeTests(unittest.TestCase):
    def test_complete_dimension_flow_and_mask(self):
        model = TransformerClassifyRegress_sep(**config()).eval()
        seen = {}
        def record(name):
            def hook(module,inputs,output):
                seen[name] = tuple(output[0].shape if isinstance(output,tuple) else output.shape)
            return hook
        handles = [getattr(model,name).register_forward_hook(record(name)) for name in
                   ['cnn_fusion','token_expansion','transformer_encoder','peak_guided_attention',
                    'classification_projection','mixture_aware_query_fusion','concentration_projection']]
        batch = artificial_batch()
        with torch.no_grad():
            classify, levels, attention, probabilities, logits = model(
                batch,return_attention=True,return_ordinal=True,return_ordinal_logits=True)
        for handle in handles: handle.remove()
        self.assertEqual(seen['cnn_fusion'],(2,210,64))
        self.assertEqual(seen['token_expansion'],(2,210,128))
        self.assertEqual(seen['transformer_encoder'],(2,210,128))
        self.assertEqual(seen['peak_guided_attention'],(2,3,128))
        self.assertEqual(seen['mixture_aware_query_fusion'],(2,3,128))
        self.assertEqual(seen['classification_projection'],(2,3,64))
        self.assertEqual(seen['concentration_projection'],(2,3,64))
        self.assertEqual(len(model.transformer_encoder.layers),6)
        self.assertEqual(model.transformer_encoder.layers[-1].self_attn.num_heads,8)
        self.assertEqual(model.peak_guided_attention.n_heads,8)
        self.assertEqual(model.mixture_aware_query_fusion.n_heads,8)
        self.assertEqual(tuple(classify.shape),(2,3))
        self.assertEqual(tuple(logits.shape),(2,3,2))
        self.assertTrue(torch.all(probabilities[...,1]<=probabilities[...,0]))
        token_mask = _downsample_valid_mask(batch['valid_mask'])
        self.assertEqual(int(token_mask[1].sum()),154)
        self.assertEqual(int(torch.count_nonzero(attention[1,:,~token_mask[1]])),0)
        self.assertTrue(all(torch.isfinite(x).all() for x in [classify,levels,attention,probabilities,logits]))

    def test_training_updates_deep_encoder_projections_and_not_validation_scheduler(self):
        model = TransformerClassifyRegress_sep(**config())
        before = {name:p.detach().clone() for name,p in model.named_parameters() if name in
                  ['token_expansion.weight','classification_projection.weight','concentration_projection.weight',
                   'transformer_encoder.layers.5.self_attn.in_proj_weight']}
        optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda step:
            warmup_cosine_factor(step,total_steps=4,warmup_steps=2,minimum_ratio=.01))
        shared = dict(model=model,loader=[artificial_batch()],device=torch.device('cpu'),
                      classification_loss_fn=torch.nn.BCELoss(),regression_loss_fn=torch.nn.MSELoss(),
                      concentration_head_mode='ordinal',ordinal_loss_mode='corn',ordinal_boundary_weight=0.,
                      ordinal_boundary_margin=.15,ternary_boundary_multiplier=1.,classification_weight=.5,
                      regression_weight=.5,max_batches=None,ordinal_decoding='median')
        metrics,_ = run_epoch(**shared,optimizer=optimizer,update_scheduler=scheduler)
        self.assertTrue(torch.isfinite(torch.tensor(metrics['loss_total'])))
        self.assertEqual(scheduler.last_epoch,1)
        for name,p in model.named_parameters():
            if name in before:
                self.assertTrue(p.grad is not None and torch.isfinite(p.grad).all(),name)
                self.assertFalse(torch.equal(p.detach(),before[name]),name)
        last_epoch = scheduler.last_epoch
        with torch.no_grad():
            run_epoch(**shared,optimizer=None,update_scheduler=scheduler)
        self.assertEqual(scheduler.last_epoch,last_epoch)

    def test_legacy_state_and_predictions_unchanged_when_expansion_disabled(self):
        old = LegacyModel(**config(False)).eval()
        new = TransformerClassifyRegress_sep(**config(False)).eval()
        new.load_state_dict(old.state_dict(),strict=True)
        with torch.no_grad():
            a = old(artificial_batch(),return_ordinal=True,return_ordinal_logits=True)
            b = new(artificial_batch(),return_ordinal=True,return_ordinal_logits=True)
        for first,second in zip(a,b):
            torch.testing.assert_close(first,second,rtol=0,atol=0)

    def test_expanded_checkpoint_roundtrip_and_original_incompatibility(self):
        from Inference_T3_17 import TransformerClassifyRegress_sep as EvaluationModel
        model = TransformerClassifyRegress_sep(**config()).eval()
        buffer = io.BytesIO()
        torch.save({'model_config':config(),'model_state_dict':model.state_dict(),'checkpoint_version':5},buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer,map_location='cpu',weights_only=False)
        restored = EvaluationModel(**checkpoint['model_config']).eval()
        restored.load_state_dict(checkpoint['model_state_dict'],strict=True)
        with torch.no_grad():
            a = model(artificial_batch(),return_ordinal=True)
            b = restored(artificial_batch(),return_ordinal=True)
        for first,second in zip(a,b):
            torch.testing.assert_close(first,second,rtol=0,atol=0)
        with self.assertRaises(RuntimeError):
            model.load_state_dict(LegacyModel(**config(False)).state_dict(),strict=True)


suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(StaticTests),
                           unittest.defaultTestLoader.loadTestsFromTestCase(RuntimeTests)])
result = unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful(): raise SystemExit(1)
print('T3.17 ARTIFICIAL MODEL CHECK: PASS; no real data or optimizer state files changed.',flush=True)
