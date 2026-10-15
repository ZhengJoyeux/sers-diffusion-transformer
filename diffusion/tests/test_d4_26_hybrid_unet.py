"""Behavioral checks: backward compatibility, physical masks and learned context."""
import copy
import io
import unittest
from unittest import mock
from pathlib import Path
import torch

from src.hybrid_unet_configuration import normalize_hybrid_configuration, validate_hybrid_context
from src.masked_unet import MaskConditionedUnet1D
from src.prior_bottleneck_transformer import PriorBottleneckTransformer
from src.model_builder import build_diffusion_model


def configuration(enabled=True, cross=True):
    return {"data": {"raman_axis_mode": "union_with_valid_mask"},
            "conditioning": {"enabled": True, "vector_size": 14, "embedding_dimension": 8,
                             "injection": "input_and_all_resnet_blocks_film", "prior_spectrum": {"enabled": True}},
            "prior_residual": {"enabled": True}, "broad_local_residual": {"enabled": True},
            "model": {"channels": 1, "model_dimension": 8, "dimension_multipliers": [1, 2],
                      "dropout": 0.0, "bottleneck_transformer": {"enabled": enabled, "cross_attention": cross,
                      "dropout": 0.0}},
            "diffusion": {"diffusion_steps": 4, "sampling_steps": 2, "objective": "pred_x0",
                          "auto_normalize": False, "beta_schedule": "cosine", "loss_weighting": "uniform"}}


def direct_adapter(cross=True):
    return PriorBottleneckTransformer(dim=16, spectrum_channels=1, condition_dim=8,
        time_dim=32, downsample_factor=4,
        configuration={"enabled": True, "cross_attention": cross, "dropout": 0.0})


def activate(adapter):
    with torch.no_grad():
        adapter.output_projection.weight.copy_(0.1 * torch.eye(adapter.output_projection.out_features))


class HybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_disabled_keys_weights_rng_and_strict_old_state(self):
        cfg = configuration(False)
        cfg["model"].pop("bottleneck_transformer")
        torch.manual_seed(12); a, _ = build_diffusion_model(cfg, 64); state = torch.get_rng_state()
        torch.manual_seed(12); b, _ = build_diffusion_model(configuration(False), 64)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(set(a.state_dict()), set(b.state_dict()))
        b.load_state_dict(a.state_dict(), strict=True)
        for k, v in a.state_dict().items():
            self.assertTrue(torch.equal(v, b.state_dict()[k]), k)

    def test_zero_initialized_adapter_matches_old_forward_in_eval(self):
        torch.manual_seed(14); old, _ = build_diffusion_model(configuration(False), 64)
        torch.manual_seed(14); new, _ = build_diffusion_model(configuration(), 64)
        missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(all(k.startswith("bottleneck_transformer.") for k in missing))
        old.eval(); new.eval()
        x = torch.randn(2, 1, 64); mask = torch.ones_like(x); mask[0, :, 41:] = 0
        kwargs = dict(valid_mask=mask, condition=torch.randn(2, 14), prior_conditioning=torch.randn_like(x))
        with torch.no_grad():
            self.assertTrue(torch.equal(old(x, torch.tensor([0, 3]), **kwargs), new(x, torch.tensor([0, 3]), **kwargs)))

    def test_1401_and_1901_geometry_and_partial_boundary(self):
        adapter = direct_adapter()
        mask = torch.zeros(2, 1, 1904); mask[0, :, :1401] = 1; mask[1, :, :1901] = 1
        coverage, valid, center = adapter.token_geometry(mask)
        self.assertEqual(valid.sum(1).tolist(), [351, 476])
        self.assertEqual(coverage[0, 0, 350].item(), 0.25)
        self.assertEqual(coverage[1, 0, 475].item(), 0.25)
        self.assertEqual(center[0, 350].item(), 2000.0)
        self.assertEqual(center[1, 475].item(), 2500.0)
        self.assertTrue(torch.equal(center[0, :350], center[1, :350]))

    def test_invalid_tokens_and_prior_cannot_affect_valid_adapter_output(self):
        torch.manual_seed(1); adapter = direct_adapter().eval(); activate(adapter)
        f = torch.randn(2, 16, 16); mask = torch.ones(2, 1, 64); mask[0, :, 40:] = 0
        base = torch.randn(2, 1, 64); c = torch.randn(2, 8); t = torch.randn(2, 32)
        kwargs = dict(valid_mask=mask, condition_embedding=c, time_embedding=t)
        y = adapter(f, prior_conditioning=base, **kwargs)
        changed = f.clone(); changed[0, :, 10:] = 1e4
        base2 = base.clone(); base2[0, :, 40:] = -1e5
        y2 = adapter(changed, prior_conditioning=base2, **kwargs)
        torch.testing.assert_close(y[0, :, :10], y2[0, :, :10], rtol=0, atol=0)

    def test_actual_prior_changes_cross_attention_output(self):
        torch.manual_seed(2); adapter = direct_adapter().eval(); activate(adapter)
        f = torch.randn(2, 16, 16); mask = torch.ones(2, 1, 64)
        kwargs = dict(valid_mask=mask, condition_embedding=torch.randn(2, 8), time_embedding=torch.randn(2, 32))
        y1 = adapter(f, prior_conditioning=torch.zeros_like(mask), **kwargs)
        y2 = adapter(f, prior_conditioning=torch.linspace(-1, 1, 64).reshape(1, 1, 64).expand_as(mask), **kwargs)
        self.assertGreater(float((y1-y2).detach().abs().max()), 1e-5)

    def test_self_ablation_does_not_consume_adapter_prior(self):
        adapter = direct_adapter(False).eval(); activate(adapter)
        f = torch.randn(2, 16, 16); mask = torch.ones(2, 1, 64)
        kwargs = dict(valid_mask=mask, condition_embedding=torch.randn(2, 8), time_embedding=torch.randn(2, 32))
        self.assertTrue(torch.equal(adapter(f, prior_conditioning=None, **kwargs),
                                    adapter(f, prior_conditioning=torch.randn_like(mask), **kwargs)))

    def test_zero_projection_opens_then_attention_and_film_receive_gradients(self):
        torch.manual_seed(3); model, _ = build_diffusion_model(configuration(), 64)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randn(2, 1, 64); mask = torch.ones_like(x); mask[0, :, 41:] = 0
        c = torch.randn(2, 14); base = torch.randn_like(x)
        for _ in range(3):
            optimizer.zero_grad()
            prediction = model(x, torch.tensor([0, 3]), valid_mask=mask, condition=c, prior_conditioning=base)
            loss = (((prediction-x)*mask)**2).sum() / mask.sum()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for name, parameter in model.named_parameters():
                if any(s in name for s in ("self_attention.in_proj_weight", "cross_attention.in_proj_weight", "condition_film.weight")):
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            optimizer.step()
        for block in model.bottleneck_transformer.blocks:
            for p in (block.self_attention.in_proj_weight, block.cross_attention.in_proj_weight):
                self.assertGreater(float(p.grad.abs().sum()), 0)
        film = [p.grad for n,p in model.named_parameters() if "condition_film.weight" in n]
        self.assertTrue(all(float(g.abs().sum()) > 0 for g in film))
        self.assertEqual(torch.count_nonzero(prediction[0, :, 41:]).item(), 0)

    def test_masked_values_full_network_and_output_tail(self):
        model, _ = build_diffusion_model(configuration(), 64); model.eval(); activate(model.bottleneck_transformer)
        x = torch.randn(2, 1, 64); mask = torch.ones_like(x); mask[0, :, 41:] = 0
        base = torch.randn_like(x); x2=x.clone(); x2[0,:,41:]=10000; b2=base.clone(); b2[0,:,41:]=-10000
        kwargs = dict(valid_mask=mask, condition=torch.randn(2, 14))
        with torch.no_grad():
            y1=model(x,torch.tensor([1,2]),prior_conditioning=base,**kwargs)
            y2=model(x2,torch.tensor([1,2]),prior_conditioning=b2,**kwargs)
        self.assertTrue(torch.equal(y1,y2))
        self.assertEqual(torch.count_nonzero(y1[0,:,41:]).item(),0)

    def test_diffusion_loss_ddim_and_checkpoint_roundtrip(self):
        cfg=configuration(); _, d=build_diffusion_model(cfg,64)
        x=torch.randn(2,1,64)*0.1; m=torch.ones_like(x);m[0,:,41:]=0;c=torch.randn(2,14);b=torch.randn_like(x)
        loss=d.p_losses(x,torch.tensor([0,3]),noise=torch.randn_like(x),valid_mask=m,condition=c,prior_conditioning=b)
        self.assertTrue(torch.isfinite(loss));loss.backward()
        activate(d.model.bottleneck_transformer); d.eval()
        buffer=io.BytesIO();torch.save({'configuration':cfg,'state':d.state_dict()},buffer);buffer.seek(0)
        ckpt=torch.load(buffer,weights_only=False);_,restored=build_diffusion_model(ckpt['configuration'],64)
        restored.load_state_dict(ckpt['state'],strict=True);restored.eval()
        torch.manual_seed(55);y=d.sample(batch_size=2,valid_mask=m,condition=c,prior_conditioning=b)
        torch.manual_seed(55);z=restored.sample(batch_size=2,valid_mask=m,condition=c,prior_conditioning=b)
        self.assertTrue(torch.equal(y,z));self.assertTrue(torch.isfinite(y).all())
        self.assertEqual(torch.count_nonzero(y[0,:,41:]).item(),0)

    def test_ema_deepcopy_state_has_no_stale_context(self):
        from ema_pytorch import EMA
        model,_=build_diffusion_model(configuration(),64)
        ema=EMA(model,beta=0.995,update_every=1,update_after_step=0)
        ema.update()
        x=torch.randn(2,1,64);kw=dict(valid_mask=torch.ones_like(x),condition=torch.randn(2,14),prior_conditioning=torch.randn_like(x))
        model.eval();ema.ema_model.eval()
        self.assertTrue(torch.equal(model(x,torch.tensor([0,2]),**kw),ema.ema_model(x,torch.tensor([0,2]),**kw)))
        self.assertTrue(all(getattr(b,'_d4_condition_embedding',None) is None for b in model.conditioned_resnet_blocks))

    def test_real_length_full_network_forward(self):
        cfg=configuration();cfg['model'].update(model_dimension=32,dimension_multipliers=[1,2,4]);cfg['conditioning']['embedding_dimension']=32
        model,_=build_diffusion_model(cfg,1904);model.eval()
        x=torch.randn(2,1,1904)*0.1;m=torch.ones_like(x);m[0,:,1401:]=0;m[1,:,1901:]=0
        with torch.no_grad():y=model(x,torch.tensor([0,3]),valid_mask=m,condition=torch.randn(2,14),prior_conditioning=torch.randn_like(x))
        self.assertEqual(y.shape,x.shape);self.assertTrue(torch.isfinite(y).all());self.assertEqual(torch.count_nonzero(y*(1-m)).item(),0)

    def test_invalid_config_and_all_masked_sample_rejected(self):
        for raw in [{'enabled':'true'}, {'enabled':True,'depth':0}, {'enabled':True,'num_heads':2.5}, {'enabled':True,'dropout':float('nan')}, {'enabled':True,'unknown':1}]:
            with self.assertRaises((ValueError,TypeError)):normalize_hybrid_configuration(raw)
        cfg=configuration();cfg['conditioning']['prior_spectrum']['enabled']=False
        with self.assertRaises(ValueError):validate_hybrid_context(cfg)
        with self.assertRaises(ValueError):direct_adapter().token_geometry(torch.zeros(2,1,64))

    def test_resume_rejects_old_or_different_transformer_architecture(self):
        from scripts.start_ddpm_training import validate_resume_stage
        old=configuration(False)
        with mock.patch('scripts.start_ddpm_training.load_checkpoint_file',return_value={'configuration':old}):
            with self.assertRaisesRegex(ValueError,'Transformer'):
                validate_resume_stage(checkpoint_path=Path('synthetic.pt'),configuration=configuration())
        self_variant=configuration(True,False)
        with mock.patch('scripts.start_ddpm_training.load_checkpoint_file',return_value={'configuration':self_variant}):
            with self.assertRaisesRegex(ValueError,'Transformer'):
                validate_resume_stage(checkpoint_path=Path('synthetic.pt'),configuration=configuration())


if __name__ == '__main__':
    unittest.main()
