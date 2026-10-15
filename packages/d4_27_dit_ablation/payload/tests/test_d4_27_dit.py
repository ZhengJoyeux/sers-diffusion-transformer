import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from src.dit_1d import SpectrumDiT1D
from src.broad_local_residual import BroadLocalResidualDecomposer
from src.conditional_prior_residual import ConditionalPriorResidualBank
from src.model_builder import build_diffusion_model

torch.set_num_threads(2)


def tiny_model(checkpointing=False):
    return SpectrumDiT1D(channels=1, condition_dimension=14, prior_conditioning_enabled=True,
                         hidden_size=32, depth=2, num_heads=4, patch_size=8,
                         gradient_checkpointing=checkpointing)


@pytest.mark.parametrize('length', [1401, 1901, 1904])
def test_shape_zero_initialization_and_no_convolution(length):
    model = tiny_model()
    x = torch.randn(2, 1, length)
    result = model(x, torch.tensor([1, 2]), condition=torch.randn(2, 14), prior_conditioning=x)
    assert result.shape == x.shape
    assert torch.count_nonzero(result) == 0
    assert not any(isinstance(m, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)) for m in model.modules())


def activate(model):
    with torch.no_grad():
        model.final_linear.weight.normal_(0, .05)
        for block in model.blocks:
            block.adaLN_modulation[-1].weight.normal_(0, .02)


def test_masked_padding_cannot_change_valid_output():
    model = tiny_model().eval()
    activate(model)
    x = torch.randn(2, 1, 1904); base = torch.randn_like(x)
    mask = torch.zeros_like(x); mask[0, :, :1401] = 1; mask[1, :, :1901] = 1
    args = dict(valid_mask=mask, condition=torch.randn(2, 14))
    expected = model(x, torch.tensor([1, 2]), prior_conditioning=base, **args)
    changed = model(x + (1 - mask) * 10000, torch.tensor([1, 2]),
                    prior_conditioning=base - (1 - mask) * 10000, **args)
    torch.testing.assert_close(expected, changed)
    assert torch.count_nonzero(changed * (1 - mask)) == 0


def test_time_chemical_condition_and_base_change_predictions():
    model = tiny_model().eval(); activate(model)
    x = torch.randn(2, 1, 64); c = torch.randn(2, 14); t = torch.tensor([1, 2])
    y = model(x, t, condition=c, prior_conditioning=x)
    for result in (model(x, t + 10, condition=c, prior_conditioning=x),
                   model(x, t, condition=c + 1, prior_conditioning=x),
                   model(x, t, condition=c, prior_conditioning=x + 1)):
        assert not torch.allclose(y, result)


def test_gradient_checkpointing_and_weight_roundtrip():
    model = tiny_model(True); activate(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(2, 1, 64); c = torch.randn(2, 14); t = torch.tensor([1, 2])
    loss = (model(x, t, condition=c, prior_conditioning=x) - x).square().mean()
    loss.backward()
    assert model.blocks[0].qkv.weight.grad.abs().sum() > 0
    optimizer.step()
    restored = tiny_model(); restored.load_state_dict(model.state_dict())
    model.eval(); restored.eval()
    torch.testing.assert_close(model(x, t, condition=c, prior_conditioning=x),
                               restored(x, t, condition=c, prior_conditioning=x))


@pytest.mark.parametrize('mode', ['broad_local', 'full_residual'])
def test_residual_roundtrip_and_sampling(mode):
    rng = np.random.default_rng(2026)
    raw = rng.normal(0, .1, (12, 128)).astype(np.float32)
    decomposer = BroadLocalResidualDecomposer.from_configuration({'enabled': True, 'residual_mode': mode})
    decomposer.fit(raw, raman_shift=np.arange(128) + 600)
    broad, learned = decomposer.split_raw_residuals(raw)
    restored = BroadLocalResidualDecomposer.from_state_dict(json.loads(json.dumps(decomposer.state_dict())))
    assert restored.residual_mode == mode
    np.testing.assert_allclose(restored.inverse_local_transform(restored.transform_raw_residuals(raw)) + broad, raw, atol=2e-7)
    if mode == 'full_residual':
        np.testing.assert_array_equal(learned, raw)
        assert not broad.any()
        before = copy.deepcopy(rng.bit_generator.state)
        assert not restored.sample_broad_residuals(100, random_generator=rng).any()
        assert before == rng.bit_generator.state
    else:
        assert broad.any()


def test_old_state_defaults_to_original_mode():
    d = BroadLocalResidualDecomposer.from_configuration({'enabled': True})
    d.fit(np.random.default_rng(1).normal(size=(12, 64)), raman_shift=np.arange(64) + 600)
    state = d.state_dict(); state.pop('residual_mode')
    assert BroadLocalResidualDecomposer.from_state_dict(state).residual_mode == 'broad_local'


@pytest.mark.parametrize('mode', ['broad_local', 'full_residual'])
def test_conditional_bank_cross_fit_masks_and_reconstruction(mode):
    rng = np.random.default_rng(4); axis = np.arange(1901) + 600
    values = rng.normal(0, .04, (40, 1901)).astype(np.float32)
    values += np.sin(axis / 25).astype(np.float32)
    masks = np.ones_like(values); masks[20:, 1401:] = 0; values *= masks
    ids = ['a'] * 20 + ['b'] * 20; train = list(range(12)) + list(range(20, 32))
    prior = dict(enabled=True, prior_method='pca_reconstruction', pca_max_components=3,
                 training_cross_fit=dict(enabled=True, method='leave_one_out'),
                 raman_variance_equalization=dict(enabled=True, equalization_power=.5))
    bank = ConditionalPriorResidualBank(prior_configuration=prior,
                                      broad_local_configuration=dict(enabled=True, residual_mode=mode))
    bank.fit(values, valid_masks=masks, condition_ids=ids, training_indices=train, raman_shift=axis)
    targets, bases = bank.transform_training_aware_with_conditioning(values, valid_masks=masks,
                                                                   condition_ids=ids, training_indices=train)
    for condition, rows in [('a', slice(0, 20)), ('b', slice(20, 40))]:
        restored = bank.reconstruct_generated_with_conditioning(targets[rows], prior_conditioning=bases[rows], condition_id=condition)
        np.testing.assert_allclose(restored, values[rows, :bank.valid_length(condition)], atol=3e-6)
    restored = ConditionalPriorResidualBank.from_state_dict(json.loads(json.dumps(bank.state_dict())))
    components = []
    base = restored.sample_generation_conditioning(100, condition_id='b', prior_random_generator=np.random.default_rng(1),
                                                   broad_random_generator=np.random.default_rng(2), sampled_components=components)
    assert base.shape == (100, 1901) and not base[:, 1401:].any()
    if mode == 'full_residual':
        assert not components[0]['broad'].any()
        np.testing.assert_array_equal(base[:, :1401], components[0]['outer'])
    # Holdout intensities cannot affect any fitted state.
    altered = values.copy(); altered[[i for i in range(40) if i not in train]] += masks[[i for i in range(40) if i not in train]] * 20
    other = ConditionalPriorResidualBank(prior_configuration=prior, broad_local_configuration=dict(enabled=True, residual_mode=mode))
    other.fit(altered, valid_masks=masks, condition_ids=ids, training_indices=train, raman_shift=axis)
    assert json.dumps(other.state_dict(), sort_keys=True) == json.dumps(bank.state_dict(), sort_keys=True)


def test_real_builder_loss_sampling_and_ema():
    from src.ddpm_trainer import ExponentialMovingAverage
    config = dict(model=dict(architecture='dit', model_dimension=32, dimension_multipliers=[1,2,4], channels=1,
                             dit=dict(hidden_size=32, depth=2, num_heads=4, patch_size=8, gradient_checkpointing=True)),
                  diffusion=dict(diffusion_steps=8, sampling_steps=4, objective='pred_x0', beta_schedule='cosine', auto_normalize=False),
                  data=dict(raman_axis_mode='union_with_valid_mask'), prior_residual=dict(enabled=True),
                  broad_local_residual=dict(enabled=True, residual_mode='full_residual'),
                  conditioning=dict(enabled=True, vector_size=14, prior_spectrum=dict(enabled=True)))
    model, diffusion = build_diffusion_model(config, 64)
    x = torch.randn(2, 1, 64) * .1; mask = torch.ones_like(x); mask[1, :, 48:] = 0
    kwargs = dict(valid_mask=mask, condition=torch.randn(2,14), prior_conditioning=x.clone())
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-3)
    for _ in range(3):
        optimizer.zero_grad(); loss = diffusion(x * mask, **kwargs); assert torch.isfinite(loss); loss.backward(); optimizer.step()
    ema = ExponentialMovingAverage(diffusion, decay=.995, update_every=1)
    ema.update(diffusion)
    samples = ema.ema_model.sample(batch_size=2, **kwargs)
    assert samples.shape == x.shape and torch.isfinite(samples).all()
    assert not (samples * (1-mask)).any()
