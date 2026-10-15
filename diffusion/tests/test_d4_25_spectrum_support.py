from __future__ import annotations

import copy
import io

import numpy as np
import pytest
import torch

from src.conditional_prior_residual import ConditionalPriorResidualBank
from src.conditional_spectrum_constraints import (
    ConditionalSpectrumSupport, fit_spectrum_support_state,
    normalize_spectrum_configuration,
)
from src.conditional_diversity_constraints import fit_condition_aware_diversity_constraint_state
from src.model_builder import build_diffusion_model


@pytest.fixture(scope="module")
def fixture():
    rng = np.random.default_rng(425)
    length = 128
    axis = np.arange(600.0, 600.0 + length)
    ids = np.asarray(["DEL-M_water"] * 20 + ["CHL-M_TEB-M_soil"] * 20)
    vectors = np.zeros((40, 14), np.float32)
    vectors[:20, 0] = 1; vectors[20:, 1] = 1
    masks = np.ones((40, length), np.float32); masks[:20, 96:] = 0
    values = np.zeros_like(masks)
    for i in range(40):
        valid = 96 if i < 20 else length
        x = axis[:valid]
        shift = rng.normal(0, 0.5)
        values[i, :valid] = (
            -0.25 + 0.45 * rng.uniform(0.8, 1.2) * np.exp(-0.5 * ((x - 645 - shift) / rng.uniform(2.5, 3.5)) ** 2)
            + 0.25 * rng.uniform(0.8, 1.2) * np.exp(-0.5 * ((x - 675 + shift) / 4) ** 2)
            + rng.normal(0, 0.006, valid)
        )
    train = np.r_[np.arange(12), np.arange(20, 32)]
    prior = {"enabled": True, "prior_method": "pca_reconstruction",
             "pca_explained_variance_ratio": .95, "pca_max_components": 3,
             "pca_sampling_strategy": "independent_truncated_gaussian_scores",
             "pca_score_clip_standard_deviations": 2.5,
             "residual_normalization": "robust_asinh", "residual_quantile": 99.5,
             "training_cross_fit": {"enabled": True, "method": "leave_one_out"},
             "raman_variance_equalization": {"enabled": True, "equalization_power": .5}}
    broad = {"enabled": True, "broad_filter": {"sigma_cm1": 7},
             "broad_prior": {"pca_max_components": 3, "score_clip_standard_deviations": 2.5},
             "local_normalization": {"method": "robust_asinh", "residual_quantile": 99.5}}
    bank = ConditionalPriorResidualBank(prior_configuration=prior, broad_local_configuration=broad).fit(
        values, valid_masks=masks, condition_ids=ids, training_indices=train, raman_shift=axis)
    scaled, base = bank.transform_training_aware_with_conditioning(
        values, valid_masks=masks, condition_ids=ids, training_indices=train)
    configuration = {"full_spectrum_support": {"enabled": True}}
    state = fit_spectrum_support_state(
        bank=bank, training_full_spectra=values[train], training_scaled_residuals=scaled[train],
        training_valid_masks=masks[train], training_condition_vectors=vectors[train],
        training_condition_ids=ids[train], configuration=configuration)
    return dict(bank=bank, scaled=scaled, base=base, values=values, masks=masks,
                ids=ids, train=train, vectors=vectors, state=state, configuration=configuration)


def tensor(value):
    return torch.tensor(value, dtype=torch.float32).unsqueeze(1)


def test_exact_inverse_matches_actual_generation_after_equalization(fixture):
    f = fixture; module = ConditionalSpectrumSupport(f["state"], 128)
    for condition, indices in (("DEL-M_water", np.arange(4)), ("CHL-M_TEB-M_soil", np.arange(20, 24))):
        perturbation = np.linspace(-0.4, 0.4, 128)[None].astype(np.float32)
        prediction = f["scaled"][indices] + perturbation * f["masks"][indices]
        actual = f["bank"].reconstruct_generated_with_conditioning(
            prediction, prior_conditioning=f["base"][indices], condition_id=condition)
        exact = module.reconstruct(tensor(prediction), tensor(f["scaled"][indices]),
                                   tensor(f["values"][indices]), torch.tensor(f["vectors"][indices]), tensor(f["masks"][indices]))
        np.testing.assert_allclose(exact.detach().numpy()[:, 0, :actual.shape[1]], actual, atol=2e-6)
        assert torch.all(exact[:, :, actual.shape[1]:] == 0)


def test_state_uses_only_training_and_accepts_negative_background(fixture):
    f = fixture; changed = f["values"].copy(); changed[12:20] = 1e6; changed[32:] = -1e6
    state = fit_spectrum_support_state(
        bank=f["bank"], training_full_spectra=changed[f["train"]],
        training_scaled_residuals=f["scaled"][f["train"]], training_valid_masks=f["masks"][f["train"]],
        training_condition_vectors=f["vectors"][f["train"]], training_condition_ids=f["ids"][f["train"]],
        configuration=f["configuration"])
    assert state == f["state"]
    assert state["training_counts"] == [12, 12]
    module = ConditionalSpectrumSupport(state, 128)
    scaled = tensor(f["scaled"][:4]); masks = tensor(f["masks"][:4]); condition = torch.tensor(f["vectors"][:4])
    guarded = module.soft_guard(scaled, tensor(f["base"][:4]), condition, masks)
    torch.testing.assert_close(guarded, scaled, atol=2e-6, rtol=1e-5)
    assert torch.any(tensor(f["values"][:4]) < 0)


def test_soft_guard_controls_both_tails_and_keeps_interior_diversity(fixture):
    f = fixture; module = ConditionalSpectrumSupport(f["state"], 128)
    mask = tensor(f["masks"][:4]); condition = torch.tensor(f["vectors"][:4]); base = tensor(f["base"][:4])
    original = tensor(f["scaled"][:4]); extreme = original.clone()
    extreme[:, :, 20] = -10; extreme[:, :, 45] = 10
    guarded = module.soft_guard(extreme, base, condition, mask)
    delivered = (base + module.decode(guarded, condition)) * mask
    idx = module.indices(condition)
    lower = module.field("lower", idx, delivered); upper = module.field("upper", idx, delivered)
    extension = module.field("scale", idx, delivered) * module.configuration["soft_extension_scale"]
    assert torch.all(delivered >= (lower - extension - 2e-6) * mask)
    assert torch.all(delivered <= (upper + extension + 2e-6) * mask)
    torch.testing.assert_close(guarded[:, :, 30:40], original[:, :, 30:40], atol=2e-6, rtol=1e-5)
    assert torch.all(guarded[:, :, 96:] == 0)


def test_full_spectrum_losses_have_finite_corrective_gradients(fixture):
    f = fixture; module = ConditionalSpectrumSupport(f["state"], 128)
    condition = torch.tensor(f["vectors"][:4]); mask = tensor(f["masks"][:4])
    full = tensor(f["values"][:4]); pred = full.clone(); pred[:, :, 45] += .4; pred[:, :, 20] -= .5
    pred.requires_grad_()
    result = module.losses(pred, full, condition, mask, torch.ones(4))
    assert result["envelope_loss"] > 0 and result["peak_derivative_loss"] > 0
    result["raw_loss"].backward()
    assert torch.isfinite(pred.grad).all()
    assert torch.all(pred.grad[:, :, 45] > 0) and torch.all(pred.grad[:, :, 20] < 0)
    gated = module.losses(pred, full, condition, mask, torch.zeros(4))
    assert gated["raw_loss"].item() == 0


def test_no_invalid_tail_enters_derivative_loss(fixture):
    f = fixture; module = ConditionalSpectrumSupport(f["state"], 128)
    full = tensor(f["values"][:4]); condition = torch.tensor(f["vectors"][:4]); mask = tensor(f["masks"][:4])
    changed = full.clone(); changed[:, :, 96:] = 1e8
    result = module.losses(changed, full, condition, mask, torch.ones(4))
    assert result["raw_loss"].item() == 0


def model_configuration(fixture):
    return {
        "data": {"raman_axis_mode": "union_with_valid_mask"},
        "conditioning": {"enabled": True, "vector_size": 14, "embedding_dimension": 8,
                         "injection": "input_and_all_resnet_blocks_film", "prior_spectrum": {"enabled": True}},
        "normalization": {"enabled": True}, "prior_residual": {"enabled": True},
        "broad_local_residual": {"enabled": True},
        "model": {"channels": 1, "base_dimension": 8, "dimension_multipliers": [1, 2], "self_condition": False},
        "diffusion": {"diffusion_timesteps": 10, "sampling_timesteps": 2, "objective": "pred_x0",
                      "beta_schedule": "cosine", "auto_normalize": False, "loss_weighting": "uniform"},
        "diversity_constraints": {"enabled": True, "total_weight": .02, "maximum_total_ratio_to_ddpm": .02,
            "condition_grouping": {"samples_per_condition": 4, "shared_timestep": True},
            "quality_fidelity": {"enabled": True, "total_weight": .1, "maximum_total_ratio_to_ddpm": .1,
                                 "full_spectrum_support": {"enabled": True}}},
    }


@pytest.mark.parametrize("sampling_steps", [2, 10])
def test_training_sampling_checkpoint_and_legacy_tensor_compatibility(fixture, sampling_steps):
    f = fixture; cfg = model_configuration(f)
    cfg["diffusion"]["sampling_timesteps"] = sampling_steps
    state = fit_condition_aware_diversity_constraint_state(
        training_scaled_residuals=f["scaled"][f["train"]], training_valid_masks=f["masks"][f["train"]],
        training_condition_vectors=f["vectors"][f["train"]], configuration=cfg["diversity_constraints"])
    state["full_spectrum_support_state"] = f["state"]
    _, diffusion = build_diffusion_model(cfg, sequence_length=128)
    before = set(diffusion.state_dict())
    diffusion.configure_diversity_constraints(diversity_constraint_state=state)
    assert before == set(diffusion.state_dict())
    scaled = tensor(f["scaled"][:4]); mask = tensor(f["masks"][:4]); condition = torch.tensor(f["vectors"][:4]); base = tensor(f["base"][:4])
    loss = diffusion.p_losses(scaled, torch.zeros(4, dtype=torch.long), noise=torch.zeros_like(scaled),
        valid_mask=mask, condition=condition, prior_conditioning=base,
        full_spectrum_target=tensor(f["values"][:4]), local_inverse_slope=torch.ones_like(scaled))
    loss.backward(); assert torch.isfinite(loss)
    components = diffusion.get_latest_loss_components()
    assert components["spectrum_support_loss"] <= .05 * components["ddpm_loss"] + 1e-6
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in diffusion.parameters())
    generated = diffusion.sample(4, valid_mask=mask, condition=condition, prior_conditioning=base)
    assert torch.isfinite(generated).all() and torch.all(generated[:, :, 96:] == 0)
    output = io.BytesIO(); torch.save({"model": diffusion.state_dict(), "reference": state}, output); output.seek(0)
    checkpoint = torch.load(output, weights_only=False)
    _, restored = build_diffusion_model(cfg, sequence_length=128)
    restored.configure_diversity_constraints(diversity_constraint_state=checkpoint["reference"])
    restored.load_state_dict(checkpoint["model"], strict=True)
    restored.eval(); diffusion.eval()
    torch.manual_seed(425); first = diffusion.sample(4, valid_mask=mask, condition=condition, prior_conditioning=base)
    torch.manual_seed(425); second = restored.sample(4, valid_mask=mask, condition=condition, prior_conditioning=base)
    torch.testing.assert_close(first, second)
    with pytest.raises(ValueError, match="旧checkpoint"):
        restored.configure_diversity_constraints(diversity_constraint_state={k: v for k, v in state.items() if k != "full_spectrum_support_state"})


def test_valid_residual_above_one_survives_soft_guard(fixture):
    f = fixture; cfg = model_configuration(f)
    _, diffusion = build_diffusion_model(cfg, sequence_length=128)
    state = copy.deepcopy(f["state"])
    # Explicitly wide valid support: ±1 is not a valid model-domain bound.
    state["lower"] = (np.asarray(state["lower"]) - 100).tolist()
    state["upper"] = (np.asarray(state["upper"]) + 100).tolist()
    diffusion.spectrum_support_module = ConditionalSpectrumSupport(state, 128)
    x = tensor(f["scaled"][:4]); x[:, :, 30] = 1.25
    clipped = diffusion._clip_prediction(x, tensor(f["base"][:4]), torch.tensor(f["vectors"][:4]), tensor(f["masks"][:4]))
    torch.testing.assert_close(clipped[:, :, 30], x[:, :, 30], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("override", [{"lower_quantile": .9}, {"total_weight": -1}, {"minimum_alpha_cumprod": 0}, {"epsilon": float("nan")}])
def test_bad_configuration_rejected(override):
    with pytest.raises(ValueError):
        normalize_spectrum_configuration({"full_spectrum_support": {"enabled": True, **override}})
