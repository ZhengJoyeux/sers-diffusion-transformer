from __future__ import annotations

import numpy as np
import torch

from src.condition_grouped_batch_sampler import ConditionGroupedBatchSampler
from src.conditional_diversity_constraints import (
    D4_3_METHOD_VERSION,
    DifferentiableConditionAwareDiversityLoss,
    evaluate_condition_aware_diversity_residuals,
    fit_condition_aware_diversity_constraint_state,
)


def _configuration() -> dict:
    return {
        "enabled": True,
        "total_weight": 0.02,
        "maximum_total_ratio_to_ddpm": 0.02,
        "low_noise_gate": {
            "minimum_alpha_cumprod": 0.5,
            "minimum_samples": 4,
        },
        "condition_grouping": {
            "samples_per_condition": 4,
            "shared_timestep": True,
        },
        "pairwise_distance": {
            "enabled": True,
            "penalty_mode": "floor_only",
            "minimum_distance_ratio": 0.8,
            "maximum_distance_ratio": 1.25,
            "transition_fraction": 0.2,
            "weight": 1.0,
        },
        "pairwise_correlation": {
            "enabled": True,
            "maximum_excess_correlation": 0.02,
            "transition_fraction": 0.2,
            "weight": 0.5,
        },
        "pointwise_variance_floor": {
            "enabled": True,
            "penalty_mode": "floor_only",
            "active_std_quantile": 20.0,
            "minimum_std_ratio": 0.8,
            "maximum_std_ratio": 1.25,
            "reference_std_floor_fraction": 0.25,
            "weight": 0.5,
        },
        "peak_morphology": {"enabled": False},
    }


def _training() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(2026)
    length = 32
    first = generator.normal(0.0, 0.12, size=(8, length))
    second = generator.normal(0.0, 0.08, size=(8, length))
    spectra = np.concatenate([first, second]).astype(np.float32)
    masks = np.ones_like(spectra, dtype=np.float32)
    masks[8:, 24:] = 0.0
    spectra[8:, 24:] = 0.0
    conditions = np.concatenate(
        [
            np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (8, 1)),
            np.tile(np.asarray([[0.0, 1.0]], dtype=np.float32), (8, 1)),
        ]
    )
    return spectra, masks, conditions


def test_fit_is_condition_and_mask_aware() -> None:
    spectra, masks, conditions = _training()
    state = fit_condition_aware_diversity_constraint_state(
        training_scaled_residuals=spectra,
        training_valid_masks=masks,
        training_condition_vectors=conditions,
        configuration=_configuration(),
    )
    assert state["method_version"] == D4_3_METHOD_VERSION
    assert state["number_of_conditions"] == 2
    assert sorted(state["training_counts"]) == [8, 8]
    assert sorted(sum(row) for row in state["valid_masks"]) == [24.0, 32.0]


def test_matched_geometry_is_better_than_contracted_geometry() -> None:
    spectra, masks, conditions = _training()
    state = fit_condition_aware_diversity_constraint_state(
        training_scaled_residuals=spectra,
        training_valid_masks=masks,
        training_condition_vectors=conditions,
        configuration=_configuration(),
    )
    module = DifferentiableConditionAwareDiversityLoss(
        diversity_constraint_state=state,
        padded_length=spectra.shape[1],
    )
    target = torch.as_tensor(spectra[:4]).unsqueeze(1)
    mask = torch.as_tensor(masks[:4]).unsqueeze(1)
    condition = torch.as_tensor(conditions[:4])
    timesteps = torch.zeros(4, dtype=torch.long)
    alphas = torch.linspace(0.99, 0.01, 100)

    matched = module(
        predicted_scaled_residual=target.clone(),
        target_scaled_residual=target,
        timesteps=timesteps,
        alphas_cumprod=alphas,
        valid_mask=mask,
        condition=condition,
    )
    contracted_prediction = (target.mean(dim=0, keepdim=True) + 0.2 * (
        target - target.mean(dim=0, keepdim=True)
    )).requires_grad_(True)
    contracted = module(
        predicted_scaled_residual=contracted_prediction,
        target_scaled_residual=target,
        timesteps=timesteps,
        alphas_cumprod=alphas,
        valid_mask=mask,
        condition=condition,
    )
    assert torch.isfinite(matched["diversity_raw_loss"])
    assert contracted["diversity_raw_loss"] > matched["diversity_raw_loss"]
    contracted["diversity_timestep_weighted_loss"].backward()
    assert contracted_prediction.grad is not None
    assert torch.isfinite(contracted_prediction.grad).all()


def test_floor_only_does_not_penalize_expanded_geometry() -> None:
    spectra, masks, conditions = _training()
    configuration = _configuration()
    state = fit_condition_aware_diversity_constraint_state(
        training_scaled_residuals=spectra,
        training_valid_masks=masks,
        training_condition_vectors=conditions,
        configuration=configuration,
    )
    module = DifferentiableConditionAwareDiversityLoss(
        diversity_constraint_state=state,
        padded_length=spectra.shape[1],
    )
    target = torch.as_tensor(spectra[:4]).unsqueeze(1)
    arguments = {
        "target_scaled_residual": target,
        "timesteps": torch.zeros(4, dtype=torch.long),
        "alphas_cumprod": torch.linspace(0.99, 0.01, 100),
        "valid_mask": torch.as_tensor(masks[:4]).unsqueeze(1),
        "condition": torch.as_tensor(conditions[:4]),
    }
    floor_only = module(
        predicted_scaled_residual=2.0 * target,
        **arguments,
    )
    assert floor_only["pairwise_distance_loss"].item() == 0.0
    assert floor_only["pointwise_variance_floor_loss"].item() == 0.0

    legacy_configuration = _configuration()
    del legacy_configuration["pairwise_distance"]["penalty_mode"]
    del legacy_configuration["pointwise_variance_floor"]["penalty_mode"]
    legacy_state = fit_condition_aware_diversity_constraint_state(
        training_scaled_residuals=spectra,
        training_valid_masks=masks,
        training_condition_vectors=conditions,
        configuration=legacy_configuration,
    )
    legacy_module = DifferentiableConditionAwareDiversityLoss(
        diversity_constraint_state=legacy_state,
        padded_length=spectra.shape[1],
    )
    legacy = legacy_module(
        predicted_scaled_residual=2.0 * target,
        **arguments,
    )
    assert legacy["pairwise_distance_loss"] > 0.0
    assert legacy["pointwise_variance_floor_loss"] > 0.0


def test_generation_diagnostic_matches_training_geometry() -> None:
    spectra, masks, conditions = _training()
    state = fit_condition_aware_diversity_constraint_state(
        training_scaled_residuals=spectra,
        training_valid_masks=masks,
        training_condition_vectors=conditions,
        configuration=_configuration(),
    )
    result = evaluate_condition_aware_diversity_residuals(
        generated_scaled_residuals=spectra[:8],
        condition_vector=conditions[0],
        diversity_constraint_state=state,
    )
    assert result["generated_sample_count"] == 8
    assert np.isclose(result["pairwise_distance_ratio"], 1.0)
    assert np.isclose(
        result["pairwise_correlation_excess"],
        0.0,
        atol=1.0e-7,
    )
    assert np.isclose(result["active_pointwise_std_ratio_median"], 1.0)


def test_invalid_tail_does_not_change_loss() -> None:
    spectra, masks, conditions = _training()
    state = fit_condition_aware_diversity_constraint_state(
        training_scaled_residuals=spectra,
        training_valid_masks=masks,
        training_condition_vectors=conditions,
        configuration=_configuration(),
    )
    module = DifferentiableConditionAwareDiversityLoss(
        diversity_constraint_state=state,
        padded_length=spectra.shape[1],
    )
    target = torch.as_tensor(spectra[8:12]).unsqueeze(1)
    mask = torch.as_tensor(masks[8:12]).unsqueeze(1)
    condition = torch.as_tensor(conditions[8:12])
    changed = target.clone()
    changed[:, :, 24:] = 1000.0
    arguments = {
        "target_scaled_residual": target,
        "timesteps": torch.zeros(4, dtype=torch.long),
        "alphas_cumprod": torch.linspace(0.99, 0.01, 100),
        "valid_mask": mask,
        "condition": condition,
    }
    original = module(predicted_scaled_residual=target, **arguments)
    modified = module(predicted_scaled_residual=changed, **arguments)
    assert torch.allclose(
        original["diversity_raw_loss"], modified["diversity_raw_loss"]
    )


def test_grouped_sampler_preserves_four_samples_per_condition() -> None:
    _, _, conditions = _training()
    sampler = ConditionGroupedBatchSampler(
        conditions,
        batch_size=8,
        samples_per_condition=4,
        shuffle=True,
        random_seed=2026,
    )
    seen: list[int] = []
    for batch in sampler:
        assert len(batch) == 8
        seen.extend(batch)
        rows = conditions[batch]
        _, counts = np.unique(rows, axis=0, return_counts=True)
        assert all(int(count) % 4 == 0 for count in counts)
    assert sorted(seen) == list(range(16))
