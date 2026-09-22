from __future__ import annotations

import numpy as np
import torch

from src.conditional_diversity_constraints import (
    DifferentiableConditionFullSpectrumVarianceProfileLoss,
    fit_condition_aware_diversity_constraint_state,
)


def _configuration() -> dict:
    return {
        "enabled": True,
        "total_weight": 0.05,
        "maximum_total_ratio_to_ddpm": 0.05,
        "epsilon": 1.0e-8,
        "condition_grouping": {
            "samples_per_condition": 4,
            "shared_timestep": True,
        },
        "low_noise_gate": {
            "minimum_alpha_cumprod": 0.5,
            "minimum_samples": 4,
        },
        "high_frequency_filter": {
            "smoothing_sigma_points": 1.0,
            "kernel_truncate": 3.0,
            "distance_reference_quantile": 50.0,
            "correlation_reference_quantile": 90.0,
        },
        "pairwise_distance": {
            "enabled": True,
            "metric": "whitened_l2",
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
        "peak_morphology": {
            "enabled": False,
        },
        "quality_fidelity": {
            "enabled": True,
            "condition_full_spectrum_variance_profile": {
                "enabled": True,
                "scale_floor_quantile": 10.0,
                "standardized_error_deadband": 0.25,
                "maximum_timestep_fraction": 0.5,
                "smooth_l1_beta": 0.1,
                "weight": 0.35,
                "epsilon": 1.0e-6,
            },
        },
    }


def _training():
    rng = np.random.default_rng(
        2026
    )

    count = 12
    length = 32

    scaled = rng.normal(
        0.0,
        0.10,
        size=(
            count,
            length,
        ),
    ).astype(
        np.float32
    )

    full = np.zeros(
        (
            count,
            length,
        ),
        dtype=np.float32,
    )

    # 前16点是真实稳定区。
    full[
        :,
        :16,
    ] = rng.normal(
        0.0,
        0.02,
        size=(
            count,
            16,
        ),
    )

    # 后16点是真实高自然方差区。
    full[
        :,
        16:,
    ] = rng.normal(
        0.0,
        0.30,
        size=(
            count,
            16,
        ),
    )

    masks = np.ones_like(
        scaled,
        dtype=np.float32,
    )

    conditions = np.tile(
        np.asarray(
            [
                [
                    1.0,
                    0.0,
                ]
            ],
            dtype=np.float32,
        ),
        (
            count,
            1,
        ),
    )

    return (
        scaled,
        full,
        masks,
        conditions,
    )


def test_train_only_profile_preserves_stable_and_variable_regions():
    (
        scaled,
        full,
        masks,
        conditions,
    ) = _training()

    state = (
        fit_condition_aware_diversity_constraint_state(
            training_scaled_residuals=scaled,
            training_valid_masks=masks,
            training_condition_vectors=conditions,
            training_full_spectra=full,
            configuration=_configuration(),
        )
    )

    assert state[
        "full_spectrum_variance_profile_enabled"
    ]

    scale = np.asarray(
        state[
            "full_spectrum_scale"
        ],
        dtype=np.float64,
    )[0]

    assert np.isfinite(
        scale
    ).all()

    assert np.all(
        scale > 0.0
    )

    assert (
        np.median(
            scale[
                16:
            ]
        )
        > 3.0
        * np.median(
            scale[
                :16
            ]
        )
    )


def test_same_absolute_error_is_penalized_more_in_stable_raman_region():
    (
        scaled,
        full,
        masks,
        conditions,
    ) = _training()

    state = (
        fit_condition_aware_diversity_constraint_state(
            training_scaled_residuals=scaled,
            training_valid_masks=masks,
            training_condition_vectors=conditions,
            training_full_spectra=full,
            configuration=_configuration(),
        )
    )

    profile_configuration = (
        _configuration()[
            "quality_fidelity"
        ][
            "condition_full_spectrum_variance_profile"
        ]
    )

    module = (
        DifferentiableConditionFullSpectrumVarianceProfileLoss(
            diversity_constraint_state=state,
            padded_length=scaled.shape[1],
            configuration=profile_configuration,
        )
    )

    target = torch.zeros(
        (
            4,
            1,
            scaled.shape[1],
        ),
        dtype=torch.float32,
    )

    full_target = torch.as_tensor(
        full[
            :4
        ],
        dtype=torch.float32,
    ).unsqueeze(
        1
    )

    slope = torch.ones_like(
        full_target
    )

    mask = torch.ones_like(
        full_target
    )

    condition = torch.as_tensor(
        conditions[
            :4
        ],
        dtype=torch.float32,
    )

    timesteps = torch.zeros(
        4,
        dtype=torch.long,
    )

    matched_prediction = (
        target.clone()
        .requires_grad_(
            True
        )
    )

    matched = module(
        prediction=matched_prediction,
        target=target,
        full_spectrum_target=full_target,
        local_inverse_slope=slope,
        valid_mask=mask,
        condition=condition,
        timesteps=timesteps,
        number_of_timesteps=200,
    )

    stable_error = (
        target.clone()
    )

    stable_error[
        :,
        :,
        :16,
    ] = 0.05

    stable_error.requires_grad_()

    stable_result = module(
        prediction=stable_error,
        target=target,
        full_spectrum_target=full_target,
        local_inverse_slope=slope,
        valid_mask=mask,
        condition=condition,
        timesteps=timesteps,
        number_of_timesteps=200,
    )

    variable_error = (
        target.clone()
    )

    variable_error[
        :,
        :,
        16:,
    ] = 0.05

    variable_error.requires_grad_()

    variable_result = module(
        prediction=variable_error,
        target=target,
        full_spectrum_target=full_target,
        local_inverse_slope=slope,
        valid_mask=mask,
        condition=condition,
        timesteps=timesteps,
        number_of_timesteps=200,
    )

    assert torch.isfinite(
        matched["loss"]
    )

    assert torch.isfinite(
        stable_result["loss"]
    )

    assert torch.isfinite(
        variable_result["loss"]
    )

    assert matched[
        "loss"
    ].item() == 0.0

    assert (
        stable_result[
            "loss"
        ]
        > variable_result[
            "loss"
        ]
    )

    assert (
        stable_result[
            "violation_fraction"
        ]
        > 0.0
    )

    stable_result[
        "loss"
    ].backward()

    assert (
        stable_error.grad
        is not None
    )

    assert torch.isfinite(
        stable_error.grad
    ).all()


def test_high_timestep_disables_profile_loss():
    (
        scaled,
        full,
        masks,
        conditions,
    ) = _training()

    state = (
        fit_condition_aware_diversity_constraint_state(
            training_scaled_residuals=scaled,
            training_valid_masks=masks,
            training_condition_vectors=conditions,
            training_full_spectra=full,
            configuration=_configuration(),
        )
    )

    module = (
        DifferentiableConditionFullSpectrumVarianceProfileLoss(
            diversity_constraint_state=state,
            padded_length=scaled.shape[1],
            configuration=(
                _configuration()[
                    "quality_fidelity"
                ][
                    "condition_full_spectrum_variance_profile"
                ]
            ),
        )
    )

    target = torch.zeros(
        (
            4,
            1,
            scaled.shape[1],
        ),
        dtype=torch.float32,
    )

    full_target = torch.as_tensor(
        full[
            :4
        ],
        dtype=torch.float32,
    ).unsqueeze(1)

    prediction = (
        target
        + 0.5
    )

    result = module(
        prediction=prediction,
        target=target,
        full_spectrum_target=full_target,
        local_inverse_slope=torch.ones_like(
            full_target
        ),
        valid_mask=torch.ones_like(
            full_target
        ),
        condition=torch.as_tensor(
            conditions[
                :4
            ],
            dtype=torch.float32,
        ),
        timesteps=torch.full(
            (
                4,
            ),
            199,
            dtype=torch.long,
        ),
        number_of_timesteps=200,
    )

    assert result[
        "loss"
    ].item() == 0.0

    assert result[
        "active_sample_fraction"
    ].item() == 0.0


def test_profile_disabled_keeps_old_state_path_compatible():
    (
        scaled,
        _full,
        masks,
        conditions,
    ) = _training()

    configuration = (
        _configuration()
    )

    configuration[
        "quality_fidelity"
    ][
        "condition_full_spectrum_variance_profile"
    ][
        "enabled"
    ] = False

    state = (
        fit_condition_aware_diversity_constraint_state(
            training_scaled_residuals=scaled,
            training_valid_masks=masks,
            training_condition_vectors=conditions,
            configuration=configuration,
        )
    )

    assert not state[
        "full_spectrum_variance_profile_enabled"
    ]

    assert (
        "full_spectrum_scale"
        not in state
    )
