from __future__ import annotations

import math

import numpy as np
import torch

from src.conditional_diversity_constraints import (
    DifferentiableConditionFullSpectrumGroupVarianceLoss,
    fit_condition_aware_diversity_constraint_state,
)


def _configuration() -> dict:
    return {
        "enabled": True,
        "total_weight": 0.10,
        "maximum_total_ratio_to_ddpm": 0.10,
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

            # 关键：
            # D4.3.2.16关闭。
            "condition_full_spectrum_variance_profile": {
                "enabled": False,
            },

            # D4.3.2.17单独开启。
            "condition_full_spectrum_group_variance": {
                "enabled": True,
                "minimum_group_size": 4,
                "maximum_timestep_fraction": 0.50,
                "overdispersion_deadband": 0.25,
                "underdispersion_deadband": 0.20,
                "smooth_l1_beta": 0.05,
                "overdispersion_weight": 1.0,
                "underdispersion_weight": 1.0,
                "weight": 0.35,
                "epsilon": 1.0e-6,
            },
        },
    }


def _make_training_data():
    rng = np.random.default_rng(2026)

    number = 12
    length = 40

    scaled_residuals = rng.normal(
        0.0,
        0.10,
        size=(number, length),
    ).astype(np.float32)

    full_spectra = np.zeros(
        (number, length),
        dtype=np.float32,
    )

    # 前20点：稳定区。
    full_spectra[:, :20] = rng.normal(
        0.0,
        0.02,
        size=(number, 20),
    )

    # 后20点：天然高变化区。
    full_spectra[:, 20:] = rng.normal(
        0.0,
        0.30,
        size=(number, 20),
    )

    valid_masks = np.ones_like(
        scaled_residuals,
        dtype=np.float32,
    )

    condition_vectors = np.tile(
        np.asarray(
            [[1.0, 0.0]],
            dtype=np.float32,
        ),
        (number, 1),
    )

    return (
        scaled_residuals,
        full_spectra,
        valid_masks,
        condition_vectors,
    )


def _fit_state():
    (
        scaled_residuals,
        full_spectra,
        valid_masks,
        condition_vectors,
    ) = _make_training_data()

    state = fit_condition_aware_diversity_constraint_state(
        training_scaled_residuals=scaled_residuals,
        training_valid_masks=valid_masks,
        training_condition_vectors=condition_vectors,
        configuration=_configuration(),
        training_full_spectra=full_spectra,
    )

    return state


def test_d417_fits_reference_when_d416_is_disabled():
    """D4.3.2.17必须能够独立触发train-only variance reference。"""

    state = _fit_state()

    assert state[
        "full_spectrum_variance_profile_enabled"
    ]

    assert state[
        "training_counts"
    ] == [12]

    assert "full_spectrum_std" in state
    assert "full_spectrum_scale" in state

    reference_std = np.asarray(
        state["full_spectrum_std"],
        dtype=np.float32,
    )

    reference_scale = np.asarray(
        state["full_spectrum_scale"],
        dtype=np.float32,
    )

    assert reference_std.shape == (1, 40)
    assert reference_scale.shape == (1, 40)

    assert np.isfinite(reference_std).all()
    assert np.isfinite(reference_scale).all()

    assert np.all(reference_std >= 0.0)
    assert np.all(reference_scale > 0.0)

    # scale使用robust floor，因此不能小于reference std。
    assert np.all(
        reference_scale
        >= reference_std
    )


def test_d417_group_variance_two_sided_loss_has_gradient():
    """同时制造稳定区过分散和高变化区欠分散。"""

    state = _fit_state()

    configuration = _configuration()[
        "quality_fidelity"
    ][
        "condition_full_spectrum_group_variance"
    ]

    module = (
        DifferentiableConditionFullSpectrumGroupVarianceLoss(
            diversity_constraint_state=state,
            padded_length=40,
            configuration=configuration,
        )
    )

    reference_std = torch.as_tensor(
        state[
            "full_spectrum_std"
        ][0],
        dtype=torch.float32,
    )

    # 对4个样本 [-a,-a,+a,+a]：
    #
    # sample std = 2a/sqrt(3)
    #
    # 因此：
    amplitude = (
        reference_std
        * math.sqrt(3.0)
        / 2.0
    )

    prediction = torch.stack(
        [
            -amplitude,
            -amplitude,
            amplitude,
            amplitude,
        ],
        dim=0,
    ).unsqueeze(1)

    # 稳定区过分散。
    prediction[:, :, :20] *= 2.5

    # 高自然方差区欠分散。
    prediction[:, :, 20:] *= 0.6

    prediction = prediction.clone().requires_grad_()

    zeros = torch.zeros_like(
        prediction
    )

    ones = torch.ones_like(
        prediction
    )

    condition = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    result = module(
        prediction=prediction,
        target=zeros,
        full_spectrum_target=zeros,
        local_inverse_slope=ones,
        valid_mask=ones,
        condition=condition,
        timesteps=torch.zeros(
            4,
            dtype=torch.long,
        ),
        number_of_timesteps=200,
    )

    assert torch.isfinite(
        result["loss"]
    )

    assert result[
        "loss"
    ].item() > 0.0

    assert result[
        "active_groups"
    ].item() == 1.0

    assert result[
        "overdispersion_fraction"
    ].item() > 0.0

    assert result[
        "underdispersion_fraction"
    ].item() > 0.0

    result[
        "loss"
    ].backward()

    assert prediction.grad is not None

    assert torch.isfinite(
        prediction.grad
    ).all()


def test_d417_group_variance_is_off_at_high_timestep():
    """高噪声阶段不强行匹配完整光谱方差。"""

    state = _fit_state()

    configuration = _configuration()[
        "quality_fidelity"
    ][
        "condition_full_spectrum_group_variance"
    ]

    module = (
        DifferentiableConditionFullSpectrumGroupVarianceLoss(
            diversity_constraint_state=state,
            padded_length=40,
            configuration=configuration,
        )
    )

    prediction = torch.randn(
        4,
        1,
        40,
        dtype=torch.float32,
    )

    zeros = torch.zeros_like(
        prediction
    )

    result = module(
        prediction=prediction,
        target=zeros,
        full_spectrum_target=zeros,
        local_inverse_slope=torch.ones_like(
            prediction
        ),
        valid_mask=torch.ones_like(
            prediction
        ),
        condition=torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        timesteps=torch.full(
            (4,),
            199,
            dtype=torch.long,
        ),
        number_of_timesteps=200,
    )

    assert result[
        "loss"
    ].item() == 0.0

    assert result[
        "active_groups"
    ].item() == 0.0
