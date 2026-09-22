"""具有逐样本Raman有效区掩码的一维高斯扩散。"""

from __future__ import annotations

from collections import namedtuple
from random import random
from typing import Any

import torch
from torch.nn import functional as F

from src.one_dimensional_ddpm import GaussianDiffusion1D
from src.conditional_diversity_constraints import (
    DifferentiableConditionAwareDiversityLoss,
    DifferentiableConditionFullSpectrumGroupVarianceLoss,
    DifferentiableConditionFullSpectrumVarianceProfileLoss,
    normalize_condition_aware_diversity_configuration,
)


MaskedModelPrediction = namedtuple(
    "MaskedModelPrediction",
    ["pred_noise", "pred_x_start"],
)


def _extract(
    values: torch.Tensor,
    timesteps: torch.Tensor,
    target_shape: torch.Size,
) -> torch.Tensor:
    gathered = values.gather(-1, timesteps)
    return gathered.reshape(
        timesteps.shape[0],
        *((1,) * (len(target_shape) - 1)),
    )


def _normalize_quality_fidelity_configuration(
    diversity_configuration: dict[str, Any],
) -> dict[str, Any]:
    """Validate the optional D4.3.2 fidelity block.

    It deliberately lives below ``diversity_constraints`` so the existing
    model-builder call remains checkpoint-compatible: no new constructor
    argument or public interface is required.
    """

    source = diversity_configuration.get("quality_fidelity", {}) or {}
    if not isinstance(source, dict):
        raise ValueError("quality_fidelity必须是字典。")
    enabled = bool(source.get("enabled", False))
    normalized: dict[str, Any] = {
        "enabled": enabled,
        "total_weight": float(source.get("total_weight", 0.0)),
        "maximum_total_ratio_to_ddpm": float(
            source.get("maximum_total_ratio_to_ddpm", 0.0)
        ),
    }
    if not enabled:
        return normalized
    if normalized["total_weight"] <= 0.0:
        raise ValueError("quality_fidelity.total_weight必须大于0。")
    if not 0.0 < normalized["maximum_total_ratio_to_ddpm"] <= 0.25:
        raise ValueError(
            "quality_fidelity.maximum_total_ratio_to_ddpm必须位于(0,0.25]。"
        )

    high_noise_sampling = source.get("high_noise_sampling", {}) or {}
    high_noise = source.get("high_noise_weighting", {}) or {}
    high_noise_recovery = (
        source.get("high_noise_standardized_recovery", {}) or {}
    )
    derivative = source.get("first_derivative", {}) or {}
    multiscale = source.get("multiscale_shape", {}) or {}
    multiscale_variance = source.get(
        "condition_multiscale_variance_floor", {}
    ) or {}
    derivative_variance = source.get(
        "condition_derivative_variance_floor", {}
    ) or {}
    condition_mean = source.get("condition_mean", {}) or {}
    envelope = source.get("condition_pointwise_envelope", {}) or {}
    full_tail = source.get("full_spectrum_tail_distribution", {}) or {}
    variance_profile = source.get(
        "condition_full_spectrum_variance_profile",
        {},
    ) or {}

    group_variance = source.get(
        "condition_full_spectrum_group_variance",
        {},
    ) or {}

    independent_full_spectrum = (
        source.get("independent_full_spectrum_budget", {}) or {}
    )
    if not all(
        isinstance(item, dict)
        for item in (
            high_noise_sampling,
            high_noise,
            high_noise_recovery,
            derivative,
            multiscale,
            multiscale_variance,
            derivative_variance,
            condition_mean,
            envelope,
            full_tail,
            variance_profile,
            group_variance,
            independent_full_spectrum,
        )
    ):
        raise ValueError("quality_fidelity的子配置必须是字典。")

    normalized["independent_full_spectrum_budget"] = {
        "enabled": bool(
            independent_full_spectrum.get("enabled", False)
        ),
        "total_weight": float(
            independent_full_spectrum.get("total_weight", 0.0)
        ),
        "maximum_total_ratio_to_ddpm": float(
            independent_full_spectrum.get(
                "maximum_total_ratio_to_ddpm",
                0.0,
            )
        ),
    }
    independent_budget = normalized[
        "independent_full_spectrum_budget"
    ]
    if independent_budget["enabled"]:
        if independent_budget["total_weight"] <= 0.0:
            raise ValueError(
                "independent_full_spectrum_budget.total_weight"
                "必须大于0。"
            )
        if not (
            0.0
            < independent_budget["maximum_total_ratio_to_ddpm"]
            <= 0.25
        ):
            raise ValueError(
                "independent_full_spectrum_budget."
                "maximum_total_ratio_to_ddpm必须位于(0,0.25]。"
            )

    normalized["high_noise_sampling"] = {
        "enabled": bool(high_noise_sampling.get("enabled", False)),
        "sampling_probability": float(
            high_noise_sampling.get("sampling_probability", 0.0)
        ),
        "minimum_timestep_fraction": float(
            high_noise_sampling.get("minimum_timestep_fraction", 0.75)
        ),
    }
    high_sampling = normalized["high_noise_sampling"]
    if high_sampling["enabled"] and not (
        0.0 < high_sampling["sampling_probability"] <= 0.75
        and 0.50 <= high_sampling["minimum_timestep_fraction"] < 1.0
    ):
        raise ValueError(
            "high_noise_sampling要求sampling_probability位于(0,0.75]且"
            "minimum_timestep_fraction位于[0.5,1)。"
        )

    normalized["high_noise_weighting"] = {
        "enabled": bool(high_noise.get("enabled", False)),
        "maximum_weight": float(high_noise.get("maximum_weight", 1.0)),
        "power": float(high_noise.get("power", 1.0)),
    }
    high = normalized["high_noise_weighting"]
    if high["enabled"] and (
        not 1.0 <= high["maximum_weight"] <= 4.0 or high["power"] <= 0.0
    ):
        raise ValueError("high_noise_weighting要求maximum_weight位于[1,4]且power>0。")

    normalized["high_noise_standardized_recovery"] = {
        "enabled": bool(high_noise_recovery.get("enabled", False)),
        "minimum_timestep_fraction": float(
            high_noise_recovery.get("minimum_timestep_fraction", 0.50)
        ),
        "full_strength_timestep_fraction": float(
            high_noise_recovery.get("full_strength_timestep_fraction", 0.85)
        ),
        "gate_power": float(high_noise_recovery.get("gate_power", 1.5)),
        "smooth_l1_beta": float(
            high_noise_recovery.get("smooth_l1_beta", 0.10)
        ),
        "total_weight": float(high_noise_recovery.get("total_weight", 0.05)),
        "maximum_total_ratio_to_ddpm": float(
            high_noise_recovery.get("maximum_total_ratio_to_ddpm", 0.05)
        ),
        "epsilon": float(high_noise_recovery.get("epsilon", 1.0e-6)),
    }
    recovery = normalized["high_noise_standardized_recovery"]
    if recovery["enabled"]:
        if not (
            0.0 <= recovery["minimum_timestep_fraction"] < 1.0
            and recovery["minimum_timestep_fraction"]
            < recovery["full_strength_timestep_fraction"] <= 1.0
        ):
            raise ValueError(
                "high_noise_standardized_recovery要求"
                "0<=minimum_timestep_fraction<"
                "full_strength_timestep_fraction<=1。"
            )
        if (
            recovery["gate_power"] <= 0.0
            or recovery["smooth_l1_beta"] <= 0.0
            or recovery["total_weight"] <= 0.0
            or not 0.0 < recovery["maximum_total_ratio_to_ddpm"] <= 0.25
            or recovery["epsilon"] <= 0.0
        ):
            raise ValueError(
                "high_noise_standardized_recovery的gate_power、"
                "smooth_l1_beta、total_weight、epsilon必须大于0，"
                "maximum_total_ratio_to_ddpm必须位于(0,0.25]。"
            )

    normalized["first_derivative"] = {
        "enabled": bool(derivative.get("enabled", False)),
        "smooth_l1_beta": float(derivative.get("smooth_l1_beta", 0.05)),
        "weight": float(derivative.get("weight", 1.0)),
    }
    first = normalized["first_derivative"]
    if first["enabled"] and (
        first["smooth_l1_beta"] <= 0.0 or first["weight"] <= 0.0
    ):
        raise ValueError("first_derivative要求smooth_l1_beta和weight均大于0。")

    kernel_sizes = [int(value) for value in multiscale.get("kernel_sizes", [])]
    normalized["multiscale_shape"] = {
        "enabled": bool(multiscale.get("enabled", False)),
        "kernel_sizes": kernel_sizes,
        "smooth_l1_beta": float(multiscale.get("smooth_l1_beta", 0.05)),
        "weight": float(multiscale.get("weight", 1.0)),
    }
    multi = normalized["multiscale_shape"]
    if multi["enabled"] and (
        not kernel_sizes
        or any(value < 3 or value % 2 == 0 or value > 127 for value in kernel_sizes)
        or len(set(kernel_sizes)) != len(kernel_sizes)
        or multi["smooth_l1_beta"] <= 0.0
        or multi["weight"] <= 0.0
    ):
        raise ValueError(
            "multiscale_shape要求互不重复的奇数kernel_sizes位于[3,127]，"
            "且smooth_l1_beta和weight均大于0。"
        )

    variance_kernel_sizes = [
        int(value) for value in multiscale_variance.get("kernel_sizes", [])
    ]
    normalized["condition_multiscale_variance_floor"] = {
        "enabled": bool(multiscale_variance.get("enabled", False)),
        "kernel_sizes": variance_kernel_sizes,
        "minimum_group_size": int(
            multiscale_variance.get("minimum_group_size", 4)
        ),
        "minimum_std_ratio": float(
            multiscale_variance.get("minimum_std_ratio", 0.85)
        ),
        "active_std_quantile": float(
            multiscale_variance.get("active_std_quantile", 20.0)
        ),
        "maximum_timestep_fraction": float(
            multiscale_variance.get("maximum_timestep_fraction", 0.50)
        ),
        "smooth_l1_beta": float(
            multiscale_variance.get("smooth_l1_beta", 0.05)
        ),
        "weight": float(multiscale_variance.get("weight", 1.0)),
    }
    variance_floor = normalized["condition_multiscale_variance_floor"]
    if variance_floor["enabled"] and (
        not variance_kernel_sizes
        or any(
            value < 3 or value % 2 == 0 or value > 127
            for value in variance_kernel_sizes
        )
        or len(set(variance_kernel_sizes)) != len(variance_kernel_sizes)
        or variance_floor["minimum_group_size"] < 3
        or not 0.0 < variance_floor["minimum_std_ratio"] <= 1.0
        or not 0.0 <= variance_floor["active_std_quantile"] <= 80.0
        or not 0.0 < variance_floor["maximum_timestep_fraction"] <= 1.0
        or variance_floor["smooth_l1_beta"] <= 0.0
        or variance_floor["weight"] <= 0.0
    ):
        raise ValueError("condition_multiscale_variance_floor配置无效。")

    derivative_kernel_sizes = [
        int(value)
        for value in derivative_variance.get("smoothing_kernel_sizes", [])
    ]
    normalized["condition_derivative_variance_floor"] = {
        "enabled": bool(derivative_variance.get("enabled", False)),
        "smoothing_kernel_sizes": derivative_kernel_sizes,
        "minimum_group_size": int(
            derivative_variance.get("minimum_group_size", 4)
        ),
        "minimum_std_ratio": float(
            derivative_variance.get("minimum_std_ratio", 0.90)
        ),
        "active_std_quantile": float(
            derivative_variance.get("active_std_quantile", 30.0)
        ),
        "maximum_timestep_fraction": float(
            derivative_variance.get("maximum_timestep_fraction", 0.50)
        ),
        "smooth_l1_beta": float(
            derivative_variance.get("smooth_l1_beta", 0.05)
        ),
        "weight": float(derivative_variance.get("weight", 1.0)),
    }
    derivative_floor = normalized["condition_derivative_variance_floor"]
    if derivative_floor["enabled"] and (
        not derivative_kernel_sizes
        or any(
            value < 3 or value % 2 == 0 or value > 127
            for value in derivative_kernel_sizes
        )
        or len(set(derivative_kernel_sizes)) != len(derivative_kernel_sizes)
        or derivative_floor["minimum_group_size"] < 3
        or not 0.0 < derivative_floor["minimum_std_ratio"] <= 1.0
        or not 0.0 <= derivative_floor["active_std_quantile"] <= 80.0
        or not 0.0 < derivative_floor["maximum_timestep_fraction"] <= 1.0
        or derivative_floor["smooth_l1_beta"] <= 0.0
        or derivative_floor["weight"] <= 0.0
    ):
        raise ValueError("condition_derivative_variance_floor配置无效。")

    normalized["condition_mean"] = {
        "enabled": bool(condition_mean.get("enabled", False)),
        "smooth_l1_beta": float(condition_mean.get("smooth_l1_beta", 0.05)),
        "minimum_group_size": int(condition_mean.get("minimum_group_size", 4)),
        "weight": float(condition_mean.get("weight", 1.0)),
    }
    mean = normalized["condition_mean"]
    if mean["enabled"] and (
        mean["smooth_l1_beta"] <= 0.0
        or mean["minimum_group_size"] < 2
        or mean["weight"] <= 0.0
    ):
        raise ValueError(
            "condition_mean要求smooth_l1_beta和weight大于0且"
            "minimum_group_size至少为2。"
        )

    normalized["condition_pointwise_envelope"] = {
        "enabled": bool(envelope.get("enabled", False)),
        "lower_quantile": float(envelope.get("lower_quantile", 0.05)),
        "upper_quantile": float(envelope.get("upper_quantile", 0.95)),
        "iqr_margin": float(envelope.get("iqr_margin", 0.75)),
        "smooth_l1_beta": float(envelope.get("smooth_l1_beta", 0.05)),
        "minimum_group_size": int(envelope.get("minimum_group_size", 4)),
        "weight": float(envelope.get("weight", 1.0)),
    }
    bounds = normalized["condition_pointwise_envelope"]
    if bounds["enabled"] and not (
        0.0 <= bounds["lower_quantile"] < 0.25
        and 0.75 < bounds["upper_quantile"] <= 1.0
        and bounds["lower_quantile"] < bounds["upper_quantile"]
        and bounds["iqr_margin"] >= 0.0
        and bounds["smooth_l1_beta"] > 0.0
        and bounds["minimum_group_size"] >= 2
        and bounds["weight"] > 0.0
    ):
        raise ValueError("condition_pointwise_envelope配置无效。")

    probabilities = [
        float(value)
        for value in full_tail.get(
            "piecewise_probabilities", [0.01, 0.10, 0.50, 0.90, 0.99]
        )
    ]
    segment_weights = [
        float(value)
        for value in full_tail.get("segment_weights", [1.5, 0.5, 0.5, 1.5])
    ]
    normalized["full_spectrum_tail_distribution"] = {
        "enabled": bool(full_tail.get("enabled", False)),
        "use_train_reference": bool(
            full_tail.get(
                "use_train_reference",
                False,
            )
        ),
        "variance_stratified_enabled": bool(
            full_tail.get(
                "variance_stratified_enabled",
                False,
            )
        ),
        "number_of_variance_strata": int(
            full_tail.get(
                "number_of_variance_strata",
                5,
            )
        ),
        "variance_strata_weights": [
            float(value)
            for value in full_tail.get(
                "variance_strata_weights",
                [1.0, 1.0, 1.0, 1.0, 1.0],
            )
        ],
        "piecewise_probabilities": probabilities,
        "segment_weights": segment_weights,
        "minimum_segment_span_ratio": float(
            full_tail.get("minimum_segment_span_ratio", 0.90)
        ),
        "maximum_segment_span_ratio": float(
            full_tail.get("maximum_segment_span_ratio", 1.10)
        ),
        "maximum_timestep_fraction": float(
            full_tail.get("maximum_timestep_fraction", 0.50)
        ),
        "minimum_group_size": int(full_tail.get("minimum_group_size", 4)),
        "pointwise_scale_floor_quantile": float(
            full_tail.get("pointwise_scale_floor_quantile", 10.0)
        ),
        "reconstruction_smooth_l1_beta": float(
            full_tail.get("reconstruction_smooth_l1_beta", 0.01)
        ),
        "span_smooth_l1_beta": float(
            full_tail.get("span_smooth_l1_beta", 0.02)
        ),
        "reconstruction_weight": float(
            full_tail.get("reconstruction_weight", 0.5)
        ),
        "span_weight": float(full_tail.get("span_weight", 1.0)),
        "short_axis_maximum_points": int(
            full_tail.get("short_axis_maximum_points", 1500)
        ),
        "short_axis_weight_multiplier": float(
            full_tail.get("short_axis_weight_multiplier", 1.5)
        ),
        "epsilon": float(full_tail.get("epsilon", 1.0e-6)),
    }
    tail = normalized["full_spectrum_tail_distribution"]
    if tail["enabled"] and not (
        len(probabilities) == 5
        and all(0.0 < value < 1.0 for value in probabilities)
        and all(
            probabilities[index] < probabilities[index + 1]
            for index in range(4)
        )
        and len(segment_weights) == 4
        and all(value > 0.0 for value in segment_weights)
        and 0.50 <= tail["minimum_segment_span_ratio"] < 1.0
        and 1.0 < tail["maximum_segment_span_ratio"] <= 1.50
        and 0.0 < tail["maximum_timestep_fraction"] <= 0.75
        and tail["minimum_group_size"] >= 3
        and 0.0 <= tail["pointwise_scale_floor_quantile"] <= 50.0
        and tail["reconstruction_smooth_l1_beta"] > 0.0
        and tail["span_smooth_l1_beta"] > 0.0
        and tail["reconstruction_weight"] > 0.0
        and tail["span_weight"] > 0.0
        and tail["short_axis_maximum_points"] >= 2
        and tail["short_axis_weight_multiplier"] >= 1.0
        and (
            not tail["variance_stratified_enabled"]
            or (
                tail["use_train_reference"]
                and tail["number_of_variance_strata"] >= 2
                and len(tail["variance_strata_weights"])
                == tail["number_of_variance_strata"]
                and all(
                    value > 0.0
                    for value in tail["variance_strata_weights"]
                )
            )
        )
        and tail["epsilon"] > 0.0
    ):
        raise ValueError("full_spectrum_tail_distribution配置无效。")

    normalized[
        "condition_full_spectrum_variance_profile"
    ] = {
        "enabled": bool(
            variance_profile.get(
                "enabled",
                False,
            )
        ),
        "scale_floor_quantile": float(
            variance_profile.get(
                "scale_floor_quantile",
                10.0,
            )
        ),
        "standardized_error_deadband": float(
            variance_profile.get(
                "standardized_error_deadband",
                0.25,
            )
        ),
        "maximum_timestep_fraction": float(
            variance_profile.get(
                "maximum_timestep_fraction",
                0.50,
            )
        ),
        "smooth_l1_beta": float(
            variance_profile.get(
                "smooth_l1_beta",
                0.10,
            )
        ),
        "weight": float(
            variance_profile.get(
                "weight",
                0.35,
            )
        ),
        "epsilon": float(
            variance_profile.get(
                "epsilon",
                1.0e-6,
            )
        ),
    }

    profile = normalized[
        "condition_full_spectrum_variance_profile"
    ]

    if profile["enabled"] and not (
        0.0
        <= profile["scale_floor_quantile"]
        <= 50.0
        and 0.0
        <= profile["standardized_error_deadband"]
        <= 2.0
        and 0.0
        < profile["maximum_timestep_fraction"]
        <= 0.75
        and profile["smooth_l1_beta"]
        > 0.0
        and profile["weight"]
        > 0.0
        and profile["epsilon"]
        > 0.0
    ):
        raise ValueError(
            "condition_full_spectrum_variance_profile配置无效。"
        )

    normalized[
        "condition_full_spectrum_group_variance"
    ] = {
        "enabled": bool(
            group_variance.get(
                "enabled",
                False,
            )
        ),
        "minimum_group_size": int(
            group_variance.get(
                "minimum_group_size",
                4,
            )
        ),
        "maximum_timestep_fraction": float(
            group_variance.get(
                "maximum_timestep_fraction",
                0.50,
            )
        ),
        "overdispersion_deadband": float(
            group_variance.get(
                "overdispersion_deadband",
                0.25,
            )
        ),
        "underdispersion_deadband": float(
            group_variance.get(
                "underdispersion_deadband",
                0.20,
            )
        ),
        "smooth_l1_beta": float(
            group_variance.get(
                "smooth_l1_beta",
                0.05,
            )
        ),
        "overdispersion_weight": float(
            group_variance.get(
                "overdispersion_weight",
                1.0,
            )
        ),
        "underdispersion_weight": float(
            group_variance.get(
                "underdispersion_weight",
                1.0,
            )
        ),
        "weight": float(
            group_variance.get(
                "weight",
                0.35,
            )
        ),
        "epsilon": float(
            group_variance.get(
                "epsilon",
                1.0e-6,
            )
        ),
    }

    group_profile = normalized[
        "condition_full_spectrum_group_variance"
    ]

    if group_profile["enabled"] and not (
        group_profile["minimum_group_size"] >= 3
        and 0.0
        < group_profile["maximum_timestep_fraction"]
        <= 0.75
        and 0.0
        <= group_profile["overdispersion_deadband"]
        <= 2.0
        and 0.0
        <= group_profile["underdispersion_deadband"]
        <= 2.0
        and group_profile["smooth_l1_beta"] > 0.0
        and group_profile["overdispersion_weight"] > 0.0
        and group_profile["underdispersion_weight"] > 0.0
        and group_profile["weight"] > 0.0
        and group_profile["epsilon"] > 0.0
    ):
        raise ValueError(
            "condition_full_spectrum_group_variance配置无效。"
        )

    if not any(
        (
            first["enabled"],
            multi["enabled"],
            variance_floor["enabled"],
            derivative_floor["enabled"],
            mean["enabled"],
            bounds["enabled"],
            tail["enabled"],
            profile["enabled"],
            group_profile["enabled"],
        )
    ):
        raise ValueError("quality_fidelity至少需要启用一个保真损失。")

    # --------------------------------------------------------
    # D4.23 condition-wise equalized-residual tail envelope
    # --------------------------------------------------------
    equalized_residual_tail = source.get(
        "condition_equalized_residual_tail",
        {},
    ) or {}

    if not isinstance(
        equalized_residual_tail,
        dict,
    ):
        raise ValueError(
            "quality_fidelity."
            "condition_equalized_residual_tail必须是字典。"
        )

    equalized_residual_tail_enabled = bool(
        equalized_residual_tail.get(
            "enabled",
            False,
        )
    )

    normalized[
        "condition_equalized_residual_tail"
    ] = {
        "enabled": equalized_residual_tail_enabled,
    }

    if equalized_residual_tail_enabled:
        lower_quantile = float(
            equalized_residual_tail.get(
                "lower_quantile",
                5.0,
            )
        )

        upper_quantile = float(
            equalized_residual_tail.get(
                "upper_quantile",
                95.0,
            )
        )

        span_floor_quantile = float(
            equalized_residual_tail.get(
                "span_floor_quantile",
                10.0,
            )
        )

        margin_fraction = float(
            equalized_residual_tail.get(
                "margin_fraction",
                0.10,
            )
        )

        smooth_l1_beta = float(
            equalized_residual_tail.get(
                "smooth_l1_beta",
                0.05,
            )
        )

        maximum_timestep_fraction = float(
            equalized_residual_tail.get(
                "maximum_timestep_fraction",
                0.50,
            )
        )

        timestep_weight_power = float(
            equalized_residual_tail.get(
                "timestep_weight_power",
                1.0,
            )
        )

        total_weight = float(
            equalized_residual_tail.get(
                "total_weight",
                1.0,
            )
        )

        maximum_total_ratio_to_ddpm = float(
            equalized_residual_tail.get(
                "maximum_total_ratio_to_ddpm",
                0.05,
            )
        )

        if not (
            0.0 <= lower_quantile < 50.0
            and 50.0 < upper_quantile <= 100.0
            and lower_quantile < upper_quantile
        ):
            raise ValueError(
                "D4.23 lower/upper_quantile范围无效。"
            )

        if not (
            0.0
            <= span_floor_quantile
            <= 50.0
        ):
            raise ValueError(
                "D4.23 span_floor_quantile必须位于[0,50]。"
            )

        if not (
            0.0
            <= margin_fraction
            <= 1.0
        ):
            raise ValueError(
                "D4.23 margin_fraction必须位于[0,1]。"
            )

        if smooth_l1_beta <= 0.0:
            raise ValueError(
                "D4.23 smooth_l1_beta必须大于0。"
            )

        if not (
            0.0
            < maximum_timestep_fraction
            <= 1.0
        ):
            raise ValueError(
                "D4.23 maximum_timestep_fraction"
                "必须位于(0,1]。"
            )

        if timestep_weight_power < 0.0:
            raise ValueError(
                "D4.23 timestep_weight_power不能小于0。"
            )

        if total_weight <= 0.0:
            raise ValueError(
                "D4.23 total_weight必须大于0。"
            )

        if not (
            0.0
            < maximum_total_ratio_to_ddpm
            <= 0.25
        ):
            raise ValueError(
                "D4.23 maximum_total_ratio_to_ddpm"
                "必须位于(0,0.25]。"
            )

        normalized[
            "condition_equalized_residual_tail"
        ] = {
            "enabled": True,
            "use_group_overdispersion_gate": bool(
                equalized_residual_tail.get(
                    "use_group_overdispersion_gate",
                    False,
                )
            ),
            "severe_deadband_fraction_of_envelope_width": float(
                equalized_residual_tail.get(
                    "severe_deadband_fraction_of_envelope_width",
                    0.0,
                )
            ),
            "lower_quantile": lower_quantile,
            "upper_quantile": upper_quantile,
            "span_floor_quantile": span_floor_quantile,
            "margin_fraction": margin_fraction,
            "smooth_l1_beta": smooth_l1_beta,
            "maximum_timestep_fraction": (
                maximum_timestep_fraction
            ),
            "timestep_weight_power": (
                timestep_weight_power
            ),
            "total_weight": total_weight,
            "maximum_total_ratio_to_ddpm": (
                maximum_total_ratio_to_ddpm
            ),
        }

    return normalized


class MaskedGaussianDiffusion1D(GaussianDiffusion1D):
    """只在真实测量的Raman点上加噪、计算损失和采样。"""

    supports_valid_mask = True
    supports_condition = True
    supports_prior_conditioning = True
    supports_full_spectrum_training_context = True

    def __init__(
        self,
        *args,
        diversity_configuration: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        if bool(kwargs.get("auto_normalize", True)):
            raise ValueError(
                "掩码扩散要求diffusion.auto_normalize=false，"
                "并使用项目的train-only归一化。"
            )
        super().__init__(*args, **kwargs)
        self._sampling_valid_mask: torch.Tensor | None = None
        self._sampling_condition: torch.Tensor | None = None
        self._sampling_prior_conditioning: torch.Tensor | None = None
        self._latest_loss_components: dict[str, torch.Tensor] = {}
        self.diversity_configuration = (
            normalize_condition_aware_diversity_configuration(
                diversity_configuration or {"enabled": False}
            )
        )
        self.diversity_enabled = bool(
            self.diversity_configuration.get("enabled", False)
        )
        self.diversity_total_weight = (
            float(self.diversity_configuration["total_weight"])
            if self.diversity_enabled
            else 0.0
        )
        self.diversity_maximum_ratio = (
            float(
                self.diversity_configuration[
                    "maximum_total_ratio_to_ddpm"
                ]
            )
            if self.diversity_enabled
            else 0.0
        )
        self.diversity_loss_module: (
            DifferentiableConditionAwareDiversityLoss | None
        ) = None

        self.full_spectrum_variance_profile_loss_module: (
            DifferentiableConditionFullSpectrumVarianceProfileLoss
            | None
        ) = None

        self.full_spectrum_group_variance_loss_module: (
            DifferentiableConditionFullSpectrumGroupVarianceLoss
            | None
        ) = None

        # D4.3.2.20：高噪声恢复使用的train-only condition reference。
        self.high_noise_recovery_condition_vectors: torch.Tensor | None = None
        self.high_noise_recovery_full_spectrum_scale: torch.Tensor | None = None


        # D4.3.2.18 train-only tail reference。
        self.full_spectrum_tail_reference_condition_vectors: (
            torch.Tensor | None
        ) = None

        self.full_spectrum_tail_reference_mean: (
            torch.Tensor | None
        ) = None

        self.full_spectrum_tail_reference_scale: (
            torch.Tensor | None
        ) = None

        self.full_spectrum_tail_reference_probabilities: (
            torch.Tensor | None
        ) = None

        self.full_spectrum_tail_reference_quantiles: (
            torch.Tensor | None
        ) = None

        # D4.3.2.19 variance-stratified tail reference。
        self.full_spectrum_tail_reference_variance_strata: (
            torch.Tensor | None
        ) = None

        self.full_spectrum_tail_reference_stratified_quantiles: (
            torch.Tensor | None
        ) = None

        # D4.23 condition-wise equalized-residual tail envelope.
        # 仅保存train-only reference；首次loss调用时搬到GPU。
        self.equalized_residual_tail_condition_vectors: (
            torch.Tensor | None
        ) = None

        self.equalized_residual_tail_lower: (
            torch.Tensor | None
        ) = None

        self.equalized_residual_tail_upper: (
            torch.Tensor | None
        ) = None

        self.quality_fidelity_configuration = (
            _normalize_quality_fidelity_configuration(
                diversity_configuration or {"enabled": False}
            )
        )
        self.quality_fidelity_enabled = bool(
            self.quality_fidelity_configuration.get("enabled", False)
        )
        if self.quality_fidelity_enabled and self.objective != "pred_x0":
            raise ValueError("D4.3.2 quality_fidelity当前只支持objective=pred_x0。")

    def _high_noise_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        configuration = self.quality_fidelity_configuration.get(
            "high_noise_weighting", {}
        )
        if not self.quality_fidelity_enabled or not bool(
            configuration.get("enabled", False)
        ):
            return torch.ones_like(timesteps, dtype=torch.float32)
        denominator = float(max(self.num_timesteps - 1, 1))
        fraction = timesteps.to(dtype=torch.float32) / denominator
        maximum = float(configuration["maximum_weight"])
        power = float(configuration["power"])
        return 1.0 + (maximum - 1.0) * fraction.pow(power)

    def _high_noise_standardized_recovery_losses(
        self,
        *,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        condition: torch.Tensor | None,
        timesteps: torch.Tensor,
        local_inverse_slope: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """D4.3.2.20：直接约束高噪声pred_x0恢复。

        不在高噪声区执行完整robust-asinh反变换，而把真实target处的
        local inverse slope作为Jacobian，将模型空间x0预测误差换算成近似
        full-spectrum intensity误差，再除以condition train-only pointwise
        scale。低自然方差Raman点因此得到更强的恢复约束，高方差峰区则
        保留更大的允许变化范围。
        """

        zero = prediction.sum() * 0.0
        configuration = self.quality_fidelity_configuration.get(
            "high_noise_standardized_recovery", {}
        )
        if not bool(configuration.get("enabled", False)):
            return {
                "raw_loss": zero,
                "active_fraction": zero,
                "mean_gate": zero,
                "mean_abs_standardized_error": zero,
            }
        if condition is None:
            raise RuntimeError("D4.3.2.20高噪声恢复缺少condition。")
        if local_inverse_slope is None:
            raise RuntimeError(
                "D4.3.2.20高噪声恢复缺少local_inverse_slope。"
            )
        if self.objective != "pred_x0":
            raise RuntimeError("D4.3.2.20当前只支持objective=pred_x0。")
        if (
            self.high_noise_recovery_condition_vectors is None
            or self.high_noise_recovery_full_spectrum_scale is None
        ):
            raise RuntimeError(
                "D4.3.2.20高噪声恢复train-only reference尚未配置。"
            )

        epsilon = float(configuration["epsilon"])
        reference_conditions = self.high_noise_recovery_condition_vectors
        reference_scale = self.high_noise_recovery_full_spectrum_scale

        # 首次调用时一次性搬到GPU；之后保持缓存，避免CPU<->GPU同步。
        if (
            reference_conditions.device != prediction.device
            or reference_conditions.dtype != prediction.dtype
        ):
            reference_conditions = reference_conditions.to(
                device=prediction.device, dtype=prediction.dtype
            )
            self.high_noise_recovery_condition_vectors = reference_conditions
        if (
            reference_scale.device != prediction.device
            or reference_scale.dtype != prediction.dtype
        ):
            reference_scale = reference_scale.to(
                device=prediction.device, dtype=prediction.dtype
            )
            self.high_noise_recovery_full_spectrum_scale = reference_scale

        condition_distance = torch.sum(
            torch.square(
                condition.unsqueeze(1) - reference_conditions.unsqueeze(0)
            ),
            dim=2,
        )
        condition_index = torch.argmin(condition_distance, dim=1)

        # D4.3.2.20 shape compatibility:
        # train-only full-spectrum scale may be stored as
        #   [condition, Raman]
        # or
        #   [condition, 1, Raman].
        #
        # Normalize it once to [condition, Raman].
        if reference_scale.ndim == 3:
            if reference_scale.shape[1] != 1:
                raise RuntimeError(
                    "D4.3.2.20 high-noise reference三维形状异常："
                    f"{tuple(reference_scale.shape)}。"
                )
            reference_scale = reference_scale[:, 0, :]
            self.high_noise_recovery_full_spectrum_scale = (
                reference_scale
            )
        elif reference_scale.ndim != 2:
            raise RuntimeError(
                "D4.3.2.20 high-noise reference必须为二维"
                "或单通道三维tensor，实际形状="
                f"{tuple(reference_scale.shape)}。"
            )

        # D4.3.2.20 physical-axis alignment:
        #
        # 一维U-Net可能把1901点模型轴padding到1904点。
        # train-only Raman reference只对应真实1901个物理点，
        # 因此高噪声恢复loss必须裁掉模型末端padding点，
        # 而不能要求reference也扩展到1904点。
        physical_length = int(reference_scale.shape[-1])
        model_length = int(prediction.shape[-1])

        if model_length < physical_length:
            raise RuntimeError(
                "D4.3.2.20模型输出长度小于真实Raman reference长度："
                f"reference={tuple(reference_scale.shape)}, "
                f"prediction={tuple(prediction.shape)}。"
            )

        prediction = prediction[..., :physical_length]
        target = target[..., :physical_length]
        local_inverse_slope = local_inverse_slope[..., :physical_length]
        valid_mask = valid_mask[..., :physical_length]

        selected_scale = (
            reference_scale.index_select(
                0,
                condition_index,
            )[:, :physical_length]
            .unsqueeze(1)
            .clamp_min(epsilon)
        )

        standardized_error = (
            local_inverse_slope
            * (prediction - target)
            / selected_scale
        ) * valid_mask
        pointwise = F.smooth_l1_loss(
            standardized_error,
            torch.zeros_like(standardized_error),
            reduction="none",
            beta=float(configuration["smooth_l1_beta"]),
        )
        valid_count = valid_mask.flatten(start_dim=1).sum(dim=1).clamp_min(1.0)
        per_sample = (
            (pointwise * valid_mask).flatten(start_dim=1).sum(dim=1)
            / valid_count
        )

        fraction = timesteps.to(dtype=prediction.dtype) / float(
            max(self.num_timesteps - 1, 1)
        )
        minimum = float(configuration["minimum_timestep_fraction"])
        full_strength = float(configuration["full_strength_timestep_fraction"])
        gate = ((fraction - minimum) / max(full_strength - minimum, epsilon))
        gate = gate.clamp(0.0, 1.0).pow(float(configuration["gate_power"]))
        gate_sum = gate.sum().clamp_min(epsilon)

        raw_loss = (per_sample * gate).sum() / gate_sum
        active_fraction = (gate > 0.0).to(dtype=prediction.dtype).mean()
        mean_abs_error_per_sample = (
            (standardized_error.abs() * valid_mask)
            .flatten(start_dim=1)
            .sum(dim=1)
            / valid_count
        )
        mean_abs_error = (mean_abs_error_per_sample * gate).sum() / gate_sum

        return {
            "raw_loss": raw_loss,
            "active_fraction": active_fraction,
            "mean_gate": gate.mean(),
            "mean_abs_standardized_error": mean_abs_error,
        }

    def _sample_training_timesteps(
        self,
        count: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample all noise levels while reserving coverage for the hard tail."""

        number = int(count)
        if number <= 0:
            raise ValueError("训练时间步抽样数量必须大于0。")
        uniform = torch.randint(
            0,
            self.num_timesteps,
            (number,),
            device=device,
        ).long()
        configuration = self.quality_fidelity_configuration.get(
            "high_noise_sampling", {}
        )
        if not self.quality_fidelity_enabled or not bool(
            configuration.get("enabled", False)
        ):
            return uniform

        probability = float(configuration["sampling_probability"])
        minimum_fraction = float(configuration["minimum_timestep_fraction"])
        high_start = min(
            max(int(round(minimum_fraction * (self.num_timesteps - 1))), 0),
            self.num_timesteps - 1,
        )
        high = torch.randint(
            high_start,
            self.num_timesteps,
            (number,),
            device=device,
        ).long()
        choose_high = torch.rand(number, device=device) < probability
        return torch.where(choose_high, high, uniform)

    @staticmethod
    def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        count = mask.sum().clamp_min(1.0)
        return (value * mask).sum() / count

    def _condition_equalized_residual_tail_losses(
        self,
        *,
        prediction: torch.Tensor,
        valid_mask: torch.Tensor,
        condition: torch.Tensor | None,
        timesteps: torch.Tensor,
        sample_overdispersion_gate: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """D4.23：只处罚超出train-only residual envelope的pred_x0。

        Reference位于D4.21 Raman variance equalization之后的模型域，
        因此prediction和reference处于完全相同的pred_x0空间。

        正常训练分布内部没有任何处罚。
        """

        zero = prediction.sum() * 0.0

        configuration = (
            self.quality_fidelity_configuration.get(
                "condition_equalized_residual_tail",
                {},
            )
        )

        if not bool(
            configuration.get(
                "enabled",
                False,
            )
        ):
            return {
                "raw_loss": zero,
                "active_fraction": zero,
                "mean_abs_excess": zero,
                "mean_gate": zero,
                "overdispersion_gate_mean": zero,
            }

        if condition is None:
            raise RuntimeError(
                "D4.23 residual tail约束缺少condition。"
            )

        references = (
            self.equalized_residual_tail_condition_vectors
        )
        lower_reference = (
            self.equalized_residual_tail_lower
        )
        upper_reference = (
            self.equalized_residual_tail_upper
        )

        if (
            references is None
            or lower_reference is None
            or upper_reference is None
        ):
            raise RuntimeError(
                "D4.23 train-only residual tail reference尚未配置。"
            )

        # 首次使用时一次性搬到当前设备。
        if references.device != prediction.device:
            references = references.to(
                device=prediction.device,
                dtype=prediction.dtype,
            )
            lower_reference = lower_reference.to(
                device=prediction.device,
                dtype=prediction.dtype,
            )
            upper_reference = upper_reference.to(
                device=prediction.device,
                dtype=prediction.dtype,
            )

            self.equalized_residual_tail_condition_vectors = (
                references
            )
            self.equalized_residual_tail_lower = (
                lower_reference
            )
            self.equalized_residual_tail_upper = (
                upper_reference
            )

        active_condition = condition.detach()

        if active_condition.ndim != 2:
            active_condition = active_condition.reshape(
                active_condition.shape[0],
                -1,
            )

        reference_conditions = references

        if reference_conditions.ndim != 2:
            raise RuntimeError(
                "D4.23 reference condition必须是二维矩阵。"
            )

        if (
            active_condition.shape[1]
            != reference_conditions.shape[1]
        ):
            raise RuntimeError(
                "D4.23 condition维度与reference不一致。"
            )

        matches = torch.isclose(
            active_condition[:, None, :],
            reference_conditions[None, :, :],
            atol=1.0e-6,
            rtol=0.0,
        ).all(dim=-1)

        match_count = matches.sum(dim=1)

        if not torch.all(match_count == 1):
            raise RuntimeError(
                "D4.23无法唯一匹配condition-specific reference。"
            )

        reference_index = matches.to(
            dtype=torch.int64
        ).argmax(dim=1)

        lower = lower_reference[
            reference_index
        ]
        upper = upper_reference[
            reference_index
        ]

        target_length = prediction.shape[-1]

        if lower.shape[-1] > target_length:
            lower = lower[..., :target_length]
            upper = upper[..., :target_length]

        elif lower.shape[-1] < target_length:
            padding = (
                target_length
                - lower.shape[-1]
            )

            lower = F.pad(
                lower,
                (0, padding),
                value=0.0,
            )

            upper = F.pad(
                upper,
                (0, padding),
                value=0.0,
            )

        lower = lower.unsqueeze(1)
        upper = upper.unsqueeze(1)

        lower_excess = F.relu(
            lower - prediction
        )

        upper_excess = F.relu(
            prediction - upper
        )

        # ------------------------------------------------
        # D4.23.2 severe-tail-only
        #
        # D4.23原逻辑：
        # prediction一旦越过train-only envelope立即处罚。
        #
        # D4.23.2：
        # envelope之外再保留一段与该Raman点
        # train-only envelope宽度成比例的允许区。
        #
        # 轻微越界不处罚；
        # 只有明显的极端负谷/高峰才进入SmoothL1。
        #
        # fraction=0时严格退化为原D4.23。
        # ------------------------------------------------

        envelope_excess = (
            lower_excess
            + upper_excess
        )

        severe_deadband_fraction = float(
            configuration.get(
                "severe_deadband_fraction_of_envelope_width",
                0.0,
            )
        )

        if not (
            0.0
            <= severe_deadband_fraction
            <= 2.0
        ):
            raise ValueError(
                "D4.23.2 severe_deadband_fraction_of_"
                "envelope_width必须位于[0, 2]。"
            )

        envelope_width = (
            upper
            - lower
        ).clamp_min(
            1.0e-6
        )

        severe_deadband = (
            severe_deadband_fraction
            * envelope_width
        )

        excess = F.relu(
            envelope_excess
            - severe_deadband
        )

        pointwise_loss = F.smooth_l1_loss(
            excess,
            torch.zeros_like(excess),
            reduction="none",
            beta=float(
                configuration[
                    "smooth_l1_beta"
                ]
            ),
        )

        mask = valid_mask.to(
            dtype=prediction.dtype
        )

        valid_count = (
            mask.flatten(start_dim=1)
            .sum(dim=1)
            .clamp_min(1.0)
        )

        per_sample = (
            (
                pointwise_loss
                * mask
            )
            .flatten(start_dim=1)
            .sum(dim=1)
            / valid_count
        )

        # ------------------------------------------------
        # D4.23.1 condition-level overdispersion soft gate
        # ------------------------------------------------
        if sample_overdispersion_gate is None:
            overdispersion_gate = torch.ones_like(
                per_sample
            )
        else:
            if (
                sample_overdispersion_gate.ndim != 1
                or sample_overdispersion_gate.shape[0]
                != per_sample.shape[0]
            ):
                raise ValueError(
                    "D4.23.1 sample_overdispersion_gate"
                    "必须为[B]。"
                )

            overdispersion_gate = (
                sample_overdispersion_gate
                .to(
                    device=prediction.device,
                    dtype=prediction.dtype,
                )
                .detach()
                .clamp(
                    min=0.0,
                    max=1.0,
                )
            )

        timestep_fraction = (
            timesteps.to(
                dtype=prediction.dtype
            )
            / float(
                max(
                    self.num_timesteps - 1,
                    1,
                )
            )
        )

        maximum_fraction = float(
            configuration[
                "maximum_timestep_fraction"
            ]
        )

        normalized_fraction = (
            timestep_fraction
            / maximum_fraction
        )

        gate = (
            1.0
            - normalized_fraction
        ).clamp(
            min=0.0,
            max=1.0,
        )

        gate = gate.pow(
            float(
                configuration[
                    "timestep_weight_power"
                ]
            )
        )

        # timestep gate仍负责低/中噪声选择。
        #
        # overdispersion gate只乘在分子，不进入分母：
        # 这样gate=0.3就真的只保留约30%的tail penalty，
        # 不会因为分子分母同时乘0.3而被抵消。
        raw_loss = (
            (
                per_sample
                * gate
                * overdispersion_gate
            ).sum()
            / gate.sum().clamp_min(
                1.0e-12
            )
        )

        active_fraction = (
            (
                (excess > 0.0)
                .to(
                    dtype=prediction.dtype
                )
                * mask
            ).sum()
            / mask.sum().clamp_min(
                1.0
            )
        )

        mean_abs_excess = (
            (
                excess
                * mask
            ).sum()
            / mask.sum().clamp_min(
                1.0
            )
        )

        return {
            "raw_loss": raw_loss,
            "active_fraction": active_fraction,
            "mean_abs_excess": mean_abs_excess,
            "mean_gate": gate.mean(),
            "overdispersion_gate_mean": (
                overdispersion_gate.mean()
            ),
        }

    def _full_spectrum_tail_losses(
        self,
        *,
        prediction: torch.Tensor,
        target: torch.Tensor,
        full_spectrum_target: torch.Tensor,
        local_inverse_slope: torch.Tensor,
        valid_mask: torch.Tensor,
        condition: torch.Tensor,
        timesteps: torch.Tensor | None,
        configuration: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Constrain the actual reconstructed spectrum, not only its residual.

        The robust-asinh inverse is locally linearized around each real local
        residual.  This loss is used only at low/middle noise, where pred_x0 is
        close enough to the target for that train-only linearization to be
        reliable.  Four standardized QQ spans are matched with a dead band,
        so both over-wide and collapsed tails are corrected without forcing
        every condition to an identical global distribution.
        """

        zero = prediction.sum() * 0.0
        for name, value in (
            ("full_spectrum_target", full_spectrum_target),
            ("local_inverse_slope", local_inverse_slope),
        ):
            if value.shape != prediction.shape:
                raise ValueError(f"{name}形状必须与prediction一致。")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name}包含NaN或无穷值。")
        if torch.any(local_inverse_slope < 0.0):
            raise ValueError("local_inverse_slope不能为负。")

        reconstructed_prediction = (
            full_spectrum_target
            + local_inverse_slope * (prediction - target)
        ) * valid_mask
        probabilities = prediction.new_tensor(
            configuration["piecewise_probabilities"]
        )
        segment_weights = prediction.new_tensor(
            configuration["segment_weights"]
        )
        minimum_ratio = float(configuration["minimum_segment_span_ratio"])
        maximum_ratio = float(configuration["maximum_segment_span_ratio"])
        maximum_timestep_fraction = float(
            configuration["maximum_timestep_fraction"]
        )
        epsilon = float(configuration["epsilon"])

        _, inverse = torch.unique(
            condition.detach(), dim=0, sorted=True, return_inverse=True
        )
        reconstruction_losses: list[torch.Tensor] = []
        span_losses: list[torch.Tensor] = []
        active_group_count = 0
        for group_index in range(int(inverse.max().item()) + 1):
            selected = inverse == group_index
            if int(selected.sum().item()) < int(
                configuration["minimum_group_size"]
            ):
                continue
            if timesteps is not None:
                group_fraction = (
                    timesteps[selected].to(dtype=torch.float32).mean()
                    / float(max(self.num_timesteps - 1, 1))
                )
                if bool(group_fraction.item() > maximum_timestep_fraction):
                    continue

            group_mask = valid_mask[selected]
            group_target = full_spectrum_target[selected]
            group_prediction = reconstructed_prediction[selected]
            common_mask = torch.min(group_mask, dim=0, keepdim=True).values
            valid_length = int(common_mask.sum().item())
            if valid_length < 2:
                continue
            axis_weight = (
                float(configuration["short_axis_weight_multiplier"])
                if valid_length
                <= int(configuration["short_axis_maximum_points"])
                else 1.0
            )

            reconstruction_pointwise = F.smooth_l1_loss(
                group_prediction,
                group_target,
                reduction="none",
                beta=float(configuration["reconstruction_smooth_l1_beta"]),
            )
            reconstruction_losses.append(
                axis_weight
                * self._masked_mean(reconstruction_pointwise, group_mask)
            )

            if bool(
                configuration.get(
                    "use_train_reference",
                    False,
                )
            ):
                reference_items = (
                    self.full_spectrum_tail_reference_condition_vectors,
                    self.full_spectrum_tail_reference_mean,
                    self.full_spectrum_tail_reference_scale,
                    self.full_spectrum_tail_reference_probabilities,
                    self.full_spectrum_tail_reference_quantiles,
                )
                if any(item is None for item in reference_items):
                    raise RuntimeError(
                        "D4.3.2.19启用了use_train_reference，"
                        "但固定tail reference未加载。"
                    )

                # 工程加速：首次调用时一次性将全部126条件reference
                # 搬到当前GPU。之后不再逐batch CPU<->GPU往返。
                float_reference_names = (
                    "full_spectrum_tail_reference_condition_vectors",
                    "full_spectrum_tail_reference_mean",
                    "full_spectrum_tail_reference_scale",
                    "full_spectrum_tail_reference_probabilities",
                    "full_spectrum_tail_reference_quantiles",
                    "full_spectrum_tail_reference_stratified_quantiles",
                )
                for reference_name in float_reference_names:
                    reference_value = getattr(self, reference_name)
                    if reference_value is not None and (
                        reference_value.device != prediction.device
                        or reference_value.dtype != prediction.dtype
                    ):
                        setattr(
                            self,
                            reference_name,
                            reference_value.to(
                                device=prediction.device,
                                dtype=prediction.dtype,
                            ),
                        )
                if (
                    self.full_spectrum_tail_reference_variance_strata
                    is not None
                    and self.full_spectrum_tail_reference_variance_strata.device
                    != prediction.device
                ):
                    self.full_spectrum_tail_reference_variance_strata = (
                        self.full_spectrum_tail_reference_variance_strata.to(
                            device=prediction.device
                        )
                    )

                reference_conditions = (
                    self.full_spectrum_tail_reference_condition_vectors
                )
                group_condition = condition[selected][0].detach().to(
                    device=prediction.device,
                    dtype=prediction.dtype,
                )

                # 全GPU匹配，不使用.cpu()/.item()/nonzero().item()。
                condition_distance = torch.sum(
                    torch.square(
                        reference_conditions
                        - group_condition.unsqueeze(0)
                    ),
                    dim=1,
                )
                reference_index = torch.argmin(condition_distance)

                reference_mean = (
                    self.full_spectrum_tail_reference_mean[reference_index]
                    .reshape(1, 1, -1)
                )
                reference_scale = (
                    self.full_spectrum_tail_reference_scale[reference_index]
                    .reshape(1, 1, -1)
                    .clamp_min(epsilon)
                )
                reference_length = reference_mean.shape[-1]
                if group_prediction.shape[-1] < reference_length:
                    raise RuntimeError(
                        "D4.3.2.19 prediction长度小于reference长度。"
                    )

                group_prediction_for_tail = group_prediction[
                    ..., :reference_length
                ]
                common_mask_for_tail = common_mask[
                    ..., :reference_length
                ]
                prediction_z = (
                    group_prediction_for_tail - reference_mean
                ) / reference_scale

                if bool(
                    configuration.get(
                        "variance_stratified_enabled",
                        False,
                    )
                ):
                    if (
                        self.full_spectrum_tail_reference_variance_strata
                        is None
                        or self.full_spectrum_tail_reference_stratified_quantiles
                        is None
                    ):
                        raise RuntimeError(
                            "D4.3.2.19 variance-stratified reference未加载。"
                        )

                    stratum_index_vector = (
                        self.full_spectrum_tail_reference_variance_strata[
                            reference_index, :reference_length
                        ]
                    )
                    target_quantiles_by_stratum = (
                        self.full_spectrum_tail_reference_stratified_quantiles[
                            reference_index
                        ]
                    )
                    stratum_weights = prediction.new_tensor(
                        configuration["variance_strata_weights"]
                    )
                    number_of_strata = int(
                        configuration["number_of_variance_strata"]
                    )
                    per_stratum_losses: list[torch.Tensor] = []
                    per_stratum_weights: list[torch.Tensor] = []
                    common_bool = common_mask_for_tail > 0.5

                    for stratum_index in range(number_of_strata):
                        point_mask = (
                            stratum_index_vector == stratum_index
                        ).reshape(1, 1, -1) & common_bool
                        expanded_mask = point_mask.expand_as(prediction_z)
                        prediction_values = prediction_z[expanded_mask]
                        if prediction_values.numel() < 5:
                            continue

                        prediction_quantiles_stratum = torch.quantile(
                            prediction_values,
                            probabilities,
                        )
                        target_quantiles_stratum = (
                            target_quantiles_by_stratum[stratum_index]
                        ).detach()
                        target_spans_stratum = torch.diff(
                            target_quantiles_stratum
                        ).clamp_min(epsilon)
                        prediction_spans_stratum = torch.diff(
                            prediction_quantiles_stratum
                        ).clamp_min(epsilon)
                        ratios_stratum = (
                            prediction_spans_stratum
                            / target_spans_stratum
                        )
                        violation_stratum = (
                            F.relu(minimum_ratio - ratios_stratum)
                            + F.relu(ratios_stratum - maximum_ratio)
                        )
                        segment_loss_stratum = F.smooth_l1_loss(
                            violation_stratum,
                            torch.zeros_like(violation_stratum),
                            reduction="none",
                            beta=float(
                                configuration["span_smooth_l1_beta"]
                            ),
                        )
                        per_stratum_losses.append(
                            (segment_loss_stratum * segment_weights).sum()
                            / segment_weights.sum()
                        )
                        per_stratum_weights.append(
                            stratum_weights[stratum_index]
                        )

                    if not per_stratum_losses:
                        continue
                    stacked_losses = torch.stack(per_stratum_losses)
                    stacked_weights = torch.stack(per_stratum_weights)
                    span_losses.append(
                        axis_weight
                        * (stacked_losses * stacked_weights).sum()
                        / stacked_weights.sum()
                    )
                    active_group_count += 1
                    continue

                target_quantiles = (
                    self.full_spectrum_tail_reference_quantiles[
                        reference_index
                    ]
                ).detach()
                expanded_common = (
                    common_mask_for_tail.expand_as(
                        group_prediction_for_tail
                    )
                    > 0.5
                )
                prediction_quantiles = torch.quantile(
                    prediction_z[expanded_common],
                    probabilities,
                )

            else:
                center = torch.quantile(
                    group_target.detach(), 0.50, dim=0, keepdim=True
                )
                q25 = torch.quantile(
                    group_target.detach(), 0.25, dim=0, keepdim=True
                )
                q75 = torch.quantile(
                    group_target.detach(), 0.75, dim=0, keepdim=True
                )
                pointwise_scale = (q75 - q25).abs() / 1.349
                valid_scales = pointwise_scale[common_mask > 0.5]
                scale_floor = torch.quantile(
                    valid_scales,
                    float(configuration["pointwise_scale_floor_quantile"])
                    / 100.0,
                ).clamp_min(epsilon)
                pointwise_scale = pointwise_scale.clamp_min(scale_floor).detach()
                target_z = (group_target - center) / pointwise_scale
                prediction_z = (group_prediction - center) / pointwise_scale
                expanded_common = common_mask.expand_as(group_target) > 0.5
                target_quantiles = torch.quantile(
                    target_z[expanded_common], probabilities
                ).detach()
                prediction_quantiles = torch.quantile(
                    prediction_z[expanded_common], probabilities
                )
            target_spans = torch.diff(target_quantiles).clamp_min(epsilon)
            prediction_spans = torch.diff(prediction_quantiles).clamp_min(
                epsilon
            )
            ratios = prediction_spans / target_spans
            violation = F.relu(minimum_ratio - ratios) + F.relu(
                ratios - maximum_ratio
            )
            segment_loss = F.smooth_l1_loss(
                violation,
                torch.zeros_like(violation),
                reduction="none",
                beta=float(configuration["span_smooth_l1_beta"]),
            )
            span_losses.append(
                axis_weight
                * (segment_loss * segment_weights).sum()
                / segment_weights.sum()
            )
            active_group_count += 1

        reconstruction_loss = (
            torch.stack(reconstruction_losses).mean()
            if reconstruction_losses
            else zero
        )
        span_loss = torch.stack(span_losses).mean() if span_losses else zero
        return (
            reconstruction_loss,
            span_loss,
            prediction.new_tensor(float(active_group_count)),
        )

    def _quality_fidelity_losses(
        self,
        *,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        condition: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        full_spectrum_target: torch.Tensor | None = None,
        local_inverse_slope: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        zero = prediction.sum() * 0.0
        derivative_loss = zero
        multiscale_loss = zero
        multiscale_variance_loss = zero
        derivative_variance_loss = zero
        condition_mean_loss = zero
        envelope_loss = zero
        full_reconstruction_loss = zero
        full_tail_span_loss = zero
        full_tail_active_groups = zero

        full_spectrum_variance_profile_loss = zero
        full_spectrum_variance_profile_active_fraction = zero
        full_spectrum_variance_profile_mean_abs_z = zero
        full_spectrum_variance_profile_violation_fraction = zero

        full_spectrum_group_variance_loss = zero
        full_spectrum_group_variance_active_groups = zero
        full_spectrum_group_variance_mean_abs_z = zero
        full_spectrum_group_variance_over_fraction = zero
        full_spectrum_group_variance_under_fraction = zero

        # D4.23.1：
        # 默认全部为0；只有D4.17 active group才写入soft gate。
        full_spectrum_group_variance_sample_overdispersion_gate = (
            prediction.new_zeros(
                (
                    prediction.shape[0],
                )
            )
        )

        active_groups = zero

        derivative = self.quality_fidelity_configuration.get(
            "first_derivative", {}
        )
        if bool(derivative.get("enabled", False)):
            derivative_mask = valid_mask[..., 1:] * valid_mask[..., :-1]
            predicted_derivative = prediction[..., 1:] - prediction[..., :-1]
            target_derivative = target[..., 1:] - target[..., :-1]
            pointwise = F.smooth_l1_loss(
                predicted_derivative,
                target_derivative,
                reduction="none",
                beta=float(derivative["smooth_l1_beta"]),
            )
            derivative_loss = self._masked_mean(pointwise, derivative_mask)

        multiscale = self.quality_fidelity_configuration.get(
            "multiscale_shape", {}
        )
        if bool(multiscale.get("enabled", False)):
            scale_losses: list[torch.Tensor] = []
            for kernel_size in multiscale["kernel_sizes"]:
                padding = int(kernel_size) // 2
                pooled_mask = F.avg_pool1d(
                    valid_mask,
                    kernel_size=int(kernel_size),
                    stride=1,
                    padding=padding,
                    count_include_pad=True,
                )
                predicted_pool = F.avg_pool1d(
                    prediction * valid_mask,
                    kernel_size=int(kernel_size),
                    stride=1,
                    padding=padding,
                    count_include_pad=True,
                ) / pooled_mask.clamp_min(1.0e-12)
                target_pool = F.avg_pool1d(
                    target * valid_mask,
                    kernel_size=int(kernel_size),
                    stride=1,
                    padding=padding,
                    count_include_pad=True,
                ) / pooled_mask.clamp_min(1.0e-12)
                pointwise = F.smooth_l1_loss(
                    predicted_pool,
                    target_pool,
                    reduction="none",
                    beta=float(multiscale["smooth_l1_beta"]),
                )
                pooled_valid = (pooled_mask > 1.0e-12).to(pointwise.dtype)
                scale_losses.append(self._masked_mean(pointwise, pooled_valid))
            if scale_losses:
                multiscale_loss = torch.stack(scale_losses).mean()

        condition_mean = self.quality_fidelity_configuration.get(
            "condition_mean", {}
        )
        envelope = self.quality_fidelity_configuration.get(
            "condition_pointwise_envelope", {}
        )
        variance_floor = self.quality_fidelity_configuration.get(
            "condition_multiscale_variance_floor", {}
        )
        derivative_floor = self.quality_fidelity_configuration.get(
            "condition_derivative_variance_floor", {}
        )
        full_tail = self.quality_fidelity_configuration.get(
            "full_spectrum_tail_distribution", {}
        )

        variance_profile = (
            self.quality_fidelity_configuration.get(
                "condition_full_spectrum_variance_profile",
                {},
            )
        )

        group_variance = (
            self.quality_fidelity_configuration.get(
                "condition_full_spectrum_group_variance",
                {},
            )
        )

        if bool(
            variance_profile.get(
                "enabled",
                False,
            )
        ):
            if (
                full_spectrum_target is None
                or local_inverse_slope is None
            ):
                raise RuntimeError(
                    "完整谱variance-profile约束缺少"
                    "full_spectrum_target或local_inverse_slope。"
                )

            if (
                self.full_spectrum_variance_profile_loss_module
                is None
            ):
                raise RuntimeError(
                    "完整谱variance-profile约束"
                    "尚未配置train-only condition state。"
                )

            profile_result = (
                self.full_spectrum_variance_profile_loss_module(
                    prediction=prediction,
                    target=target,
                    full_spectrum_target=(
                        full_spectrum_target
                    ),
                    local_inverse_slope=(
                        local_inverse_slope
                    ),
                    valid_mask=valid_mask,
                    condition=condition,
                    timesteps=timesteps,
                    number_of_timesteps=(
                        self.num_timesteps
                    ),
                )
            )

            full_spectrum_variance_profile_loss = (
                profile_result[
                    "loss"
                ]
            )

            full_spectrum_variance_profile_active_fraction = (
                profile_result[
                    "active_sample_fraction"
                ]
            )

            full_spectrum_variance_profile_mean_abs_z = (
                profile_result[
                    "mean_abs_standardized_error"
                ]
            )

            full_spectrum_variance_profile_violation_fraction = (
                profile_result[
                    "violation_fraction"
                ]
            )

        if bool(
            group_variance.get(
                "enabled",
                False,
            )
        ):
            if (
                full_spectrum_target is None
                or local_inverse_slope is None
            ):
                raise RuntimeError(
                    "D4.3.2.17 group-variance约束缺少"
                    "full_spectrum_target或local_inverse_slope。"
                )

            if (
                self.full_spectrum_group_variance_loss_module
                is None
            ):
                raise RuntimeError(
                    "D4.3.2.17 group-variance约束"
                    "尚未配置train-only condition state。"
                )

            group_variance_result = (
                self.full_spectrum_group_variance_loss_module(
                    prediction=prediction,
                    target=target,
                    full_spectrum_target=(
                        full_spectrum_target
                    ),
                    local_inverse_slope=(
                        local_inverse_slope
                    ),
                    valid_mask=valid_mask,
                    condition=condition,
                    timesteps=timesteps,
                    number_of_timesteps=(
                        self.num_timesteps
                    ),
                )
            )

            full_spectrum_group_variance_loss = (
                group_variance_result[
                    "loss"
                ]
            )

            full_spectrum_group_variance_active_groups = (
                group_variance_result[
                    "active_groups"
                ]
            )

            full_spectrum_group_variance_mean_abs_z = (
                group_variance_result[
                    "mean_abs_standardized_difference"
                ]
            )

            full_spectrum_group_variance_over_fraction = (
                group_variance_result[
                    "overdispersion_fraction"
                ]
            )

            full_spectrum_group_variance_under_fraction = (
                group_variance_result[
                    "underdispersion_fraction"
                ]
            )

        if bool(
            group_variance.get(
                "enabled",
                False,
            )
        ):
            full_spectrum_group_variance_sample_overdispersion_gate = (
                group_variance_result[
                    "sample_overdispersion_gate"
                ]
            )

        if bool(full_tail.get("enabled", False)):
            if full_spectrum_target is None or local_inverse_slope is None:
                raise RuntimeError(
                    "完整谱尾部分布损失缺少full_spectrum_target或"
                    "local_inverse_slope。"
                )
            (
                full_reconstruction_loss,
                full_tail_span_loss,
                full_tail_active_groups,
            ) = self._full_spectrum_tail_losses(
                prediction=prediction,
                target=target,
                full_spectrum_target=full_spectrum_target,
                local_inverse_slope=local_inverse_slope,
                valid_mask=valid_mask,
                condition=condition,
                timesteps=timesteps,
                configuration=full_tail,
            )
        needs_condition_groups = bool(condition_mean.get("enabled", False)) or bool(
            envelope.get("enabled", False)
        ) or bool(variance_floor.get("enabled", False))
        needs_condition_groups = needs_condition_groups or bool(
            derivative_floor.get("enabled", False)
        )
        if needs_condition_groups:
            _, inverse = torch.unique(
                condition.detach(), dim=0, sorted=True, return_inverse=True
            )
            mean_group_losses: list[torch.Tensor] = []
            envelope_group_losses: list[torch.Tensor] = []
            variance_group_losses: list[torch.Tensor] = []
            derivative_variance_group_losses: list[torch.Tensor] = []
            active_group_count = 0
            for group_index in range(int(inverse.max().item()) + 1):
                selected = inverse == group_index
                selected_count = int(selected.sum().item())
                group_target = target[selected]
                group_prediction = prediction[selected]
                group_mask = valid_mask[selected]
                group_valid = torch.min(
                    group_mask, dim=0, keepdim=True
                ).values
                group_is_active = False
                timestep_is_active = True
                if timesteps is not None:
                    timestep_is_active = bool(
                        (
                            timesteps[selected].to(dtype=torch.float32).mean()
                            / float(max(self.num_timesteps - 1, 1))
                        ).item()
                        <= float(
                            variance_floor.get("maximum_timestep_fraction", 1.0)
                        )
                    )

                if (
                    bool(variance_floor.get("enabled", False))
                    and selected_count
                    >= int(variance_floor["minimum_group_size"])
                    and timestep_is_active
                ):
                    kernel_losses: list[torch.Tensor] = []
                    for kernel_size in variance_floor["kernel_sizes"]:
                        padding = int(kernel_size) // 2
                        pooled_mask = F.avg_pool1d(
                            group_mask,
                            kernel_size=int(kernel_size),
                            stride=1,
                            padding=padding,
                            count_include_pad=True,
                        )
                        predicted_pool = F.avg_pool1d(
                            group_prediction * group_mask,
                            kernel_size=int(kernel_size),
                            stride=1,
                            padding=padding,
                            count_include_pad=True,
                        ) / pooled_mask.clamp_min(1.0e-12)
                        target_pool = F.avg_pool1d(
                            group_target * group_mask,
                            kernel_size=int(kernel_size),
                            stride=1,
                            padding=padding,
                            count_include_pad=True,
                        ) / pooled_mask.clamp_min(1.0e-12)
                        predicted_std = torch.sqrt(
                            torch.mean(
                                torch.square(
                                    predicted_pool
                                    - predicted_pool.mean(dim=0, keepdim=True)
                                ),
                                dim=0,
                                keepdim=True,
                            ).clamp_min(1.0e-12)
                        )
                        target_std = torch.sqrt(
                            torch.mean(
                                torch.square(
                                    target_pool
                                    - target_pool.mean(dim=0, keepdim=True)
                                ),
                                dim=0,
                                keepdim=True,
                            ).clamp_min(1.0e-12)
                        ).detach()
                        valid_std = target_std[group_valid > 0.5]
                        active_threshold = torch.quantile(
                            valid_std,
                            float(variance_floor["active_std_quantile"]) / 100.0,
                        )
                        active_mask = group_valid * (
                            target_std >= active_threshold
                        ).to(group_valid.dtype)
                        minimum_std = (
                            float(variance_floor["minimum_std_ratio"])
                            * target_std
                        )
                        relative_deficit = F.relu(
                            minimum_std - predicted_std
                        ) / target_std.clamp_min(1.0e-6)
                        pointwise = F.smooth_l1_loss(
                            relative_deficit,
                            torch.zeros_like(relative_deficit),
                            reduction="none",
                            beta=float(variance_floor["smooth_l1_beta"]),
                        )
                        kernel_losses.append(
                            self._masked_mean(pointwise, active_mask)
                        )
                    if kernel_losses:
                        variance_group_losses.append(
                            torch.stack(kernel_losses).mean()
                        )
                        group_is_active = True

                derivative_timestep_is_active = True
                if timesteps is not None:
                    derivative_timestep_is_active = bool(
                        (
                            timesteps[selected].to(dtype=torch.float32).mean()
                            / float(max(self.num_timesteps - 1, 1))
                        ).item()
                        <= float(
                            derivative_floor.get(
                                "maximum_timestep_fraction", 1.0
                            )
                        )
                    )
                if (
                    bool(derivative_floor.get("enabled", False))
                    and selected_count
                    >= int(derivative_floor["minimum_group_size"])
                    and derivative_timestep_is_active
                ):
                    kernel_losses: list[torch.Tensor] = []
                    for kernel_size in derivative_floor[
                        "smoothing_kernel_sizes"
                    ]:
                        padding = int(kernel_size) // 2
                        pooled_mask = F.avg_pool1d(
                            group_mask,
                            kernel_size=int(kernel_size),
                            stride=1,
                            padding=padding,
                            count_include_pad=True,
                        )
                        predicted_pool = F.avg_pool1d(
                            group_prediction * group_mask,
                            kernel_size=int(kernel_size),
                            stride=1,
                            padding=padding,
                            count_include_pad=True,
                        ) / pooled_mask.clamp_min(1.0e-12)
                        target_pool = F.avg_pool1d(
                            group_target * group_mask,
                            kernel_size=int(kernel_size),
                            stride=1,
                            padding=padding,
                            count_include_pad=True,
                        ) / pooled_mask.clamp_min(1.0e-12)
                        predicted_derivative = (
                            predicted_pool[..., 1:] - predicted_pool[..., :-1]
                        )
                        target_derivative = (
                            target_pool[..., 1:] - target_pool[..., :-1]
                        )
                        predicted_std = torch.sqrt(torch.mean(torch.square(
                            predicted_derivative
                            - predicted_derivative.mean(dim=0, keepdim=True)
                        ), dim=0, keepdim=True).clamp_min(1.0e-12))
                        target_std = torch.sqrt(torch.mean(torch.square(
                            target_derivative
                            - target_derivative.mean(dim=0, keepdim=True)
                        ), dim=0, keepdim=True).clamp_min(1.0e-12)).detach()
                        full_window = (pooled_mask > 1.0 - 1.0e-6).to(
                            group_mask.dtype
                        )
                        derivative_valid = (
                            full_window[..., 1:] * full_window[..., :-1]
                        ).min(dim=0, keepdim=True).values
                        valid_std = target_std[derivative_valid > 0.5]
                        if valid_std.numel() == 0:
                            continue
                        active_threshold = torch.quantile(
                            valid_std,
                            float(derivative_floor["active_std_quantile"])
                            / 100.0,
                        )
                        active_mask = derivative_valid * (
                            target_std >= active_threshold
                        ).to(derivative_valid.dtype)
                        minimum_std = (
                            float(derivative_floor["minimum_std_ratio"])
                            * target_std
                        )
                        relative_deficit = F.relu(
                            minimum_std - predicted_std
                        ) / target_std.clamp_min(1.0e-6)
                        pointwise = F.smooth_l1_loss(
                            relative_deficit,
                            torch.zeros_like(relative_deficit),
                            reduction="none",
                            beta=float(derivative_floor["smooth_l1_beta"]),
                        )
                        kernel_losses.append(
                            self._masked_mean(pointwise, active_mask)
                        )
                    if kernel_losses:
                        derivative_variance_group_losses.append(
                            torch.stack(kernel_losses).mean()
                        )
                        group_is_active = True

                if bool(condition_mean.get("enabled", False)) and (
                    selected_count >= int(condition_mean["minimum_group_size"])
                ):
                    predicted_mean = group_prediction.mean(dim=0, keepdim=True)
                    target_mean = group_target.mean(dim=0, keepdim=True)
                    mean_pointwise = F.smooth_l1_loss(
                        predicted_mean,
                        target_mean,
                        reduction="none",
                        beta=float(condition_mean["smooth_l1_beta"]),
                    )
                    mean_group_losses.append(
                        self._masked_mean(mean_pointwise, group_valid)
                    )
                    group_is_active = True

                if bool(envelope.get("enabled", False)) and (
                    selected_count >= int(envelope["minimum_group_size"])
                ):
                    lower_q = torch.quantile(
                        group_target.detach(),
                        float(envelope["lower_quantile"]),
                        dim=0,
                        keepdim=True,
                    )
                    upper_q = torch.quantile(
                        group_target.detach(),
                        float(envelope["upper_quantile"]),
                        dim=0,
                        keepdim=True,
                    )
                    q25 = torch.quantile(
                        group_target.detach(), 0.25, dim=0, keepdim=True
                    )
                    q75 = torch.quantile(
                        group_target.detach(), 0.75, dim=0, keepdim=True
                    )
                    spread = (q75 - q25).clamp_min(0.0)
                    margin = float(envelope["iqr_margin"]) * spread
                    lower = lower_q - margin
                    upper = upper_q + margin
                    violation = F.relu(lower - group_prediction) + F.relu(
                        group_prediction - upper
                    )
                    envelope_pointwise = F.smooth_l1_loss(
                        violation,
                        torch.zeros_like(violation),
                        reduction="none",
                        beta=float(envelope["smooth_l1_beta"]),
                    )
                    envelope_group_losses.append(
                        self._masked_mean(envelope_pointwise, group_mask)
                    )
                    group_is_active = True
                active_group_count += int(group_is_active)

            if mean_group_losses:
                condition_mean_loss = torch.stack(mean_group_losses).mean()
            if envelope_group_losses:
                envelope_loss = torch.stack(envelope_group_losses).mean()
            if variance_group_losses:
                multiscale_variance_loss = torch.stack(
                    variance_group_losses
                ).mean()
            if derivative_variance_group_losses:
                derivative_variance_loss = torch.stack(
                    derivative_variance_group_losses
                ).mean()
            active_groups = prediction.new_tensor(float(active_group_count))

        raw = (
            float(derivative.get("weight", 0.0)) * derivative_loss
            + float(multiscale.get("weight", 0.0)) * multiscale_loss
            + float(variance_floor.get("weight", 0.0))
            * multiscale_variance_loss
            + float(derivative_floor.get("weight", 0.0))
            * derivative_variance_loss
            + float(condition_mean.get("weight", 0.0)) * condition_mean_loss
            + float(envelope.get("weight", 0.0)) * envelope_loss
            + float(full_tail.get("reconstruction_weight", 0.0))
            * full_reconstruction_loss
            + float(full_tail.get("span_weight", 0.0))
            * full_tail_span_loss
            + float(
                variance_profile.get(
                    "weight",
                    0.0,
                )
            )
            * full_spectrum_variance_profile_loss
            + float(
                group_variance.get(
                    "weight",
                    0.0,
                )
            )
            * full_spectrum_group_variance_loss
        )
        return {
            "quality_fidelity_raw_loss": raw,
            "first_derivative_fidelity_loss": derivative_loss,
            "multiscale_shape_loss": multiscale_loss,
            "condition_multiscale_variance_floor_loss": multiscale_variance_loss,
            "condition_derivative_variance_floor_loss": derivative_variance_loss,
            "condition_mean_fidelity_loss": condition_mean_loss,
            "condition_envelope_loss": envelope_loss,
            "full_spectrum_reconstruction_loss": full_reconstruction_loss,
            "full_spectrum_tail_span_loss": full_tail_span_loss,
            "full_spectrum_tail_active_groups": full_tail_active_groups,

            "full_spectrum_variance_profile_loss": (
                full_spectrum_variance_profile_loss
            ),

            "full_spectrum_variance_profile_active_fraction": (
                full_spectrum_variance_profile_active_fraction
            ),

            "full_spectrum_variance_profile_mean_abs_z": (
                full_spectrum_variance_profile_mean_abs_z
            ),

            "full_spectrum_variance_profile_violation_fraction": (
                full_spectrum_variance_profile_violation_fraction
            ),

            "full_spectrum_group_variance_loss": (
                full_spectrum_group_variance_loss
            ),
            "full_spectrum_group_variance_active_groups": (
                full_spectrum_group_variance_active_groups
            ),
            "full_spectrum_group_variance_mean_abs_z": (
                full_spectrum_group_variance_mean_abs_z
            ),
            "full_spectrum_group_variance_over_fraction": (
                full_spectrum_group_variance_over_fraction
            ),
            "full_spectrum_group_variance_under_fraction": (
                full_spectrum_group_variance_under_fraction
            ),

            "full_spectrum_group_variance_sample_overdispersion_gate": (
                full_spectrum_group_variance_sample_overdispersion_gate
            ),

            "quality_fidelity_active_groups": active_groups,
        }

    def configure_diversity_constraints(
        self,
        *,
        diversity_constraint_state: dict[str, Any],
    ) -> None:
        if not self.diversity_enabled:
            return
        self.diversity_loss_module = DifferentiableConditionAwareDiversityLoss(
            diversity_constraint_state=diversity_constraint_state,
            padded_length=self.seq_length,
        )


        # ----------------------------------------------------
        # D4.23 condition-wise equalized-residual tail state
        # ----------------------------------------------------
        equalized_tail_configuration = (
            self.quality_fidelity_configuration.get(
                "condition_equalized_residual_tail",
                {},
            )
        )

        if bool(
            equalized_tail_configuration.get(
                "enabled",
                False,
            )
        ):
            required_keys = (
                "equalized_residual_tail_condition_vectors",
                "equalized_residual_tail_lower",
                "equalized_residual_tail_upper",
            )

            missing = [
                name
                for name in required_keys
                if name not in diversity_constraint_state
            ]

            if missing:
                raise ValueError(
                    "D4.23 train-only residual tail state缺少字段："
                    + ", ".join(missing)
                )

            self.equalized_residual_tail_condition_vectors = (
                torch.as_tensor(
                    diversity_constraint_state[
                        "equalized_residual_tail_condition_vectors"
                    ],
                    dtype=torch.float32,
                )
            )

            lower = torch.as_tensor(
                diversity_constraint_state[
                    "equalized_residual_tail_lower"
                ],
                dtype=torch.float32,
            )

            upper = torch.as_tensor(
                diversity_constraint_state[
                    "equalized_residual_tail_upper"
                ],
                dtype=torch.float32,
            )

            if (
                lower.ndim != 2
                or upper.ndim != 2
                or lower.shape != upper.shape
            ):
                raise ValueError(
                    "D4.23 lower/upper reference形状无效。"
                )

            if (
                lower.shape[0]
                != self.equalized_residual_tail_condition_vectors.shape[0]
            ):
                raise ValueError(
                    "D4.23 condition数量与tail reference不一致。"
                )

            if lower.shape[1] > self.seq_length:
                raise ValueError(
                    "D4.23 residual tail reference长度"
                    "超过模型sequence length。"
                )

            self.equalized_residual_tail_lower = lower
            self.equalized_residual_tail_upper = upper

        # ----------------------------------------------------
        # D4.3.2.18
        # 从train-only diversity state加载固定QQ reference。
        # ----------------------------------------------------
        tail_configuration = (
            self.quality_fidelity_configuration.get(
                "full_spectrum_tail_distribution",
                {},
            )
        )

        if (
            bool(
                tail_configuration.get(
                    "enabled",
                    False,
                )
            )
            and bool(
                tail_configuration.get(
                    "use_train_reference",
                    False,
                )
            )
        ):
            required_tail_reference_keys = [
                "condition_vectors",
                "full_spectrum_mean",
                "full_spectrum_scale",
                "full_spectrum_tail_probabilities",
                "full_spectrum_tail_quantiles",
            ]
            if bool(
                tail_configuration.get(
                    "variance_stratified_enabled",
                    False,
                )
            ):
                required_tail_reference_keys.extend(
                    [
                        "full_spectrum_variance_stratum_index",
                        "full_spectrum_stratified_tail_quantiles",
                    ]
                )

            missing_keys = [
                name
                for name in required_tail_reference_keys
                if name not in diversity_constraint_state
            ]
            if missing_keys:
                raise ValueError(
                    "D4.3.2.19 train-reference tail state缺少字段："
                    + ", ".join(missing_keys)
                )

            # 先保存在CPU；首次训练loss调用时一次性搬到当前GPU并缓存。
            self.full_spectrum_tail_reference_condition_vectors = (
                torch.as_tensor(
                    diversity_constraint_state["condition_vectors"],
                    dtype=torch.float32,
                )
            )
            self.full_spectrum_tail_reference_mean = torch.as_tensor(
                diversity_constraint_state["full_spectrum_mean"],
                dtype=torch.float32,
            )
            self.full_spectrum_tail_reference_scale = torch.as_tensor(
                diversity_constraint_state["full_spectrum_scale"],
                dtype=torch.float32,
            )
            self.full_spectrum_tail_reference_probabilities = (
                torch.as_tensor(
                    diversity_constraint_state[
                        "full_spectrum_tail_probabilities"
                    ],
                    dtype=torch.float32,
                )
            )
            self.full_spectrum_tail_reference_quantiles = torch.as_tensor(
                diversity_constraint_state["full_spectrum_tail_quantiles"],
                dtype=torch.float32,
            )

            stored_probabilities = (
                self.full_spectrum_tail_reference_probabilities
            )
            configured_probabilities = torch.as_tensor(
                tail_configuration["piecewise_probabilities"],
                dtype=torch.float32,
            )
            if (
                stored_probabilities.shape != configured_probabilities.shape
                or not torch.allclose(
                    stored_probabilities,
                    configured_probabilities,
                    rtol=0.0,
                    atol=1.0e-6,
                )
            ):
                raise ValueError(
                    "D4.3.2.19配置piecewise_probabilities与"
                    "train-reference state不一致。"
                )

            if bool(
                tail_configuration.get(
                    "variance_stratified_enabled",
                    False,
                )
            ):
                expected_strata = int(
                    tail_configuration["number_of_variance_strata"]
                )
                stored_strata = int(
                    diversity_constraint_state.get(
                        "full_spectrum_variance_stratum_count",
                        -1,
                    )
                )
                if stored_strata != expected_strata:
                    raise ValueError(
                        "D4.3.2.19 variance strata数量与"
                        "train-reference state不一致。"
                    )
                self.full_spectrum_tail_reference_variance_strata = (
                    torch.as_tensor(
                        diversity_constraint_state[
                            "full_spectrum_variance_stratum_index"
                        ],
                        dtype=torch.long,
                    )
                )
                self.full_spectrum_tail_reference_stratified_quantiles = (
                    torch.as_tensor(
                        diversity_constraint_state[
                            "full_spectrum_stratified_tail_quantiles"
                        ],
                        dtype=torch.float32,
                    )
                )

        high_noise_recovery = self.quality_fidelity_configuration.get(
            "high_noise_standardized_recovery", {}
        )
        if bool(high_noise_recovery.get("enabled", False)):
            required_recovery_keys = (
                "condition_vectors",
                "full_spectrum_scale",
            )
            missing_recovery_keys = [
                name
                for name in required_recovery_keys
                if name not in diversity_constraint_state
            ]
            if missing_recovery_keys:
                raise ValueError(
                    "D4.3.2.20高噪声恢复state缺少字段："
                    + ", ".join(missing_recovery_keys)
                )
            self.high_noise_recovery_condition_vectors = torch.as_tensor(
                diversity_constraint_state["condition_vectors"],
                dtype=torch.float32,
            )
            self.high_noise_recovery_full_spectrum_scale = torch.as_tensor(
                diversity_constraint_state["full_spectrum_scale"],
                dtype=torch.float32,
            )

        variance_profile = (
            self.quality_fidelity_configuration.get(
                "condition_full_spectrum_variance_profile",
                {},
            )
        )

        if bool(
            variance_profile.get(
                "enabled",
                False,
            )
        ):
            self.full_spectrum_variance_profile_loss_module = (
                DifferentiableConditionFullSpectrumVarianceProfileLoss(
                    diversity_constraint_state=(
                        diversity_constraint_state
                    ),
                    padded_length=self.seq_length,
                    configuration=variance_profile,
                )
            )

        group_variance = (
            self.quality_fidelity_configuration.get(
                "condition_full_spectrum_group_variance",
                {},
            )
        )

        if bool(
            group_variance.get(
                "enabled",
                False,
            )
        ):
            self.full_spectrum_group_variance_loss_module = (
                DifferentiableConditionFullSpectrumGroupVarianceLoss(
                    diversity_constraint_state=(
                        diversity_constraint_state
                    ),
                    padded_length=self.seq_length,
                    configuration=group_variance,
                )
            )

    def get_latest_loss_components(self) -> dict[str, torch.Tensor]:
        return dict(self._latest_loss_components)

    def _prepare_condition(
        self,
        condition: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        expected = int(getattr(self.model, "condition_dimension", 0))
        if expected == 0:
            if condition is not None:
                raise ValueError("无条件模型不能接收condition。")
            return None
        if condition is None:
            raise ValueError("D4.1条件模型必须提供condition。")
        if not torch.is_tensor(condition):
            raise TypeError("condition必须是torch.Tensor。")
        prepared = condition.to(device=device, dtype=dtype)
        if prepared.ndim == 1:
            prepared = prepared.unsqueeze(0)
        if prepared.shape[0] == 1 and batch_size > 1:
            prepared = prepared.expand(batch_size, -1)
        if prepared.shape != (batch_size, expected):
            raise ValueError(
                "condition形状必须为[B,C]："
                f"实际{tuple(prepared.shape)}，期望({batch_size}, {expected})。"
            )
        if not torch.isfinite(prepared).all():
            raise ValueError("condition包含NaN或无穷值。")
        return prepared

    def _prepare_mask(
        self,
        valid_mask: torch.Tensor | None,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if valid_mask is None:
            return torch.ones_like(reference)
        if not torch.is_tensor(valid_mask):
            raise TypeError("valid_mask必须是torch.Tensor。")

        mask = valid_mask.to(
            device=reference.device,
            dtype=reference.dtype,
        )
        if mask.ndim == 1:
            mask = mask.reshape(1, 1, -1)
        elif mask.ndim == 2:
            mask = mask.unsqueeze(1)

        if mask.shape[0] == 1 and reference.shape[0] > 1:
            mask = mask.expand(reference.shape[0], -1, -1)
        if mask.shape != reference.shape:
            raise ValueError(
                "valid_mask形状必须与光谱一致："
                f"掩码={tuple(mask.shape)}，光谱={tuple(reference.shape)}。"
            )
        if not torch.isfinite(mask).all():
            raise ValueError("valid_mask包含NaN或无穷值。")
        if not torch.logical_or(mask == 0.0, mask == 1.0).all():
            raise ValueError("valid_mask只能包含0和1。")
        if torch.any(mask.flatten(start_dim=1).sum(dim=1) <= 0.0):
            raise ValueError("每条光谱至少需要一个有效Raman点。")
        return mask

    def _prepare_prior_conditioning(
        self,
        prior_conditioning: torch.Tensor | None,
        reference: torch.Tensor,
    ) -> torch.Tensor | None:
        enabled = bool(
            getattr(self.model, "prior_conditioning_enabled", False)
        )
        if not enabled:
            if prior_conditioning is not None:
                raise ValueError("当前U-Net未启用prior_conditioning。")
            return None
        if prior_conditioning is None:
            raise ValueError("先验条件化扩散必须提供prior_conditioning。")
        if not torch.is_tensor(prior_conditioning):
            raise TypeError("prior_conditioning必须是torch.Tensor。")
        prepared = prior_conditioning.to(
            device=reference.device, dtype=reference.dtype
        )
        if prepared.ndim == 1:
            prepared = prepared.reshape(1, 1, -1)
        elif prepared.ndim == 2:
            prepared = prepared.unsqueeze(1)
        if prepared.shape[0] == 1 and reference.shape[0] > 1:
            prepared = prepared.expand(reference.shape[0], -1, -1)
        if prepared.shape != reference.shape:
            raise ValueError(
                "prior_conditioning形状必须与光谱一致："
                f"先验={tuple(prepared.shape)}，光谱={tuple(reference.shape)}。"
            )
        if not torch.isfinite(prepared).all():
            raise ValueError("prior_conditioning包含NaN或无穷值。")
        return prepared

    def forward(
        self,
        img: torch.Tensor,
        *args,
        valid_mask: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
        prior_conditioning: torch.Tensor | None = None,
        full_spectrum_target: torch.Tensor | None = None,
        local_inverse_slope: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, channels, sequence_length = img.shape
        if channels != self.channels or sequence_length != self.seq_length:
            raise ValueError(
                "输入光谱形状与扩散模型不一致："
                f"得到{tuple(img.shape)}，期望通道={self.channels}、"
                f"长度={self.seq_length}。"
            )

        mask = self._prepare_mask(valid_mask, img)
        prepared_condition = self._prepare_condition(
            condition,
            batch_size,
            img.device,
            img.dtype,
        )
        prepared_prior = self._prepare_prior_conditioning(
            prior_conditioning, img
        )
        if prepared_prior is not None:
            prepared_prior = prepared_prior * mask
        prepared_full_target = None
        prepared_inverse_slope = None
        if full_spectrum_target is not None or local_inverse_slope is not None:
            if full_spectrum_target is None or local_inverse_slope is None:
                raise ValueError("完整谱训练上下文必须成对提供。")
            prepared_full_target = full_spectrum_target.to(
                device=img.device, dtype=img.dtype
            )
            prepared_inverse_slope = local_inverse_slope.to(
                device=img.device, dtype=img.dtype
            )
            if (
                prepared_full_target.shape != img.shape
                or prepared_inverse_slope.shape != img.shape
            ):
                raise ValueError("完整谱训练上下文形状必须与输入光谱一致。")
            if not torch.isfinite(prepared_full_target).all() or not torch.isfinite(
                prepared_inverse_slope
            ).all():
                raise ValueError("完整谱训练上下文包含NaN或无穷值。")
            if torch.any(prepared_inverse_slope < 0.0):
                raise ValueError("local_inverse_slope不能为负。")
            prepared_full_target = prepared_full_target * mask
            prepared_inverse_slope = prepared_inverse_slope * mask
        if self.diversity_enabled:
            if prepared_condition is None:
                raise RuntimeError("D4.3多样性约束要求提供condition。")
            _, inverse = torch.unique(
                prepared_condition,
                dim=0,
                sorted=True,
                return_inverse=True,
            )
            grouped_timestep = self._sample_training_timesteps(
                int(inverse.max().item()) + 1,
                img.device,
            )
            timestep = grouped_timestep[inverse]
        else:
            timestep = self._sample_training_timesteps(batch_size, img.device)
        normalized = self.normalize(img) * mask
        return self.p_losses(
            normalized,
            timestep,
            *args,
            valid_mask=mask,
            condition=prepared_condition,
            prior_conditioning=prepared_prior,
            full_spectrum_target=prepared_full_target,
            local_inverse_slope=prepared_inverse_slope,
            **kwargs,
        )

    def p_losses(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
        model_forward_kwargs: dict[str, Any] | None = None,
        return_reduced_loss: bool = True,
        loss_reduction: str = "mean",
        valid_mask: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
        prior_conditioning: torch.Tensor | None = None,
        full_spectrum_target: torch.Tensor | None = None,
        local_inverse_slope: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if model_forward_kwargs is None:
            model_forward_kwargs = {}
        if loss_reduction not in {"mean", "none"}:
            raise ValueError("loss_reduction只支持mean或none。")

        mask = self._prepare_mask(valid_mask, x_start)
        prepared_condition = self._prepare_condition(
            condition,
            x_start.shape[0],
            x_start.device,
            x_start.dtype,
        )
        prepared_prior = self._prepare_prior_conditioning(
            prior_conditioning, x_start
        )
        if prepared_prior is not None:
            prepared_prior = prepared_prior * mask
        if full_spectrum_target is not None or local_inverse_slope is not None:
            if full_spectrum_target is None or local_inverse_slope is None:
                raise ValueError("完整谱训练上下文必须成对提供。")
            full_spectrum_target = full_spectrum_target.to(
                device=x_start.device, dtype=x_start.dtype
            )
            local_inverse_slope = local_inverse_slope.to(
                device=x_start.device, dtype=x_start.dtype
            )
            if (
                full_spectrum_target.shape != x_start.shape
                or local_inverse_slope.shape != x_start.shape
            ):
                raise ValueError("完整谱训练上下文形状必须与x_start一致。")
            full_spectrum_target = full_spectrum_target * mask
            local_inverse_slope = local_inverse_slope * mask
        x_start = x_start * mask
        if noise is None:
            noise = torch.randn_like(x_start)
        noise = noise * mask

        noisy_input = self.q_sample(
            x_start=x_start,
            t=t,
            noise=noise,
        ) * mask

        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                self_condition = self.model_predictions(
                    noisy_input,
                    t,
                ).pred_x_start
                self_condition = self_condition.detach() * mask
            model_forward_kwargs = {
                **model_forward_kwargs,
                "self_cond": self_condition,
            }

        model_out = self.model(
            noisy_input,
            t,
            valid_mask=mask,
            condition=prepared_condition,
            prior_conditioning=prepared_prior,
            **model_forward_kwargs,
        )

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        elif self.objective == "pred_v":
            target = self.predict_v(x_start, t, noise)
        else:
            raise ValueError(f"未知扩散预测目标：{self.objective}")

        pointwise_loss = F.mse_loss(
            model_out,
            target,
            reduction="none",
        )
        weighted_pointwise = (
            pointwise_loss
            * mask
            * _extract(self.loss_weight, t, pointwise_loss.shape)
        )

        if not return_reduced_loss:
            return weighted_pointwise

        valid_count = mask.flatten(start_dim=1).sum(dim=1).clamp_min(1.0)
        per_sample = (
            weighted_pointwise.flatten(start_dim=1).sum(dim=1)
            / valid_count
        )
        if loss_reduction == "none":
            return per_sample
        ddpm_uniform_loss = per_sample.mean()
        timestep_fidelity_weight = self._high_noise_weights(t).to(
            device=per_sample.device,
            dtype=per_sample.dtype,
        )
        ddpm_loss = (
            (per_sample * timestep_fidelity_weight).sum()
            / timestep_fidelity_weight.sum().clamp_min(1.0e-12)
        )
        zero = torch.zeros_like(ddpm_loss)

        high_noise_recovery = self._high_noise_standardized_recovery_losses(
            prediction=model_out,
            target=target,
            valid_mask=mask,
            condition=prepared_condition,
            timesteps=t,
            local_inverse_slope=local_inverse_slope,
        )
        high_noise_recovery_configuration = (
            self.quality_fidelity_configuration.get(
                "high_noise_standardized_recovery", {}
            )
        )
        if bool(high_noise_recovery_configuration.get("enabled", False)):
            uncapped_high_noise_recovery = (
                float(high_noise_recovery_configuration["total_weight"])
                * high_noise_recovery["raw_loss"]
            )
            high_noise_recovery_cap = (
                ddpm_loss.detach()
                * float(
                    high_noise_recovery_configuration[
                        "maximum_total_ratio_to_ddpm"
                    ]
                )
            )
            high_noise_recovery_scale = torch.clamp(
                high_noise_recovery_cap
                / uncapped_high_noise_recovery.detach()
                .abs()
                .clamp_min(1.0e-12),
                max=1.0,
            )
            weighted_high_noise_recovery = (
                uncapped_high_noise_recovery * high_noise_recovery_scale
            )
        else:
            uncapped_high_noise_recovery = zero
            weighted_high_noise_recovery = zero
            high_noise_recovery_scale = zero

        # ----------------------------------------------------
        # D4.23 / D4.23.1 residual-tail状态初始化
        #
        # D4.23.1需要先得到D4.17 same-condition variance gate，
        # 因此真正tail loss计算移动到quality_fidelity之后。
        # ----------------------------------------------------
        equalized_tail_configuration = (
            self.quality_fidelity_configuration.get(
                "condition_equalized_residual_tail",
                {},
            )
        )

        equalized_residual_tail = {
            "raw_loss": zero,
            "active_fraction": zero,
            "mean_abs_excess": zero,
            "mean_gate": zero,
            "overdispersion_gate_mean": zero,
        }

        uncapped_equalized_residual_tail = zero
        weighted_equalized_residual_tail = zero
        equalized_tail_scale = zero

        if self.diversity_enabled:
            if self.diversity_loss_module is None:
                raise RuntimeError(
                    "D4.3扩散模型尚未配置diversity_constraint_state。"
                )
            if prepared_condition is None:
                raise RuntimeError("D4.3多样性损失缺少condition。")
            diversity = self.diversity_loss_module(
                predicted_scaled_residual=model_out,
                target_scaled_residual=x_start,
                timesteps=t,
                alphas_cumprod=self.alphas_cumprod,
                valid_mask=mask,
                condition=prepared_condition,
            )
            uncapped_diversity = (
                self.diversity_total_weight
                * diversity["diversity_timestep_weighted_loss"]
            )
            diversity_cap = ddpm_loss.detach() * self.diversity_maximum_ratio
            diversity_scale = torch.clamp(
                diversity_cap
                / uncapped_diversity.detach().abs().clamp_min(1.0e-12),
                max=1.0,
            )
            weighted_diversity = uncapped_diversity * diversity_scale
        else:
            diversity = {}
            uncapped_diversity = zero
            weighted_diversity = zero

        independent_full_spectrum_raw = zero
        uncapped_legacy_quality = zero
        weighted_legacy_quality = zero
        legacy_quality_scale = zero
        uncapped_full_spectrum_quality = zero
        weighted_full_spectrum_quality = zero
        full_spectrum_quality_scale = zero

        if self.quality_fidelity_enabled:
            if prepared_condition is None:
                raise RuntimeError("D4.3.2保真约束缺少condition。")

            quality = self._quality_fidelity_losses(
                prediction=model_out,
                target=x_start,
                valid_mask=mask,
                condition=prepared_condition,
                timesteps=t,
                full_spectrum_target=full_spectrum_target,
                local_inverse_slope=local_inverse_slope,
            )

            # ------------------------------------------------
            # D4.23.1：
            # 使用已经计算完成的D4.17 same-condition
            # overdispersion gate控制residual-tail penalty。
            # ------------------------------------------------
            if bool(
                equalized_tail_configuration.get(
                    "enabled",
                    False,
                )
            ):
                use_group_gate = bool(
                    equalized_tail_configuration.get(
                        "use_group_overdispersion_gate",
                        False,
                    )
                )

                if use_group_gate:
                    group_variance_configuration = (
                        self.quality_fidelity_configuration.get(
                            "condition_full_spectrum_group_variance",
                            {},
                        )
                    )

                    if not bool(
                        group_variance_configuration.get(
                            "enabled",
                            False,
                        )
                    ):
                        raise RuntimeError(
                            "D4.23.1要求"
                            "condition_full_spectrum_group_variance"
                            "保持enabled=True。"
                        )

                    sample_overdispersion_gate = quality.get(
                        "full_spectrum_group_variance_"
                        "sample_overdispersion_gate"
                    )

                    if sample_overdispersion_gate is None:
                        raise RuntimeError(
                            "D4.23.1没有获得D4.17 "
                            "sample overdispersion gate。"
                        )
                else:
                    # 保持旧D4.23完全兼容：
                    # 没有gate时相当于每个sample gate=1。
                    sample_overdispersion_gate = None

                equalized_residual_tail = (
                    self._condition_equalized_residual_tail_losses(
                        prediction=model_out,
                        valid_mask=mask,
                        condition=prepared_condition,
                        timesteps=t,
                        sample_overdispersion_gate=(
                            sample_overdispersion_gate
                        ),
                    )
                )

                uncapped_equalized_residual_tail = (
                    float(
                        equalized_tail_configuration[
                            "total_weight"
                        ]
                    )
                    * equalized_residual_tail[
                        "raw_loss"
                    ]
                )

                equalized_tail_cap = (
                    ddpm_loss.detach()
                    * float(
                        equalized_tail_configuration[
                            "maximum_total_ratio_to_ddpm"
                        ]
                    )
                )

                equalized_tail_scale = torch.clamp(
                    equalized_tail_cap
                    / uncapped_equalized_residual_tail
                    .detach()
                    .abs()
                    .clamp_min(
                        1.0e-12
                    ),
                    max=1.0,
                )

                weighted_equalized_residual_tail = (
                    uncapped_equalized_residual_tail
                    * equalized_tail_scale
                )

            independent_budget = self.quality_fidelity_configuration.get(
                "independent_full_spectrum_budget",
                {},
            )

            if bool(independent_budget.get("enabled", False)):
                full_tail_configuration = (
                    self.quality_fidelity_configuration.get(
                        "full_spectrum_tail_distribution",
                        {},
                    )
                )

                independent_full_spectrum_raw = (
                    float(
                        full_tail_configuration.get(
                            "reconstruction_weight",
                            0.0,
                        )
                    )
                    * quality["full_spectrum_reconstruction_loss"]
                    + float(
                        full_tail_configuration.get(
                            "span_weight",
                            0.0,
                        )
                    )
                    * quality["full_spectrum_tail_span_loss"]
                    + float(
                        self.quality_fidelity_configuration.get(
                            "condition_full_spectrum_variance_profile",
                            {},
                        ).get(
                            "weight",
                            0.0,
                        )
                    )
                    * quality.get(
                        "full_spectrum_variance_profile_loss",
                        zero,
                    )
                    + float(
                        self.quality_fidelity_configuration.get(
                            "condition_full_spectrum_group_variance",
                            {},
                        ).get(
                            "weight",
                            0.0,
                        )
                    )
                    * quality.get(
                        "full_spectrum_group_variance_loss",
                        zero,
                    )
                )

                legacy_quality_raw = (
                    quality["quality_fidelity_raw_loss"]
                    - independent_full_spectrum_raw
                )

                uncapped_legacy_quality = (
                    float(
                        self.quality_fidelity_configuration[
                            "total_weight"
                        ]
                    )
                    * legacy_quality_raw
                )

                legacy_quality_cap = (
                    ddpm_loss.detach()
                    * float(
                        self.quality_fidelity_configuration[
                            "maximum_total_ratio_to_ddpm"
                        ]
                    )
                )
                legacy_quality_scale = torch.clamp(
                    legacy_quality_cap
                    / uncapped_legacy_quality.detach()
                    .abs()
                    .clamp_min(1.0e-12),
                    max=1.0,
                )
                weighted_legacy_quality = (
                    uncapped_legacy_quality * legacy_quality_scale
                )

                uncapped_full_spectrum_quality = (
                    float(independent_budget["total_weight"])
                    * independent_full_spectrum_raw
                )
                full_spectrum_quality_cap = (
                    ddpm_loss.detach()
                    * float(
                        independent_budget[
                            "maximum_total_ratio_to_ddpm"
                        ]
                    )
                )
                full_spectrum_quality_scale = torch.clamp(
                    full_spectrum_quality_cap
                    / uncapped_full_spectrum_quality.detach()
                    .abs()
                    .clamp_min(1.0e-12),
                    max=1.0,
                )
                weighted_full_spectrum_quality = (
                    uncapped_full_spectrum_quality
                    * full_spectrum_quality_scale
                )

                uncapped_quality = (
                    uncapped_legacy_quality
                    + uncapped_full_spectrum_quality
                )
                weighted_quality = (
                    weighted_legacy_quality
                    + weighted_full_spectrum_quality
                )

            else:
                # Backward-compatible D4.3.2.11 behavior.
                uncapped_quality = (
                    float(
                        self.quality_fidelity_configuration[
                            "total_weight"
                        ]
                    )
                    * quality["quality_fidelity_raw_loss"]
                )
                quality_cap = ddpm_loss.detach() * float(
                    self.quality_fidelity_configuration[
                        "maximum_total_ratio_to_ddpm"
                    ]
                )
                quality_scale = torch.clamp(
                    quality_cap
                    / uncapped_quality.detach()
                    .abs()
                    .clamp_min(1.0e-12),
                    max=1.0,
                )
                weighted_quality = uncapped_quality * quality_scale

                uncapped_legacy_quality = uncapped_quality
                weighted_legacy_quality = weighted_quality
                legacy_quality_scale = quality_scale

        else:
            quality = {}
            uncapped_quality = zero
            weighted_quality = zero

        reduced = (
            ddpm_loss
            + weighted_high_noise_recovery
            + weighted_equalized_residual_tail
            + weighted_diversity
            + weighted_quality
        )
        timestep_fraction = t.to(dtype=torch.float32) / float(
            max(self.num_timesteps - 1, 1)
        )
        high_sampling = self.quality_fidelity_configuration.get(
            "high_noise_sampling", {}
        )
        high_threshold = float(
            high_sampling.get("minimum_timestep_fraction", 0.75)
        )
        self._latest_loss_components = {
            "total_loss": reduced.detach(),
            "ddpm_loss": ddpm_loss.detach(),
            "ddpm_uniform_loss": ddpm_uniform_loss.detach(),
            "mean_high_noise_weight": timestep_fidelity_weight.mean().detach(),
            "mean_training_timestep_fraction": timestep_fraction.mean().detach(),
            "high_noise_training_sample_fraction": (
                timestep_fraction >= high_threshold
            ).to(dtype=torch.float32).mean().detach(),
            "high_noise_recovery_raw_loss": (
                high_noise_recovery["raw_loss"].detach()
            ),
            "high_noise_recovery_uncapped_loss": (
                uncapped_high_noise_recovery.detach()
            ),
            "high_noise_recovery_loss": (
                weighted_high_noise_recovery.detach()
            ),
            "high_noise_recovery_scale": high_noise_recovery_scale.detach(),
            "high_noise_recovery_active_fraction": (
                high_noise_recovery["active_fraction"].detach()
            ),
            "high_noise_recovery_mean_gate": (
                high_noise_recovery["mean_gate"].detach()
            ),
            "high_noise_recovery_mean_abs_standardized_error": (
                high_noise_recovery[
                    "mean_abs_standardized_error"
                ].detach()
            ),
            "equalized_residual_tail_raw_loss": (
                equalized_residual_tail["raw_loss"].detach()
            ),
            "equalized_residual_tail_uncapped_loss": (
                uncapped_equalized_residual_tail.detach()
            ),
            "equalized_residual_tail_loss": (
                weighted_equalized_residual_tail.detach()
            ),
            "equalized_residual_tail_scale": (
                equalized_tail_scale.detach()
            ),
            "equalized_residual_tail_active_fraction": (
                equalized_residual_tail[
                    "active_fraction"
                ].detach()
            ),
            "equalized_residual_tail_mean_abs_excess": (
                equalized_residual_tail[
                    "mean_abs_excess"
                ].detach()
            ),
            "equalized_residual_tail_mean_gate": (
                equalized_residual_tail[
                    "mean_gate"
                ].detach()
            ),
            "equalized_residual_tail_overdispersion_gate_mean": (
                equalized_residual_tail[
                    "overdispersion_gate_mean"
                ].detach()
            ),
            "physics_loss": zero,
            "diversity_loss": weighted_diversity.detach(),
            "diversity_uncapped_loss": uncapped_diversity.detach(),
            "diversity_raw_loss": diversity.get(
                "diversity_raw_loss", zero
            ).detach(),
            "pairwise_distance_loss": diversity.get(
                "pairwise_distance_loss", zero
            ).detach(),
            "pairwise_correlation_loss": diversity.get(
                "pairwise_correlation_loss", zero
            ).detach(),
            "pointwise_variance_floor_loss": diversity.get(
                "pointwise_variance_floor_loss", zero
            ).detach(),
            "diversity_active_samples": diversity.get(
                "diversity_active_samples", zero
            ).detach(),
            "conditional_diversity_active_groups": diversity.get(
                "conditional_diversity_active_groups", zero
            ).detach(),
            "mean_diversity_timestep_weight": diversity.get(
                "mean_diversity_timestep_weight", zero
            ).detach(),
            "quality_fidelity_loss": weighted_quality.detach(),
            "quality_fidelity_uncapped_loss": uncapped_quality.detach(),
            "legacy_quality_loss": weighted_legacy_quality.detach(),
            "legacy_quality_uncapped_loss": (
                uncapped_legacy_quality.detach()
            ),
            "legacy_quality_scale": legacy_quality_scale.detach(),
            "independent_full_spectrum_raw_loss": (
                independent_full_spectrum_raw.detach()
            ),
            "independent_full_spectrum_uncapped_loss": (
                uncapped_full_spectrum_quality.detach()
            ),
            "independent_full_spectrum_loss": (
                weighted_full_spectrum_quality.detach()
            ),
            "independent_full_spectrum_scale": (
                full_spectrum_quality_scale.detach()
            ),
            "quality_fidelity_raw_loss": quality.get(
                "quality_fidelity_raw_loss", zero
            ).detach(),
            "first_derivative_fidelity_loss": quality.get(
                "first_derivative_fidelity_loss", zero
            ).detach(),
            "multiscale_shape_loss": quality.get(
                "multiscale_shape_loss", zero
            ).detach(),
            "condition_multiscale_variance_floor_loss": quality.get(
                "condition_multiscale_variance_floor_loss", zero
            ).detach(),
            "condition_derivative_variance_floor_loss": quality.get(
                "condition_derivative_variance_floor_loss", zero
            ).detach(),
            "condition_mean_fidelity_loss": quality.get(
                "condition_mean_fidelity_loss", zero
            ).detach(),
            "condition_envelope_loss": quality.get(
                "condition_envelope_loss", zero
            ).detach(),
            "full_spectrum_reconstruction_loss": quality.get(
                "full_spectrum_reconstruction_loss", zero
            ).detach(),
            "full_spectrum_tail_span_loss": quality.get(
                "full_spectrum_tail_span_loss", zero
            ).detach(),
            "full_spectrum_tail_active_groups": quality.get(
                "full_spectrum_tail_active_groups", zero
            ).detach(),

            "full_spectrum_variance_profile_loss": quality.get(
                "full_spectrum_variance_profile_loss",
                zero,
            ).detach(),

            "full_spectrum_variance_profile_active_fraction": quality.get(
                "full_spectrum_variance_profile_active_fraction",
                zero,
            ).detach(),

            "full_spectrum_variance_profile_mean_abs_z": quality.get(
                "full_spectrum_variance_profile_mean_abs_z",
                zero,
            ).detach(),

            "full_spectrum_variance_profile_violation_fraction": quality.get(
                "full_spectrum_variance_profile_violation_fraction",
                zero,
            ).detach(),

            "full_spectrum_group_variance_loss": quality.get(
                "full_spectrum_group_variance_loss",
                zero,
            ).detach(),

            "full_spectrum_group_variance_active_groups": quality.get(
                "full_spectrum_group_variance_active_groups",
                zero,
            ).detach(),

            "full_spectrum_group_variance_mean_abs_z": quality.get(
                "full_spectrum_group_variance_mean_abs_z",
                zero,
            ).detach(),

            "full_spectrum_group_variance_over_fraction": quality.get(
                "full_spectrum_group_variance_over_fraction",
                zero,
            ).detach(),

            "full_spectrum_group_variance_under_fraction": quality.get(
                "full_spectrum_group_variance_under_fraction",
                zero,
            ).detach(),

            "quality_fidelity_active_groups": quality.get(
                "quality_fidelity_active_groups", zero
            ).detach(),
        }
        return reduced

    def model_predictions(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        x_self_cond: torch.Tensor | None = None,
        clip_x_start: bool = False,
        rederive_pred_noise: bool = False,
        model_forward_kwargs: dict[str, Any] | None = None,
        valid_mask: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
        prior_conditioning: torch.Tensor | None = None,
        **kwargs,
    ):
        if self._sampling_valid_mask is not None and valid_mask is not None:
            raise RuntimeError("采样内部掩码不能与显式valid_mask同时提供。")
        if self._sampling_condition is not None and condition is not None:
            raise RuntimeError("采样内部条件不能与显式condition同时提供。")
        if (
            self._sampling_prior_conditioning is not None
            and prior_conditioning is not None
        ):
            raise RuntimeError(
                "采样内部先验条件不能与显式prior_conditioning同时提供。"
            )

        mask = (
            self._sampling_valid_mask
            if self._sampling_valid_mask is not None
            else valid_mask
        )
        active_condition = (
            self._sampling_condition
            if self._sampling_condition is not None
            else condition
        )
        active_prior = (
            self._sampling_prior_conditioning
            if self._sampling_prior_conditioning is not None
            else prior_conditioning
        )
        if mask is None and active_condition is None and active_prior is None:
            return super().model_predictions(
                x,
                t,
                x_self_cond,
                clip_x_start,
                rederive_pred_noise,
            )

        prepared = self._prepare_mask(mask, x)
        prepared_condition = self._prepare_condition(
            active_condition,
            x.shape[0],
            x.device,
            x.dtype,
        )
        prepared_prior = self._prepare_prior_conditioning(active_prior, x)
        if prepared_prior is not None:
            prepared_prior = prepared_prior * prepared
        if x_self_cond is not None:
            x_self_cond = x_self_cond * prepared

        forward_kwargs = dict(model_forward_kwargs or {})
        forward_kwargs.update(kwargs)
        forward_kwargs.pop("self_cond", None)
        model_output = self.model(
            x * prepared,
            t,
            valid_mask=prepared,
            condition=prepared_condition,
            prior_conditioning=prepared_prior,
            **forward_kwargs,
        )

        if self.objective == "pred_noise":
            pred_noise = model_output
            pred_x_start = self.predict_start_from_noise(
                x * prepared,
                t,
                pred_noise,
            )
            if clip_x_start:
                pred_x_start = pred_x_start.clamp(-1.0, 1.0)
            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(
                    x * prepared,
                    t,
                    pred_x_start,
                )
        elif self.objective == "pred_x0":
            pred_x_start = model_output
            if clip_x_start:
                pred_x_start = pred_x_start.clamp(-1.0, 1.0)
            pred_noise = self.predict_noise_from_start(
                x * prepared,
                t,
                pred_x_start,
            )
        elif self.objective == "pred_v":
            pred_x_start = self.predict_start_from_v(
                x * prepared,
                t,
                model_output,
            )
            if clip_x_start:
                pred_x_start = pred_x_start.clamp(-1.0, 1.0)
            pred_noise = self.predict_noise_from_start(
                x * prepared,
                t,
                pred_x_start,
            )
        else:
            raise ValueError(f"未知扩散预测目标：{self.objective}")

        return MaskedModelPrediction(
            pred_noise * prepared,
            pred_x_start * prepared,
        )

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 16,
        *args,
        valid_mask: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
        prior_conditioning: torch.Tensor | None = None,
        **kwargs,
    ):
        reference = torch.empty(
            (int(batch_size), self.channels, self.seq_length),
            device=self.betas.device,
        )
        mask = self._prepare_mask(valid_mask, reference)
        prepared_condition = self._prepare_condition(
            condition,
            int(batch_size),
            reference.device,
            reference.dtype,
        )
        prepared_prior = self._prepare_prior_conditioning(
            prior_conditioning, reference
        )
        if prepared_prior is not None:
            prepared_prior = prepared_prior * mask

        if (
            self._sampling_valid_mask is not None
            or self._sampling_condition is not None
            or self._sampling_prior_conditioning is not None
        ):
            raise RuntimeError("同一个扩散对象不能同时执行两次掩码采样。")

        self._sampling_valid_mask = mask
        self._sampling_condition = prepared_condition
        self._sampling_prior_conditioning = prepared_prior
        try:
            generated = super().sample(
                batch_size,
                *args,
                **kwargs,
            )
        finally:
            self._sampling_valid_mask = None
            self._sampling_condition = None
            self._sampling_prior_conditioning = None

        if isinstance(generated, tuple):
            primary = generated[0] * mask
            return (primary, *generated[1:])
        return generated * mask
