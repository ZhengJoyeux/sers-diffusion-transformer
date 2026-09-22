"""Generate spectra from a trained one-dimensional diffusion model."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.prior_residual import (
    PriorResidualTransformer,
)
from src.broad_local_residual import (
    BroadLocalResidualDecomposer,
)
from src.conditional_prior_residual import (
    ConditionalPriorResidualBank,
)
from src.feature_peak_residual_limiter import (
    FeaturePeakResidualLimiter,
)
from src.spectrum_length_adapter import (
    SpectrumLengthAdapter,
)
from src.sers_sampling_calibrator import (
    SersSamplingCalibrator,
)


def _condition_arrays(
    generated_spectra: np.ndarray,
    training_spectra: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    generated = np.asarray(generated_spectra, dtype=np.float64)
    training = np.asarray(training_spectra, dtype=np.float64)
    if generated.ndim != 2 or training.ndim != 2:
        raise ValueError("条件校准要求generated_spectra和training_spectra为[N,L]。")
    if generated.shape[0] == 0 or training.shape[0] < 4:
        raise ValueError("条件校准至少需要1条生成谱和4条同条件训练谱。")
    if generated.shape[1] != training.shape[1] or generated.shape[1] < 2:
        raise ValueError("条件校准的生成谱与训练谱Raman点数必须一致。")
    if not np.isfinite(generated).all() or not np.isfinite(training).all():
        raise ValueError("条件校准输入包含NaN或无穷值。")
    return generated, training


def _training_standardized_values(
    generated: np.ndarray,
    training: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = np.mean(training, axis=0)
    scale = np.std(training, axis=0, ddof=1)
    positive = scale[scale > 1.0e-12]
    scale_floor = max(
        float(np.percentile(positive, 10.0)) if positive.size else 1.0,
        1.0e-12,
    )
    scale = np.maximum(scale, scale_floor)
    return (
        (generated - center[np.newaxis, :]) / scale[np.newaxis, :],
        (training - center[np.newaxis, :]) / scale[np.newaxis, :],
        center,
        scale,
    )


def apply_condition_mean_fidelity_calibration(
    generated_spectra: np.ndarray,
    training_spectra: np.ndarray,
    *,
    configuration: dict,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """Correct significant condition-mean bias while preserving pairwise MSE."""

    generated, training = _condition_arrays(generated_spectra, training_spectra)
    if not isinstance(configuration, dict):
        raise ValueError("mean_fidelity_calibration配置必须是字典。")
    standard_error_multiplier = float(
        configuration.get("standard_error_multiplier", 1.96)
    )
    correction_strength = float(configuration.get("correction_strength", 0.35))
    maximum_iqr_fraction = float(
        configuration.get("maximum_correction_iqr_fraction", 0.25)
    )
    minimum_cap = float(
        configuration.get("minimum_maximum_correction_intensity", 1.0)
    )
    maximum_modified_fraction = float(
        configuration.get("maximum_modified_raman_fraction", 0.75)
    )
    if not 0.0 < standard_error_multiplier <= 4.0:
        raise ValueError("standard_error_multiplier必须位于(0,4]。")
    if not 0.0 < correction_strength <= 1.0:
        raise ValueError("correction_strength必须位于(0,1]。")
    if maximum_iqr_fraction < 0.0 or minimum_cap <= 0.0:
        raise ValueError("均值保真校准的修正上限配置无效。")
    if not 0.0 < maximum_modified_fraction <= 1.0:
        raise ValueError("maximum_modified_raman_fraction必须位于(0,1]。")

    training_mean = np.mean(training, axis=0)
    generated_mean = np.mean(generated, axis=0)
    training_std = np.std(training, axis=0, ddof=1)
    q25, q75 = np.quantile(training, [0.25, 0.75], axis=0)
    standard_error = training_std / np.sqrt(float(training.shape[0]))
    allowed_bias = standard_error_multiplier * standard_error
    bias = generated_mean - training_mean
    excess = np.maximum(np.abs(bias) - allowed_bias, 0.0)
    correction_cap = np.maximum(
        maximum_iqr_fraction * np.maximum(q75 - q25, 0.0), minimum_cap
    )
    correction = -np.sign(bias) * np.minimum(
        correction_strength * excess, correction_cap
    )
    active = np.abs(correction) > 0.0
    modified_fraction = float(np.mean(active))
    if modified_fraction > maximum_modified_fraction:
        raise RuntimeError(
            "条件均值显著偏移的Raman点比例超过安全上限："
            f"{modified_fraction:.6%} > {maximum_modified_fraction:.6%}。"
            "这说明生成分布存在广泛系统偏差，应优先重新训练。"
        )
    corrected = generated + correction[np.newaxis, :]
    diagnostics: dict[str, float | int | bool] = {
        "enabled": True,
        "modified_raman_count": int(np.sum(active)),
        "modified_raman_fraction": modified_fraction,
        "mean_bias_rmse_before": float(np.sqrt(np.mean(np.square(bias)))),
        "mean_bias_rmse_after": float(
            np.sqrt(
                np.mean(
                    np.square(np.mean(corrected, axis=0) - training_mean)
                )
            )
        ),
        "maximum_absolute_correction": float(np.max(np.abs(correction))),
        "pairwise_mse_is_translation_invariant": True,
    }
    return corrected.astype(np.float32, copy=False), diagnostics


def _mean_pairwise_mse_from_variance(spectra: np.ndarray) -> float:
    """Return the exact mean MSE across all unordered sample pairs."""

    if spectra.shape[0] < 2:
        return 0.0
    pointwise_variance = np.var(spectra, axis=0, ddof=1)
    return float(2.0 * np.mean(pointwise_variance))


def _pairwise_indices(
    number: int,
    *,
    maximum_pair_count: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return every small-batch pair or a reproducible large-batch sample."""

    total = number * (number - 1) // 2
    requested = min(int(maximum_pair_count), total)
    if requested <= 0:
        raise ValueError("maximum_pair_count必须大于0。")
    if requested == total:
        return np.triu_indices(number, k=1)

    random_generator = np.random.default_rng(int(random_seed))
    all_left, all_right = np.triu_indices(number, k=1)
    selected = random_generator.choice(total, size=requested, replace=False)
    return all_left[selected], all_right[selected]


def _pairwise_mse_values(
    spectra: np.ndarray,
    *,
    maximum_pair_count: int = 10000,
    random_seed: int = 2026,
) -> np.ndarray:
    """Return exact small-batch or sampled large-batch pairwise MSE values."""

    array = np.asarray(spectra, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2:
        raise ValueError("pairwise-MSE至少需要二维数组中的2条光谱。")
    left, right = _pairwise_indices(
        array.shape[0],
        maximum_pair_count=maximum_pair_count,
        random_seed=random_seed,
    )
    values = np.empty(left.size, dtype=np.float64)
    for start in range(0, left.size, 256):
        stop = min(start + 256, left.size)
        difference = array[left[start:stop]] - array[right[start:stop]]
        values[start:stop] = np.mean(np.square(difference), axis=1)
    return values


def _protect_pairwise_diversity_after_calibration(
    original: np.ndarray,
    calibrated: np.ndarray,
    training: np.ndarray,
    *,
    minimum_ratio: float,
    search_iterations: int,
    minimum_calibration_blend_factor: float = 0.0,
    pair_count: int = 10000,
    random_seed: int = 2026,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """Blend back only enough calibration to retain condition diversity.

    The floor is measured with the same median all-pairs MSE used by the
    evaluator.  When the original batch is already below the configured
    training ratio, calibration is allowed only if it does not reduce that
    batch's diversity further.
    """

    original = np.asarray(original, dtype=np.float64)
    calibrated = np.asarray(calibrated, dtype=np.float64)
    training = np.asarray(training, dtype=np.float64)
    if original.shape != calibrated.shape or original.ndim != 2:
        raise ValueError("多样性保护前后的生成谱形状必须一致且为二维。")
    if training.ndim != 2 or training.shape[1] != original.shape[1]:
        raise ValueError("多样性保护的training与生成谱Raman长度不一致。")
    if not 0.0 <= minimum_ratio <= 1.0:
        raise ValueError(
            "minimum_postcalibration_pairwise_mse_ratio必须位于[0,1]。"
        )
    if not 0.0 <= float(minimum_calibration_blend_factor) <= 1.0:
        raise ValueError(
            "minimum_calibration_blend_factor必须位于[0,1]。"
        )
    if not 4 <= int(search_iterations) <= 64:
        raise ValueError("diversity_guard_search_iterations必须位于[4,64]。")
    if int(pair_count) <= 0:
        raise ValueError("diversity_guard_pair_count必须大于0。")

    training_median = float(
        np.median(
            _pairwise_mse_values(
                training,
                maximum_pair_count=pair_count,
                random_seed=random_seed,
            )
        )
    )
    left, right = _pairwise_indices(
        original.shape[0],
        maximum_pair_count=pair_count,
        random_seed=random_seed,
    )
    constant = np.empty(left.size, dtype=np.float64)
    linear = np.empty(left.size, dtype=np.float64)
    quadratic = np.empty(left.size, dtype=np.float64)
    delta = calibrated - original
    for start in range(0, left.size, 256):
        stop = min(start + 256, left.size)
        base_difference = (
            original[left[start:stop]] - original[right[start:stop]]
        )
        delta_difference = (
            delta[left[start:stop]] - delta[right[start:stop]]
        )
        constant[start:stop] = np.mean(np.square(base_difference), axis=1)
        linear[start:stop] = np.mean(base_difference * delta_difference, axis=1)
        quadratic[start:stop] = np.mean(np.square(delta_difference), axis=1)
    original_median = float(np.median(constant))
    calibrated_median = float(np.median(constant + 2.0 * linear + quadratic))
    denominator = max(training_median, 1.0e-12)
    ratio_before = original_median / denominator
    ratio_unguarded = calibrated_median / denominator
    required_ratio = min(float(minimum_ratio), ratio_before)

    blend = 1.0
    activated = bool(
        minimum_ratio > 0.0 and ratio_unguarded < required_ratio - 1.0e-12
    )
    if activated:
        low = 0.0
        high = 1.0
        for _ in range(int(search_iterations)):
            middle = 0.5 * (low + high)
            pairwise = constant + 2.0 * middle * linear + middle * middle * quadratic
            ratio = float(np.median(pairwise)) / denominator
            if ratio >= required_ratio:
                low = middle
            else:
                high = middle
        blend = low

    unconstrained_blend = float(blend)
    tail_priority_limited_rollback = False

    if (
        activated
        and blend < float(minimum_calibration_blend_factor)
    ):
        # D4.3.2.12:
        # pairwise-MSE diversity只是软目标，不能为了恢复一个总体
        # diversity指标而几乎完全撤销train-supported QQ尾部修正。
        blend = float(minimum_calibration_blend_factor)
        tail_priority_limited_rollback = True

    protected = original + blend * delta
    protected_pairwise = (
        constant + 2.0 * blend * linear + blend * blend * quadratic
    )
    ratio_after = float(np.median(protected_pairwise)) / denominator

    diversity_floor_satisfied = bool(
        ratio_after >= required_ratio - 1.0e-9
    )

    # minimum_calibration_blend_factor=0保持旧版本行为。
    if (
        activated
        and not diversity_floor_satisfied
        and float(minimum_calibration_blend_factor) <= 0.0
    ):
        blend = 0.0
        protected = original.copy()
        ratio_after = ratio_before
        diversity_floor_satisfied = True
    diagnostics: dict[str, float | int | bool] = {
        "diversity_guard_enabled": bool(minimum_ratio > 0.0),
        "diversity_guard_activated": activated,
        "pairwise_mse_ratio_before": ratio_before,
        "pairwise_mse_ratio_unguarded": ratio_unguarded,
        "pairwise_mse_ratio_required": required_ratio,
        "pairwise_mse_ratio_after": ratio_after,
        "calibration_blend_factor": float(blend),
        "unconstrained_calibration_blend_factor": float(
            unconstrained_blend
        ),
        "minimum_calibration_blend_factor": float(
            minimum_calibration_blend_factor
        ),
        "diversity_floor_satisfied": bool(
            diversity_floor_satisfied
        ),
        "tail_priority_limited_rollback": bool(
            tail_priority_limited_rollback
        ),
        "diversity_guard_search_iterations": int(search_iterations),
        "diversity_guard_pair_count": int(left.size),
    }
    return protected, diagnostics


def apply_condition_pca_spread_calibration(
    generated_spectra: np.ndarray,
    training_spectra: np.ndarray,
    *,
    configuration: dict,
) -> tuple[np.ndarray, dict[str, float | int | bool | list[float]]]:
    """Restore only missing spread along train-supported PCA directions.

    The calibration is fitted independently for every condition from its 12
    training spectra.  It never uses validation/test spectra and never creates
    a direction outside the training PCA subspace.  Generated means are kept
    exactly unchanged; only centered component scores whose standard deviation
    falls below the configured floor are expanded.  Components that already
    have enough (or excessive) spread are left untouched.
    """

    generated, training = _condition_arrays(generated_spectra, training_spectra)
    if not isinstance(configuration, dict):
        raise ValueError("pca_spread_calibration配置必须是字典。")

    maximum_components = int(configuration.get("maximum_components", 6))
    explained_variance_ratio = float(
        configuration.get("explained_variance_ratio", 0.95)
    )
    minimum_component_ratio = float(
        configuration.get("minimum_component_std_ratio", 0.85)
    )
    maximum_scale_factor = float(
        configuration.get("maximum_scale_factor", 1.35)
    )
    minimum_pairwise_ratio = float(
        configuration.get("minimum_pairwise_mse_ratio", 0.0)
    )
    maximum_global_scale_factor = float(
        configuration.get("maximum_global_scale_factor", 1.15)
    )
    maximum_component_std_ratio_after_global = float(
        configuration.get(
            "maximum_component_std_ratio_after_global",
            1.0,
        )
    )
    pairwise_pair_count = int(
        configuration.get("pairwise_mse_pair_count", 10000)
    )
    pairwise_search_iterations = int(
        configuration.get("pairwise_mse_search_iterations", 24)
    )
    pairwise_random_seed = int(configuration.get("random_seed", 2026))
    minimum_component_variance_fraction = float(
        configuration.get("minimum_component_variance_fraction", 0.01)
    )
    maximum_correction_fraction = float(
        configuration.get("maximum_correction_range_fraction", 0.08)
    )
    minimum_generated_count = int(
        configuration.get("minimum_generated_count", 20)
    )
    epsilon = float(configuration.get("epsilon", 1.0e-12))

    if maximum_components < 1:
        raise ValueError("maximum_components至少为1。")
    if not 0.50 <= explained_variance_ratio <= 1.0:
        raise ValueError("explained_variance_ratio必须位于[0.50,1]。")
    if not 0.0 < minimum_component_ratio <= 1.0:
        raise ValueError("minimum_component_std_ratio必须位于(0,1]。")
    if not 1.0 <= maximum_scale_factor <= 2.0:
        raise ValueError("maximum_scale_factor必须位于[1,2]。")
    if not 0.0 <= minimum_pairwise_ratio <= 1.0:
        raise ValueError("minimum_pairwise_mse_ratio必须位于[0,1]。")
    if not 1.0 <= maximum_global_scale_factor <= 2.0:
        raise ValueError("maximum_global_scale_factor必须位于[1,2]。")
    if not (
        minimum_component_ratio
        <= maximum_component_std_ratio_after_global
        <= 1.25
    ):
        raise ValueError(
            "maximum_component_std_ratio_after_global必须位于"
            "[minimum_component_std_ratio,1.25]。"
        )
    if pairwise_pair_count <= 0:
        raise ValueError("pairwise_mse_pair_count必须大于0。")
    if not 4 <= pairwise_search_iterations <= 64:
        raise ValueError("pairwise_mse_search_iterations必须位于[4,64]。")
    if not 0.0 <= minimum_component_variance_fraction <= 0.25:
        raise ValueError("minimum_component_variance_fraction必须位于[0,0.25]。")
    if not 0.0 < maximum_correction_fraction <= 0.25:
        raise ValueError("maximum_correction_range_fraction必须位于(0,0.25]。")
    if minimum_generated_count < 4 or epsilon <= 0.0:
        raise ValueError("PCA spread最少生成数或epsilon无效。")
    if generated.shape[0] < minimum_generated_count:
        raise ValueError(
            "PCA spread校准要求至少"
            f"{minimum_generated_count}条生成谱，当前只有{generated.shape[0]}条。"
        )

    training_center = np.mean(training, axis=0)
    training_centered = training - training_center[np.newaxis, :]
    _, singular_values, right_vectors = np.linalg.svd(
        training_centered, full_matrices=False
    )
    component_variances = np.square(singular_values) / max(
        training.shape[0] - 1, 1
    )
    total_variance = float(np.sum(component_variances))
    if not np.isfinite(total_variance) or total_variance <= epsilon:
        raise ValueError("同条件training光谱没有可用于PCA spread的方差。")
    variance_fractions = component_variances / total_variance
    cumulative = np.cumsum(variance_fractions)
    required = int(np.searchsorted(cumulative, explained_variance_ratio) + 1)
    component_count = min(
        maximum_components,
        training.shape[0] - 1,
        right_vectors.shape[0],
        required,
    )
    components = right_vectors[:component_count]
    training_scores = training_centered @ components.T

    generated_mean = np.mean(generated, axis=0)
    generated_centered = generated - generated_mean[np.newaxis, :]
    generated_scores = generated_centered @ components.T
    training_std = np.std(training_scores, axis=0, ddof=1)
    generated_std = np.std(generated_scores, axis=0, ddof=1)
    component_ratio_before = generated_std / np.maximum(training_std, epsilon)
    eligible = np.logical_and(
        variance_fractions[:component_count]
        >= minimum_component_variance_fraction,
        training_std > epsilon,
    )
    scale_factors = np.ones(component_count, dtype=np.float64)
    deficient = np.logical_and(
        eligible, component_ratio_before < minimum_component_ratio
    )
    scale_factors[deficient] = np.minimum(
        minimum_component_ratio
        / np.maximum(component_ratio_before[deficient], epsilon),
        maximum_scale_factor,
    )

    score_delta = generated_scores * (scale_factors - 1.0)[np.newaxis, :]
    correction = score_delta @ components
    pairwise_training = float(np.median(_pairwise_mse_values(
        training,
        maximum_pair_count=pairwise_pair_count,
        random_seed=pairwise_random_seed,
    )))
    provisional = generated + correction
    provisional += (
        generated_mean - np.mean(provisional, axis=0)
    )[np.newaxis, :]
    provisional_pairwise = float(np.median(_pairwise_mse_values(
        provisional,
        maximum_pair_count=pairwise_pair_count,
        random_seed=pairwise_random_seed,
    )))
    provisional_ratio = provisional_pairwise / max(pairwise_training, epsilon)

    provisional_centered = (
        provisional - generated_mean[np.newaxis, :]
    )
    provisional_scores = provisional_centered @ components.T
    provisional_std = np.std(
        provisional_scores,
        axis=0,
        ddof=1,
    )
    provisional_component_ratio = (
        provisional_std / np.maximum(training_std, epsilon)
    )

    global_scale_factor = 1.0
    effective_maximum_global_scale_factor = 1.0

    global_spread_activated = bool(
        minimum_pairwise_ratio > 0.0
        and np.any(deficient)
        and provisional_ratio < minimum_pairwise_ratio
    )

    if global_spread_activated:
        # D4.3.2.12:
        # 第一阶段已经找出真正spread不足的PC。
        # 第二阶段global spread只允许这些deficient PC继续扩展，
        # 不能再把所有eligible PC一起放大。
        supported_direction = (
            provisional_scores
            * deficient.astype(np.float64)[np.newaxis, :]
        ) @ components

        deficient_ratios = provisional_component_ratio[deficient]

        component_safe_scale = float(
            np.min(
                maximum_component_std_ratio_after_global
                / np.maximum(deficient_ratios, epsilon)
            )
        )

        effective_maximum_global_scale_factor = max(
            1.0,
            min(
                maximum_global_scale_factor,
                component_safe_scale,
            ),
        )

        left, right = _pairwise_indices(
            provisional.shape[0],
            maximum_pair_count=pairwise_pair_count,
            random_seed=pairwise_random_seed,
        )
        constant = np.empty(left.size, dtype=np.float64)
        linear = np.empty(left.size, dtype=np.float64)
        quadratic = np.empty(left.size, dtype=np.float64)
        for start in range(0, left.size, 256):
            stop = min(start + 256, left.size)
            base_difference = (
                provisional[left[start:stop]] - provisional[right[start:stop]]
            )
            direction_difference = (
                supported_direction[left[start:stop]]
                - supported_direction[right[start:stop]]
            )
            constant[start:stop] = np.mean(
                np.square(base_difference), axis=1
            )
            linear[start:stop] = np.mean(
                base_difference * direction_difference, axis=1
            )
            quadratic[start:stop] = np.mean(
                np.square(direction_difference), axis=1
            )

        def ratio_at(scale_factor: float) -> float:
            alpha = float(scale_factor) - 1.0
            value = float(np.median(
                constant + 2.0 * alpha * linear + alpha * alpha * quadratic
            ))
            return value / max(pairwise_training, epsilon)

        if (
            ratio_at(effective_maximum_global_scale_factor)
            < minimum_pairwise_ratio
        ):
            global_scale_factor = effective_maximum_global_scale_factor
        else:
            low = 1.0
            high = effective_maximum_global_scale_factor

            for _ in range(pairwise_search_iterations):
                middle = 0.5 * (low + high)

                if ratio_at(middle) >= minimum_pairwise_ratio:
                    high = middle
                else:
                    low = middle

            global_scale_factor = high
        correction += (
            (global_scale_factor - 1.0) * supported_direction
        )
    training_range = max(
        float(np.percentile(training, 95.0) - np.percentile(training, 5.0)),
        epsilon,
    )
    correction_rmse_before_cap = float(
        np.sqrt(np.mean(np.square(correction)))
    )
    maximum_correction_rmse = maximum_correction_fraction * training_range
    safety_scale = 1.0
    if correction_rmse_before_cap > maximum_correction_rmse:
        safety_scale = maximum_correction_rmse / max(
            correction_rmse_before_cap, epsilon
        )
        correction *= safety_scale
    corrected = generated + correction
    # Numerical round-off must not turn a spread-only operation into a mean
    # calibration.  This also makes the invariant directly testable.
    corrected += (
        generated_mean - np.mean(corrected, axis=0)
    )[np.newaxis, :]

    corrected_scores = (corrected - generated_mean[np.newaxis, :]) @ components.T
    corrected_std = np.std(corrected_scores, axis=0, ddof=1)
    component_ratio_after = corrected_std / np.maximum(training_std, epsilon)
    pairwise_before = float(np.median(_pairwise_mse_values(
        generated,
        maximum_pair_count=pairwise_pair_count,
        random_seed=pairwise_random_seed,
    )))
    pairwise_after = float(np.median(_pairwise_mse_values(
        corrected,
        maximum_pair_count=pairwise_pair_count,
        random_seed=pairwise_random_seed,
    )))
    diagnostics: dict[str, float | int | bool | list[float]] = {
        "enabled": True,
        "training_only": True,
        "preserves_generated_mean": True,
        "component_count": int(component_count),
        "eligible_component_count": int(np.sum(eligible)),
        "expanded_component_count": int(np.sum(deficient)),
        "component_std_ratio_before": component_ratio_before.tolist(),
        "component_std_ratio_after": component_ratio_after.tolist(),
        "component_scale_factors": scale_factors.tolist(),
        "component_std_ratio_median_before": float(
            np.median(component_ratio_before[eligible]) if np.any(eligible) else 1.0
        ),
        "component_std_ratio_median_after": float(
            np.median(component_ratio_after[eligible]) if np.any(eligible) else 1.0
        ),
        # Keep the legacy field names for old manifest readers.  From 2.9
        # onward their values use the evaluator's median all-pairs statistic.
        "mean_pairwise_mse_ratio_before": pairwise_before
        / max(pairwise_training, epsilon),
        "mean_pairwise_mse_ratio_after": pairwise_after
        / max(pairwise_training, epsilon),
        "global_pairwise_spread_activated": global_spread_activated,
        "global_pairwise_scale_factor": float(global_scale_factor),
        "effective_maximum_global_scale_factor": float(
            effective_maximum_global_scale_factor
        ),
        "maximum_component_std_ratio_after_global": float(
            maximum_component_std_ratio_after_global
        ),
        "global_pairwise_target_met": bool(
            minimum_pairwise_ratio <= 0.0
            or (
                pairwise_after / max(pairwise_training, epsilon)
                >= minimum_pairwise_ratio - 1.0e-9
            )
        ),
        "pairwise_mse_statistic": "median_all_pairs_or_reproducible_sample",
        "median_pairwise_mse_ratio_before": pairwise_before
        / max(pairwise_training, epsilon),
        "median_pairwise_mse_ratio_after": pairwise_after
        / max(pairwise_training, epsilon),
        "pairwise_mse_pair_count": int(pairwise_pair_count),
        "minimum_pairwise_mse_ratio": float(minimum_pairwise_ratio),
        "correction_rmse": float(np.sqrt(np.mean(np.square(corrected - generated)))),
        "correction_rmse_before_safety_cap": correction_rmse_before_cap,
        "safety_scale": float(safety_scale),
        "maximum_scale_factor_used": float(np.max(scale_factors)),
    }
    return corrected.astype(np.float32, copy=False), diagnostics


def resolve_condition_tail_calibration_configuration(
    configuration: dict,
    *,
    point_count: int,
    condition_name: str,
) -> tuple[dict, dict[str, float | int | str]]:
    """Resolve axis/analyte calibration multipliers without data leakage."""

    if not isinstance(configuration, dict):
        raise ValueError("tail_calibration配置必须是字典。")
    if int(point_count) < 2:
        raise ValueError("point_count至少为2。")
    axis_profiles = configuration.get("axis_profiles", {}) or {}
    analyte_profiles = configuration.get("analyte_count_profiles", {}) or {}
    if not isinstance(axis_profiles, dict) or not isinstance(
        analyte_profiles, dict
    ):
        raise ValueError("tail calibration分层配置必须是字典。")

    analytes = {"DEL", "CHL", "TEB"}
    analyte_count = sum(
        token.split("-", 1)[0].upper() in analytes
        for token in str(condition_name).split("_")
    )
    if analyte_count not in {1, 2, 3}:
        raise ValueError(f"无法从条件名解析1--3种农药：{condition_name}")

    effective = {
        key: value
        for key, value in configuration.items()
        if key not in {"axis_profiles", "analyte_count_profiles"}
    }
    correction_multiplier = 1.0
    maximum_delta_multiplier = 1.0
    selected_profiles: list[str] = []
    for label, profile in (
        (f"axis:{int(point_count)}", axis_profiles.get(str(int(point_count)))),
        (f"analytes:{analyte_count}", analyte_profiles.get(str(analyte_count))),
    ):
        if profile is None:
            continue
        if not isinstance(profile, dict):
            raise ValueError(f"tail calibration profile {label}必须是字典。")
        selected_profiles.append(label)
        profile_values = dict(profile)
        correction_multiplier *= float(
            profile_values.pop("correction_strength_multiplier", 1.0)
        )
        maximum_delta_multiplier *= float(
            profile_values.pop(
                "maximum_absolute_correction_z_multiplier", 1.0
            )
        )
        effective.update(profile_values)

    if not 0.25 <= correction_multiplier <= 2.0:
        raise ValueError("累计correction_strength_multiplier必须位于[0.25,2]。")
    if not 0.25 <= maximum_delta_multiplier <= 2.0:
        raise ValueError(
            "累计maximum_absolute_correction_z_multiplier必须位于[0.25,2]。"
        )
    effective["correction_strength"] = min(
        1.0,
        float(effective.get("correction_strength", 0.60))
        * correction_multiplier,
    )
    effective["maximum_absolute_correction_z"] = (
        float(effective.get("maximum_absolute_correction_z", 0.40))
        * maximum_delta_multiplier
    )
    diagnostics: dict[str, float | int | str] = {
        "point_count": int(point_count),
        "analyte_count": int(analyte_count),
        "selected_profiles": ",".join(selected_profiles) or "base",
        "effective_correction_strength": float(
            effective["correction_strength"]
        ),
        "effective_maximum_absolute_correction_z": float(
            effective["maximum_absolute_correction_z"]
        ),
    }
    return effective, diagnostics


def apply_condition_oracle_tail_calibration(
    generated_spectra: np.ndarray,
    training_spectra: np.ndarray,
    *,
    configuration: dict,
) -> tuple[
    np.ndarray,
    dict[str, float | int | bool | list[float] | str],
]:
    """Compress only QQ tails exceeding train-only real-vs-real uncertainty.

    The middle distribution is unchanged.  The threshold is not a hand-picked
    1.0 line: repeated 4-vs-8 splits of the 12 training spectra provide the
    condition-specific oracle upper bound expected from small-sample variation.
    """

    generated, training = _condition_arrays(generated_spectra, training_spectra)
    if not isinstance(configuration, dict):
        raise ValueError("tail_calibration配置必须是字典。")

    strategy = str(configuration.get("strategy", "severe_tail_compression"))
    if strategy == "oracle_piecewise_quantile":
        return _apply_condition_oracle_piecewise_quantile_calibration(
            generated,
            training,
            configuration=configuration,
        )
    if strategy != "severe_tail_compression":
        raise ValueError(
            "tail_calibration.strategy必须是severe_tail_compression或"
            "oracle_piecewise_quantile。"
        )

    lower_probability = float(configuration.get("lower_probability", 0.01))
    upper_probability = float(configuration.get("upper_probability", 0.99))
    lower_anchor = float(configuration.get("lower_anchor_probability", 0.05))
    upper_anchor = float(configuration.get("upper_anchor_probability", 0.95))
    reference_count = int(configuration.get("oracle_reference_count", 4))
    repeats = int(configuration.get("oracle_bootstrap_repeats", 256))
    confidence = float(configuration.get("oracle_confidence", 0.975))
    minimum_allowed = float(
        configuration.get("minimum_allowed_tail_ratio", 1.15)
    )
    minimum_trigger = float(
        configuration.get("minimum_trigger_tail_ratio", 1.50)
    )
    minimum_compression_factor = float(
        configuration.get("minimum_compression_factor", 0.25)
    )
    maximum_modified_fraction = float(
        configuration.get("maximum_modified_fraction", 0.10)
    )
    random_seed = int(configuration.get("random_seed", 2026))
    if not (
        0.0 < lower_probability < lower_anchor < 0.5
        and 0.5 < upper_anchor < upper_probability < 1.0
    ):
        raise ValueError("tail_calibration的分位数顺序无效。")
    if not 2 <= reference_count < training.shape[0] - 1:
        raise ValueError("oracle_reference_count必须小于训练谱数且至少为2。")
    if repeats < 32 or not 0.90 <= confidence < 1.0:
        raise ValueError("oracle_bootstrap_repeats至少32且confidence位于[0.90,1)。")
    if not 1.0 <= minimum_allowed <= 2.0:
        raise ValueError("minimum_allowed_tail_ratio必须位于[1,2]。")
    if not minimum_allowed <= minimum_trigger <= 3.0:
        raise ValueError(
            "minimum_trigger_tail_ratio必须不小于允许尾宽且不大于3。"
        )
    if not 0.0 < minimum_compression_factor <= 1.0:
        raise ValueError("minimum_compression_factor必须位于(0,1]。")
    if not 0.0 < maximum_modified_fraction <= 0.20:
        raise ValueError("maximum_modified_fraction必须位于(0,0.20]。")

    generated_z, training_z, center, scale = _training_standardized_values(
        generated, training
    )
    training_flat = training_z.reshape(-1)
    generated_flat = generated_z.reshape(-1)
    probabilities = np.asarray(
        [lower_probability, lower_anchor, 0.5, upper_anchor, upper_probability],
        dtype=np.float64,
    )
    train_q = np.quantile(training_flat, probabilities)
    generated_q = np.quantile(generated_flat, probabilities)
    training_lower_span = max(float(train_q[2] - train_q[0]), 1.0e-12)
    training_upper_span = max(float(train_q[4] - train_q[2]), 1.0e-12)

    random_generator = np.random.default_rng(random_seed)
    oracle_lower: list[float] = []
    oracle_upper: list[float] = []
    all_indices = np.arange(training.shape[0], dtype=np.int64)
    for _ in range(repeats):
        shuffled = random_generator.permutation(all_indices)
        reference_indices = shuffled[:reference_count]
        population_indices = shuffled[reference_count:]
        reference_q = np.quantile(
            training_z[reference_indices].reshape(-1), probabilities[[0, 2, 4]]
        )
        population_q = np.quantile(
            training_z[population_indices].reshape(-1), probabilities[[0, 2, 4]]
        )
        reference_lower = max(float(reference_q[1] - reference_q[0]), 1.0e-12)
        reference_upper = max(float(reference_q[2] - reference_q[1]), 1.0e-12)
        oracle_lower.append(float((population_q[1] - population_q[0]) / reference_lower))
        oracle_upper.append(float((population_q[2] - population_q[1]) / reference_upper))

    allowed_lower = max(
        minimum_allowed, float(np.quantile(oracle_lower, confidence))
    )
    allowed_upper = max(
        minimum_allowed, float(np.quantile(oracle_upper, confidence))
    )
    lower_ratio_before = float(
        (generated_q[2] - generated_q[0]) / training_lower_span
    )
    upper_ratio_before = float(
        (generated_q[4] - generated_q[2]) / training_upper_span
    )

    lower_factor = 1.0
    lower_trigger = max(allowed_lower, minimum_trigger)
    upper_trigger = max(allowed_upper, minimum_trigger)
    if lower_ratio_before > lower_trigger:
        target_lower = generated_q[2] - allowed_lower * training_lower_span
        denominator = max(float(generated_q[1] - generated_q[0]), 1.0e-12)
        lower_factor = float(
            np.clip(
                (generated_q[1] - target_lower) / denominator,
                minimum_compression_factor,
                1.0,
            )
        )
    upper_factor = 1.0
    if upper_ratio_before > upper_trigger:
        target_upper = generated_q[2] + allowed_upper * training_upper_span
        denominator = max(float(generated_q[4] - generated_q[3]), 1.0e-12)
        upper_factor = float(
            np.clip(
                (target_upper - generated_q[3]) / denominator,
                minimum_compression_factor,
                1.0,
            )
        )

    corrected_z = generated_z.copy()
    lower_mask = generated_z < generated_q[1]
    upper_mask = generated_z > generated_q[3]
    if lower_factor < 1.0:
        corrected_z[lower_mask] = generated_q[1] - lower_factor * (
            generated_q[1] - generated_z[lower_mask]
        )
    else:
        lower_mask.fill(False)
    if upper_factor < 1.0:
        corrected_z[upper_mask] = generated_q[3] + upper_factor * (
            generated_z[upper_mask] - generated_q[3]
        )
    else:
        upper_mask.fill(False)
    modified = np.logical_or(lower_mask, upper_mask)
    modified_fraction = float(np.mean(modified))
    if modified_fraction > maximum_modified_fraction + 1.0e-12:
        raise RuntimeError(
            "QQ尾部校准修改比例超过安全上限："
            f"{modified_fraction:.6%} > {maximum_modified_fraction:.6%}。"
        )
    corrected = center[np.newaxis, :] + scale[np.newaxis, :] * corrected_z
    corrected_q = np.quantile(corrected_z.reshape(-1), probabilities)
    diagnostics: dict[str, float | int | bool] = {
        "enabled": True,
        "oracle_bootstrap_repeats": repeats,
        "oracle_lower_tail_ratio_upper": allowed_lower,
        "oracle_upper_tail_ratio_upper": allowed_upper,
        "lower_tail_trigger_ratio": lower_trigger,
        "upper_tail_trigger_ratio": upper_trigger,
        "lower_tail_ratio_before": lower_ratio_before,
        "upper_tail_ratio_before": upper_ratio_before,
        "lower_tail_ratio_after": float(
            (corrected_q[2] - corrected_q[0]) / training_lower_span
        ),
        "upper_tail_ratio_after": float(
            (corrected_q[4] - corrected_q[2]) / training_upper_span
        ),
        "lower_compression_factor": lower_factor,
        "upper_compression_factor": upper_factor,
        "lower_tail_calibration_activated": bool(lower_factor < 1.0),
        "upper_tail_calibration_activated": bool(upper_factor < 1.0),
        "modified_point_fraction": modified_fraction,
        "modified_spectrum_count": int(np.sum(np.any(modified, axis=1))),
    }
    return corrected.astype(np.float32, copy=False), diagnostics


def _apply_condition_oracle_piecewise_quantile_calibration(
    generated: np.ndarray,
    training: np.ndarray,
    *,
    configuration: dict,
) -> tuple[np.ndarray, dict[str, float | int | bool | list[float] | str]]:
    """Correct significant QQ curvature with a monotone train-only mapping.

    Four spans (1--10, 10--50, 50--90 and 90--99 percent) are handled
    independently.  A span is changed only when its generated/training ratio
    lies outside the real-vs-real bootstrap interval.  The mapping is
    monotone, preserves the generated median and is capped in standardized
    units.  The pointwise training envelope guard still runs afterwards.
    """

    probabilities = np.asarray(
        configuration.get(
            "piecewise_probabilities", [0.01, 0.10, 0.50, 0.90, 0.99]
        ),
        dtype=np.float64,
    )
    reference_count = int(configuration.get("oracle_reference_count", 4))
    repeats = int(configuration.get("oracle_bootstrap_repeats", 256))
    confidence = float(configuration.get("oracle_confidence", 0.975))
    minimum_ratio = float(configuration.get("minimum_segment_span_ratio", 0.90))
    maximum_ratio = float(configuration.get("maximum_segment_span_ratio", 1.10))
    correction_strength = float(configuration.get("correction_strength", 0.60))
    maximum_correction_z = float(
        configuration.get("maximum_absolute_correction_z", 0.40)
    )
    maximum_modified_fraction = float(
        configuration.get("maximum_modified_fraction", 1.0)
    )
    minimum_pairwise_ratio = float(
        configuration.get(
            "minimum_postcalibration_pairwise_mse_ratio", 0.0
        )
    )
    minimum_calibration_blend_factor = float(
        configuration.get(
            "minimum_calibration_blend_factor",
            0.0,
        )
    )
    diversity_search_iterations = int(
        configuration.get("diversity_guard_search_iterations", 20)
    )
    diversity_pair_count = int(
        configuration.get("diversity_guard_pair_count", 10000)
    )
    random_seed = int(configuration.get("random_seed", 2026))
    epsilon = float(configuration.get("epsilon", 1.0e-12))

    if probabilities.shape != (5,) or not np.all(np.diff(probabilities) > 0.0):
        raise ValueError("piecewise_probabilities必须是5个严格递增分位数。")
    if not (
        0.0 < probabilities[0]
        < probabilities[1]
        < probabilities[2]
        < probabilities[3]
        < probabilities[4]
        < 1.0
    ):
        raise ValueError("piecewise_probabilities必须位于(0,1)。")
    if not 2 <= reference_count < training.shape[0] - 1:
        raise ValueError("oracle_reference_count必须小于训练谱数且至少为2。")
    if repeats < 32 or not 0.90 <= confidence < 1.0:
        raise ValueError("oracle_bootstrap_repeats至少32且confidence位于[0.90,1)。")
    if not 0.50 <= minimum_ratio <= 1.0:
        raise ValueError("minimum_segment_span_ratio必须位于[0.50,1]。")
    if not 1.0 <= maximum_ratio <= 1.50 or minimum_ratio >= maximum_ratio:
        raise ValueError("maximum_segment_span_ratio必须位于[1,1.50]且大于下界。")
    if not 0.0 < correction_strength <= 1.0:
        raise ValueError("correction_strength必须位于(0,1]。")
    if maximum_correction_z <= 0.0:
        raise ValueError("maximum_absolute_correction_z必须为正数。")
    if not 0.0 < maximum_modified_fraction <= 1.0 or epsilon <= 0.0:
        raise ValueError("maximum_modified_fraction或epsilon无效。")
    if not 0.0 <= minimum_pairwise_ratio <= 1.0:
        raise ValueError(
            "minimum_postcalibration_pairwise_mse_ratio必须位于[0,1]。"
        )
    if not 4 <= diversity_search_iterations <= 64:
        raise ValueError(
            "diversity_guard_search_iterations必须位于[4,64]。"
        )
    if diversity_pair_count <= 0:
        raise ValueError("diversity_guard_pair_count必须大于0。")

    generated_z, training_z, center, scale = _training_standardized_values(
        generated, training
    )
    training_q = np.quantile(training_z.reshape(-1), probabilities)
    generated_q = np.quantile(generated_z.reshape(-1), probabilities)
    training_spans = np.maximum(np.diff(training_q), epsilon)
    generated_spans = np.maximum(np.diff(generated_q), epsilon)
    ratios_before = generated_spans / training_spans

    random_generator = np.random.default_rng(random_seed)
    indices = np.arange(training.shape[0], dtype=np.int64)
    oracle_ratios: list[np.ndarray] = []
    cache: dict[tuple[int, ...], np.ndarray] = {}
    for _ in range(repeats):
        shuffled = random_generator.permutation(indices)
        reference_indices = tuple(
            sorted(int(value) for value in shuffled[:reference_count])
        )
        cached = cache.get(reference_indices)
        if cached is not None:
            oracle_ratios.append(cached)
            continue
        reference_array = np.asarray(reference_indices, dtype=np.int64)
        population_array = np.asarray(
            [value for value in indices if int(value) not in reference_indices],
            dtype=np.int64,
        )
        reference_q = np.quantile(
            training_z[reference_array].reshape(-1), probabilities
        )
        population_q = np.quantile(
            training_z[population_array].reshape(-1), probabilities
        )
        ratio = np.diff(population_q) / np.maximum(
            np.diff(reference_q), epsilon
        )
        cache[reference_indices] = ratio
        oracle_ratios.append(ratio)

    oracle_array = np.stack(oracle_ratios, axis=0)
    oracle_lower = np.quantile(oracle_array, 1.0 - confidence, axis=0)
    oracle_upper = np.quantile(oracle_array, confidence, axis=0)
    outside_oracle = np.logical_or(
        ratios_before < oracle_lower,
        ratios_before > oracle_upper,
    )
    outside_target_band = np.logical_or(
        ratios_before < minimum_ratio,
        ratios_before > maximum_ratio,
    )
    significant = np.logical_and(outside_oracle, outside_target_band)
    desired_ratios = np.clip(ratios_before, minimum_ratio, maximum_ratio)
    desired_ratios = np.where(significant, desired_ratios, ratios_before)
    target_spans = generated_spans + correction_strength * (
        desired_ratios * training_spans - generated_spans
    )

    target_q = np.empty_like(generated_q)
    target_q[2] = generated_q[2]
    target_q[1] = target_q[2] - target_spans[1]
    target_q[0] = target_q[1] - target_spans[0]
    target_q[3] = target_q[2] + target_spans[2]
    target_q[4] = target_q[3] + target_spans[3]

    flat = generated_z.reshape(-1)
    if np.any(significant):
        mapped = np.interp(flat, generated_q, target_q)
        left = flat < generated_q[0]
        right = flat > generated_q[-1]
        left_slope = target_spans[0] / generated_spans[0]
        right_slope = target_spans[-1] / generated_spans[-1]
        mapped[left] = target_q[0] + left_slope * (flat[left] - generated_q[0])
        mapped[right] = target_q[-1] + right_slope * (
            flat[right] - generated_q[-1]
        )
        delta = np.clip(
            mapped - flat, -maximum_correction_z, maximum_correction_z
        )
    else:
        delta = np.zeros_like(flat)
    corrected_z = (flat + delta).reshape(generated_z.shape)
    unguarded = center[np.newaxis, :] + scale[np.newaxis, :] * corrected_z
    corrected, diversity_guard = _protect_pairwise_diversity_after_calibration(
        generated,
        unguarded,
        training,
        minimum_ratio=minimum_pairwise_ratio,
        search_iterations=diversity_search_iterations,
        minimum_calibration_blend_factor=(
            minimum_calibration_blend_factor
        ),
        pair_count=diversity_pair_count,
        random_seed=random_seed,
    )
    corrected_z = (corrected - center[np.newaxis, :]) / scale[np.newaxis, :]
    delta = (corrected_z - generated_z).reshape(-1)
    modified = np.abs(delta) > 1.0e-12
    modified_fraction = float(np.mean(modified))
    if modified_fraction > maximum_modified_fraction + 1.0e-12:
        raise RuntimeError(
            "分段QQ校准修改比例超过安全上限："
            f"{modified_fraction:.6%} > {maximum_modified_fraction:.6%}。"
        )

    corrected_q = np.quantile(corrected_z.reshape(-1), probabilities)
    ratios_after = np.diff(corrected_q) / training_spans
    lower_before = float(
        (generated_q[2] - generated_q[0])
        / max(float(training_q[2] - training_q[0]), epsilon)
    )
    upper_before = float(
        (generated_q[4] - generated_q[2])
        / max(float(training_q[4] - training_q[2]), epsilon)
    )
    lower_after = float(
        (corrected_q[2] - corrected_q[0])
        / max(float(training_q[2] - training_q[0]), epsilon)
    )
    upper_after = float(
        (corrected_q[4] - corrected_q[2])
        / max(float(training_q[4] - training_q[2]), epsilon)
    )
    lower_active = bool(np.any(significant[:2]))
    upper_active = bool(np.any(significant[2:]))
    diagnostics: dict[str, float | int | bool | list[float] | str] = {
        "enabled": True,
        "strategy": "oracle_piecewise_quantile",
        "training_only": True,
        "preserves_generated_median": True,
        "oracle_bootstrap_repeats": repeats,
        "piecewise_probabilities": probabilities.tolist(),
        "segment_span_ratio_before": ratios_before.tolist(),
        "segment_span_ratio_after": ratios_after.tolist(),
        "segment_oracle_lower": oracle_lower.tolist(),
        "segment_oracle_upper": oracle_upper.tolist(),
        "segment_calibration_activated": significant.tolist(),
        "active_segment_count": int(np.sum(significant)),
        "lower_tail_ratio_before": lower_before,
        "upper_tail_ratio_before": upper_before,
        "lower_tail_ratio_after": lower_after,
        "upper_tail_ratio_after": upper_after,
        "lower_compression_factor": min(lower_after / max(lower_before, epsilon), 1.0),
        "upper_compression_factor": min(upper_after / max(upper_before, epsilon), 1.0),
        "lower_tail_calibration_activated": lower_active,
        "upper_tail_calibration_activated": upper_active,
        "lower_tail_expansion_activated": bool(
            np.any(np.logical_and(significant[:2], ratios_after[:2] > ratios_before[:2]))
        ),
        "upper_tail_expansion_activated": bool(
            np.any(np.logical_and(significant[2:], ratios_after[2:] > ratios_before[2:]))
        ),
        "modified_point_fraction": modified_fraction,
        "modified_spectrum_count": int(
            np.sum(np.any(modified.reshape(generated_z.shape), axis=1))
        ),
        "maximum_absolute_correction_z_used": float(np.max(np.abs(delta))),
        **diversity_guard,
    }
    return corrected.astype(np.float32, copy=False), diagnostics


def _apply_condition_local_residual_negative_valley_guard(
    generated_spectra: np.ndarray,
    training_spectra: np.ndarray,
    *,
    configuration: dict,
    raman_shift: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    """Repair only unsupported coherent negative valleys in local-residual space."""

    from scipy.ndimage import gaussian_filter1d

    generated, training = _condition_arrays(
        generated_spectra,
        training_spectra,
    )

    if generated.shape[1] < 3:
        raise ValueError("局部负谷保护要求至少3个Raman点。")

    if raman_shift is None:
        axis = np.arange(generated.shape[1], dtype=np.float64)
    else:
        axis = np.asarray(raman_shift, dtype=np.float64).reshape(-1)
        if axis.size != generated.shape[1]:
            raise ValueError("raman_shift长度与生成光谱长度不一致。")
        if not np.isfinite(axis).all() or not np.all(np.diff(axis) > 0.0):
            raise ValueError("raman_shift必须是严格递增有限数轴。")

    spacing = float(np.median(np.diff(axis)))
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("Raman轴间隔无效。")

    baseline_sigma_cm1 = float(
        configuration.get("local_residual_baseline_sigma_cm1", 30.0)
    )
    training_lower_quantile = float(
        configuration.get("local_residual_training_lower_quantile", 0.10)
    )
    training_iqr_margin = float(
        configuration.get("local_residual_training_iqr_margin", 0.50)
    )
    training_mad_multiplier = float(
        configuration.get(
            "local_residual_training_mad_multiplier",
            3.0,
        )
    )
    minimum_scale = float(
        configuration.get("local_residual_minimum_scale_intensity", 2.0)
    )
    activation_maximum = float(
        configuration.get("activation_maximum_intensity", -18.0)
    )
    minimum_deficit = float(
        configuration.get("minimum_deficit_intensity", 5.0)
    )
    minimum_contiguous_points = int(
        configuration.get("minimum_negative_valley_contiguous_points", 3)
    )
    softness_fraction = float(
        configuration.get("local_residual_softness_scale_fraction", 0.20)
    )
    minimum_softness = float(
        configuration.get("minimum_softness_intensity", 0.25)
    )
    maximum_softness = float(
        configuration.get("maximum_softness_intensity", 2.0)
    )
    maximum_modified_fraction = float(
        configuration.get("maximum_modified_fraction", 0.05)
    )

    if not np.isfinite(baseline_sigma_cm1) or baseline_sigma_cm1 <= 0.0:
        raise ValueError("local_residual_baseline_sigma_cm1必须大于0。")
    if not 0.0 <= training_lower_quantile <= 0.25:
        raise ValueError("local_residual_training_lower_quantile必须位于[0,0.25]。")
    if not np.isfinite(training_iqr_margin) or training_iqr_margin < 0.0:
        raise ValueError("local_residual_training_iqr_margin必须是非负有限数。")
    if (
        not np.isfinite(training_mad_multiplier)
        or training_mad_multiplier <= 0.0
    ):
        raise ValueError(
            "local_residual_training_mad_multiplier必须大于0。"
        )
    if not np.isfinite(minimum_scale) or minimum_scale <= 0.0:
        raise ValueError("local_residual_minimum_scale_intensity必须大于0。")
    if not np.isfinite(activation_maximum):
        raise ValueError("activation_maximum_intensity必须是有限数。")
    if not np.isfinite(minimum_deficit) or minimum_deficit < 0.0:
        raise ValueError("minimum_deficit_intensity必须是非负有限数。")
    if not 1 <= minimum_contiguous_points <= 31:
        raise ValueError("minimum_negative_valley_contiguous_points必须位于[1,31]。")
    if not np.isfinite(softness_fraction) or softness_fraction < 0.0:
        raise ValueError("local_residual_softness_scale_fraction必须是非负有限数。")
    if (
        not np.isfinite(minimum_softness)
        or not np.isfinite(maximum_softness)
        or minimum_softness <= 0.0
        or maximum_softness < minimum_softness
    ):
        raise ValueError("负谷softness配置无效。")
    if not 0.0 < maximum_modified_fraction <= 0.20:
        raise ValueError("maximum_modified_fraction必须位于(0,0.20]。")

    sigma_samples = baseline_sigma_cm1 / spacing
    training_baseline = gaussian_filter1d(
        training,
        sigma=sigma_samples,
        axis=1,
        mode="nearest",
        truncate=4.0,
    )
    generated_baseline = gaussian_filter1d(
        generated,
        sigma=sigma_samples,
        axis=1,
        mode="nearest",
        truncate=4.0,
    )

    training_local = training - training_baseline
    generated_local = generated - generated_baseline

    train_q = np.quantile(
        training_local,
        training_lower_quantile,
        axis=0,
    )
    train_q25 = np.quantile(
        training_local,
        0.25,
        axis=0,
    )
    train_q75 = np.quantile(
        training_local,
        0.75,
        axis=0,
    )

    train_iqr = np.maximum(
        train_q75 - train_q25,
        minimum_scale,
    )

    # D4.3.2.14b:
    # 再建立一个对单个极端training mapping更稳健的
    # median/MAD负尾边界。
    train_median = np.median(
        training_local,
        axis=0,
    )

    train_mad = np.median(
        np.abs(
            training_local
            - train_median[np.newaxis, :]
        ),
        axis=0,
    )

    train_robust_sigma = np.maximum(
        1.4826 * train_mad,
        minimum_scale,
    )

    # q10/IQR边界
    quantile_floor = (
        train_q
        - training_iqr_margin
        * train_iqr
    )

    # median/MAD边界
    mad_floor = (
        train_median
        - training_mad_multiplier
        * train_robust_sigma
    )

    # 取两者中较严格、较不负的下界。
    #
    # 单独1条极端training谱不能把允许范围拖到-100；
    # 如果多数training本身都有真实负谷，
    # median和q10都会同步下降，因此仍然允许。
    local_floor = np.maximum(
        quantile_floor,
        mad_floor,
    )

    # 仍然只做negative-valley guard。
    local_floor = np.minimum(
        local_floor,
        0.0,
    )

    local_deficit = local_floor[np.newaxis, :] - generated_local
    candidate = (
        (generated_local < local_floor[np.newaxis, :])
        & (generated < activation_maximum)
        & (local_deficit > minimum_deficit)
    )

    retained = np.zeros_like(candidate, dtype=bool)
    for spectrum_index in range(candidate.shape[0]):
        row = candidate[spectrum_index]
        padded = np.pad(row.astype(np.int8), (1, 1), constant_values=0)
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        stops = np.flatnonzero(changes == -1)
        for start, stop in zip(starts, stops, strict=True):
            if int(stop - start) >= minimum_contiguous_points:
                retained[spectrum_index, start:stop] = True

    modified_point_count = int(np.sum(retained))
    modified_fraction = modified_point_count / float(generated.size)
    if modified_fraction > maximum_modified_fraction:
        raise RuntimeError(
            "局部负谷修正点比例超过安全上限："
            f"{modified_fraction:.6%} > {maximum_modified_fraction:.6%}。"
            "这说明异常不是少数局部深谷，应停止导出并检查模型。"
        )

    corrected = generated.copy()
    if modified_point_count:
        softness = np.clip(
            softness_fraction * train_iqr,
            minimum_softness,
            maximum_softness,
        )
        soft_local = (
            local_floor[np.newaxis, :]
            - softness[np.newaxis, :]
            * np.tanh(
                np.maximum(local_deficit, 0.0)
                / softness[np.newaxis, :]
            )
        )
        corrected_local = generated_local.copy()
        corrected_local[retained] = soft_local[retained]
        candidate_corrected = generated_baseline + corrected_local
        corrected[retained] = candidate_corrected[retained]

    difference = corrected - generated
    diagnostics: dict[str, float | int | bool | str] = {
        "enabled": True,
        "mode": "local_residual_negative_valley",
        "baseline_sigma_cm1": baseline_sigma_cm1,
        "raman_spacing_cm1": spacing,
        "training_lower_quantile": training_lower_quantile,
        "training_iqr_margin": training_iqr_margin,
        "training_mad_multiplier": training_mad_multiplier,
        "robust_sigma_minimum": float(
            np.min(train_robust_sigma)
        ),
        "robust_sigma_median": float(
            np.median(train_robust_sigma)
        ),
        "activation_maximum_intensity": activation_maximum,
        "minimum_deficit_intensity": minimum_deficit,
        "minimum_negative_valley_contiguous_points": minimum_contiguous_points,
        "negative_valley_candidate_point_count": int(np.sum(candidate)),
        "negative_valley_retained_point_count": modified_point_count,
        "modified_point_count": modified_point_count,
        "modified_point_fraction": float(modified_fraction),
        "modified_spectrum_count": int(np.sum(np.any(retained, axis=1))),
        "minimum_before": float(np.min(generated)),
        "minimum_after": float(np.min(corrected)),
        "correction_rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "maximum_absolute_correction": float(np.max(np.abs(difference))),
        "local_floor_minimum": float(np.min(local_floor)),
        "local_floor_median": float(np.median(local_floor)),
    }
    return corrected.astype(np.float32, copy=False), diagnostics


def _apply_condition_pointwise_shape_preserving_negative_valley_guard(
    generated_spectra: np.ndarray,
    training_spectra: np.ndarray,
    *,
    configuration: dict,
    raman_shift: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    # Repair coherent unsupported negative valleys while preserving shape.

    generated, training = _condition_arrays(
        generated_spectra,
        training_spectra,
    )

    if generated.shape[1] < 3:
        raise ValueError("保形负谷保护要求至少3个Raman点。")

    if raman_shift is not None:
        axis = np.asarray(raman_shift, dtype=np.float64).reshape(-1)
        if axis.size != generated.shape[1]:
            raise ValueError("raman_shift长度与生成光谱长度不一致。")
        if not np.isfinite(axis).all() or not np.all(np.diff(axis) > 0.0):
            raise ValueError("raman_shift必须是严格递增有限数轴。")

    lower_quantile = float(
        configuration.get("pointwise_negative_lower_quantile", 0.10)
    )
    reference_quantile = float(
        configuration.get("pointwise_negative_reference_quantile", 0.25)
    )
    minimum_margin = float(
        configuration.get("pointwise_negative_minimum_margin_intensity", 2.0)
    )
    maximum_margin = float(
        configuration.get("pointwise_negative_maximum_margin_intensity", 8.0)
    )
    activation_maximum = float(
        configuration.get("activation_maximum_intensity", -18.0)
    )
    minimum_deficit = float(
        configuration.get("minimum_deficit_intensity", 5.0)
    )
    minimum_contiguous_points = int(
        configuration.get("minimum_negative_valley_contiguous_points", 3)
    )
    raw_retention = float(
        configuration.get("shape_preserving_raw_retention", 0.20)
    )
    maximum_modified_fraction = float(
        configuration.get("maximum_modified_fraction", 0.05)
    )

    if not 0.0 <= lower_quantile < reference_quantile <= 0.50:
        raise ValueError(
            "pointwise_negative_lower_quantile和"
            "pointwise_negative_reference_quantile配置无效。"
        )
    if (
        not np.isfinite(minimum_margin)
        or not np.isfinite(maximum_margin)
        or minimum_margin < 0.0
        or maximum_margin < minimum_margin
    ):
        raise ValueError("逐Raman负尾margin配置无效。")
    if not np.isfinite(activation_maximum):
        raise ValueError("activation_maximum_intensity必须是有限数。")
    if not np.isfinite(minimum_deficit) or minimum_deficit < 0.0:
        raise ValueError("minimum_deficit_intensity必须是非负有限数。")
    if not 1 <= minimum_contiguous_points <= 31:
        raise ValueError(
            "minimum_negative_valley_contiguous_points必须位于[1,31]。"
        )
    if not np.isfinite(raw_retention) or not 0.0 < raw_retention < 1.0:
        raise ValueError("shape_preserving_raw_retention必须位于(0,1)。")
    if not 0.0 < maximum_modified_fraction <= 0.20:
        raise ValueError("maximum_modified_fraction必须位于(0,0.20]。")

    train_lower = np.quantile(training, lower_quantile, axis=0)
    train_reference = np.quantile(training, reference_quantile, axis=0)
    train_minimum = np.min(training, axis=0)

    lower_tail_spread = np.maximum(train_reference - train_lower, 0.0)
    pointwise_margin = np.clip(
        lower_tail_spread,
        minimum_margin,
        maximum_margin,
    )

    # 逐Raman训练下边界；如果training全为正值，也至少允许到-18附近的普通负噪声。
    pointwise_floor = np.minimum(
        train_minimum - pointwise_margin,
        activation_maximum,
    )

    deficit = pointwise_floor[np.newaxis, :] - generated
    candidate = (
        (generated < activation_maximum)
        & (deficit > minimum_deficit)
    )

    # 只保留连续负谷；单点极端噪声不在本模块处理。
    retained = np.zeros_like(candidate, dtype=bool)
    for spectrum_index in range(candidate.shape[0]):
        row = candidate[spectrum_index]
        padded = np.pad(row.astype(np.int8), (1, 1), constant_values=0)
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        stops = np.flatnonzero(changes == -1)
        for start, stop in zip(starts, stops, strict=True):
            if int(stop - start) >= minimum_contiguous_points:
                retained[spectrum_index, start:stop] = True

    modified_point_count = int(np.sum(retained))
    modified_fraction = modified_point_count / float(generated.size)
    if modified_fraction > maximum_modified_fraction:
        raise RuntimeError(
            "保形负谷修正点比例超过安全上限："
            f"{modified_fraction:.6%} > {maximum_modified_fraction:.6%}。"
            "这说明异常不是少数局部深谷，应停止导出并检查模型。"
        )

    corrected = generated.copy()
    if modified_point_count:
        # 20%保留RAW谷形，80%向training逐点下边界靠近；不会tanh饱和成水平地板。
        blended = (
            raw_retention * generated
            + (1.0 - raw_retention) * pointwise_floor[np.newaxis, :]
        )
        corrected[retained] = blended[retained]

    difference = corrected - generated
    diagnostics: dict[str, float | int | bool | str] = {
        "enabled": True,
        "mode": "pointwise_shape_preserving_negative_valley",
        "training_count": int(training.shape[0]),
        "generated_count": int(generated.shape[0]),
        "pointwise_negative_lower_quantile": lower_quantile,
        "pointwise_negative_reference_quantile": reference_quantile,
        "pointwise_negative_minimum_margin_intensity": minimum_margin,
        "pointwise_negative_maximum_margin_intensity": maximum_margin,
        "activation_maximum_intensity": activation_maximum,
        "minimum_deficit_intensity": minimum_deficit,
        "minimum_negative_valley_contiguous_points": int(minimum_contiguous_points),
        "shape_preserving_raw_retention": raw_retention,
        "negative_valley_candidate_point_count": int(np.sum(candidate)),
        "negative_valley_retained_point_count": modified_point_count,
        "modified_point_count": modified_point_count,
        "modified_point_fraction": float(modified_fraction),
        "modified_spectrum_count": int(np.sum(np.any(retained, axis=1))),
        "minimum_before": float(np.min(generated)),
        "minimum_after": float(np.min(corrected)),
        "pointwise_floor_minimum": float(np.min(pointwise_floor)),
        "pointwise_floor_median": float(np.median(pointwise_floor)),
        "correction_rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "maximum_absolute_correction": float(np.max(np.abs(difference))),
    }
    return corrected.astype(np.float32, copy=False), diagnostics


def apply_condition_intensity_envelope_guard(
    generated_spectra: np.ndarray,
    training_spectra: np.ndarray,
    *,
    configuration: dict,
    raman_shift: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    """Softly pull back pointwise extreme negative valleys and high peaks."""

    generated, training = _condition_arrays(generated_spectra, training_spectra)
    if not isinstance(configuration, dict):
        raise ValueError("intensity_envelope_guard配置必须是字典。")

    mode = str(
        configuration.get("mode", "legacy_intensity_envelope")
    ).strip().lower()
    if mode == "local_residual_negative_valley":
        return _apply_condition_local_residual_negative_valley_guard(
            generated,
            training,
            configuration=configuration,
            raman_shift=raman_shift,
        )
    if mode == "pointwise_shape_preserving_negative_valley":
        return _apply_condition_pointwise_shape_preserving_negative_valley_guard(
            generated,
            training,
            configuration=configuration,
            raman_shift=raman_shift,
        )
    if mode != "legacy_intensity_envelope":
        raise ValueError(
            "intensity_envelope_guard.mode必须为"
            "legacy_intensity_envelope、"
            "local_residual_negative_valley或"
            "pointwise_shape_preserving_negative_valley。"
        )

    training_quantile = float(configuration.get("training_lower_quantile", 0.05))
    training_upper_quantile = float(
        configuration.get("training_upper_quantile", 0.95)
    )
    training_iqr_margin = float(configuration.get("training_iqr_margin", 0.50))
    extrema_margin_fraction = float(
        configuration.get("training_extrema_margin_iqr_fraction", 0.10)
    )
    minimum_extrema_margin = float(
        configuration.get("minimum_extrema_margin_intensity", 2.0)
    )
    generated_quantile = float(
        configuration.get("generated_lower_quantile", 0.025)
    )
    generated_upper_quantile = float(
        configuration.get("generated_upper_quantile", 0.975)
    )
    minimum_generated_count = int(configuration.get("minimum_generated_count", 20))
    activation_maximum = float(
        configuration.get("activation_maximum_intensity", -18.0)
    )
    minimum_deficit = float(
        configuration.get("minimum_deficit_intensity", 5.0)
    )
    minimum_excess = float(
        configuration.get("minimum_excess_intensity", 5.0)
    )
    softness_fraction = float(configuration.get("softness_iqr_fraction", 0.05))
    minimum_softness = float(
        configuration.get("minimum_softness_intensity", 0.25)
    )
    maximum_softness = float(
        configuration.get("maximum_softness_intensity", 2.0)
    )
    maximum_modified_fraction = float(
        configuration.get("maximum_modified_fraction", 0.02)
    )
    enforce_global_extrema = bool(
        configuration.get("enforce_training_global_extrema", False)
    )
    background_quantile = float(
        configuration.get("condition_negative_background_quantile", 0.01)
    )
    background_iqr_margin = float(
        configuration.get("condition_negative_background_iqr_margin", 0.50)
    )
    minimum_negative_count = int(
        configuration.get("minimum_condition_negative_count", 64)
    )
    local_support_quantile = float(
        configuration.get("local_negative_support_quantile", 0.10)
    )
    minimum_contiguous_points = int(
        configuration.get("minimum_negative_valley_contiguous_points", 3)
    )
    severe_depth_iqr_multiplier = float(
        configuration.get("severe_negative_depth_iqr_multiplier", 3.0)
    )

    if not 0.0 <= training_quantile <= 0.25:
        raise ValueError("training_lower_quantile必须位于[0, 0.25]。")
    if not 0.75 <= training_upper_quantile <= 1.0:
        raise ValueError("training_upper_quantile必须位于[0.75, 1]。")
    if not 0.0 <= generated_quantile <= 0.10:
        raise ValueError("generated_lower_quantile必须位于[0, 0.10]。")
    if not 0.90 <= generated_upper_quantile <= 1.0:
        raise ValueError("generated_upper_quantile必须位于[0.90, 1]。")
    if not np.isfinite(training_iqr_margin) or training_iqr_margin < 0.0:
        raise ValueError("training_iqr_margin必须是非负有限数。")
    if not np.isfinite(extrema_margin_fraction) or extrema_margin_fraction < 0.0:
        raise ValueError("training_extrema_margin_iqr_fraction必须是非负有限数。")
    if not np.isfinite(minimum_extrema_margin) or minimum_extrema_margin < 0.0:
        raise ValueError("minimum_extrema_margin_intensity必须是非负有限数。")
    if minimum_generated_count < 2:
        raise ValueError("minimum_generated_count至少为2。")
    if not np.isfinite(activation_maximum):
        raise ValueError("activation_maximum_intensity必须是有限数。")
    if not np.isfinite(minimum_deficit) or minimum_deficit < 0.0:
        raise ValueError("minimum_deficit_intensity必须是非负有限数。")
    if not np.isfinite(minimum_excess) or minimum_excess < 0.0:
        raise ValueError("minimum_excess_intensity必须是非负有限数。")
    if (
        not np.isfinite(softness_fraction)
        or softness_fraction < 0.0
        or not np.isfinite(minimum_softness)
        or not np.isfinite(maximum_softness)
        or minimum_softness <= 0.0
        or maximum_softness < minimum_softness
    ):
        raise ValueError("负值回拉softness配置无效。")
    if not 0.0 < maximum_modified_fraction <= 0.20:
        raise ValueError("maximum_modified_fraction必须位于(0, 0.20]。")
    if not 0.0 <= background_quantile <= 0.10:
        raise ValueError("condition_negative_background_quantile必须位于[0,0.10]。")
    if background_iqr_margin < 0.0 or minimum_negative_count < 4:
        raise ValueError("条件负背景分布配置无效。")
    if not 0.0 <= local_support_quantile <= 0.25:
        raise ValueError("local_negative_support_quantile必须位于[0,0.25]。")
    if not 1 <= minimum_contiguous_points <= 31:
        raise ValueError("minimum_negative_valley_contiguous_points必须位于[1,31]。")
    if severe_depth_iqr_multiplier <= 0.0:
        raise ValueError("severe_negative_depth_iqr_multiplier必须大于0。")

    train_q = np.quantile(training, training_quantile, axis=0)
    train_upper_q = np.quantile(training, training_upper_quantile, axis=0)
    train_q25 = np.quantile(training, 0.25, axis=0)
    train_q75 = np.quantile(training, 0.75, axis=0)
    train_iqr = np.maximum(train_q75 - train_q25, 0.0)
    extrema_margin = np.maximum(
        extrema_margin_fraction * train_iqr,
        minimum_extrema_margin,
    )
    training_floor = np.minimum(
        train_q - training_iqr_margin * train_iqr,
        np.min(training, axis=0) - extrema_margin,
    )
    training_ceiling = np.maximum(
        train_upper_q + training_iqr_margin * train_iqr,
        np.max(training, axis=0) + extrema_margin,
    )

    used_generated_reference = generated.shape[0] >= minimum_generated_count
    if used_generated_reference:
        generated_floor = np.quantile(generated, generated_quantile, axis=0)
        generated_ceiling = np.quantile(
            generated, generated_upper_quantile, axis=0
        )
        # Training owns the physical boundary.  Generated quantiles are only
        # rank gates, so a malformed generated batch cannot legitimize its own
        # extreme tail by moving the allowed boundary outward.
        lower_rank_gate = generated <= generated_floor[np.newaxis, :]
        upper_rank_gate = generated >= generated_ceiling[np.newaxis, :]
    else:
        lower_rank_gate = np.ones_like(generated, dtype=bool)
        upper_rank_gate = np.ones_like(generated, dtype=bool)

    allowed_floor = training_floor.copy()
    allowed_ceiling = training_ceiling.copy()

    # Build one condition-level negative background boundary from all of this
    # condition's train-only points.  This is intentionally not a universal
    # -40 rule: a condition whose ordinary background reaches -100 keeps that
    # range, while a condition whose noise lives around -10..-20 cannot create
    # an unsupported -70 valley.  A Raman position with repeated training
    # support below that boundary is exempted and retains its pointwise floor.
    negative_training = training[training < 0.0]
    used_negative_background = negative_training.size >= minimum_negative_count
    # ``minimum_negative_count`` is the preferred production sample size, not
    # a reason to mix positive peaks into the negative-background estimate.
    # Small smoke/unit-test conditions can still provide a valid lower-tail
    # reference when at least four genuine negative observations are present.
    # Falling back to all intensities too early makes peak heights inflate the
    # IQR and can hide even an isolated -200-style numerical deformation.
    used_small_negative_background = (
        4 <= negative_training.size < minimum_negative_count
    )
    if used_negative_background or used_small_negative_background:
        background_q = float(
            np.quantile(negative_training, background_quantile)
        )
        background_q25 = float(np.quantile(negative_training, 0.25))
        background_q75 = float(np.quantile(negative_training, 0.75))
        background_iqr = max(background_q75 - background_q25, 0.0)
        condition_background_floor = (
            background_q - background_iqr_margin * background_iqr
        )
    else:
        flattened = training.reshape(-1)
        background_q = float(np.quantile(flattened, training_quantile))
        background_q25 = float(np.quantile(flattened, 0.25))
        background_q75 = float(np.quantile(flattened, 0.75))
        background_iqr = max(background_q75 - background_q25, 0.0)
        condition_background_floor = (
            background_q - background_iqr_margin * background_iqr
        )
    condition_background_floor = min(
        float(condition_background_floor), activation_maximum
    )
    pointwise_support = np.quantile(
        training, local_support_quantile, axis=0
    ) <= condition_background_floor
    condition_limited_floor = np.maximum(
        allowed_floor, condition_background_floor
    )
    allowed_floor = np.where(
        pointwise_support, allowed_floor, condition_limited_floor
    )

    # The lower side is exclusively a negative-valley guard: a low but
    # positive peak is not lifted merely because training intensities are
    # higher.  The upper side independently limits unsupported peak strength.
    allowed_floor = np.minimum(allowed_floor, activation_maximum)
    training_global_minimum = float(np.min(training))
    training_global_maximum = float(np.max(training))
    if enforce_global_extrema:
        allowed_floor = np.maximum(allowed_floor, training_global_minimum)
        allowed_ceiling = np.minimum(allowed_ceiling, training_global_maximum)

    lower_candidate = np.logical_and(
        lower_rank_gate,
        np.logical_and(
            np.logical_and(
                generated < allowed_floor[np.newaxis, :],
                generated < activation_maximum,
            ),
            allowed_floor[np.newaxis, :] - generated > minimum_deficit,
        ),
    )
    # Keep only coherent valleys (or an exceptionally deep isolated point).
    # This leaves ordinary one-point negative noise untouched while catching
    # the broad 1270--1288 cm-1 type deformation seen in generated spectra.
    lower_violation = np.zeros_like(lower_candidate, dtype=bool)
    severe_depth = max(
        minimum_deficit,
        severe_depth_iqr_multiplier * max(background_iqr, minimum_softness),
    )
    lower_deficit = allowed_floor[np.newaxis, :] - generated
    for spectrum_index in range(lower_candidate.shape[0]):
        row = lower_candidate[spectrum_index]
        padded = np.pad(row.astype(np.int8), (1, 1), constant_values=0)
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        stops = np.flatnonzero(changes == -1)
        for start, stop in zip(starts, stops, strict=True):
            coherent = int(stop - start) >= minimum_contiguous_points
            exceptionally_deep = bool(
                np.max(lower_deficit[spectrum_index, start:stop])
                >= severe_depth
            )
            if coherent or exceptionally_deep:
                lower_violation[spectrum_index, start:stop] = True
    upper_violation = np.logical_and(
        upper_rank_gate,
        np.logical_and(
            generated > allowed_ceiling[np.newaxis, :],
            generated - allowed_ceiling[np.newaxis, :] > minimum_excess,
        ),
    )
    violation = np.logical_or(lower_violation, upper_violation)
    modified_point_count = int(np.sum(violation))
    modified_fraction = modified_point_count / float(generated.size)
    if modified_fraction > maximum_modified_fraction:
        raise RuntimeError(
            "强度包络修正点比例超过安全上限："
            f"{modified_fraction:.6%} > {maximum_modified_fraction:.6%}。"
            "这说明不是少数离群负谷，应停止导出并检查模型。"
        )

    corrected = generated.copy()
    if modified_point_count:
        softness = np.clip(
            softness_fraction * train_iqr,
            minimum_softness,
            maximum_softness,
        )
        deficit = allowed_floor[np.newaxis, :] - generated
        # Small violations remain almost unchanged.  Large violations approach
        # the pointwise allowed floor without forming a constant horizontal cap.
        soft_value = allowed_floor[np.newaxis, :] - softness[np.newaxis, :] * np.tanh(
            np.maximum(deficit, 0.0) / softness[np.newaxis, :]
        )
        corrected[lower_violation] = soft_value[lower_violation]
        excess = generated - allowed_ceiling[np.newaxis, :]
        soft_upper = allowed_ceiling[np.newaxis, :] + softness[np.newaxis, :] * np.tanh(
            np.maximum(excess, 0.0) / softness[np.newaxis, :]
        )
        corrected[upper_violation] = soft_upper[upper_violation]
        if enforce_global_extrema:
            corrected[lower_violation] = np.maximum(
                corrected[lower_violation], training_global_minimum
            )
            corrected[upper_violation] = np.minimum(
                corrected[upper_violation], training_global_maximum
            )

    difference = corrected - generated
    diagnostics: dict[str, float | int | bool] = {
        "enabled": True,
        "training_count": int(training.shape[0]),
        "generated_count": int(generated.shape[0]),
        "used_generated_batch_reference": used_generated_reference,
        "generated_batch_is_rank_gate_only": True,
        "modified_spectrum_count": int(np.sum(np.any(violation, axis=1))),
        "modified_point_count": modified_point_count,
        "lower_modified_point_count": int(np.sum(lower_violation)),
        "upper_modified_point_count": int(np.sum(upper_violation)),
        "modified_point_fraction": float(modified_fraction),
        "minimum_before": float(np.min(generated)),
        "minimum_after": float(np.min(corrected)),
        "training_global_minimum": training_global_minimum,
        "training_global_maximum": training_global_maximum,
        "maximum_before": float(np.max(generated)),
        "maximum_after": float(np.max(corrected)),
        "correction_rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "maximum_absolute_correction": float(np.max(np.abs(difference))),
        "condition_negative_background_floor": float(
            condition_background_floor
        ),
        "condition_negative_background_quantile": background_quantile,
        "condition_negative_background_iqr": float(background_iqr),
        "condition_negative_training_point_count": int(
            negative_training.size
        ),
        "used_condition_negative_background": bool(
            used_negative_background
        ),
        "used_small_sample_negative_background": bool(
            used_small_negative_background
        ),
        "minimum_negative_valley_contiguous_points": int(
            minimum_contiguous_points
        ),
        "negative_valley_candidate_point_count": int(
            np.sum(lower_candidate)
        ),
        "negative_valley_retained_point_count": int(
            np.sum(lower_violation)
        ),
    }
    return corrected.astype(np.float32, copy=False), diagnostics


def apply_condition_negative_floor_guard(
    generated_spectra: np.ndarray,
    training_spectra: np.ndarray,
    *,
    configuration: dict,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """Backward-compatible lower-tail-only wrapper for older configurations."""

    legacy = dict(configuration)
    legacy.setdefault("training_upper_quantile", 1.0)
    legacy.setdefault("generated_upper_quantile", 1.0)
    legacy.setdefault("minimum_excess_intensity", 1.0e30)
    legacy.setdefault(
        "enforce_training_global_extrema",
        bool(legacy.pop("enforce_training_global_minimum", True)),
    )
    return apply_condition_intensity_envelope_guard(
        generated_spectra,
        training_spectra,
        configuration=legacy,
    )


@torch.inference_mode()
def generate_spectra(
    *,
    diffusion: nn.Module,
    number_of_spectra: int,
    generation_batch_size: int,
    device: torch.device,
    length_adapter: SpectrumLengthAdapter,
    output_raman_shifts: np.ndarray | None = None,
    condition_vector: np.ndarray | None = None,
    condition_id: str | None = None,
    conditional_prior_residual_bank: (
        ConditionalPriorResidualBank | None
    ) = None,
    prior_residual_transformer: (
        PriorResidualTransformer | None
    ) = None,
    broad_local_residual_decomposer: (
        BroadLocalResidualDecomposer | None
    ) = None,
    feature_peak_residual_limiter: (
        FeaturePeakResidualLimiter | None
    ) = None,
    prior_random_seed: int | None = None,
    generated_scaled_residual_batches: list[np.ndarray] | None = None,
    variation_scale: float = 1.0,
    sampling_calibrator: (
        SersSamplingCalibrator | None
    ) = None,
) -> np.ndarray:
    """
    分批生成光谱。

    首先删除模型输入末尾的补齐点；如果提供了
    output_raman_shifts，再将光谱插值恢复到指定的
    原始拉曼位移轴。
    """

    if number_of_spectra <= 0:
        raise ValueError(
            "number_of_spectra必须大于0。"
        )

    if conditional_prior_residual_bank is not None:
        if prior_residual_transformer is not None or broad_local_residual_decomposer is not None:
            raise ValueError(
                "conditional_prior_residual_bank不能与旧单条件先验参数同时使用。"
            )
        if condition_id is None or not str(condition_id).strip():
            raise ValueError("使用条件先验库生成时必须提供condition_id。")

    if generation_batch_size <= 0:
        raise ValueError(
            "generation_batch_size必须大于0。"
        )

    if (
        feature_peak_residual_limiter is not None
        and prior_residual_transformer is None
    ):
        raise ValueError(
            "D3.5特征峰残差软上限必须与D2先验残差变换器一起使用。"
        )

    if broad_local_residual_decomposer is not None:
        if prior_residual_transformer is None:
            raise ValueError(
                "D2.6 broad-local residual必须与"
                "PCA prior_residual_transformer一起使用。"
            )

        if (
            prior_residual_transformer.prior_method
            != "pca_reconstruction"
        ):
            raise ValueError(
                "D2.6当前只支持pca_reconstruction outer prior。"
            )

        if feature_peak_residual_limiter is not None:
            raise ValueError(
                "D2.6第一轮消融不与D3.5 limiter组合。"
            )

    variation_scale = float(variation_scale)

    if (
        not np.isfinite(variation_scale)
        or variation_scale <= 0.0
        or variation_scale > 1.0
    ):
        raise ValueError(
            "variation_scale必须位于(0, 1]范围内。"
        )

    variation_center: np.ndarray | None = None

    if variation_scale < 1.0:
        if prior_residual_transformer is None:
            raise ValueError(
                "variation_scale小于1时，必须启用"
                "prior_residual_transformer。"
            )

        if (
            prior_residual_transformer.prior_method
            != "pca_reconstruction"
        ):
            raise ValueError(
                "当前variation_scale校准只支持"
                "pca_reconstruction先验。"
            )

        if prior_residual_transformer.pca_mean is None:
            raise RuntimeError(
                "checkpoint中的PCA状态缺少pca_mean。"
            )

        variation_center = np.asarray(
            prior_residual_transformer.pca_mean,
            dtype=np.float32,
        ).reshape(-1)

        if variation_center.size < 2:
            raise RuntimeError(
                "checkpoint中的pca_mean长度无效。"
            )

        if not np.isfinite(variation_center).all():
            raise RuntimeError(
                "checkpoint中的pca_mean包含NaN或无穷值。"
            )

    target_axis: np.ndarray | None = None

    if output_raman_shifts is not None:
        target_axis = np.asarray(
            output_raman_shifts,
            dtype=np.float64,
        ).reshape(-1)

        if target_axis.size < 2:
            raise ValueError(
                "输出拉曼位移轴至少需要包含两个点。"
            )

        if not np.isfinite(target_axis).all():
            raise ValueError(
                "输出拉曼位移轴包含NaN或无穷值。"
            )

        if not np.all(
            np.diff(target_axis) > 0.0
        ):
            raise ValueError(
                "输出拉曼位移轴必须严格递增。"
            )

    diffusion = diffusion.to(device)
    diffusion.eval()

    sampling_arguments: dict[str, torch.Tensor] = {}
    if length_adapter.raman_axis_mode == "union_with_valid_mask":
        mask_axis = (
            length_adapter.model_axis
            if target_axis is None
            else target_axis
        )
        sampling_mask = length_adapter.valid_mask_for_axis(
            mask_axis
        )
        if not bool(
            getattr(diffusion, "supports_valid_mask", False)
        ):
            raise RuntimeError(
                "该检查点要求掩码采样，但扩散模型不支持valid_mask。"
            )
        sampling_arguments["valid_mask"] = torch.from_numpy(
            sampling_mask
        ).to(device=device, dtype=torch.float32)

    if condition_vector is not None:
        condition_array = np.asarray(
            condition_vector,
            dtype=np.float32,
        ).reshape(-1)
        if condition_array.size == 0 or not np.isfinite(condition_array).all():
            raise ValueError("condition_vector无效。")
        if not bool(getattr(diffusion, "supports_condition", False)):
            raise RuntimeError("该扩散模型不支持condition。")
        sampling_arguments["condition"] = torch.from_numpy(
            condition_array
        ).to(device=device, dtype=torch.float32)

    prior_random_generator = np.random.default_rng(
        prior_random_seed
    )

    broad_random_generator = None

    if (
        broad_local_residual_decomposer is not None
        or conditional_prior_residual_bank is not None
    ):
        broad_random_seed = (
            None
            if prior_random_seed is None
            else int(prior_random_seed)
            + 1_000_003
        )

        broad_random_generator = (
            np.random.default_rng(
                broad_random_seed
            )
        )

    generated_batches: list[np.ndarray] = []
    number_generated = 0

    while number_generated < number_of_spectra:
        current_batch_size = min(
            generation_batch_size,
            number_of_spectra - number_generated,
        )

        batch_sampling_arguments = dict(sampling_arguments)
        sampled_prior_conditioning = None
        prior_conditioning_enabled = bool(
            getattr(diffusion, "configured_prior_conditioning_enabled", False)
            or getattr(
                getattr(diffusion, "model", None),
                "prior_conditioning_enabled",
                False,
            )
        )
        if prior_conditioning_enabled:
            if conditional_prior_residual_bank is None:
                raise RuntimeError(
                    "先验条件化检查点生成时必须提供条件先验残差库。"
                )
            if broad_random_generator is None:
                raise RuntimeError("条件先验库缺少broad随机数生成器。")
            sampled_prior_conditioning = (
                conditional_prior_residual_bank.sample_generation_conditioning(
                    current_batch_size,
                    condition_id=str(condition_id),
                    prior_random_generator=prior_random_generator,
                    broad_random_generator=broad_random_generator,
                )
            )
            padded_prior_conditioning = length_adapter.adapt(
                sampled_prior_conditioning
            )
            batch_sampling_arguments["prior_conditioning"] = (
                torch.from_numpy(padded_prior_conditioning).to(
                    device=device, dtype=torch.float32
                )
            )

        generated = diffusion.sample(
            batch_size=current_batch_size,
            **batch_sampling_arguments,
        )

        if generated.ndim != 3:
            raise RuntimeError(
                "生成张量应为[B,C,L]，实际为"
                f"{tuple(generated.shape)}。"
            )

        if generated.shape[0] != current_batch_size:
            raise RuntimeError(
                "模型返回的生成光谱数量与请求的"
                "批次大小不一致。"
            )

        if generated.shape[1] != 1:
            raise RuntimeError(
                "当前项目要求生成结果只有一个光谱通道。"
            )

        generated_numpy = (
            generated[:, 0, :]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        # 删除模型输入末尾的补齐点，
        # 恢复到训练使用的统一拉曼位移轴。
        restored = length_adapter.restore(
            generated_numpy
        )

        if generated_scaled_residual_batches is not None:
            generated_scaled_residual_batches.append(
                np.asarray(restored, dtype=np.float32).copy()
            )

        active_model_axis = np.asarray(
            length_adapter.model_axis,
            dtype=np.float64,
        )

        if conditional_prior_residual_bank is not None:
            if sampled_prior_conditioning is not None:
                restored = (
                    conditional_prior_residual_bank
                    .reconstruct_generated_with_conditioning(
                        restored,
                        prior_conditioning=sampled_prior_conditioning,
                        condition_id=str(condition_id),
                    )
                )
            else:
                if broad_random_generator is None:
                    raise RuntimeError("条件先验库缺少broad随机数生成器。")
                restored = conditional_prior_residual_bank.reconstruct_generated(
                    restored,
                    condition_id=str(condition_id),
                    prior_random_generator=prior_random_generator,
                    broad_random_generator=broad_random_generator,
                )
            valid_length = conditional_prior_residual_bank.valid_length(
                str(condition_id)
            )
            active_model_axis = active_model_axis[:valid_length]

        # D2：此时数据已经删除末尾补齐点，
        # 但仍位于统一训练拉曼轴上。
        # 先取消残差缩放并加回逐点中位数先验，
        # 再恢复到标签对应的原始拉曼轴。
        if (
            prior_residual_transformer
            is not None
        ):
            reference_priors = None

            if (
                prior_residual_transformer.prior_method
                == "pca_reconstruction"
            ):
                reference_priors = (
                    prior_residual_transformer.sample_reference_priors(
                        current_batch_size,
                        random_generator=prior_random_generator,
                    )
                )

            if broad_local_residual_decomposer is not None:
                if reference_priors is None:
                    raise RuntimeError(
                        "D2.6生成缺少outer PCA reference priors。"
                    )

                if broad_random_generator is None:
                    raise RuntimeError(
                        "D2.6生成缺少broad随机数生成器。"
                    )

                local_raw_residuals = (
                    broad_local_residual_decomposer
                    .inverse_local_transform(
                        restored
                    )
                )

                sampled_broad_residuals = (
                    broad_local_residual_decomposer
                    .sample_broad_residuals(
                        current_batch_size,
                        random_generator=(
                            broad_random_generator
                        ),
                    )
                )

                restored = (
                    reference_priors
                    + sampled_broad_residuals
                    + local_raw_residuals
                ).astype(
                    np.float32,
                    copy=False,
                )

            else:
                restored = (
                    prior_residual_transformer
                    .inverse_transform(
                        restored,
                        reference_priors=reference_priors,
                    )
                )

        # D2.5最终生成离散度校准：
        # 完整归一化光谱重建完成后，以checkpoint中训练集
        # 拟合的PCA均值谱为中心，温和收缩样本间离散度。
        # 不进行平滑，也不写死任何特征峰位置。
        if variation_center is not None:
            if restored.ndim != 2:
                raise RuntimeError(
                    "离散度校准要求光谱数组为[N,L]，"
                    f"实际形状为{restored.shape}。"
                )

            if restored.shape[1] != variation_center.size:
                raise RuntimeError(
                    "重建光谱与PCA均值谱长度不一致："
                    f"{restored.shape[1]} != "
                    f"{variation_center.size}。"
                )

            restored = (
                variation_center[np.newaxis, :]
                + variation_scale
                * (
                    restored
                    - variation_center[np.newaxis, :]
                )
            ).astype(
                np.float32,
                copy=False,
            )

        # D3.5：先恢复到完整归一化光谱，再相对训练集先验
        # 只软限制自动识别的特征峰窗口中的残差幅度。
        # 此步骤必须发生在轴插值和全局反归一化之前。
        if feature_peak_residual_limiter is not None:
            restored = (
                feature_peak_residual_limiter
                .apply_to_normalized_spectra(restored)
            )

        # 如果指定了标签或模板文件的原始位移轴，
        # 再从统一训练轴插值回该输出轴。
        if target_axis is not None:
            if (
                target_axis.size == active_model_axis.size
                and np.allclose(target_axis, active_model_axis, rtol=0.0, atol=1.0e-8)
            ):
                restored = restored.astype(np.float32, copy=False)
            elif active_model_axis.size == length_adapter.model_axis.size:
                restored = length_adapter.interpolate_from_model_axis(
                    restored,
                    target_axis,
                )
            else:
                restored = np.stack(
                    [
                        np.interp(target_axis, active_model_axis, spectrum)
                        for spectrum in restored
                    ],
                    axis=0,
                ).astype(np.float32)

        restored = np.asarray(
            restored,
            dtype=np.float32,
        )

        if restored.ndim != 2:
            raise RuntimeError(
                "恢复后的生成光谱应为二维数组"
                "[光谱数量, 光谱点数]，实际为"
                f"{restored.shape}。"
            )

        if restored.shape[0] != current_batch_size:
            raise RuntimeError(
                "恢复后的生成光谱数量与当前"
                "生成批次大小不一致。"
            )

        if not np.isfinite(restored).all():
            raise RuntimeError(
                "生成结果包含NaN或无穷值。"
            )

        generated_batches.append(
            restored
        )

        number_generated += (
            current_batch_size
        )

    generated_spectra = np.concatenate(
        generated_batches,
        axis=0,
    )

    if (
        generated_spectra.shape[0]
        != number_of_spectra
    ):
        raise RuntimeError(
            "最终生成的光谱数量与请求数量不一致。"
        )

    # D2.5 targeted sampling calibration：
    # 在所有批次完成并恢复到最终输出轴后统一处理，
    # 使随机峰位微漂移不受generation batch size影响。
    if sampling_calibrator is not None:
        generated_spectra = sampling_calibrator.apply(
            generated_spectra
        )

    return generated_spectra
