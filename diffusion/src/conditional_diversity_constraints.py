"""D4.3 condition-aware and Raman-mask-aware residual diversity loss."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from torch import nn
from torch.nn import functional as F

from src.sers_diversity_constraints import normalize_diversity_configuration


D4_3_METHOD_VERSION = "d4_3_condition_mask_aware_residual_diversity"
D4_3_STATE_SCHEMA_VERSION = 1
_PENALTY_MODES = {"symmetric_band", "floor_only"}


def normalize_condition_aware_diversity_configuration(
    configuration: dict[str, Any] | None,
) -> dict[str, Any]:
    """Normalize legacy diversity terms plus D4.3 grouping/safety fields."""

    config = normalize_diversity_configuration(configuration)
    if not bool(config.get("enabled", False)):
        return config

    config = deepcopy(config)
    grouping = config.get("condition_grouping", {}) or {}
    if not isinstance(grouping, dict):
        raise TypeError("diversity_constraints.condition_grouping必须是字典。")
    grouping = deepcopy(grouping)
    grouping["samples_per_condition"] = int(
        grouping.get("samples_per_condition", 4)
    )

    grouping["training_samples_per_condition"] = int(
        grouping.get(
            "training_samples_per_condition",
            grouping["samples_per_condition"],
        )
    )

    grouping["validation_samples_per_condition"] = int(
        grouping.get(
            "validation_samples_per_condition",
            grouping["samples_per_condition"],
        )
    )

    grouping["shared_timestep"] = bool(
        grouping.get("shared_timestep", True)
    )

    if grouping["samples_per_condition"] < 2:
        raise ValueError(
            "condition_grouping.samples_per_condition至少为2。"
        )

    if grouping["training_samples_per_condition"] < 2:
        raise ValueError(
            "condition_grouping.training_samples_per_condition至少为2。"
        )

    if grouping["validation_samples_per_condition"] < 2:
        raise ValueError(
            "condition_grouping.validation_samples_per_condition至少为2。"
        )
    if not grouping["shared_timestep"]:
        raise ValueError(
            "D4.3首轮实验要求condition_grouping.shared_timestep=true，"
            "避免同组样本因噪声等级不同而产生伪多样性。"
        )
    config["condition_grouping"] = grouping

    maximum_ratio = float(config.get("maximum_total_ratio_to_ddpm", 0.02))
    if not 0.0 < maximum_ratio <= 0.20:
        raise ValueError(
            "diversity_constraints.maximum_total_ratio_to_ddpm"
            "必须在(0,0.20]内。"
        )
    config["maximum_total_ratio_to_ddpm"] = maximum_ratio

    for section_name in (
        "pairwise_distance",
        "pointwise_variance_floor",
    ):
        section = config[section_name]
        penalty_mode = str(
            section.get("penalty_mode", "symmetric_band")
        ).strip().lower()
        if penalty_mode not in _PENALTY_MODES:
            raise ValueError(
                f"diversity_constraints.{section_name}.penalty_mode"
                "必须为symmetric_band或floor_only。"
            )
        section["penalty_mode"] = penalty_mode

    morphology = config.get("peak_morphology", {}) or {}
    if bool(morphology.get("enabled", False)):
        raise ValueError(
            "D4.3首轮只启用条件感知高频残差多样性；"
            "peak_morphology必须为false。"
        )
    config["method_version"] = D4_3_METHOD_VERSION
    return config


def _as_spectra(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 3 and array.shape[1] == 1:
        array = array[:, 0, :]
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] < 9:
        raise ValueError("training_scaled_residuals必须为非空[N,L]数组。")
    if not np.isfinite(array).all():
        raise ValueError("training_scaled_residuals包含NaN或无穷值。")
    return array


def fit_condition_aware_diversity_constraint_state(
    *,
    training_scaled_residuals: np.ndarray,
    training_valid_masks: np.ndarray,
    training_condition_vectors: np.ndarray,
    configuration: dict[str, Any] | None,
    training_full_spectra: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit per-condition whitening statistics using training spectra only."""

    config = normalize_condition_aware_diversity_configuration(configuration)
    if not bool(config.get("enabled", False)):
        return {
            "schema_version": D4_3_STATE_SCHEMA_VERSION,
            "enabled": False,
            "configuration": config,
        }

    spectra = _as_spectra(training_scaled_residuals)
    masks = np.asarray(training_valid_masks, dtype=np.float64)
    conditions = np.asarray(training_condition_vectors, dtype=np.float64)
    if masks.shape != spectra.shape:
        raise ValueError("training_valid_masks形状必须与训练残差一致。")
    if conditions.ndim != 2 or conditions.shape[0] != spectra.shape[0]:
        raise ValueError("training_condition_vectors形状必须为[N,C]。")
    if not np.isfinite(masks).all() or not np.isfinite(conditions).all():
        raise ValueError("训练掩码或条件向量包含NaN/无穷值。")
    if not np.logical_or(masks == 0.0, masks == 1.0).all():
        raise ValueError("training_valid_masks只能包含0和1。")

    quality_configuration = (
        config.get(
            "quality_fidelity",
            {},
        )
        or {}
    )

    if not isinstance(
        quality_configuration,
        dict,
    ):
        raise TypeError(
            "diversity_constraints.quality_fidelity必须是字典。"
        )

    variance_profile_configuration = (
        quality_configuration.get(
            "condition_full_spectrum_variance_profile",
            {},
        )
        or {}
    )

    group_variance_configuration = (
        quality_configuration.get(
            "condition_full_spectrum_group_variance",
            {},
        )
        or {}
    )

    high_noise_recovery_configuration = (
        quality_configuration.get(
            "high_noise_standardized_recovery",
            {},
        )
        or {}
    )

    if not isinstance(
        variance_profile_configuration,
        dict,
    ):
        raise TypeError(
            "condition_full_spectrum_variance_profile必须是字典。"
        )

    if not isinstance(
        group_variance_configuration,
        dict,
    ):
        raise TypeError(
            "condition_full_spectrum_group_variance必须是字典。"
        )

    if not isinstance(
        high_noise_recovery_configuration,
        dict,
    ):
        raise TypeError(
            "high_noise_standardized_recovery必须是字典。"
        )

    # D4.3.2.16和D4.3.2.17共用完全相同的
    # train-only condition-specific Raman variance reference。
    #
    # 只要其中任意一个模块启用，都必须从training spectra拟合：
    #   full_spectrum_std
    #   full_spectrum_scale
    full_spectrum_profile_enabled = bool(
        variance_profile_configuration.get(
            "enabled",
            False,
        )
        or group_variance_configuration.get(
            "enabled",
            False,
        )
        or high_noise_recovery_configuration.get(
            "enabled",
            False,
        )
    )

    full_spectrum_scale_floor_quantile = float(
        variance_profile_configuration.get(
            "scale_floor_quantile",
            10.0,
        )
    )

    full_spectra = None

    if full_spectrum_profile_enabled:
        if not (
            0.0
            <= full_spectrum_scale_floor_quantile
            <= 50.0
        ):
            raise ValueError(
                "condition_full_spectrum_variance_profile."
                "scale_floor_quantile必须位于[0,50]。"
            )

        if training_full_spectra is None:
            raise ValueError(
                "启用D4.3.2.16 variance-profile或"
                "D4.3.2.17 group-variance时，"
                "必须提供training_full_spectra。"
            )

        full_spectra = np.asarray(
            training_full_spectra,
            dtype=np.float64,
        )

        if (
            full_spectra.ndim == 3
            and full_spectra.shape[1] == 1
        ):
            full_spectra = full_spectra[
                :,
                0,
                :,
            ]

        if full_spectra.shape != spectra.shape:
            raise ValueError(
                "training_full_spectra形状必须与"
                "training_scaled_residuals一致。"
            )

        if not np.isfinite(
            full_spectra
        ).all():
            raise ValueError(
                "training_full_spectra包含NaN或无穷值。"
            )

    unique_conditions, inverse = np.unique(conditions, axis=0, return_inverse=True)
    number_of_conditions = int(unique_conditions.shape[0])
    length = int(spectra.shape[1])
    whitening = np.ones((number_of_conditions, length), dtype=np.float32)
    valid_masks = np.zeros((number_of_conditions, length), dtype=np.float32)
    active_masks = np.zeros((number_of_conditions, length), dtype=np.float32)

    full_spectrum_std = (
        np.zeros(
            (
                number_of_conditions,
                length,
            ),
            dtype=np.float32,
        )
        if full_spectrum_profile_enabled
        else None
    )

    full_spectrum_scale = (
        np.ones(
            (
                number_of_conditions,
                length,
            ),
            dtype=np.float32,
        )
        if full_spectrum_profile_enabled
        else None
    )

    full_spectrum_scale_floors: list[float] = []

    training_counts: list[int] = []
    distance_medians: list[float] = []
    correlation_medians: list[float] = []
    epsilon = float(config["epsilon"])
    high_config = config["high_frequency_filter"]
    variance_config = config["pointwise_variance_floor"]
    minimum_samples = int(
        config["condition_grouping"].get(
            "training_samples_per_condition",
            config["condition_grouping"]["samples_per_condition"],
        )
    )

    for condition_index in range(number_of_conditions):
        members = np.flatnonzero(inverse == condition_index)
        if members.size < minimum_samples:
            raise ValueError(
                f"条件索引{condition_index}只有{members.size}条训练光谱，"
                f"少于{minimum_samples}。"
            )
        group_masks = masks[members]
        reference_mask = group_masks[0]
        if not np.array_equal(group_masks, np.broadcast_to(reference_mask, group_masks.shape)):
            raise ValueError(f"条件索引{condition_index}内部Raman有效区不一致。")
        valid_indices = np.flatnonzero(reference_mask > 0.5)
        if valid_indices.size < 9:
            raise ValueError(f"条件索引{condition_index}有效Raman点少于9个。")

        if full_spectrum_profile_enabled:
            if (
                full_spectra is None
                or full_spectrum_std is None
                or full_spectrum_scale is None
            ):
                raise RuntimeError(
                    "full-spectrum variance profile内部状态缺失。"
                )

            group_full = (
                full_spectra[
                    members
                ][
                    :,
                    valid_indices,
                ]
            )

            local_full_std = np.std(
                group_full,
                axis=0,
                ddof=1,
            )

            positive_full_std = (
                local_full_std[
                    local_full_std
                    > epsilon
                ]
            )

            if (
                positive_full_std.size
                == 0
            ):
                raise ValueError(
                    f"条件索引{condition_index}"
                    "完整谱没有可测方差。"
                )

            scale_floor = max(
                float(
                    np.percentile(
                        positive_full_std,
                        full_spectrum_scale_floor_quantile,
                    )
                ),
                epsilon,
            )

            local_scale = np.maximum(
                local_full_std,
                scale_floor,
            )

            full_spectrum_std[
                condition_index,
                valid_indices,
            ] = (
                local_full_std.astype(
                    np.float32
                )
            )

            full_spectrum_scale[
                condition_index,
                valid_indices,
            ] = (
                local_scale.astype(
                    np.float32
                )
            )

            full_spectrum_scale_floors.append(
                float(scale_floor)
            )

        group = spectra[members][:, valid_indices]
        high_frequency = group - gaussian_filter1d(
            group,
            sigma=float(high_config["smoothing_sigma_points"]),
            axis=1,
            mode="reflect",
            truncate=float(high_config["kernel_truncate"]),
        )
        pointwise_std = np.std(high_frequency, axis=0, ddof=1)
        positive = pointwise_std[pointwise_std > epsilon]
        if positive.size == 0:
            raise ValueError(f"条件索引{condition_index}高频残差没有可测方差。")
        floor = max(float(np.percentile(positive, 10.0)), epsilon)
        local_whitening = np.maximum(pointwise_std, floor)
        active_threshold = max(
            float(np.percentile(
                pointwise_std,
                float(variance_config["active_std_quantile"]),
            )),
            epsilon,
        )

        whitened = high_frequency / local_whitening[None, :]
        centered = whitened - whitened.mean(axis=1, keepdims=True)
        norms = np.sqrt(np.mean(np.square(centered), axis=1)).clip(min=epsilon)
        distances: list[float] = []
        correlations: list[float] = []
        for first in range(members.size - 1):
            for second in range(first + 1, members.size):
                distances.append(float(np.sqrt(np.mean(np.square(
                    whitened[first] - whitened[second]
                )))))
                correlations.append(float(np.clip(
                    np.mean(centered[first] * centered[second])
                    / (norms[first] * norms[second]),
                    -1.0,
                    1.0,
                )))

        whitening[condition_index, valid_indices] = local_whitening.astype(np.float32)
        valid_masks[condition_index, valid_indices] = 1.0
        active_masks[condition_index, valid_indices] = (
            pointwise_std >= active_threshold
        ).astype(np.float32)
        training_counts.append(int(members.size))
        distance_medians.append(float(np.median(distances)))
        correlation_medians.append(float(np.median(correlations)))


    # --------------------------------------------------------
    # D4.3.2.19
    #
    # Condition-specific train-reference standardized tail
    # with variance stratification.
    #
    # 1. 使用与正式QQ evaluator完全一致的train-only标准化：
    #       z(r) = [x(r) - train_mean(r)] / train_scale(r)
    #    其中train_scale(r)=max(pointwise_std, condition Q10 std floor)。
    # 2. 再按真实pointwise training std的秩将Raman点等数量分成5层。
    # 3. 每一层独立保存q01/q10/q50/q90/q99。
    #
    # 这样低方差Raman区的过分散不能再被高方差峰区的欠分散抵消。
    # 所有condition共用完全相同规则，只使用各自training数据。
    # --------------------------------------------------------
    full_spectrum_mean = None
    full_spectrum_tail_probabilities = None
    full_spectrum_tail_quantiles = None
    full_spectrum_variance_stratum_index = None
    full_spectrum_stratified_tail_quantiles = None
    number_of_variance_strata = 5

    if full_spectrum_profile_enabled:
        full_spectrum_mean = np.zeros(
            (number_of_conditions, length),
            dtype=np.float32,
        )
        full_spectrum_tail_probabilities = np.asarray(
            [0.01, 0.10, 0.50, 0.90, 0.99],
            dtype=np.float64,
        )
        full_spectrum_tail_quantiles = np.zeros(
            (number_of_conditions, 5),
            dtype=np.float32,
        )
        full_spectrum_variance_stratum_index = np.full(
            (number_of_conditions, length),
            -1,
            dtype=np.int16,
        )
        full_spectrum_stratified_tail_quantiles = np.zeros(
            (number_of_conditions, number_of_variance_strata, 5),
            dtype=np.float32,
        )

        for condition_index in range(number_of_conditions):
            selected_condition = inverse == condition_index
            condition_count = int(np.sum(selected_condition))
            if condition_count < 2:
                raise ValueError(
                    "D4.3.2.19每个condition至少需要2条training光谱。"
                )

            valid_indices = np.flatnonzero(
                valid_masks[condition_index] > 0.5
            )
            if valid_indices.size < number_of_variance_strata:
                raise ValueError(
                    "D4.3.2.19 condition有效Raman点数量不足。"
                )

            condition_full_spectra = full_spectra[
                selected_condition
            ][:, valid_indices]
            condition_mean = np.mean(
                condition_full_spectra,
                axis=0,
            )
            condition_scale = np.maximum(
                full_spectrum_scale[condition_index, valid_indices],
                1.0e-6,
            )
            standardized = (
                condition_full_spectra - condition_mean[None, :]
            ) / condition_scale[None, :]
            if not np.isfinite(standardized).all():
                raise ValueError(
                    "D4.3.2.19标准化training full spectra出现NaN或Inf。"
                )

            full_spectrum_mean[
                condition_index, valid_indices
            ] = condition_mean.astype(np.float32)

            # 保留D4.18 pooled reference用于旧配置兼容。
            full_spectrum_tail_quantiles[condition_index] = np.quantile(
                standardized.reshape(-1),
                full_spectrum_tail_probabilities,
            ).astype(np.float32)

            # 用真实pointwise std排序，而不是用加floor后的scale排序。
            # rank-based等数量分层对std并列值也稳定，不会产生空层。
            condition_pointwise_std = full_spectrum_std[
                condition_index, valid_indices
            ].astype(np.float64)
            order = np.argsort(condition_pointwise_std, kind="stable")
            local_strata = np.empty(valid_indices.size, dtype=np.int16)
            ranks = np.arange(valid_indices.size, dtype=np.int64)
            ordered_strata = np.minimum(
                ranks * number_of_variance_strata // valid_indices.size,
                number_of_variance_strata - 1,
            ).astype(np.int16)
            local_strata[order] = ordered_strata

            full_spectrum_variance_stratum_index[
                condition_index, valid_indices
            ] = local_strata

            for stratum_index in range(number_of_variance_strata):
                stratum_mask = local_strata == stratum_index
                if not np.any(stratum_mask):
                    raise RuntimeError(
                        "D4.3.2.19出现空variance stratum。"
                    )
                full_spectrum_stratified_tail_quantiles[
                    condition_index, stratum_index
                ] = np.quantile(
                    standardized[:, stratum_mask].reshape(-1),
                    full_spectrum_tail_probabilities,
                ).astype(np.float32)

    state = {
        "schema_version": D4_3_STATE_SCHEMA_VERSION,
        "method_version": D4_3_METHOD_VERSION,
        "enabled": True,
        "configuration": config,
        "number_of_conditions": number_of_conditions,
        "condition_vectors": unique_conditions.astype(np.float32).tolist(),
        "training_counts": training_counts,
        "original_length": length,
        "valid_masks": valid_masks.tolist(),
        "whitening_std": whitening.tolist(),
        "active_masks": active_masks.tolist(),
        "training_pairwise_distance_median": distance_medians,
        "training_pairwise_correlation_median": correlation_medians,
        "full_spectrum_variance_profile_enabled": bool(
            full_spectrum_profile_enabled
        ),
    }

    if full_spectrum_profile_enabled:
        if (
            full_spectrum_std is None
            or full_spectrum_scale is None
        ):
            raise RuntimeError(
                "full-spectrum variance profile拟合未完成。"
            )

        state.update(
            {
                "full_spectrum_scale_floor_quantile": (
                    full_spectrum_scale_floor_quantile
                ),
                "full_spectrum_scale_floors": (
                    full_spectrum_scale_floors
                ),
                "full_spectrum_std": (
                    full_spectrum_std.tolist()
                ),
                "full_spectrum_scale": (
                    full_spectrum_scale.tolist()
                ),
            }
        )


    # D4.3.2.18固定train-only tail reference。
    if full_spectrum_profile_enabled:
        state["full_spectrum_mean"] = (
            full_spectrum_mean.tolist()
        )

        state[
            "full_spectrum_tail_probabilities"
        ] = (
            full_spectrum_tail_probabilities.tolist()
        )

        state[
            "full_spectrum_tail_quantiles"
        ] = (
            full_spectrum_tail_quantiles.tolist()
        )

        state["full_spectrum_variance_stratum_count"] = int(
            number_of_variance_strata
        )
        state["full_spectrum_variance_stratum_index"] = (
            full_spectrum_variance_stratum_index.tolist()
        )
        state["full_spectrum_stratified_tail_quantiles"] = (
            full_spectrum_stratified_tail_quantiles.tolist()
        )

    return state


def evaluate_condition_aware_diversity_residuals(
    *,
    generated_scaled_residuals: np.ndarray,
    condition_vector: np.ndarray,
    diversity_constraint_state: dict[str, Any],
) -> dict[str, float | int | str]:
    """Measure generated residual geometry in the same domain as D4.3.

    The generated samples are not paired with individual training samples, so
    the diagnostic compares distribution summaries with the train-only state.
    Filtering, valid-point masking and whitening exactly match the loss domain.
    """

    state = diversity_constraint_state
    if int(state.get("schema_version", 0)) != D4_3_STATE_SCHEMA_VERSION:
        raise ValueError("D4.3 diversity state版本不受支持。")
    if state.get("method_version") != D4_3_METHOD_VERSION:
        raise ValueError("D4.3 diversity state方法版本不受支持。")
    if not bool(state.get("enabled", False)):
        raise ValueError("D4.3 diversity state没有启用。")

    config = normalize_condition_aware_diversity_configuration(
        state["configuration"]
    )
    generated = _as_spectra(generated_scaled_residuals)
    original_length = int(state["original_length"])
    if generated.shape[1] != original_length:
        raise ValueError(
            "生成scaled residual长度与D4.3状态不一致："
            f"{generated.shape[1]} != {original_length}。"
        )
    if generated.shape[0] < 2:
        raise ValueError("D4.3多样性诊断至少需要2条生成残差。")

    condition = np.asarray(condition_vector, dtype=np.float32).reshape(-1)
    known_conditions = np.asarray(state["condition_vectors"], dtype=np.float32)
    if known_conditions.ndim != 2 or condition.size != known_conditions.shape[1]:
        raise ValueError("condition_vector维度与D4.3状态不一致。")
    matches = np.isclose(
        known_conditions,
        condition[None, :],
        rtol=0.0,
        atol=1.0e-6,
    ).all(axis=1)
    if int(matches.sum()) != 1:
        raise ValueError("condition_vector在D4.3状态中不存在或不唯一。")
    condition_index = int(np.flatnonzero(matches)[0])

    valid_mask = np.asarray(
        state["valid_masks"][condition_index], dtype=np.float64
    )
    active_mask = np.asarray(
        state["active_masks"][condition_index], dtype=np.float64
    )
    whitening = np.asarray(
        state["whitening_std"][condition_index], dtype=np.float64
    )
    valid_indices = np.flatnonzero(valid_mask > 0.5)
    if valid_indices.size < 9:
        raise ValueError("D4.3状态中的有效Raman点不足9个。")

    high_config = config["high_frequency_filter"]
    active = generated[:, valid_indices]
    high_frequency = active - gaussian_filter1d(
        active,
        sigma=float(high_config["smoothing_sigma_points"]),
        axis=1,
        mode="reflect",
        truncate=float(high_config["kernel_truncate"]),
    )
    epsilon = float(config["epsilon"])
    local_whitening = np.maximum(whitening[valid_indices], epsilon)
    whitened = high_frequency / local_whitening[None, :]

    distances: list[float] = []
    for first in range(whitened.shape[0] - 1):
        difference = whitened[first + 1 :] - whitened[first]
        distances.extend(
            np.sqrt(np.mean(np.square(difference), axis=1)).tolist()
        )
    generated_distance = float(np.median(distances))
    training_distance = float(
        state["training_pairwise_distance_median"][condition_index]
    )
    distance_ratio = generated_distance / max(training_distance, epsilon)

    centered = whitened - whitened.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    normalized = centered / np.maximum(norms[:, None], epsilon)
    correlations = normalized @ normalized.T
    generated_correlation = float(
        np.median(correlations[np.triu_indices(whitened.shape[0], k=1)])
    )
    training_correlation = float(
        state["training_pairwise_correlation_median"][condition_index]
    )
    allowed_excess = float(
        config["pairwise_correlation"]["maximum_excess_correlation"]
    )

    generated_std = np.std(high_frequency, axis=0, ddof=1)
    std_ratio = generated_std / local_whitening
    active_valid = active_mask[valid_indices] > 0.5
    if not np.any(active_valid):
        raise ValueError("D4.3状态中没有有效的active variance点。")
    active_std_ratios = std_ratio[active_valid]

    return {
        "method_version": D4_3_METHOD_VERSION,
        "condition_index": condition_index,
        "generated_sample_count": int(generated.shape[0]),
        "valid_point_count": int(valid_indices.size),
        "generated_pairwise_distance_median": generated_distance,
        "training_pairwise_distance_median": training_distance,
        "pairwise_distance_ratio": float(distance_ratio),
        "generated_pairwise_correlation_median": generated_correlation,
        "training_pairwise_correlation_median": training_correlation,
        "pairwise_correlation_excess": float(
            generated_correlation - training_correlation
        ),
        "pairwise_correlation_violation": float(
            max(generated_correlation - training_correlation - allowed_excess, 0.0)
        ),
        "active_pointwise_std_ratio_median": float(
            np.median(active_std_ratios)
        ),
        "active_pointwise_std_ratio_p10": float(
            np.percentile(active_std_ratios, 10.0)
        ),
        "active_pointwise_std_below_floor_fraction": float(
            np.mean(
                active_std_ratios
                < float(
                    config["pointwise_variance_floor"]["minimum_std_ratio"]
                )
            )
        ),
    }



class DifferentiableConditionFullSpectrumGroupVarianceLoss(
    nn.Module
):
    """D4.3.2.17：condition内Raman逐点方差匹配。

    与D4.3.2.16不同：

    D4.3.2.16：
        约束单条预测光谱相对单条target的标准化重建误差。

    D4.3.2.17：
        对同一condition的一组预测光谱计算逐Raman点sample std，
        直接与该condition全部training光谱拟合得到的reference std比较。

    目标：
        1. 抑制稳定Raman区域的过度随机变化；
        2. 防止高自然方差区域被压得过窄；
        3. 直接约束variance allocation，而不是单谱reconstruction error。
    """

    def __init__(
        self,
        *,
        diversity_constraint_state: dict[str, Any],
        padded_length: int,
        configuration: dict[str, Any],
    ) -> None:
        super().__init__()

        state = diversity_constraint_state

        required_keys = (
            "condition_vectors",
            "valid_masks",
            "full_spectrum_std",
            "full_spectrum_scale",
            "original_length",
        )

        for name in required_keys:
            if name not in state:
                raise ValueError(
                    "D4.3.2.17 train-only variance state缺少"
                    f"{name}。"
                )

        self.minimum_group_size = int(
            configuration.get(
                "minimum_group_size",
                4,
            )
        )

        self.maximum_timestep_fraction = float(
            configuration.get(
                "maximum_timestep_fraction",
                0.50,
            )
        )

        # 允许少量统计误差。
        #
        # 因为每个训练batch中同condition目前只有约4条光谱，
        # group sample std本身存在采样波动，所以不能要求严格等于
        # 12条training拟合得到的reference std。
        self.overdispersion_deadband = float(
            configuration.get(
                "overdispersion_deadband",
                0.25,
            )
        )

        self.underdispersion_deadband = float(
            configuration.get(
                "underdispersion_deadband",
                0.20,
            )
        )

        self.smooth_l1_beta = float(
            configuration.get(
                "smooth_l1_beta",
                0.05,
            )
        )

        self.overdispersion_weight = float(
            configuration.get(
                "overdispersion_weight",
                1.0,
            )
        )

        self.underdispersion_weight = float(
            configuration.get(
                "underdispersion_weight",
                1.0,
            )
        )

        self.epsilon = float(
            configuration.get(
                "epsilon",
                1.0e-6,
            )
        )

        if self.minimum_group_size < 3:
            raise ValueError(
                "minimum_group_size必须至少为3。"
            )

        if not (
            0.0
            < self.maximum_timestep_fraction
            <= 1.0
        ):
            raise ValueError(
                "maximum_timestep_fraction必须位于(0,1]。"
            )

        if (
            self.overdispersion_deadband < 0.0
            or self.underdispersion_deadband < 0.0
        ):
            raise ValueError(
                "variance deadband不能小于0。"
            )

        if self.smooth_l1_beta <= 0.0:
            raise ValueError(
                "smooth_l1_beta必须大于0。"
            )

        if (
            self.overdispersion_weight <= 0.0
            or self.underdispersion_weight <= 0.0
        ):
            raise ValueError(
                "variance方向权重必须大于0。"
            )

        if self.epsilon <= 0.0:
            raise ValueError(
                "epsilon必须大于0。"
            )

        original_length = int(
            state["original_length"]
        )

        padded_length = int(
            padded_length
        )

        if padded_length < original_length:
            raise ValueError(
                "padded_length不能小于state中的original_length。"
            )

        condition_vectors = np.asarray(
            state["condition_vectors"],
            dtype=np.float32,
        )

        valid_masks = np.asarray(
            state["valid_masks"],
            dtype=np.float32,
        )

        reference_std = np.asarray(
            state["full_spectrum_std"],
            dtype=np.float32,
        )

        reference_scale = np.asarray(
            state["full_spectrum_scale"],
            dtype=np.float32,
        )

        if condition_vectors.ndim != 2:
            raise ValueError(
                "condition_vectors必须为二维数组。"
            )

        if (
            valid_masks.ndim != 2
            or reference_std.ndim != 2
            or reference_scale.ndim != 2
        ):
            raise ValueError(
                "variance reference数组必须为二维。"
            )

        if not (
            valid_masks.shape
            == reference_std.shape
            == reference_scale.shape
        ):
            raise ValueError(
                "valid mask/std/scale形状必须一致。"
            )

        if (
            condition_vectors.shape[0]
            != valid_masks.shape[0]
        ):
            raise ValueError(
                "condition数量与variance reference数量不一致。"
            )

        if (
            valid_masks.shape[1]
            != original_length
        ):
            raise ValueError(
                "variance reference长度与original_length不一致。"
            )

        if not np.isfinite(
            reference_std
        ).all():
            raise ValueError(
                "reference std包含NaN或无穷值。"
            )

        if not np.isfinite(
            reference_scale
        ).all():
            raise ValueError(
                "reference scale包含NaN或无穷值。"
            )

        if np.any(
            reference_std < 0.0
        ):
            raise ValueError(
                "reference std不能小于0。"
            )

        if np.any(
            reference_scale <= 0.0
        ):
            raise ValueError(
                "reference scale必须大于0。"
            )

        number_of_conditions = int(
            condition_vectors.shape[0]
        )

        padded_masks = np.zeros(
            (
                number_of_conditions,
                padded_length,
            ),
            dtype=np.float32,
        )

        padded_std = np.zeros(
            (
                number_of_conditions,
                padded_length,
            ),
            dtype=np.float32,
        )

        padded_scale = np.ones(
            (
                number_of_conditions,
                padded_length,
            ),
            dtype=np.float32,
        )

        padded_masks[
            :,
            :original_length,
        ] = valid_masks

        padded_std[
            :,
            :original_length,
        ] = reference_std

        padded_scale[
            :,
            :original_length,
        ] = reference_scale

        self.register_buffer(
            "condition_vectors",
            torch.as_tensor(
                condition_vectors,
                dtype=torch.float32,
            ),
            persistent=False,
        )

        self.register_buffer(
            "condition_valid_masks",
            torch.as_tensor(
                padded_masks,
                dtype=torch.float32,
            ),
            persistent=False,
        )

        self.register_buffer(
            "condition_reference_std",
            torch.as_tensor(
                padded_std,
                dtype=torch.float32,
            ),
            persistent=False,
        )

        self.register_buffer(
            "condition_reference_scale",
            torch.as_tensor(
                padded_scale,
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def _resolve_condition_indices(
        self,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """把batch中的14维condition映射到train-only state索引。"""

        if condition.ndim != 2:
            raise ValueError(
                "condition必须为[B,C]。"
            )

        reference = self.condition_vectors.to(
            device=condition.device,
            dtype=condition.dtype,
        )

        if (
            condition.shape[1]
            != reference.shape[1]
        ):
            raise ValueError(
                "condition向量维数与train-only state不一致。"
            )

        matches = torch.isclose(
            condition.unsqueeze(1),
            reference.unsqueeze(0),
            rtol=0.0,
            atol=1.0e-6,
        ).all(
            dim=2
        )

        match_counts = matches.sum(
            dim=1
        )

        if not torch.all(
            match_counts == 1
        ):
            raise ValueError(
                "batch存在无法唯一匹配train-only state的condition。"
            )

        return matches.to(
            dtype=torch.int64
        ).argmax(
            dim=1
        )

    def forward(
        self,
        *,
        prediction: torch.Tensor,
        target: torch.Tensor,
        full_spectrum_target: torch.Tensor,
        local_inverse_slope: torch.Tensor,
        valid_mask: torch.Tensor,
        condition: torch.Tensor,
        timesteps: torch.Tensor | None,
        number_of_timesteps: int,
    ) -> dict[str, torch.Tensor]:
        """计算condition内逐Raman点variance matching loss。"""

        if (
            prediction.ndim != 3
            or prediction.shape[1] != 1
        ):
            raise ValueError(
                "prediction必须为[B,1,L]。"
            )

        for name, value in (
            (
                "target",
                target,
            ),
            (
                "full_spectrum_target",
                full_spectrum_target,
            ),
            (
                "local_inverse_slope",
                local_inverse_slope,
            ),
            (
                "valid_mask",
                valid_mask,
            ),
        ):
            if value.shape != prediction.shape:
                raise ValueError(
                    f"{name}形状必须与prediction一致。"
                )

        if (
            condition.ndim != 2
            or condition.shape[0]
            != prediction.shape[0]
        ):
            raise ValueError(
                "condition必须为[B,C]并与batch对应。"
            )

        if timesteps is not None:
            if (
                timesteps.ndim != 1
                or timesteps.shape[0]
                != prediction.shape[0]
            ):
                raise ValueError(
                    "timesteps必须为[B]。"
                )

        number_of_timesteps = int(
            number_of_timesteps
        )

        if number_of_timesteps <= 0:
            raise ValueError(
                "number_of_timesteps必须大于0。"
            )

        condition_indices = (
            self._resolve_condition_indices(
                condition
            )
        )

        # 与D4.3.2.16保持相同full-spectrum近似重建关系：
        #
        # full_pred =
        # full_target
        # + inverse_local_slope * (pred_local - target_local)
        reconstructed_prediction = (
            full_spectrum_target
            + local_inverse_slope
            * (
                prediction
                - target
            )
        )

        reconstructed_prediction = (
            reconstructed_prediction
            * valid_mask
        )

        zero = (
            prediction.sum()
            * 0.0
        )

        group_losses: list[
            torch.Tensor
        ] = []

        mean_abs_z_values: list[
            torch.Tensor
        ] = []

        over_fraction_values: list[
            torch.Tensor
        ] = []

        under_fraction_values: list[
            torch.Tensor
        ] = []

        # D4.23.1 same-condition overdispersion gate
        #
        # 每个same-condition group最终得到一个0~1 gate，
        # 再映射回该group内的每个sample。
        # gate本身不参与梯度传播。
        sample_overdispersion_gate = prediction.new_zeros(
            (
                prediction.shape[0],
            )
        )

        active_group_count = 0

        unique_indices = torch.unique(
            condition_indices,
            sorted=True,
        )

        for condition_index_tensor in unique_indices:

            condition_index = int(
                condition_index_tensor.detach().cpu()
            )

            selected = (
                condition_indices
                == condition_index
            )

            group_size = int(
                selected.sum().detach().cpu()
            )

            if (
                group_size
                < self.minimum_group_size
            ):
                continue

            # 目前condition grouping会给同condition共享timestep。
            # 这里仍使用该组平均t，兼容未来实现。
            if timesteps is not None:

                timestep_fraction = (
                    timesteps[
                        selected
                    ]
                    .to(
                        dtype=torch.float32
                    )
                    .mean()
                    / float(
                        max(
                            number_of_timesteps - 1,
                            1,
                        )
                    )
                )

                if float(
                    timestep_fraction.detach().cpu()
                ) > self.maximum_timestep_fraction:
                    continue

            group_prediction = (
                reconstructed_prediction[
                    selected
                ]
            )

            group_mask = (
                valid_mask[
                    selected
                ]
            )

            # 一个condition组内所有样本都有效的Raman点才参与。
            common_mask = torch.min(
                group_mask,
                dim=0,
            ).values

            state_mask = (
                self.condition_valid_masks[
                    condition_index
                ]
                .view(
                    1,
                    -1,
                )
                .to(
                    device=prediction.device,
                    dtype=prediction.dtype,
                )
            )

            common_mask = (
                common_mask
                * state_mask
            )

            valid_count = (
                common_mask.sum()
            )

            if float(
                valid_count.detach().cpu()
            ) <= 1.0:
                continue

            # 与12条training reference采用相同sample-std思想。
            predicted_std = torch.std(
                group_prediction,
                dim=0,
                unbiased=True,
            )

            reference_std = (
                self.condition_reference_std[
                    condition_index
                ]
                .view(
                    1,
                    -1,
                )
                .to(
                    device=prediction.device,
                    dtype=prediction.dtype,
                )
            )

            # 关键：
            # 不直接除以reference_std。
            #
            # D4.3.2.16已经拟合：
            # scale = max(pointwise train std,
            #             condition内train std的Q10 floor)
            #
            # 从而避免稳定背景点std接近0时ratio爆炸。
            reference_scale = (
                self.condition_reference_scale[
                    condition_index
                ]
                .view(
                    1,
                    -1,
                )
                .to(
                    device=prediction.device,
                    dtype=prediction.dtype,
                )
                .clamp_min(
                    self.epsilon
                )
            )

            standardized_difference = (
                predicted_std
                - reference_std
            ) / reference_scale

            # 双向variance约束：
            #
            # 正值过大：
            # 生成在稳定区过分散。
            over_violation = F.relu(
                standardized_difference
                - self.overdispersion_deadband
            )

            # 负值绝对值过大：
            # 生成在真实高方差区欠分散。
            under_violation = F.relu(
                -standardized_difference
                - self.underdispersion_deadband
            )

            over_loss = F.smooth_l1_loss(
                over_violation,
                torch.zeros_like(
                    over_violation
                ),
                reduction="none",
                beta=self.smooth_l1_beta,
            )

            under_loss = F.smooth_l1_loss(
                under_violation,
                torch.zeros_like(
                    under_violation
                ),
                reduction="none",
                beta=self.smooth_l1_beta,
            )

            pointwise_loss = (
                self.overdispersion_weight
                * over_loss
                + self.underdispersion_weight
                * under_loss
            )

            group_loss = (
                pointwise_loss
                * common_mask
            ).sum() / valid_count

            mean_abs_z = (
                standardized_difference.abs()
                * common_mask
            ).sum() / valid_count

            over_fraction = (
                (
                    standardized_difference
                    > self.overdispersion_deadband
                )
                .to(
                    dtype=prediction.dtype
                )
                * common_mask
            ).sum() / valid_count

            under_fraction = (
                (
                    standardized_difference
                    < -self.underdispersion_deadband
                )
                .to(
                    dtype=prediction.dtype
                )
                * common_mask
            ).sum() / valid_count

            # ------------------------------------------------
            # D4.23.1：
            # 使用已有D4.17双向variance诊断决定tail loss是否应该开启。
            #
            # over <= under：
            #   当前group不存在“过宽占主导”，gate=0。
            #
            # over >> under：
            #   当前group明显过宽，gate逐渐接近1。
            #
            # detach防止模型通过操纵gate逃避tail loss。
            # ------------------------------------------------
            gate_numerator = F.relu(
                over_fraction
                - under_fraction
            )

            gate_denominator = (
                over_fraction
                + under_fraction
            ).clamp_min(
                self.epsilon
            )

            group_overdispersion_gate = (
                gate_numerator
                / gate_denominator
            ).clamp(
                min=0.0,
                max=1.0,
            ).detach()

            sample_overdispersion_gate[
                selected
            ] = group_overdispersion_gate

            group_losses.append(
                group_loss
            )

            mean_abs_z_values.append(
                mean_abs_z
            )

            over_fraction_values.append(
                over_fraction
            )

            under_fraction_values.append(
                under_fraction
            )

            active_group_count += 1

        if not group_losses:
            return {
                "loss": zero,
                "active_groups": zero,
                "mean_abs_standardized_difference": zero,
                "overdispersion_fraction": zero,
                "underdispersion_fraction": zero,
                "sample_overdispersion_gate": (
                    sample_overdispersion_gate
                ),
            }

        return {
            "loss": torch.stack(
                group_losses
            ).mean(),

            "active_groups": (
                prediction.new_tensor(
                    float(
                        active_group_count
                    )
                )
            ),

            "mean_abs_standardized_difference": (
                torch.stack(
                    mean_abs_z_values
                ).mean()
            ),

            "overdispersion_fraction": (
                torch.stack(
                    over_fraction_values
                ).mean()
            ),

            "underdispersion_fraction": (
                torch.stack(
                    under_fraction_values
                ).mean()
            ),

            "sample_overdispersion_gate": (
                sample_overdispersion_gate
            ),
        }


class DifferentiableConditionFullSpectrumVarianceProfileLoss(
    nn.Module
):
    """Condition-specific train-only Raman variance-profile fidelity.

    这里不使用当前4条同condition batch去重新估计参考方差。
    每个Raman点的尺度全部来自该condition的全部training光谱。
    稳定Raman点的允许误差更小，高自然方差区域允许更大的实验变化。
    """

    def __init__(
        self,
        *,
        diversity_constraint_state: dict[str, Any],
        padded_length: int,
        configuration: dict[str, Any],
    ) -> None:
        super().__init__()

        state = diversity_constraint_state

        if not bool(
            state.get(
                "full_spectrum_variance_profile_enabled",
                False,
            )
        ):
            raise ValueError(
                "当前diversity state缺少"
                "train-only full-spectrum variance profile。"
            )

        for name in (
            "condition_vectors",
            "valid_masks",
            "full_spectrum_scale",
            "original_length",
        ):
            if name not in state:
                raise ValueError(
                    "full-spectrum variance profile状态缺少"
                    f"{name}。"
                )

        self.maximum_timestep_fraction = float(
            configuration.get(
                "maximum_timestep_fraction",
                0.50,
            )
        )

        self.standardized_error_deadband = float(
            configuration.get(
                "standardized_error_deadband",
                0.25,
            )
        )

        self.smooth_l1_beta = float(
            configuration.get(
                "smooth_l1_beta",
                0.10,
            )
        )

        self.epsilon = float(
            configuration.get(
                "epsilon",
                1.0e-6,
            )
        )

        original_length = int(
            state["original_length"]
        )

        padded_length = int(
            padded_length
        )

        if (
            padded_length
            < original_length
        ):
            raise ValueError(
                "padded_length不能小于variance profile原始长度。"
            )

        conditions = np.asarray(
            state["condition_vectors"],
            dtype=np.float32,
        )

        masks = np.asarray(
            state["valid_masks"],
            dtype=np.float32,
        )

        scales = np.asarray(
            state["full_spectrum_scale"],
            dtype=np.float32,
        )

        if (
            conditions.ndim != 2
            or masks.ndim != 2
            or scales.ndim != 2
        ):
            raise ValueError(
                "variance profile状态数组维度错误。"
            )

        if (
            masks.shape
            != scales.shape
            or masks.shape[0]
            != conditions.shape[0]
            or masks.shape[1]
            != original_length
        ):
            raise ValueError(
                "variance profile状态数组形状不一致。"
            )

        padded_masks = np.zeros(
            (
                masks.shape[0],
                padded_length,
            ),
            dtype=np.float32,
        )

        padded_scales = np.ones(
            (
                scales.shape[0],
                padded_length,
            ),
            dtype=np.float32,
        )

        padded_masks[
            :,
            :original_length,
        ] = masks

        padded_scales[
            :,
            :original_length,
        ] = scales

        self.register_buffer(
            "condition_vectors",
            torch.as_tensor(
                conditions,
                dtype=torch.float32,
            ),
            persistent=False,
        )

        self.register_buffer(
            "condition_valid_masks",
            torch.as_tensor(
                padded_masks,
                dtype=torch.float32,
            ),
            persistent=False,
        )

        self.register_buffer(
            "condition_full_spectrum_scale",
            torch.as_tensor(
                padded_scales,
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def _condition_indices(
        self,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        matches = torch.isclose(
            condition.unsqueeze(1),
            self.condition_vectors.unsqueeze(0),
            rtol=0.0,
            atol=1.0e-6,
        ).all(
            dim=2
        )

        if not torch.all(
            matches.sum(dim=1)
            == 1
        ):
            raise ValueError(
                "batch包含variance profile状态中"
                "不存在或重复的condition。"
            )

        return matches.to(
            dtype=torch.int64
        ).argmax(
            dim=1
        )

    def forward(
        self,
        *,
        prediction: torch.Tensor,
        target: torch.Tensor,
        full_spectrum_target: torch.Tensor,
        local_inverse_slope: torch.Tensor,
        valid_mask: torch.Tensor,
        condition: torch.Tensor,
        timesteps: torch.Tensor | None,
        number_of_timesteps: int,
    ) -> dict[str, torch.Tensor]:

        for name, value in (
            ("target", target),
            (
                "full_spectrum_target",
                full_spectrum_target,
            ),
            (
                "local_inverse_slope",
                local_inverse_slope,
            ),
            (
                "valid_mask",
                valid_mask,
            ),
        ):
            if (
                value.shape
                != prediction.shape
            ):
                raise ValueError(
                    f"{name}形状必须与prediction一致。"
                )

        if (
            prediction.ndim != 3
            or prediction.shape[1] != 1
        ):
            raise ValueError(
                "variance profile输入必须为[B,1,L]。"
            )

        if (
            condition.ndim != 2
            or condition.shape[0]
            != prediction.shape[0]
        ):
            raise ValueError(
                "variance profile condition形状无效。"
            )

        condition_indices = (
            self._condition_indices(
                condition
            )
        )

        expected_mask = (
            self.condition_valid_masks[
                condition_indices
            ]
            .unsqueeze(1)
            .to(
                device=prediction.device,
                dtype=prediction.dtype,
            )
        )

        if not torch.equal(
            valid_mask,
            expected_mask,
        ):
            raise ValueError(
                "variance profile batch有效Raman区"
                "与train-only状态不一致。"
            )

        scale = (
            self.condition_full_spectrum_scale[
                condition_indices
            ]
            .unsqueeze(1)
            .to(
                device=prediction.device,
                dtype=prediction.dtype,
            )
            .clamp_min(
                self.epsilon
            )
        )

        reconstructed_prediction = (
            full_spectrum_target
            + local_inverse_slope
            * (
                prediction
                - target
            )
        ) * valid_mask

        standardized_error = (
            (
                reconstructed_prediction
                - full_spectrum_target
            )
            / scale
        )

        absolute_standardized_error = (
            standardized_error.abs()
        )

        violation = F.relu(
            absolute_standardized_error
            - self.standardized_error_deadband
        )

        active_sample_mask = torch.ones(
            (
                prediction.shape[0],
                1,
                1,
            ),
            device=prediction.device,
            dtype=prediction.dtype,
        )

        if timesteps is not None:
            if timesteps.shape != (
                prediction.shape[0],
            ):
                raise ValueError(
                    "variance profile timesteps形状无效。"
                )

            fraction = (
                timesteps.to(
                    dtype=torch.float32
                )
                / float(
                    max(
                        int(number_of_timesteps)
                        - 1,
                        1,
                    )
                )
            )

            active_sample_mask = (
                fraction
                <= self.maximum_timestep_fraction
            ).to(
                dtype=prediction.dtype
            ).reshape(
                -1,
                1,
                1,
            )

        active_mask = (
            valid_mask
            * active_sample_mask
        )

        active_count = (
            active_mask.sum()
        )

        zero = (
            prediction.sum()
            * 0.0
        )

        if bool(
            (
                active_count
                <= 0.0
            ).item()
        ):
            return {
                "loss": zero,
                "active_sample_fraction": zero,
                "mean_abs_standardized_error": zero,
                "violation_fraction": zero,
            }

        pointwise = F.smooth_l1_loss(
            violation,
            torch.zeros_like(
                violation
            ),
            reduction="none",
            beta=self.smooth_l1_beta,
        )

        loss = (
            pointwise
            * active_mask
        ).sum() / active_count

        mean_abs_error = (
            absolute_standardized_error
            * active_mask
        ).sum() / active_count

        violation_fraction = (
            (
                violation
                > 0.0
            )
            .to(
                dtype=prediction.dtype
            )
            * active_mask
        ).sum() / active_count

        active_sample_fraction = (
            active_sample_mask.mean()
        )

        return {
            "loss": loss,
            "active_sample_fraction": active_sample_fraction,
            "mean_abs_standardized_error": mean_abs_error,
            "violation_fraction": violation_fraction,
        }


class DifferentiableConditionAwareDiversityLoss(nn.Module):
    """Match within-condition residual geometry on valid Raman points only."""

    def __init__(
        self,
        *,
        diversity_constraint_state: dict[str, Any],
        padded_length: int,
    ) -> None:
        super().__init__()
        state = diversity_constraint_state
        if int(state.get("schema_version", 0)) != D4_3_STATE_SCHEMA_VERSION:
            raise ValueError("D4.3 diversity state版本不受支持。")
        if state.get("method_version") != D4_3_METHOD_VERSION:
            raise ValueError("D4.3 diversity state方法版本不受支持。")
        if not bool(state.get("enabled", False)):
            raise ValueError("D4.3 diversity state没有启用。")

        self.configuration = normalize_condition_aware_diversity_configuration(
            state["configuration"]
        )
        self.original_length = int(state["original_length"])
        self.padded_length = int(padded_length)
        if self.padded_length < self.original_length:
            raise ValueError("padded_length不能小于D4.3状态长度。")

        def padded(values: Any, fill: float) -> torch.Tensor:
            tensor = torch.as_tensor(values, dtype=torch.float32)
            if tensor.ndim != 2 or tensor.shape[1] != self.original_length:
                raise ValueError("D4.3逐条件状态形状无效。")
            if self.padded_length > self.original_length:
                tensor = F.pad(
                    tensor,
                    (0, self.padded_length - self.original_length),
                    value=fill,
                )
            return tensor

        condition_vectors = torch.as_tensor(
            state["condition_vectors"], dtype=torch.float32
        )
        if condition_vectors.ndim != 2:
            raise ValueError("D4.3 condition_vectors必须为二维数组。")
        self.register_buffer("condition_vectors", condition_vectors, persistent=False)
        self.register_buffer(
            "condition_valid_masks", padded(state["valid_masks"], 0.0),
            persistent=False,
        )
        self.register_buffer(
            "condition_whitening_std", padded(state["whitening_std"], 1.0),
            persistent=False,
        )
        self.register_buffer(
            "condition_active_masks", padded(state["active_masks"], 0.0),
            persistent=False,
        )

        high = self.configuration["high_frequency_filter"]
        sigma = float(high["smoothing_sigma_points"])
        truncate = float(high["kernel_truncate"])
        radius = max(1, int(sigma * truncate + 0.5))
        positions = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel = torch.exp(-0.5 * torch.square(positions / sigma))
        kernel = (kernel / kernel.sum()).reshape(1, 1, -1)
        self.register_buffer("smoothing_kernel", kernel, persistent=False)
        self.smoothing_radius = radius
        self.epsilon = float(self.configuration["epsilon"])
        self.minimum_alpha = float(
            self.configuration["low_noise_gate"]["minimum_alpha_cumprod"]
        )
        self.minimum_samples = int(
            self.configuration["low_noise_gate"]["minimum_samples"]
        )

    def _zero(self, reference: torch.Tensor) -> dict[str, torch.Tensor]:
        zero = reference.new_zeros(())
        return {
            "diversity_raw_loss": zero,
            "diversity_timestep_weighted_loss": zero,
            "pairwise_distance_loss": zero,
            "pairwise_correlation_loss": zero,
            "pointwise_variance_floor_loss": zero,
            "diversity_active_samples": zero,
            "conditional_diversity_active_groups": zero,
            "mean_diversity_timestep_weight": zero,
        }

    def _condition_indices(self, condition: torch.Tensor) -> torch.Tensor:
        matches = torch.isclose(
            condition.unsqueeze(1),
            self.condition_vectors.unsqueeze(0),
            rtol=0.0,
            atol=1.0e-6,
        ).all(dim=2)
        if not torch.all(matches.sum(dim=1) == 1):
            raise ValueError("batch包含D4.3训练状态中不存在或重复的条件向量。")
        return matches.to(dtype=torch.int64).argmax(dim=1)

    def _smooth(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        numerator = F.conv1d(
            F.pad(values * mask, (self.smoothing_radius,) * 2, mode="reflect"),
            self.smoothing_kernel,
        )
        denominator = F.conv1d(
            F.pad(mask, (self.smoothing_radius,) * 2, mode="reflect"),
            self.smoothing_kernel,
        ).clamp_min(self.epsilon)
        return numerator / denominator

    @staticmethod
    def _upper_triangle(values: torch.Tensor) -> torch.Tensor:
        count = values.shape[0]
        indices = torch.triu_indices(count, count, offset=1, device=values.device)
        return values[indices[0], indices[1]]

    def _ratio_loss(
        self,
        ratio: torch.Tensor,
        minimum: float,
        maximum: float,
        transition: float,
        penalty_mode: str,
    ) -> torch.Tensor:
        lower_scale = max(minimum * transition, self.epsilon)
        below = F.relu(minimum - ratio) / lower_scale
        if penalty_mode == "floor_only":
            return below.square().mean()
        upper_scale = max(maximum * transition, self.epsilon)
        above = F.relu(ratio - maximum) / upper_scale
        return (below.square() + above.square()).mean()

    def _group_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        condition_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = self.condition_valid_masks[condition_index].reshape(1, 1, -1)
        whitening = self.condition_whitening_std[condition_index].reshape(1, 1, -1)
        active = self.condition_active_masks[condition_index].reshape(1, 1, -1)
        valid_count = mask.sum().clamp_min(1.0)

        predicted_high = (predicted - self._smooth(predicted, mask)) * mask
        target_high = (target - self._smooth(target, mask)) * mask
        predicted_white = predicted_high / whitening.clamp_min(self.epsilon)
        target_white = target_high / whitening.clamp_min(self.epsilon)

        pred_difference = predicted_white[:, None] - predicted_white[None, :]
        target_difference = target_white[:, None] - target_white[None, :]
        pred_distance = torch.sqrt(
            (pred_difference.square() * mask).sum(dim=(2, 3)) / valid_count
            + self.epsilon
        )
        target_distance = torch.sqrt(
            (target_difference.square() * mask).sum(dim=(2, 3)) / valid_count
            + self.epsilon
        )
        distance_ratio = self._upper_triangle(pred_distance) / self._upper_triangle(
            target_distance
        ).clamp_min(self.epsilon)
        distance_cfg = self.configuration["pairwise_distance"]
        distance_loss = self._ratio_loss(
            distance_ratio,
            float(distance_cfg["minimum_distance_ratio"]),
            float(distance_cfg["maximum_distance_ratio"]),
            float(distance_cfg["transition_fraction"]),
            str(distance_cfg["penalty_mode"]),
        )

        def correlation_matrix(values: torch.Tensor) -> torch.Tensor:
            mean = (values * mask).sum(dim=2, keepdim=True) / valid_count
            centered = ((values - mean) * mask).squeeze(1)
            numerator = centered @ centered.transpose(0, 1)
            norm = torch.sqrt(
                centered.square().sum(dim=1).clamp_min(self.epsilon)
            )
            return numerator / (norm[:, None] * norm[None, :]).clamp_min(self.epsilon)

        pred_correlation = self._upper_triangle(correlation_matrix(predicted_white))
        target_correlation = self._upper_triangle(correlation_matrix(target_white))
        corr_cfg = self.configuration["pairwise_correlation"]
        excess = F.relu(
            pred_correlation
            - target_correlation
            - float(corr_cfg["maximum_excess_correlation"])
        )
        corr_scale = max(float(corr_cfg["transition_fraction"]), self.epsilon)
        correlation_loss = (excess / corr_scale).square().mean()

        predicted_std = torch.std(predicted_white, dim=0, unbiased=True)
        target_std = torch.std(target_white, dim=0, unbiased=True)
        active_count = active.sum().clamp_min(1.0)
        variance_cfg = self.configuration["pointwise_variance_floor"]
        target_floor = max(
            float(variance_cfg["reference_std_floor_fraction"]),
            self.epsilon,
        )
        variance_ratio = predicted_std.clamp_min(
            target_floor
        ) / target_std.clamp_min(target_floor)
        minimum = float(variance_cfg["minimum_std_ratio"])
        maximum = float(variance_cfg["maximum_std_ratio"])
        below = F.relu(minimum - variance_ratio) / max(minimum * 0.2, self.epsilon)
        if str(variance_cfg["penalty_mode"]) == "floor_only":
            variance_penalty = below.square()
        else:
            above = F.relu(variance_ratio - maximum) / max(
                maximum * 0.2, self.epsilon
            )
            variance_penalty = below.square() + above.square()
        variance_loss = (variance_penalty * active).sum() / active_count
        return distance_loss, correlation_loss, variance_loss

    def forward(
        self,
        *,
        predicted_scaled_residual: torch.Tensor,
        target_scaled_residual: torch.Tensor,
        timesteps: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        valid_mask: torch.Tensor,
        condition: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        predicted = predicted_scaled_residual
        target = target_scaled_residual
        if predicted.shape != target.shape or predicted.ndim != 3 or predicted.shape[1] != 1:
            raise ValueError("D4.3预测和目标必须为同形状[B,1,L]。")
        if valid_mask.shape != predicted.shape:
            raise ValueError("D4.3 valid_mask形状必须与光谱一致。")
        if condition.ndim != 2 or condition.shape[0] != predicted.shape[0]:
            raise ValueError("D4.3 condition形状必须为[B,C]。")
        if timesteps.shape != (predicted.shape[0],):
            raise ValueError("D4.3 timesteps形状必须为[B]。")

        condition_indices = self._condition_indices(condition)
        alpha = alphas_cumprod.gather(0, timesteps)
        terms: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        active_sample_count = 0
        present_conditions = (
            torch.unique(condition_indices).detach().cpu().tolist()
        )
        for index in present_conditions:
            selected = (condition_indices == index) & (alpha >= self.minimum_alpha)
            count = int(torch.nonzero(selected, as_tuple=False).shape[0])
            if count < self.minimum_samples:
                continue
            expected_mask = self.condition_valid_masks[index].reshape(1, 1, -1)
            if not torch.equal(valid_mask[selected], expected_mask.expand(count, -1, -1)):
                raise ValueError("D4.3 batch有效轴掩码与训练状态不一致。")
            distance, correlation, variance = self._group_loss(
                predicted[selected], target[selected], index
            )
            terms.append((distance, correlation, variance, alpha[selected].sqrt().mean()))
            active_sample_count += count

        if not terms:
            return self._zero(predicted)

        distance_loss = torch.stack([term[0] for term in terms]).mean()
        correlation_loss = torch.stack([term[1] for term in terms]).mean()
        variance_loss = torch.stack([term[2] for term in terms]).mean()
        timestep_weight = torch.stack([term[3] for term in terms]).mean()
        distance_cfg = self.configuration["pairwise_distance"]
        correlation_cfg = self.configuration["pairwise_correlation"]
        variance_cfg = self.configuration["pointwise_variance_floor"]
        weighted: list[torch.Tensor] = []
        weights: list[float] = []
        for enabled, weight, value in (
            (distance_cfg["enabled"], float(distance_cfg["weight"]), distance_loss),
            (correlation_cfg["enabled"], float(correlation_cfg["weight"]), correlation_loss),
            (variance_cfg["enabled"], float(variance_cfg["weight"]), variance_loss),
        ):
            if enabled and weight > 0.0:
                weighted.append(weight * value)
                weights.append(weight)
        raw_loss = sum(weighted) / sum(weights)
        return {
            "diversity_raw_loss": raw_loss,
            "diversity_timestep_weighted_loss": raw_loss * timestep_weight,
            "pairwise_distance_loss": distance_loss,
            "pairwise_correlation_loss": correlation_loss,
            "pointwise_variance_floor_loss": variance_loss,
            "diversity_active_samples": predicted.new_tensor(
                float(active_sample_count)
            ),
            "conditional_diversity_active_groups": predicted.new_tensor(float(len(terms))),
            "mean_diversity_timestep_weight": timestep_weight,
        }
