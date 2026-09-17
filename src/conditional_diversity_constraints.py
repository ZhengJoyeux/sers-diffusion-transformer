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
    grouping["shared_timestep"] = bool(grouping.get("shared_timestep", True))
    if grouping["samples_per_condition"] < 2:
        raise ValueError("condition_grouping.samples_per_condition至少为2。")
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

    unique_conditions, inverse = np.unique(conditions, axis=0, return_inverse=True)
    number_of_conditions = int(unique_conditions.shape[0])
    length = int(spectra.shape[1])
    whitening = np.ones((number_of_conditions, length), dtype=np.float32)
    valid_masks = np.zeros((number_of_conditions, length), dtype=np.float32)
    active_masks = np.zeros((number_of_conditions, length), dtype=np.float32)
    training_counts: list[int] = []
    distance_medians: list[float] = []
    correlation_medians: list[float] = []
    epsilon = float(config["epsilon"])
    high_config = config["high_frequency_filter"]
    variance_config = config["pointwise_variance_floor"]
    minimum_samples = int(config["condition_grouping"]["samples_per_condition"])

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

    return {
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
    }


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
