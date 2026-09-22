"""D4.2 condition-aware, mask-aware D2.6 prior residual bank.

There is one shared conditional DDPM.  This module only stores the small
train-only preprocessing/generation state for every chemical condition.
Each state is fitted on the valid Raman prefix of that condition, so a
600--2000 spectrum never contributes padded values to a 600--2500 prior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from src.broad_local_residual import BroadLocalResidualDecomposer
from src.prior_residual import PriorResidualTransformer


STATE_VERSION = "d4.2_conditional_prior_residual_v1"


def _as_2d_finite(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] < 2:
        raise ValueError(f"{name}必须是非空二维数组[N,L]。")
    if not np.isfinite(array).all():
        raise ValueError(f"{name}包含NaN或无穷值。")
    return array


def _valid_prefix_length(mask: np.ndarray) -> int:
    values = np.asarray(mask, dtype=np.float32).reshape(-1)
    if not np.logical_or(values == 0.0, values == 1.0).all():
        raise ValueError("valid_mask只能包含0和1。")
    length = int(values.sum())
    if length < 2:
        raise ValueError("每个条件至少需要两个有效Raman点。")
    expected = np.zeros_like(values)
    expected[:length] = 1.0
    if not np.array_equal(values, expected):
        raise ValueError("D4.2当前要求有效Raman区间是从轴起点开始的连续前缀。")
    return length


def _build_outer(configuration: dict[str, Any]) -> PriorResidualTransformer:
    low = configuration.get("low_frequency", {}) or {}
    blended = configuration.get("blended_frequency", {}) or {}
    return PriorResidualTransformer(
        prior_method=str(configuration.get("prior_method", "pca_reconstruction")),
        normalization_method=str(
            configuration.get("residual_normalization", "robust_asinh")
        ),
        target_abs_max=float(configuration.get("target_abs_max", 1.0)),
        residual_quantile=float(configuration.get("residual_quantile", 99.5)),
        pointwise_scale_floor_quantile=float(
            configuration.get("pointwise_scale_floor_quantile", 10.0)
        ),
        mad_scale_factor=float(configuration.get("mad_scale_factor", 1.4826)),
        epsilon=float(configuration.get("epsilon", 1.0e-8)),
        pca_explained_variance_ratio=float(
            configuration.get("pca_explained_variance_ratio", 0.95)
        ),
        pca_max_components=configuration.get("pca_max_components", 6),
        pca_sampling_strategy=str(
            configuration.get(
                "pca_sampling_strategy",
                "independent_truncated_gaussian_scores",
            )
        ),
        pca_score_clip_standard_deviations=float(
            configuration.get("pca_score_clip_standard_deviations", 2.5)
        ),
        low_frequency_method=str(low.get("method", "gaussian")),
        low_frequency_sigma_cm1=float(low.get("sigma_cm1", 40.0)),
        low_frequency_truncate=float(low.get("truncate", 4.0)),
        blended_peak_component_ratio=float(
            blended.get("peak_component_ratio", 0.5)
        ),
    )


@dataclass
class ConditionalPriorEntry:
    valid_length: int
    number_of_training_spectra: int
    outer: PriorResidualTransformer
    broad_local: BroadLocalResidualDecomposer
    # D4.21: train-only, condition-specific Raman-wise variance
    # equalization applied AFTER the existing robust-asinh local transform.
    # None means legacy D4.2--D4.20 behavior.
    raman_variance_multiplier: np.ndarray | None = None
    raman_variance_floor: float = 0.0
    raman_variance_reference_scale: float = 1.0


class ConditionalPriorResidualBank:
    """A dictionary of condition-specific D2.6 states, not 126 DDPMs."""

    def __init__(
        self,
        *,
        prior_configuration: dict[str, Any],
        broad_local_configuration: dict[str, Any],
    ) -> None:
        if not isinstance(prior_configuration, dict) or not bool(
            prior_configuration.get("enabled", False)
        ):
            raise ValueError("D4.2要求启用prior_residual。")
        if str(prior_configuration.get("prior_method", "")).strip().lower() != (
            "pca_reconstruction"
        ):
            raise ValueError("D4.2条件先验当前要求pca_reconstruction。")
        if not isinstance(broad_local_configuration, dict) or not bool(
            broad_local_configuration.get("enabled", False)
        ):
            raise ValueError("D4.2要求启用broad_local_residual。")
        self.prior_configuration = dict(prior_configuration)
        self.broad_local_configuration = dict(broad_local_configuration)

        # D4.21: condition-wise Raman heteroscedastic residual
        # variance equalization. It remains disabled for legacy configs.
        variance_equalization = (
            self.prior_configuration.get(
                "raman_variance_equalization",
                {},
            )
            or {}
        )
        if not isinstance(variance_equalization, dict):
            raise TypeError(
                "prior_residual.raman_variance_equalization必须是字典。"
            )

        self.raman_variance_equalization_enabled = bool(
            variance_equalization.get("enabled", False)
        )
        self.raman_variance_floor_quantile = float(
            variance_equalization.get("floor_quantile", 10.0)
        )
        self.raman_variance_reference_quantile = float(
            variance_equalization.get("reference_quantile", 50.0)
        )
        self.raman_variance_minimum_multiplier = float(
            variance_equalization.get("minimum_multiplier", 0.25)
        )
        self.raman_variance_maximum_multiplier = float(
            variance_equalization.get("maximum_multiplier", 4.0)
        )

        # D4.24:
        # Soft Raman-wise variance equalization.
        #
        # power = 1.0:
        #     exactly reproduces the D4.21 hard/full equalization.
        #
        # 0 < power < 1:
        #     preserves part of the original condition-specific
        #     Raman heteroscedasticity instead of fully flattening it.
        self.raman_variance_equalization_power = float(
            variance_equalization.get(
                "equalization_power",
                1.0,
            )
        )

        if not 0.0 <= self.raman_variance_floor_quantile <= 50.0:
            raise ValueError(
                "raman_variance_equalization.floor_quantile必须位于[0,50]。"
            )
        if not 1.0 <= self.raman_variance_reference_quantile <= 99.0:
            raise ValueError(
                "raman_variance_equalization.reference_quantile必须位于[1,99]。"
            )
        if (
            not 0.0 < self.raman_variance_minimum_multiplier <= 1.0
            or not 1.0 <= self.raman_variance_maximum_multiplier
            or self.raman_variance_minimum_multiplier
            > self.raman_variance_maximum_multiplier
        ):
            raise ValueError(
                "raman_variance_equalization multiplier范围无效。"
            )

        if not (
            0.0
            < self.raman_variance_equalization_power
            <= 1.0
        ):
            raise ValueError(
                "raman_variance_equalization.equalization_power"
                "必须位于(0,1]。"
            )

        cross_fit = self.prior_configuration.get("training_cross_fit", {}) or {}
        if not isinstance(cross_fit, dict):
            raise TypeError("prior_residual.training_cross_fit必须是字典。")
        self.training_cross_fit_enabled = bool(cross_fit.get("enabled", False))
        self.training_cross_fit_method = str(
            cross_fit.get("method", "leave_one_out")
        ).strip().lower()
        if self.training_cross_fit_enabled and self.training_cross_fit_method != "leave_one_out":
            raise ValueError("D4.3.2.14目前只支持training_cross_fit.method=leave_one_out。")

        self.model_axis: np.ndarray | None = None
        self.entries: dict[str, ConditionalPriorEntry] = {}
        # 仅训练数据准备阶段使用；绝不保存进checkpoint。
        self._cross_fitted_reference_priors: np.ndarray | None = None
        self._cross_fitted_training_indices: np.ndarray | None = None

    @staticmethod
    def _condition_array(
        condition_ids: Sequence[str], number_of_spectra: int
    ) -> np.ndarray:
        if len(condition_ids) != number_of_spectra:
            raise ValueError("condition_ids数量必须与光谱数量一致。")
        values = np.asarray([str(value) for value in condition_ids], dtype=object)
        if any(not str(value).strip() for value in values):
            raise ValueError("condition_ids不能包含空值。")
        return values

    def _fit_raman_variance_equalization(
        self,
        scaled_training_residuals: np.ndarray,
    ) -> tuple[np.ndarray | None, float, float]:
        """Fit D4.21 train-only Raman-wise equalization for one condition.

        The input is the existing D2.6 robust-asinh scaled local residual.
        The median Raman standard-deviation scale is retained while pointwise
        heteroscedasticity is reduced. The exact multiplier is saved in the
        checkpoint and inverted during generation.
        """

        values = _as_2d_finite(
            scaled_training_residuals,
            "scaled_training_residuals",
        )

        if not self.raman_variance_equalization_enabled:
            return None, 0.0, 1.0

        if values.shape[0] < 2:
            raise ValueError(
                "D4.21逐Raman方差均衡至少需要2条training光谱。"
            )

        pointwise_std = np.std(
            values.astype(np.float64),
            axis=0,
            ddof=1,
        )
        positive = pointwise_std[
            np.isfinite(pointwise_std) & (pointwise_std > 1.0e-12)
        ]

        if positive.size == 0:
            multiplier = np.ones(
                values.shape[1],
                dtype=np.float32,
            )
            return multiplier, 1.0, 1.0

        floor = max(
            float(
                np.percentile(
                    positive,
                    self.raman_variance_floor_quantile,
                )
            ),
            1.0e-12,
        )

        denominator = np.maximum(
            np.where(
                np.isfinite(pointwise_std),
                pointwise_std,
                floor,
            ),
            floor,
        )

        reference_scale = max(
            float(
                np.percentile(
                    denominator,
                    self.raman_variance_reference_quantile,
                )
            ),
            1.0e-12,
        )

        hard_multiplier = reference_scale / denominator
        hard_multiplier = np.clip(
            hard_multiplier,
            self.raman_variance_minimum_multiplier,
            self.raman_variance_maximum_multiplier,
        )

        # D4.24 soft equalization:
        #
        # alpha = 1.0 keeps the exact D4.21 behaviour.
        # alpha < 1.0 pulls the multiplier towards 1.0,
        # thereby retaining more of the real Raman-dependent
        # heteroscedasticity.
        if abs(
            self.raman_variance_equalization_power - 1.0
        ) <= 1.0e-12:
            multiplier = hard_multiplier
        else:
            multiplier = np.power(
                hard_multiplier,
                self.raman_variance_equalization_power,
            )

        if (
            not np.isfinite(multiplier).all()
            or np.any(multiplier <= 0.0)
        ):
            raise RuntimeError(
                "D4.21逐Raman方差均衡倍率无效。"
            )

        return (
            multiplier.astype(np.float32, copy=False),
            float(floor),
            float(reference_scale),
        )

    @staticmethod
    def _apply_raman_variance_equalization(
        values: np.ndarray,
        entry: ConditionalPriorEntry,
    ) -> np.ndarray:
        array = _as_2d_finite(values, "values")
        multiplier = entry.raman_variance_multiplier
        if multiplier is None:
            return array.astype(np.float32, copy=False)
        if multiplier.ndim != 1 or multiplier.size != array.shape[1]:
            raise ValueError(
                "D4.21 Raman variance multiplier长度与local residual不一致。"
            )
        result = array * multiplier[None, :]
        if not np.isfinite(result).all():
            raise RuntimeError(
                "D4.21 forward variance equalization出现NaN或无穷值。"
            )
        return result.astype(np.float32, copy=False)

    @staticmethod
    def _remove_raman_variance_equalization(
        values: np.ndarray,
        entry: ConditionalPriorEntry,
    ) -> np.ndarray:
        array = _as_2d_finite(values, "values")
        multiplier = entry.raman_variance_multiplier
        if multiplier is None:
            return array.astype(np.float32, copy=False)
        if multiplier.ndim != 1 or multiplier.size != array.shape[1]:
            raise ValueError(
                "D4.21 Raman variance multiplier长度与local residual不一致。"
            )
        result = array / multiplier[None, :]
        if not np.isfinite(result).all():
            raise RuntimeError(
                "D4.21 inverse variance equalization出现NaN或无穷值。"
            )
        return result.astype(np.float32, copy=False)

    def fit(
        self,
        spectra: np.ndarray,
        *,
        valid_masks: np.ndarray,
        condition_ids: Sequence[str],
        training_indices: Sequence[int],
        raman_shift: np.ndarray,
    ) -> "ConditionalPriorResidualBank":
        values = _as_2d_finite(spectra, "spectra")
        masks = _as_2d_finite(valid_masks, "valid_masks")
        if masks.shape != values.shape:
            raise ValueError("valid_masks形状必须与spectra一致。")
        conditions = self._condition_array(condition_ids, values.shape[0])
        axis = np.asarray(raman_shift, dtype=np.float64).reshape(-1)
        if (
            axis.size != values.shape[1]
            or not np.isfinite(axis).all()
            or not np.all(np.diff(axis) > 0.0)
        ):
            raise ValueError("raman_shift必须与统一模型轴一致且严格递增。")
        train = np.asarray(training_indices, dtype=np.int64).reshape(-1)
        if train.size == 0 or np.any(train < 0) or np.any(train >= values.shape[0]):
            raise ValueError("training_indices为空或越界。")
        if np.unique(train).size != train.size:
            raise ValueError("training_indices不能重复。")

        self.model_axis = axis.copy()
        self.entries = {}
        cross_fitted_references = (
            np.zeros_like(values, dtype=np.float32)
            if self.training_cross_fit_enabled else None
        )
        cross_fitted_members = (
            np.zeros(values.shape[0], dtype=bool)
            if self.training_cross_fit_enabled else None
        )
        all_conditions = sorted(set(str(value) for value in conditions))
        for condition_id in all_conditions:
            group_train = train[conditions[train] == condition_id]
            if group_train.size < 4:
                raise ValueError(
                    f"条件{condition_id}只有{group_train.size}条训练光谱；"
                    "D4.2 PCA+broad/local至少需要4条。"
                )
            lengths = {_valid_prefix_length(masks[index]) for index in group_train}
            if len(lengths) != 1:
                raise ValueError(f"条件{condition_id}的训练光谱Raman范围不一致。")
            valid_length = lengths.pop()
            training_values = values[group_train, :valid_length]
            condition_axis = axis[:valid_length]

            # Final PCA：全部training拟合，保存checkpoint并用于val/test/generation。
            outer = _build_outer(self.prior_configuration)
            outer.fit(training_values, raman_shift=condition_axis)

            if self.training_cross_fit_enabled:
                if group_train.size < 5:
                    raise ValueError(
                        f"条件{condition_id}只有{group_train.size}条训练光谱；"
                        "leave-one-out cross-fit至少要求5条。"
                    )
                reference_for_training = np.zeros_like(training_values, dtype=np.float32)
                for held_position in range(group_train.size):
                    keep = np.ones(group_train.size, dtype=bool)
                    keep[held_position] = False
                    temporary_outer = _build_outer(self.prior_configuration)
                    temporary_outer.fit(training_values[keep], raman_shift=condition_axis)
                    reference_for_training[held_position:held_position + 1] = (
                        temporary_outer.reference_priors_for_spectra(
                            training_values[held_position:held_position + 1]
                        )
                    )
                assert cross_fitted_references is not None
                assert cross_fitted_members is not None
                cross_fitted_references[group_train, :valid_length] = reference_for_training
                cross_fitted_members[group_train] = True
            else:
                reference_for_training = outer.reference_priors_for_spectra(training_values)

            # broad/local也必须在OOF raw residual上拟合。
            raw_residuals = (training_values - reference_for_training).astype(np.float32)
            broad_local = BroadLocalResidualDecomposer.from_configuration(
                self.broad_local_configuration
            )
            broad_local.fit(raw_residuals, raman_shift=condition_axis)

            scaled_training_residuals = (
                broad_local.transform_raw_residuals(raw_residuals)
            )
            (
                raman_variance_multiplier,
                raman_variance_floor,
                raman_variance_reference_scale,
            ) = self._fit_raman_variance_equalization(
                scaled_training_residuals
            )

            self.entries[condition_id] = ConditionalPriorEntry(
                valid_length=valid_length,
                number_of_training_spectra=int(group_train.size),
                outer=outer,
                broad_local=broad_local,
                raman_variance_multiplier=raman_variance_multiplier,
                raman_variance_floor=raman_variance_floor,
                raman_variance_reference_scale=(
                    raman_variance_reference_scale
                ),
            )

        if self.training_cross_fit_enabled:
            assert cross_fitted_references is not None
            assert cross_fitted_members is not None
            if not np.array_equal(np.flatnonzero(cross_fitted_members), np.sort(train)):
                raise RuntimeError("cross-fit训练索引缓存与training_indices不一致。")
            self._cross_fitted_reference_priors = cross_fitted_references
            self._cross_fitted_training_indices = np.sort(train)
        else:
            self._cross_fitted_reference_priors = None
            self._cross_fitted_training_indices = None
        return self

    def _entry(self, condition_id: str) -> ConditionalPriorEntry:
        key = str(condition_id)
        if key not in self.entries:
            raise KeyError(f"条件先验库中不存在{key!r}。")
        return self.entries[key]

    def training_cross_fit_metadata(self) -> dict[str, Any]:
        return {
            "enabled": bool(getattr(self, "training_cross_fit_enabled", False)),
            "method": str(getattr(self, "training_cross_fit_method", "leave_one_out")),
        }

    def transform_training_aware_with_conditioning(
        self, spectra: np.ndarray, *, valid_masks: np.ndarray,
        condition_ids: Sequence[str], training_indices: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Training rows use LOO PCA; val/test rows use final PCA."""
        result, conditioning = self.transform_with_conditioning(
            spectra, valid_masks=valid_masks, condition_ids=condition_ids
        )
        if not bool(getattr(self, "training_cross_fit_enabled", False)):
            return result, conditioning
        if self._cross_fitted_reference_priors is None or self._cross_fitted_training_indices is None:
            raise RuntimeError("cross-fit已启用，但LOO reference缓存不存在。")
        values = _as_2d_finite(spectra, "spectra")
        masks = _as_2d_finite(valid_masks, "valid_masks")
        conditions = self._condition_array(condition_ids, values.shape[0])
        train = np.asarray(training_indices, dtype=np.int64).reshape(-1)
        if not np.array_equal(np.sort(train), self._cross_fitted_training_indices):
            raise ValueError("training_indices与fit()时不一致。")
        references = self._cross_fitted_reference_priors
        for condition_id in sorted(set(str(v) for v in conditions[train])):
            entry = self._entry(condition_id)
            indices = train[conditions[train] == condition_id]
            active = values[indices, :entry.valid_length]
            reference = references[indices, :entry.valid_length]
            for index in indices:
                if _valid_prefix_length(masks[index]) != entry.valid_length:
                    raise ValueError(f"条件{condition_id}的training光谱掩码不一致。")
            raw = (active - reference).astype(np.float32)
            broad, _ = entry.broad_local.split_raw_residuals(raw)
            scaled_local = (
                entry.broad_local.transform_raw_residuals(raw)
            )
            result[indices, :entry.valid_length] = (
                self._apply_raman_variance_equalization(
                    scaled_local,
                    entry,
                )
            )
            conditioning[indices, :entry.valid_length] = reference + broad
        if not np.isfinite(result).all() or not np.isfinite(conditioning).all():
            raise RuntimeError("cross-fit训练变换产生NaN或无穷值。")
        return result, conditioning

    def transform_training_aware(
        self, spectra: np.ndarray, *, valid_masks: np.ndarray,
        condition_ids: Sequence[str], training_indices: Sequence[int],
    ) -> np.ndarray:
        transformed, _ = self.transform_training_aware_with_conditioning(
            spectra, valid_masks=valid_masks, condition_ids=condition_ids,
            training_indices=training_indices,
        )
        return transformed

    def transform(
        self,
        spectra: np.ndarray,
        *,
        valid_masks: np.ndarray,
        condition_ids: Sequence[str],
    ) -> np.ndarray:
        transformed, _ = self.transform_with_conditioning(
            spectra,
            valid_masks=valid_masks,
            condition_ids=condition_ids,
        )
        return transformed

    def transform_with_conditioning(
        self,
        spectra: np.ndarray,
        *,
        valid_masks: np.ndarray,
        condition_ids: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return local residuals and their exact reconstruction bases.

        The second array is the condition-specific outer PCA reference plus
        the broad residual extracted from the same spectrum.  Feeding it to
        the denoiser makes training learn ``p(local | sampled_base, condition)``
        instead of silently learning only ``p(local | condition)``.
        """
        if self.model_axis is None or not self.entries:
            raise RuntimeError("条件先验库尚未拟合。")
        values = _as_2d_finite(spectra, "spectra")
        masks = _as_2d_finite(valid_masks, "valid_masks")
        if values.shape != masks.shape or values.shape[1] != self.model_axis.size:
            raise ValueError("待变换光谱、掩码与条件先验库轴不一致。")
        conditions = self._condition_array(condition_ids, values.shape[0])
        result = np.zeros_like(values, dtype=np.float32)
        conditioning = np.zeros_like(values, dtype=np.float32)
        for condition_id in sorted(set(str(value) for value in conditions)):
            entry = self._entry(condition_id)
            indices = np.flatnonzero(conditions == condition_id)
            for index in indices:
                if _valid_prefix_length(masks[index]) != entry.valid_length:
                    raise ValueError(
                        f"条件{condition_id}的光谱掩码与训练状态不一致。"
                    )
            active = values[indices, : entry.valid_length]
            reference = entry.outer.reference_priors_for_spectra(active)
            raw = (active - reference).astype(np.float32)
            broad, _ = entry.broad_local.split_raw_residuals(raw)
            scaled_local = (
                entry.broad_local.transform_raw_residuals(raw)
            )
            result[indices, : entry.valid_length] = (
                self._apply_raman_variance_equalization(
                    scaled_local,
                    entry,
                )
            )
            conditioning[indices, : entry.valid_length] = reference + broad
        if not np.isfinite(result).all() or not np.isfinite(conditioning).all():
            raise RuntimeError("条件先验变换产生NaN或无穷值。")
        return result, conditioning

    def diagnostic_transform(
        self,
        spectrum: np.ndarray,
        *,
        valid_mask: np.ndarray,
        condition_id: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the learned local domain and deterministic true-spectrum base.

        The D4.2/D4.3 DDPM learns only the scaled local residual.  A timestep
        recovery diagnostic must therefore corrupt that residual and restore a
        prediction with the same outer-PCA plus broad component extracted from
        the selected real spectrum.  Random generation priors are deliberately
        not used in a denoising-recovery test.
        """

        if self.model_axis is None or not self.entries:
            raise RuntimeError("条件先验库尚未拟合。")
        values = _as_2d_finite(spectrum, "spectrum")
        masks = _as_2d_finite(valid_mask, "valid_mask")
        if values.shape != masks.shape or values.shape[0] != 1:
            raise ValueError("诊断光谱和掩码必须为形状一致的[1,L]数组。")
        if values.shape[1] != self.model_axis.size:
            raise ValueError("诊断光谱长度与条件先验库模型轴不一致。")

        entry = self._entry(condition_id)
        if _valid_prefix_length(masks[0]) != entry.valid_length:
            raise ValueError(f"条件{condition_id}的诊断掩码与训练状态不一致。")
        active = values[:, : entry.valid_length]
        reference = entry.outer.reference_priors_for_spectra(active)
        raw = (active - reference).astype(np.float32)
        broad, _ = entry.broad_local.split_raw_residuals(raw)
        scaled_local = entry.broad_local.transform_raw_residuals(raw)
        scaled_local = self._apply_raman_variance_equalization(
            scaled_local,
            entry,
        )

        model_domain = np.zeros_like(values, dtype=np.float32)
        deterministic_base = np.zeros_like(values, dtype=np.float32)
        model_domain[:, : entry.valid_length] = scaled_local
        deterministic_base[:, : entry.valid_length] = reference + broad
        return model_domain, deterministic_base

    def restore_diagnostic_prediction(
        self,
        scaled_local_residual: np.ndarray,
        *,
        deterministic_base: np.ndarray,
        condition_id: str,
    ) -> np.ndarray:
        """Restore a timestep prediction with its matching deterministic base."""

        values = _as_2d_finite(scaled_local_residual, "scaled_local_residual")
        base = _as_2d_finite(deterministic_base, "deterministic_base")
        if values.shape != base.shape:
            raise ValueError("scaled_local_residual与deterministic_base形状不一致。")
        entry = self._entry(condition_id)
        if values.shape[1] < entry.valid_length:
            raise ValueError("诊断预测长度小于条件有效Raman长度。")
        restored = np.zeros_like(values, dtype=np.float32)
        original_scaled_local = (
            self._remove_raman_variance_equalization(
                values[:, : entry.valid_length],
                entry,
            )
        )
        local = entry.broad_local.inverse_local_transform(
            original_scaled_local
        )
        restored[:, : entry.valid_length] = (
            base[:, : entry.valid_length] + local
        )
        return restored

    def reconstruct_generated(
        self,
        scaled_local_residuals: np.ndarray,
        *,
        condition_id: str,
        prior_random_generator: np.random.Generator,
        broad_random_generator: np.random.Generator,
    ) -> np.ndarray:
        values = _as_2d_finite(scaled_local_residuals, "scaled_local_residuals")
        conditioning = self.sample_generation_conditioning(
            values.shape[0],
            condition_id=condition_id,
            prior_random_generator=prior_random_generator,
            broad_random_generator=broad_random_generator,
        )
        return self.reconstruct_generated_with_conditioning(
            values,
            prior_conditioning=conditioning,
            condition_id=condition_id,
        )

    def sample_generation_conditioning(
        self,
        number: int,
        *,
        condition_id: str,
        prior_random_generator: np.random.Generator,
        broad_random_generator: np.random.Generator,
    ) -> np.ndarray:
        """Sample the exact outer-plus-broad base used by one generation batch."""

        if self.model_axis is None or not self.entries:
            raise RuntimeError("条件先验库尚未拟合。")
        number = int(number)
        if number <= 0:
            raise ValueError("number必须大于0。")
        entry = self._entry(condition_id)
        prior = entry.outer.sample_reference_priors(
            number, random_generator=prior_random_generator
        )
        broad = entry.broad_local.sample_broad_residuals(
            number, random_generator=broad_random_generator
        )
        conditioning = np.zeros(
            (number, self.model_axis.size), dtype=np.float32
        )
        conditioning[:, : entry.valid_length] = prior + broad
        if not np.isfinite(conditioning).all():
            raise RuntimeError("条件生成先验产生NaN或无穷值。")
        return conditioning

    def reconstruct_generated_with_conditioning(
        self,
        scaled_local_residuals: np.ndarray,
        *,
        prior_conditioning: np.ndarray,
        condition_id: str,
    ) -> np.ndarray:
        """Reconstruct with the same prior base seen by the denoising U-Net."""

        values = _as_2d_finite(scaled_local_residuals, "scaled_local_residuals")
        conditioning = _as_2d_finite(prior_conditioning, "prior_conditioning")
        if conditioning.shape != values.shape:
            raise ValueError("prior_conditioning形状必须与生成残差一致。")
        entry = self._entry(condition_id)
        if values.shape[1] < entry.valid_length:
            raise ValueError("生成残差长度小于条件有效Raman长度。")
        original_scaled_local = (
            self._remove_raman_variance_equalization(
                values[:, : entry.valid_length],
                entry,
            )
        )
        local = entry.broad_local.inverse_local_transform(
            original_scaled_local
        )
        reconstructed = (
            conditioning[:, : entry.valid_length] + local
        )
        if not np.isfinite(reconstructed).all():
            raise RuntimeError("条件先验残差重建产生NaN或无穷值。")
        return reconstructed.astype(np.float32, copy=False)

    def local_inverse_linearization_slopes(
        self,
        scaled_local_residuals: np.ndarray,
        *,
        valid_masks: np.ndarray,
        condition_ids: Sequence[str],
        finite_difference_step: float = 1.0e-3,
    ) -> np.ndarray:
        """Return d(local_raw)/d(local_scaled) around every real spectrum.

        The local residual uses a nonlinear robust-asinh transform.  Training
        losses that are evaluated only in that scaled domain can therefore
        under-penalize an error that becomes a very high peak or a deep valley
        after inverse transformation.  These train-only finite-difference
        slopes provide a differentiable first-order reconstruction around the
        true local residual.  The corresponding full-spectrum loss is gated to
        low/middle noise levels where this local approximation is reliable.
        """

        values = _as_2d_finite(
            scaled_local_residuals, "scaled_local_residuals"
        )
        masks = _as_2d_finite(valid_masks, "valid_masks")
        if values.shape != masks.shape:
            raise ValueError(
                "scaled_local_residuals与valid_masks形状必须一致。"
            )
        if self.model_axis is None or values.shape[1] != self.model_axis.size:
            raise ValueError("局部残差长度与条件先验库模型轴不一致。")
        conditions = self._condition_array(condition_ids, values.shape[0])
        step = float(finite_difference_step)
        if not np.isfinite(step) or not 1.0e-5 <= step <= 1.0e-1:
            raise ValueError("finite_difference_step必须位于[1e-5,1e-1]。")

        slopes = np.zeros_like(values, dtype=np.float32)
        for condition_id in sorted(set(str(value) for value in conditions)):
            entry = self._entry(condition_id)
            indices = np.flatnonzero(conditions == condition_id)
            for index in indices:
                if _valid_prefix_length(masks[index]) != entry.valid_length:
                    raise ValueError(
                        f"条件{condition_id}的线性化掩码与训练状态不一致。"
                    )
            active = values[indices, : entry.valid_length]
            upper_scaled = self._remove_raman_variance_equalization(
                active + step,
                entry,
            )
            lower_scaled = self._remove_raman_variance_equalization(
                active - step,
                entry,
            )
            upper = entry.broad_local.inverse_local_transform(
                upper_scaled
            )
            lower = entry.broad_local.inverse_local_transform(
                lower_scaled
            )
            active_slopes = (upper - lower) / (2.0 * step)
            if not np.isfinite(active_slopes).all() or np.any(
                active_slopes <= 0.0
            ):
                raise RuntimeError(
                    f"条件{condition_id}的局部残差反变换斜率无效。"
                )
            slopes[indices, : entry.valid_length] = active_slopes.astype(
                np.float32, copy=False
            )
        return slopes

    def valid_length(self, condition_id: str) -> int:
        return self._entry(condition_id).valid_length

    def apply_score_clip_runtime_override(
        self,
        standard_deviations: float | None,
    ) -> dict[str, Any]:
        """Override PCA sampling ranges in memory without changing checkpoint state.

        Both the outer spectrum prior and the broad-residual prior are sampled
        during D4.2/D4.3 reconstruction.  Keeping their truncation ranges equal
        avoids restoring the outer variability while silently leaving the broad
        component compressed.
        """

        checkpoint_outer_values = sorted(
            {
                float(entry.outer.pca_score_clip_standard_deviations)
                for entry in self.entries.values()
            }
        )
        checkpoint_broad_values = sorted(
            {
                float(entry.broad_local.broad_score_clip_standard_deviations)
                for entry in self.entries.values()
            }
        )
        information: dict[str, Any] = {
            "active": standard_deviations is not None,
            "checkpoint_outer_values": checkpoint_outer_values,
            "checkpoint_broad_values": checkpoint_broad_values,
            "runtime_value": None,
            "overridden": False,
        }
        if standard_deviations is None:
            return information

        value = float(standard_deviations)
        if not np.isfinite(value) or not 0.5 <= value <= 4.0:
            raise ValueError(
                "PCA score采样截断范围必须位于[0.5, 4.0]个标准差。"
            )
        for entry in self.entries.values():
            entry.outer.pca_score_clip_standard_deviations = value
            entry.broad_local.broad_score_clip_standard_deviations = value
        information["runtime_value"] = value
        information["overridden"] = bool(
            any(abs(item - value) > 1.0e-12 for item in checkpoint_outer_values)
            or any(abs(item - value) > 1.0e-12 for item in checkpoint_broad_values)
        )
        return information

    def state_dict(self) -> dict[str, Any]:
        if self.model_axis is None or not self.entries:
            raise RuntimeError("条件先验库尚未拟合。")
        return {
            "version": STATE_VERSION,
            "enabled": True,
            "model_axis": self.model_axis.tolist(),
            "training_cross_fit": self.training_cross_fit_metadata(),
            "raman_variance_equalization": {
                "enabled": bool(
                    self.raman_variance_equalization_enabled
                ),
                "floor_quantile": float(
                    self.raman_variance_floor_quantile
                ),
                "reference_quantile": float(
                    self.raman_variance_reference_quantile
                ),
                "minimum_multiplier": float(
                    self.raman_variance_minimum_multiplier
                ),
                "maximum_multiplier": float(
                    self.raman_variance_maximum_multiplier
                ),
                "equalization_power": float(
                    self.raman_variance_equalization_power
                ),
            },
            "number_of_conditions": len(self.entries),
            "conditions": {
                key: {
                    "valid_length": entry.valid_length,
                    "number_of_training_spectra": entry.number_of_training_spectra,
                    "prior_residual_state": entry.outer.state_dict(),
                    "broad_local_residual_state": entry.broad_local.state_dict(),
                    "raman_variance_equalization_state": {
                        "enabled": bool(
                            entry.raman_variance_multiplier is not None
                        ),
                        "floor": float(
                            entry.raman_variance_floor
                        ),
                        "reference_scale": float(
                            entry.raman_variance_reference_scale
                        ),
                        "multiplier": (
                            None
                            if entry.raman_variance_multiplier is None
                            else entry.raman_variance_multiplier.tolist()
                        ),
                    },
                }
                for key, entry in sorted(self.entries.items())
            },
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "ConditionalPriorResidualBank":
        if not isinstance(state, dict) or state.get("version") != STATE_VERSION:
            raise ValueError("不支持的conditional_prior_residual_state。")
        records = state.get("conditions")
        if not isinstance(records, dict) or not records:
            raise ValueError("条件先验状态不包含conditions。")
        instance = cls.__new__(cls)
        instance.prior_configuration = {}
        instance.broad_local_configuration = {}

        variance_equalization_state = (
            state.get("raman_variance_equalization", {})
            or {}
        )
        if not isinstance(variance_equalization_state, dict):
            raise ValueError(
                "checkpoint中的raman_variance_equalization无效。"
            )
        instance.raman_variance_equalization_enabled = bool(
            variance_equalization_state.get("enabled", False)
        )
        instance.raman_variance_floor_quantile = float(
            variance_equalization_state.get("floor_quantile", 10.0)
        )
        instance.raman_variance_reference_quantile = float(
            variance_equalization_state.get("reference_quantile", 50.0)
        )
        instance.raman_variance_minimum_multiplier = float(
            variance_equalization_state.get("minimum_multiplier", 0.25)
        )
        instance.raman_variance_maximum_multiplier = float(
            variance_equalization_state.get("maximum_multiplier", 4.0)
        )

        # Backward compatibility:
        # D4.21-D4.23 checkpoints do not contain this field,
        # therefore they must retain the original power=1.0 behaviour.
        instance.raman_variance_equalization_power = float(
            variance_equalization_state.get(
                "equalization_power",
                1.0,
            )
        )

        if not (
            0.0
            < instance.raman_variance_equalization_power
            <= 1.0
        ):
            raise ValueError(
                "checkpoint中的raman_variance_equalization."
                "equalization_power必须位于(0,1]。"
            )

        cross_fit_state = state.get("training_cross_fit", {}) or {}
        if not isinstance(cross_fit_state, dict):
            raise ValueError("checkpoint中的training_cross_fit无效。")
        instance.training_cross_fit_enabled = bool(cross_fit_state.get("enabled", False))
        instance.training_cross_fit_method = str(
            cross_fit_state.get("method", "leave_one_out")
        ).strip().lower()
        if instance.training_cross_fit_enabled and instance.training_cross_fit_method != "leave_one_out":
            raise ValueError("checkpoint包含不支持的training cross-fit方法。")
        instance._cross_fitted_reference_priors = None
        instance._cross_fitted_training_indices = None
        instance.model_axis = np.asarray(state.get("model_axis"), dtype=np.float64)
        if (
            instance.model_axis.ndim != 1
            or instance.model_axis.size < 2
            or not np.isfinite(instance.model_axis).all()
            or not np.all(np.diff(instance.model_axis) > 0.0)
        ):
            raise ValueError("条件先验状态中的model_axis无效。")
        instance.entries = {}
        for condition_id, record in records.items():
            if not isinstance(record, dict):
                raise ValueError(f"条件{condition_id}的状态无效。")
            valid_length = int(record.get("valid_length", 0))
            if not 2 <= valid_length <= instance.model_axis.size:
                raise ValueError(f"条件{condition_id}的valid_length无效。")

            condition_variance_state = (
                record.get("raman_variance_equalization_state", {})
                or {}
            )
            if not isinstance(condition_variance_state, dict):
                raise ValueError(
                    f"条件{condition_id}的Raman variance状态无效。"
                )

            multiplier = None
            variance_floor = 0.0
            variance_reference_scale = 1.0

            if bool(condition_variance_state.get("enabled", False)):
                multiplier = np.asarray(
                    condition_variance_state.get("multiplier"),
                    dtype=np.float32,
                ).reshape(-1)
                if (
                    multiplier.size != valid_length
                    or not np.isfinite(multiplier).all()
                    or np.any(multiplier <= 0.0)
                ):
                    raise ValueError(
                        f"条件{condition_id}的Raman variance multiplier无效。"
                    )
                variance_floor = float(
                    condition_variance_state.get("floor", 0.0)
                )
                variance_reference_scale = float(
                    condition_variance_state.get(
                        "reference_scale",
                        1.0,
                    )
                )
                if (
                    not np.isfinite(variance_floor)
                    or variance_floor < 0.0
                    or not np.isfinite(variance_reference_scale)
                    or variance_reference_scale <= 0.0
                ):
                    raise ValueError(
                        f"条件{condition_id}的Raman variance统计量无效。"
                    )

            instance.entries[str(condition_id)] = ConditionalPriorEntry(
                valid_length=valid_length,
                number_of_training_spectra=int(
                    record.get("number_of_training_spectra", 0)
                ),
                outer=PriorResidualTransformer.from_state_dict(
                    record["prior_residual_state"]
                ),
                broad_local=BroadLocalResidualDecomposer.from_state_dict(
                    record["broad_local_residual_state"]
                ),
                raman_variance_multiplier=multiplier,
                raman_variance_floor=variance_floor,
                raman_variance_reference_scale=(
                    variance_reference_scale
                ),
            )
        if int(state.get("number_of_conditions", len(instance.entries))) != len(
            instance.entries
        ):
            raise ValueError("条件先验状态中的条件数量不一致。")
        return instance

    def summary(self) -> dict[str, Any]:
        lengths = [entry.valid_length for entry in self.entries.values()]
        counts = [entry.number_of_training_spectra for entry in self.entries.values()]
        multipliers = [
            entry.raman_variance_multiplier
            for entry in self.entries.values()
            if entry.raman_variance_multiplier is not None
        ]
        variance_summary = {
            "enabled": bool(
                self.raman_variance_equalization_enabled
            ),
            "equalization_power": float(
                self.raman_variance_equalization_power
            ),
            "condition_count": len(multipliers),
            "minimum_multiplier": (
                float(min(np.min(value) for value in multipliers))
                if multipliers else 1.0
            ),
            "maximum_multiplier": (
                float(max(np.max(value) for value in multipliers))
                if multipliers else 1.0
            ),
        }
        return {
            "number_of_conditions": len(self.entries),
            "training_cross_fit": self.training_cross_fit_metadata(),
            "raman_variance_equalization": variance_summary,
            "valid_lengths": {
                str(length): lengths.count(length) for length in sorted(set(lengths))
            },
            "training_spectra_per_condition": sorted(set(counts)),
            "total_training_spectra": int(sum(counts)),
        }
