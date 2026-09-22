from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class SpectrumDataset(
    Dataset[torch.Tensor]
):
    """
    保存DDPM输入张量，同时保存原始轴、长度和标签。

    __getitem__仍然只返回光谱张量，
    因此不需要修改DdpmTrainer。
    """

    def __init__(
        self,
        spectra: np.ndarray,
        *,
        original_lengths: (
            Sequence[int] | None
        ) = None,
        original_raman_shifts: (
            Sequence[np.ndarray] | None
        ) = None,
        labels: (
            Sequence[str] | None
        ) = None,
        source_files: (
            Sequence[str] | None
        ) = None,
        valid_masks: np.ndarray | None = None,
        conditions: np.ndarray | None = None,
        constraint_reference_priors: np.ndarray | None = None,
        prior_conditionings: np.ndarray | None = None,
        full_spectrum_targets: np.ndarray | None = None,
        local_inverse_slopes: np.ndarray | None = None,
    ) -> None:
        values = np.asarray(
            spectra,
            dtype=np.float32,
        )

        if values.ndim != 2:
            raise ValueError(
                "spectra必须是二维数组[N,L]，"
                f"实际为{values.shape}。"
            )

        if values.shape[0] == 0:
            raise ValueError(
                "数据集不能为空。"
            )

        if not np.isfinite(values).all():
            raise ValueError(
                "光谱中存在NaN或无穷大。"
            )

        number_of_spectra = int(
            values.shape[0]
        )

        def check_count(
            name: str,
            metadata: (
                Sequence[object] | None
            ),
        ) -> None:
            if (
                metadata is not None
                and len(metadata)
                != number_of_spectra
            ):
                raise ValueError(
                    f"{name}数量{len(metadata)}"
                    f"与光谱数量{number_of_spectra}"
                    "不一致。"
                )

        check_count(
            "original_lengths",
            original_lengths,
        )

        check_count(
            "original_raman_shifts",
            original_raman_shifts,
        )

        check_count(
            "labels",
            labels,
        )

        check_count(
            "source_files",
            source_files,
        )

        reference_priors = None
        if constraint_reference_priors is not None:
            reference_priors = np.asarray(
                constraint_reference_priors,
                dtype=np.float32,
            )
            if reference_priors.shape != values.shape:
                raise ValueError(
                    "constraint_reference_priors形状必须与spectra一致："
                    f"先验为{reference_priors.shape}，"
                    f"光谱为{values.shape}。"
                )
            if not np.isfinite(reference_priors).all():
                raise ValueError(
                    "constraint_reference_priors中存在NaN或无穷大。"
                )

        prior_values = None
        if prior_conditionings is not None:
            prior_values = np.asarray(prior_conditionings, dtype=np.float32)
            if prior_values.shape != values.shape:
                raise ValueError(
                    "prior_conditionings形状必须与spectra一致："
                    f"先验条件为{prior_values.shape}，光谱为{values.shape}。"
                )
            if not np.isfinite(prior_values).all():
                raise ValueError("prior_conditionings中存在NaN或无穷大。")

        full_targets = None
        inverse_slopes = None
        if full_spectrum_targets is not None or local_inverse_slopes is not None:
            if full_spectrum_targets is None or local_inverse_slopes is None:
                raise ValueError(
                    "full_spectrum_targets与local_inverse_slopes必须同时提供。"
                )
            full_targets = np.asarray(full_spectrum_targets, dtype=np.float32)
            inverse_slopes = np.asarray(local_inverse_slopes, dtype=np.float32)
            if full_targets.shape != values.shape:
                raise ValueError(
                    "full_spectrum_targets形状必须与spectra一致。"
                )
            if inverse_slopes.shape != values.shape:
                raise ValueError("local_inverse_slopes形状必须与spectra一致。")
            if not np.isfinite(full_targets).all():
                raise ValueError("full_spectrum_targets包含NaN或无穷值。")
            if not np.isfinite(inverse_slopes).all() or np.any(
                inverse_slopes < 0.0
            ):
                raise ValueError("local_inverse_slopes必须是有限非负数。")

        masks = None
        if valid_masks is not None:
            masks = np.asarray(
                valid_masks,
                dtype=np.float32,
            )
            if masks.shape != values.shape:
                raise ValueError(
                    "valid_masks形状必须与spectra一致："
                    f"掩码为{masks.shape}，光谱为{values.shape}。"
                )
            if not np.isfinite(masks).all():
                raise ValueError("valid_masks中存在NaN或无穷大。")
            if not np.logical_or(masks == 0.0, masks == 1.0).all():
                raise ValueError("valid_masks只能包含0和1。")
            if np.any(masks.sum(axis=1) <= 0.0):
                raise ValueError("每条光谱至少需要一个有效Raman点。")

        condition_values = None
        if conditions is not None:
            condition_values = np.asarray(conditions, dtype=np.float32)
            if condition_values.ndim != 2:
                raise ValueError(
                    "conditions必须是二维数组[N,C]，"
                    f"实际为{condition_values.shape}。"
                )
            if condition_values.shape[0] != number_of_spectra:
                raise ValueError(
                    "conditions数量必须与光谱数量一致："
                    f"{condition_values.shape[0]} != {number_of_spectra}。"
                )
            if condition_values.shape[1] <= 0:
                raise ValueError("conditions至少需要一个条件维度。")
            if not np.isfinite(condition_values).all():
                raise ValueError("conditions包含NaN或无穷值。")

        # [N,L]转换为DDPM使用的[N,1,L]。
        self.spectra = torch.from_numpy(
            values
        ).unsqueeze(1)

        self.original_lengths = (
            None
            if original_lengths is None
            else np.asarray(
                original_lengths,
                dtype=np.int64,
            )
        )

        self.original_raman_shifts = (
            None
            if original_raman_shifts is None
            else tuple(
                np.asarray(
                    axis,
                    dtype=np.float64,
                ).copy()
                for axis in original_raman_shifts
            )
        )

        self.labels = (
            None
            if labels is None
            else np.asarray(
                labels,
                dtype=object,
            )
        )

        self.source_files = (
            None
            if source_files is None
            else np.asarray(
                source_files,
                dtype=object,
            )
        )

        self.constraint_reference_priors = (
            None
            if reference_priors is None
            else torch.from_numpy(reference_priors).unsqueeze(1)
        )

        self.prior_conditionings = (
            None
            if prior_values is None
            else torch.from_numpy(prior_values).unsqueeze(1)
        )

        self.full_spectrum_targets = (
            None
            if full_targets is None
            else torch.from_numpy(full_targets).unsqueeze(1)
        )
        self.local_inverse_slopes = (
            None
            if inverse_slopes is None
            else torch.from_numpy(inverse_slopes).unsqueeze(1)
        )

        self.valid_masks = (
            None
            if masks is None
            else torch.from_numpy(masks).unsqueeze(1)
        )

        self.conditions = (
            None
            if condition_values is None
            else torch.from_numpy(condition_values)
        )

    def __len__(
        self,
    ) -> int:
        return int(
            self.spectra.shape[0]
        )

    def __getitem__(
        self,
        index: int,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        spectrum = self.spectra[index]
        if (
            self.constraint_reference_priors is None
            and self.valid_masks is None
            and self.conditions is None
            and self.prior_conditionings is None
            and self.full_spectrum_targets is None
            and self.local_inverse_slopes is None
        ):
            return spectrum

        item = {"spectrum": spectrum}
        if self.valid_masks is not None:
            item["valid_mask"] = self.valid_masks[index]
        if self.conditions is not None:
            item["condition"] = self.conditions[index]
        if self.constraint_reference_priors is not None:
            item["constraint_reference_prior"] = (
                self.constraint_reference_priors[index]
            )
        if self.prior_conditionings is not None:
            item["prior_conditioning"] = self.prior_conditionings[index]
        if self.full_spectrum_targets is not None:
            item["full_spectrum_target"] = self.full_spectrum_targets[index]
            item["local_inverse_slope"] = self.local_inverse_slopes[index]
        return item

    def get_original_metadata(
        self,
        index: int,
    ) -> dict[str, object]:
        """获取指定光谱的原始信息。"""

        return {
            "original_length": (
                None
                if self.original_lengths is None
                else int(
                    self.original_lengths[index]
                )
            ),
            "raman_shift": (
                None
                if self.original_raman_shifts
                is None
                else self.original_raman_shifts[
                    index
                ].copy()
            ),
            "label": (
                None
                if self.labels is None
                else str(
                    self.labels[index]
                )
            ),
            "source_file": (
                None
                if self.source_files is None
                else str(
                    self.source_files[index]
                )
            ),
        }
