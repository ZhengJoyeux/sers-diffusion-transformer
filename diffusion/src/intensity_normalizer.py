from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class GlobalMinMaxNormalizer:
    target_min: float = -1.0
    target_max: float = 1.0
    epsilon: float = 1.0e-12
    clip: bool = False

    data_min: float | None = None
    data_max: float | None = None

    def fit(
        self,
        train_spectra: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> "GlobalMinMaxNormalizer":
        values = self._validate_array(
            train_spectra,
            "train_spectra",
        )

        mask = self._validate_mask(
            valid_mask,
            values.shape,
            "valid_mask",
        )

        valid_values = (
            values.reshape(-1)
            if mask is None
            else values[mask]
        )

        if valid_values.size == 0:
            raise ValueError("训练掩码中没有有效Raman点。")

        self.data_min = float(
            valid_values.min()
        )

        self.data_max = float(
            valid_values.max()
        )

        if (
            self.data_max - self.data_min
            <= self.epsilon
        ):
            raise ValueError(
                "训练集最大值和最小值过于接近，无法归一化。"
            )

        return self

    def transform(
        self,
        spectra: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        self._check_fitted()

        values = self._validate_array(
            spectra,
            "spectra",
        )

        mask = self._validate_mask(
            valid_mask,
            values.shape,
            "valid_mask",
        )

        normalized_01 = (
            (values - self.data_min)
            / (self.data_max - self.data_min)
        )

        normalized = (
            normalized_01
            * (self.target_max - self.target_min)
            + self.target_min
        )

        if self.clip:
            normalized = np.clip(
                normalized,
                self.target_min,
                self.target_max,
            )

        if mask is not None:
            # 无效位置使用网络域的中性零值；不能把补值参与归一化。
            normalized = np.where(
                mask,
                normalized,
                0.0,
            )

        return normalized.astype(
            np.float32,
            copy=False,
        )

    def inverse_transform(
        self,
        normalized_spectra: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        self._check_fitted()

        values = self._validate_array(
            normalized_spectra,
            "normalized_spectra",
        )

        mask = self._validate_mask(
            valid_mask,
            values.shape,
            "valid_mask",
        )

        values_01 = (
            (values - self.target_min)
            / (self.target_max - self.target_min)
        )

        restored = (
            values_01
            * (self.data_max - self.data_min)
            + self.data_min
        )

        if mask is not None:
            restored = np.where(
                mask,
                restored,
                0.0,
            )

        return restored.astype(
            np.float32,
            copy=False,
        )

    def state_dict(
        self,
    ) -> dict[str, Any]:
        self._check_fitted()

        return {
            "method": "global_minmax",
            "data_min": self.data_min,
            "data_max": self.data_max,
            "target_min": self.target_min,
            "target_max": self.target_max,
            "epsilon": self.epsilon,
            "clip": self.clip,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, Any],
    ) -> "GlobalMinMaxNormalizer":
        if state.get("method") != "global_minmax":
            raise ValueError(
                "检查点中的归一化方法不是 global_minmax。"
            )

        normalizer = cls(
            target_min=float(
                state["target_min"]
            ),
            target_max=float(
                state["target_max"]
            ),
            epsilon=float(
                state.get(
                    "epsilon",
                    1.0e-12,
                )
            ),
            clip=bool(
                state.get(
                    "clip",
                    False,
                )
            ),
        )

        normalizer.data_min = float(
            state["data_min"]
        )

        normalizer.data_max = float(
            state["data_max"]
        )

        return normalizer

    @staticmethod
    def _validate_array(
        values: np.ndarray,
        name: str,
    ) -> np.ndarray:
        array = np.asarray(
            values,
            dtype=np.float32,
        )

        if array.ndim != 2:
            raise ValueError(
                f"{name} 必须是二维数组 [N, L]，"
                f"实际为 {array.shape}。"
            )

        if array.size == 0:
            raise ValueError(
                f"{name} 不能为空。"
            )

        if not np.isfinite(array).all():
            raise ValueError(
                f"{name} 中存在 NaN 或无穷大。"
            )

        return array

    def _check_fitted(self) -> None:
        if (
            self.data_min is None
            or self.data_max is None
        ):
            raise RuntimeError(
                "归一化器尚未使用训练集执行 fit()。"
            )

    @staticmethod
    def _validate_mask(
        valid_mask: np.ndarray | None,
        expected_shape: tuple[int, ...],
        name: str,
    ) -> np.ndarray | None:
        if valid_mask is None:
            return None

        mask = np.asarray(valid_mask)
        if mask.shape != expected_shape:
            raise ValueError(
                f"{name}形状{mask.shape}与光谱形状"
                f"{expected_shape}不一致。"
            )
        if not np.isfinite(mask).all():
            raise ValueError(f"{name}包含NaN或无穷值。")
        if not np.logical_or(mask == 0, mask == 1).all():
            raise ValueError(f"{name}只能包含0和1。")
        return mask.astype(bool, copy=False)
