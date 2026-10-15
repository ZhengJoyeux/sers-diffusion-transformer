from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import scipy.signal as signal
import torch
from torch.utils.data import Dataset


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

REAL_DATA_ROOT = PROJECT_ROOT / "data" / "real"
GENERATED_DATA_ROOT = PROJECT_ROOT / "data" / "generated"


# ============================================================
# 2. Pesticide definitions
# ============================================================

PESTICIDES = (
    "DEL",
    "CHL",
    "TEB",
)

PESTICIDE_TO_INDEX = {
    pesticide: index
    for index, pesticide in enumerate(PESTICIDES)
}

# 注意：
# 这只是浓度等级编码，不是实际物理浓度。
#
# 0 = absent
# 1 = S
# 2 = M
# 3 = H
#
# 后面做真实浓度回归时，
# 必须再接入DEL/CHL/TEB实际浓度值。
LEVEL_TO_CODE = {
    "S": 1,
    "M": 2,
    "H": 3,
}


# ============================================================
# 3. Unified Raman model axis
# ============================================================

# 当前真实数据已经确认：
#
# 长轴：
# 600–2500 cm-1
# 1901 points
#
# 短轴：
# 600–2000 cm-1
# 1401 points
#
# Transformer统一使用1901点模型轴。
MODEL_RAMAN_AXIS = np.arange(
    600.0,
    2500.0 + 1.0,
    1.0,
    dtype=np.float32,
)

MODEL_LENGTH = int(
    MODEL_RAMAN_AXIS.size
)

assert MODEL_LENGTH == 1901


# ============================================================
# 4. Fixed split
# ============================================================

REAL_TRAIN_INDICES = range(0, 12)
REAL_VALIDATION_INDICES = range(12, 16)
REAL_TEST_INDICES = range(16, 20)


# ============================================================
# 5. Metadata
# ============================================================

@dataclass(frozen=True)
class ConditionInfo:
    condition: str

    # [DEL, CHL, TEB]
    presence: np.ndarray

    # [DEL_level, CHL_level, TEB_level]
    # absent = 0, S = 1, M = 2, H = 3
    concentration_level: np.ndarray

    matrix_name: str
    matrix_code: int


@dataclass(frozen=True)
class NormalizationState:
    minimum: float
    maximum: float


# ============================================================
# 6. File reading
# ============================================================

def _collect_spectrum_files(
    root: Path,
) -> list[Path]:

    if not root.exists():
        raise FileNotFoundError(
            f"数据目录不存在：{root}"
        )

    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower()
        in {
            ".xlsx",
            ".xls",
            ".csv",
        }
    )

    return files


def _read_spectrum_table(
    path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    suffix = path.suffix.lower()

    if suffix in {
        ".xlsx",
        ".xls",
    }:
        frame = pd.read_excel(
            path
        )

    elif suffix == ".csv":
        frame = pd.read_csv(
            path
        )

    else:
        raise ValueError(
            f"不支持的文件格式：{path}"
        )

    if frame.shape[1] < 2:
        raise ValueError(
            f"文件没有光谱数据列：{path}"
        )

    axis = pd.to_numeric(
        frame.iloc[:, 0],
        errors="coerce",
    ).to_numpy(
        dtype=np.float64,
    )

    spectra = (
        frame.iloc[:, 1:]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float64,
        )
    )

    valid_rows = (
        np.isfinite(axis)
        & np.all(
            np.isfinite(spectra),
            axis=1,
        )
    )

    axis = axis[
        valid_rows
    ]

    spectra = spectra[
        valid_rows,
        :,
    ]

    if axis.size < 2:
        raise ValueError(
            f"有效Raman轴过短：{path}"
        )

    if not np.all(
        np.diff(axis) > 0
    ):
        raise ValueError(
            f"Raman轴不是严格递增：{path}"
        )

    return (
        axis,
        spectra,
    )


# ============================================================
# 7. Condition parsing
# ============================================================

def _condition_from_real_file(
    path: Path,
) -> str:

    return path.stem


def _condition_from_generated_file(
    path: Path,
) -> str:

    name = path.stem

    suffix = "_generated"

    if name.endswith(
        suffix
    ):
        name = name[
            :-len(suffix)
        ]

    return name


def parse_condition(
    condition: str,
) -> ConditionInfo:

    if condition.endswith(
        "_water"
    ):
        matrix_name = "water"
        matrix_code = 0

    elif condition.endswith(
        "_soil"
    ):
        matrix_name = "soil"
        matrix_code = 1

    else:
        raise ValueError(
            "condition必须以"
            "_water或_soil结尾："
            f"{condition}"
        )

    presence = np.zeros(
        3,
        dtype=np.float32,
    )

    concentration_level = np.zeros(
        3,
        dtype=np.int64,
    )

    matches = re.findall(
        r"(DEL|CHL|TEB)-(S|M|H)",
        condition,
    )

    if not matches:
        raise ValueError(
            "无法从文件名解析农药条件："
            f"{condition}"
        )

    seen = set()

    for pesticide, level in matches:

        if pesticide in seen:
            raise ValueError(
                f"condition中农药重复："
                f"{condition}"
            )

        seen.add(
            pesticide
        )

        index = (
            PESTICIDE_TO_INDEX[
                pesticide
            ]
        )

        presence[
            index
        ] = 1.0

        concentration_level[
            index
        ] = LEVEL_TO_CODE[
            level
        ]

    return ConditionInfo(
        condition=condition,
        presence=presence,
        concentration_level=(
            concentration_level
        ),
        matrix_name=matrix_name,
        matrix_code=matrix_code,
    )


# ============================================================
# 8. Raman-axis adaptation
# ============================================================

def _adapt_to_model_axis(
    original_axis: np.ndarray,
    spectrum: np.ndarray,
    allowed_mask: np.ndarray | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    original_axis = np.asarray(
        original_axis,
        dtype=np.float64,
    )

    spectrum = np.asarray(
        spectrum,
        dtype=np.float64,
    )

    if (
        original_axis.ndim != 1
        or spectrum.ndim != 1
    ):
        raise ValueError(
            "Raman axis和spectrum必须为1D。"
        )

    if (
        original_axis.size
        != spectrum.size
    ):
        raise ValueError(
            "Raman axis和spectrum长度不一致。"
        )

    minimum_axis = float(
        original_axis[0]
    )

    maximum_axis = float(
        original_axis[-1]
    )

    valid_mask = (
        (MODEL_RAMAN_AXIS >= minimum_axis)
        & (MODEL_RAMAN_AXIS <= maximum_axis)
    )

    if allowed_mask is not None:

        allowed_mask = np.asarray(
            allowed_mask,
            dtype=bool,
        )

        if (
            allowed_mask.shape
            != valid_mask.shape
        ):
            raise ValueError(
                "allowed_mask长度错误。"
            )

        valid_mask = (
            valid_mask
            & allowed_mask
        )

    if int(
        valid_mask.sum()
    ) < 2:
        raise ValueError(
            "原始Raman范围与模型轴"
            "没有足够重叠。"
        )

    adapted = np.zeros(
        MODEL_LENGTH,
        dtype=np.float32,
    )

    adapted[
        valid_mask
    ] = np.interp(
        MODEL_RAMAN_AXIS[
            valid_mask
        ],
        original_axis,
        spectrum,
    ).astype(
        np.float32
    )

    return (
        adapted,
        valid_mask.astype(
            np.bool_
        ),
    )


# ============================================================
# 9. Preprocessing
# ============================================================

def _normalise_spectrum(
    spectrum: np.ndarray,
    valid_mask: np.ndarray,
    state: NormalizationState,
) -> np.ndarray:

    denominator = (
        state.maximum
        - state.minimum
    )

    if denominator <= 0:
        raise ValueError(
            "归一化范围无效。"
        )

    output = (
        spectrum.astype(
            np.float32
        )
        - state.minimum
    ) / denominator

    output = output.astype(
        np.float32
    )

    # 无效Raman区域必须保持0。
    output[
        ~valid_mask
    ] = 0.0

    return output


def _build_percentile_channels(
    spectrum: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:

    output = np.zeros(
        (
            4,
            MODEL_LENGTH,
        ),
        dtype=np.float32,
    )

    valid_values = spectrum[
        valid_mask
    ]

    if (
        valid_values.size
        == 0
    ):
        return output

    percentile_values = (
        95.0,
        85.0,
        75.0,
        50.0,
    )

    for channel_index, percentile in enumerate(
        percentile_values
    ):

        threshold = float(
            np.percentile(
                valid_values,
                percentile,
            )
        )

        keep = (
            valid_mask
            & (
                spectrum
                >= threshold
            )
        )

        output[
            channel_index,
            keep,
        ] = spectrum[
            keep
        ]

    return output


def _build_smoothed_spectrum(
    spectrum: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:

    output = np.zeros(
        MODEL_LENGTH,
        dtype=np.float32,
    )

    valid_indices = (
        np.flatnonzero(
            valid_mask
        )
    )

    if (
        valid_indices.size
        == 0
    ):
        return output

    values = spectrum[
        valid_indices
    ]

    if values.size < 3:
        output[
            valid_indices
        ] = values
        return output

    window_length = min(
        32,
        int(
            values.size
        ),
    )

    window = (
        signal.windows.hann(
            window_length
        )
    )

    window_sum = float(
        window.sum()
    )

    if window_sum <= 0:
        output[
            valid_indices
        ] = values
        return output

    smoothed = signal.convolve(
        values,
        window,
        mode="same",
        method="direct",
    ) / window_sum

    output[
        valid_indices
    ] = smoothed.astype(
        np.float32
    )

    return output


# ============================================================
# 10. Repository
# ============================================================

class SERSDataRepository:

    def __init__(
        self,
        real_root: str | Path = REAL_DATA_ROOT,
        generated_root: str | Path = GENERATED_DATA_ROOT,
    ) -> None:

        self.real_root = Path(
            real_root
        )

        self.generated_root = Path(
            generated_root
        )

        self.real_data: dict[
            str,
            dict[str, Any],
        ] = {}

        self._load_real_data()

        self.normalization_state = (
            self._fit_training_normalization()
        )

    def _load_real_data(
        self,
    ) -> None:

        files = _collect_spectrum_files(
            self.real_root
        )

        if len(files) != 126:
            raise RuntimeError(
                "真实数据应该有126个源文件，"
                f"当前找到：{len(files)}"
            )

        for path in files:

            condition = (
                _condition_from_real_file(
                    path
                )
            )

            if (
                condition
                in self.real_data
            ):
                raise RuntimeError(
                    "真实condition重复："
                    f"{condition}"
                )

            condition_info = (
                parse_condition(
                    condition
                )
            )

            axis, spectra = (
                _read_spectrum_table(
                    path
                )
            )

            if (
                spectra.shape[1]
                != 20
            ):
                raise RuntimeError(
                    "每个真实文件必须包含"
                    "20条mapping光谱："
                    f"{path}，"
                    f"当前={spectra.shape[1]}"
                )

            # 计算该condition的真实有效Raman范围。
            _, valid_mask = (
                _adapt_to_model_axis(
                    axis,
                    np.zeros_like(
                        axis,
                        dtype=np.float64,
                    ),
                )
            )

            self.real_data[
                condition
            ] = {
                "path": path,
                "condition_info": (
                    condition_info
                ),
                "axis": axis,
                "spectra": spectra,
                "valid_mask": valid_mask,
            }

    def _fit_training_normalization(
        self,
    ) -> NormalizationState:

        global_minimum = np.inf
        global_maximum = -np.inf

        # 只允许使用真实训练集前12条。
        #
        # validation/test/generated均不能参与
        # normalization拟合。
        for item in (
            self.real_data.values()
        ):

            axis = item[
                "axis"
            ]

            spectra = item[
                "spectra"
            ]

            for spectrum_index in (
                REAL_TRAIN_INDICES
            ):

                adapted, mask = (
                    _adapt_to_model_axis(
                        axis,
                        spectra[
                            :,
                            spectrum_index,
                        ],
                    )
                )

                valid_values = adapted[
                    mask
                ]

                local_minimum = float(
                    valid_values.min()
                )

                local_maximum = float(
                    valid_values.max()
                )

                global_minimum = min(
                    global_minimum,
                    local_minimum,
                )

                global_maximum = max(
                    global_maximum,
                    local_maximum,
                )

        if (
            not np.isfinite(
                global_minimum
            )
            or not np.isfinite(
                global_maximum
            )
        ):
            raise RuntimeError(
                "无法拟合归一化范围。"
            )

        if (
            global_maximum
            <= global_minimum
        ):
            raise RuntimeError(
                "归一化最大值必须大于最小值。"
            )

        return NormalizationState(
            minimum=float(
                global_minimum
            ),
            maximum=float(
                global_maximum
            ),
        )

    def discover_generated_data(
        self,
        require_complete: bool = True,
    ) -> dict[
        str,
        Path,
    ]:

        files = (
            _collect_spectrum_files(
                self.generated_root
            )
        )

        output: dict[
            str,
            Path,
        ] = {}

        for path in files:

            condition = (
                _condition_from_generated_file(
                    path
                )
            )

            if (
                condition
                not in self.real_data
            ):
                raise RuntimeError(
                    "生成数据condition"
                    "在真实数据中不存在："
                    f"{condition}"
                )

            if condition in output:
                raise RuntimeError(
                    "生成condition重复："
                    f"{condition}"
                )

            output[
                condition
            ] = path

        if (
            require_complete
            and len(output) != 126
        ):
            raise RuntimeError(
                "正式使用生成数据前必须完成"
                "全部126个condition。"
                f"当前={len(output)}"
            )

        return output


# ============================================================
# 11. Dataset
# ============================================================

class SERSDataset(Dataset):

    def __init__(
        self,
        repository: SERSDataRepository,
        split: str,
        include_generated: bool = False,
        maximum_generated_per_condition: int | None = None,
        require_complete_generated: bool = True,
    ) -> None:

        super().__init__()

        if split not in {
            "train",
            "validation",
            "test",
        }:
            raise ValueError(
                "split必须是"
                "train/validation/test，"
                f"当前={split}"
            )

        if (
            include_generated
            and split != "train"
        ):
            raise ValueError(
                "生成光谱只允许加入train，"
                "validation/test必须只用真实数据。"
            )

        self.repository = (
            repository
        )

        self.split = split

        self.include_generated = (
            include_generated
        )

        self.maximum_generated_per_condition = (
            maximum_generated_per_condition
        )

        self.samples: list[
            dict[str, Any]
        ] = []

        self._add_real_samples()

        if include_generated:
            self._add_generated_samples(
                require_complete=(
                    require_complete_generated
                )
            )

    def _get_real_indices(
        self,
    ) -> range:

        if self.split == "train":
            return REAL_TRAIN_INDICES

        if (
            self.split
            == "validation"
        ):
            return (
                REAL_VALIDATION_INDICES
            )

        return REAL_TEST_INDICES

    def _add_real_samples(
        self,
    ) -> None:

        indices = (
            self._get_real_indices()
        )

        for condition in sorted(
            self.repository.real_data
        ):

            item = (
                self.repository.real_data[
                    condition
                ]
            )

            axis = item[
                "axis"
            ]

            spectra = item[
                "spectra"
            ]

            for spectrum_index in indices:

                spectrum, valid_mask = (
                    _adapt_to_model_axis(
                        axis,
                        spectra[
                            :,
                            spectrum_index,
                        ],
                    )
                )

                self.samples.append(
                    {
                        "spectrum": spectrum,
                        "valid_mask": (
                            valid_mask
                        ),
                        "condition": (
                            condition
                        ),
                        "condition_info": (
                            item[
                                "condition_info"
                            ]
                        ),
                        "source": "real",
                        "source_file": str(
                            item["path"]
                        ),
                        "spectrum_index": int(
                            spectrum_index
                        ),
                    }
                )

    def _add_generated_samples(
        self,
        require_complete: bool,
    ) -> None:

        generated_files = (
            self.repository
            .discover_generated_data(
                require_complete=(
                    require_complete
                )
            )
        )

        for condition in sorted(
            generated_files
        ):

            path = generated_files[
                condition
            ]

            axis, spectra = (
                _read_spectrum_table(
                    path
                )
            )

            real_item = (
                self.repository.real_data[
                    condition
                ]
            )

            real_valid_mask = (
                real_item[
                    "valid_mask"
                ]
            )

            number_of_generated = (
                spectra.shape[1]
            )

            if (
                self.maximum_generated_per_condition
                is not None
            ):
                number_of_generated = min(
                    number_of_generated,
                    int(
                        self.maximum_generated_per_condition
                    ),
                )

            for spectrum_index in range(
                number_of_generated
            ):

                spectrum, _ = (
                    _adapt_to_model_axis(
                        axis,
                        spectra[
                            :,
                            spectrum_index,
                        ],
                        allowed_mask=(
                            real_valid_mask
                        ),
                    )
                )

                self.samples.append(
                    {
                        "spectrum": spectrum,
                        "valid_mask": (
                            real_valid_mask.copy()
                        ),
                        "condition": condition,
                        "condition_info": (
                            real_item[
                                "condition_info"
                            ]
                        ),
                        "source": (
                            "generated"
                        ),
                        "source_file": str(
                            path
                        ),
                        "spectrum_index": int(
                            spectrum_index
                        ),
                    }
                )

    def __len__(
        self,
    ) -> int:

        return len(
            self.samples
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:

        item = self.samples[
            index
        ]

        spectrum = item[
            "spectrum"
        ]

        valid_mask = item[
            "valid_mask"
        ]

        condition_info: ConditionInfo = (
            item[
                "condition_info"
            ]
        )

        normalized = (
            _normalise_spectrum(
                spectrum,
                valid_mask,
                self.repository
                .normalization_state,
            )
        )

        percentile_channels = (
            _build_percentile_channels(
                normalized,
                valid_mask,
            )
        )

        smoothed = (
            _build_smoothed_spectrum(
                normalized,
                valid_mask,
            )
        )

        return {
            # SERSFormer2的三路光谱输入
            "raw": torch.from_numpy(
                normalized[
                    np.newaxis,
                    :,
                ]
            ).float(),

            "percentile": (
                torch.from_numpy(
                    percentile_channels
                ).float()
            ),

            "smoothed": (
                torch.from_numpy(
                    smoothed[
                        np.newaxis,
                        :,
                    ]
                ).float()
            ),

            # Raman有效区域
            "valid_mask": (
                torch.from_numpy(
                    valid_mask
                ).bool()
            ),

            "raman_axis": (
                torch.from_numpy(
                    MODEL_RAMAN_AXIS.copy()
                ).float()
            ),

            # 多标签分类
            # 顺序固定：[DEL, CHL, TEB]
            "class_target": (
                torch.from_numpy(
                    condition_info
                    .presence.copy()
                ).float()
            ),

            # 暂时只是S/M/H等级。
            # 不能直接当最终浓度回归标签。
            "concentration_level": (
                torch.from_numpy(
                    condition_info
                    .concentration_level.copy()
                ).long()
            ),

            "matrix_target": (
                torch.tensor(
                    condition_info
                    .matrix_code,
                    dtype=torch.long,
                )
            ),

            # Metadata
            "condition": item[
                "condition"
            ],

            "matrix_name": (
                condition_info
                .matrix_name
            ),

            "source": item[
                "source"
            ],

            "source_file": item[
                "source_file"
            ],

            "spectrum_index": (
                item[
                    "spectrum_index"
                ]
            ),
        }


# ============================================================
# 12. Standard dataset builder
# ============================================================

def build_datasets(
    include_generated_train: bool = False,
    maximum_generated_per_condition: int | None = None,
    require_complete_generated: bool = True,
) -> tuple[
    SERSDataset,
    SERSDataset,
    SERSDataset,
]:

    repository = (
        SERSDataRepository()
    )

    train_dataset = (
        SERSDataset(
            repository=repository,
            split="train",
            include_generated=(
                include_generated_train
            ),
            maximum_generated_per_condition=(
                maximum_generated_per_condition
            ),
            require_complete_generated=(
                require_complete_generated
            ),
        )
    )

    validation_dataset = (
        SERSDataset(
            repository=repository,
            split="validation",
            include_generated=False,
        )
    )

    test_dataset = (
        SERSDataset(
            repository=repository,
            split="test",
            include_generated=False,
        )
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
    )
