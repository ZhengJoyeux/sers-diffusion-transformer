#!/usr/bin/env python3
"""
每个 condition 使用前 12 条 training 光谱自动筛选无峰背景区。

功能：
1. 只使用固定 training（默认每个源文件前 12 条光谱）。
2. 对 12 条 training 分别进行 broad-baseline 去除和轻度平滑。
3. 检测每条 training 的候选正向 Raman 峰。
4. 统计峰在 12 条 training 中的重复支持度，构建稳定峰保护区。
5. 在非保护区内使用：
   - 条件中位谱的低导数
   - training 间的低稳健波动
   筛选候选背景点。
6. 只保留达到最小连续 Raman 宽度的背景区间。
7. 使用筛选出的原始 training 背景点计算：
   - median / Q05 / Q01 / minimum
   - MAD / robust sigma
   - negative-only statistics
   - conservative / strict negative noise floor
8. 导出 condition 汇总、背景区间、稳定峰、逐点 mask、Excel 总表和文本报告。
9. 可选导出每个 condition 的诊断图。

这只是“training-only 背景区诊断/标定”，不会修改训练数据、checkpoint 或生成光谱。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


SUPPORTED_SUFFIXES = {".xlsx", ".csv"}


@dataclass
class AnalysisConfig:
    training_count: int = 12

    baseline_sigma_cm1: float = 30.0
    smoothing_sigma_cm1: float = 2.0

    minimum_peak_distance_cm1: float = 12.0
    peak_support_half_width_cm1: float = 5.0
    stable_peak_minimum_support: int = 4
    stable_peak_protection_half_width_cm1: float = 10.0

    noise_prominence_sigma_multiplier: float = 4.0
    relative_prominence: float = 0.06

    derivative_quantile: float = 0.50
    variation_quantile: float = 0.50

    minimum_background_width_cm1: float = 12.0
    edge_exclusion_cm1: float = 10.0

    noise_lower_quantile: float = 0.01
    robust_lower_sigma_multiplier: float = 3.5

    safety_margin_sigma_fraction: float = 0.50
    minimum_safety_margin_intensity: float = 3.0
    maximum_safety_margin_intensity: float = 10.0

    minimum_background_fraction_warning: float = 0.05


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "使用每个 condition 的前 12 条 training 光谱，"
            "自动筛选无峰背景区并计算背景负噪声下界。"
        )
    )

    parser.add_argument(
        "--input-directory",
        default="data/input",
        help="输入数据根目录。",
    )
    parser.add_argument(
        "--output-directory",
        default="outputs/diagnostics/automatic_peak_free_background",
        help="诊断结果输出目录。",
    )
    parser.add_argument(
        "--training-count",
        type=int,
        default=12,
        help="每个源文件使用的 training 光谱数量；固定划分当前应为12。",
    )
    parser.add_argument(
        "--stable-peak-minimum-support",
        type=int,
        default=4,
        help=(
            "一个峰至少被多少条 training 支持后，"
            "视为稳定峰并保护；默认4/12。"
        ),
    )
    parser.add_argument(
        "--derivative-quantile",
        type=float,
        default=0.50,
        help="低导数筛选分位数，默认0.50。",
    )
    parser.add_argument(
        "--variation-quantile",
        type=float,
        default=0.50,
        help="低跨 mapping 波动筛选分位数，默认0.50。",
    )
    parser.add_argument(
        "--minimum-background-width-cm1",
        type=float,
        default=12.0,
        help="背景连续区间最小 Raman 宽度。",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="额外保存每个 condition 的 PNG 诊断图。",
    )

    return parser.parse_args()


def validate_config(config: AnalysisConfig) -> None:
    if config.training_count < 2:
        raise ValueError("training_count必须至少为2。")

    if config.baseline_sigma_cm1 <= 0:
        raise ValueError("baseline_sigma_cm1必须大于0。")

    if config.smoothing_sigma_cm1 <= 0:
        raise ValueError("smoothing_sigma_cm1必须大于0。")

    if config.minimum_peak_distance_cm1 <= 0:
        raise ValueError("minimum_peak_distance_cm1必须大于0。")

    if config.peak_support_half_width_cm1 < 0:
        raise ValueError("peak_support_half_width_cm1不能为负。")

    if not 1 <= config.stable_peak_minimum_support <= config.training_count:
        raise ValueError(
            "stable_peak_minimum_support必须位于"
            f"[1,{config.training_count}]。"
        )

    if config.stable_peak_protection_half_width_cm1 < 0:
        raise ValueError(
            "stable_peak_protection_half_width_cm1不能为负。"
        )

    if config.noise_prominence_sigma_multiplier <= 0:
        raise ValueError(
            "noise_prominence_sigma_multiplier必须大于0。"
        )

    if config.relative_prominence < 0:
        raise ValueError("relative_prominence不能为负。")

    if not 0 < config.derivative_quantile <= 1:
        raise ValueError("derivative_quantile必须位于(0,1]。")

    if not 0 < config.variation_quantile <= 1:
        raise ValueError("variation_quantile必须位于(0,1]。")

    if config.minimum_background_width_cm1 <= 0:
        raise ValueError(
            "minimum_background_width_cm1必须大于0。"
        )

    if config.edge_exclusion_cm1 < 0:
        raise ValueError("edge_exclusion_cm1不能为负。")

    if not 0 < config.noise_lower_quantile <= 0.10:
        raise ValueError(
            "noise_lower_quantile必须位于(0,0.10]。"
        )

    if config.robust_lower_sigma_multiplier <= 0:
        raise ValueError(
            "robust_lower_sigma_multiplier必须大于0。"
        )

    if config.safety_margin_sigma_fraction < 0:
        raise ValueError(
            "safety_margin_sigma_fraction不能为负。"
        )

    if config.minimum_safety_margin_intensity < 0:
        raise ValueError(
            "minimum_safety_margin_intensity不能为负。"
        )

    if (
        config.maximum_safety_margin_intensity
        < config.minimum_safety_margin_intensity
    ):
        raise ValueError(
            "maximum_safety_margin_intensity不能小于minimum。"
        )


def read_spectrum_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix.lower() == ".xlsx":
        frame = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError(f"不支持的文件类型：{path}")

    if frame.shape[1] < 2:
        raise ValueError(f"{path}至少需要Raman轴+1条光谱。")

    axis = pd.to_numeric(
        frame.iloc[:, 0],
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    spectra = (
        frame.iloc[:, 1:]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float64)
    )

    if not np.all(np.isfinite(axis)):
        raise ValueError(f"{path}的Raman轴含NaN/Inf。")

    if not np.all(np.isfinite(spectra)):
        raise ValueError(f"{path}的光谱含NaN/Inf。")

    if axis.ndim != 1 or axis.size < 3:
        raise ValueError(f"{path}的Raman轴无效。")

    differences = np.diff(axis)

    if np.all(differences < 0):
        axis = axis[::-1].copy()
        spectra = spectra[::-1, :].copy()
        differences = np.diff(axis)

    if not np.all(differences > 0):
        raise ValueError(
            f"{path}的Raman轴必须严格单调，不能包含重复点。"
        )

    return axis, spectra


def robust_sigma(values: np.ndarray, axis=None) -> np.ndarray:
    median = np.median(
        values,
        axis=axis,
        keepdims=True if axis is not None else False,
    )

    if axis is None:
        mad = np.median(np.abs(values - median))
    else:
        mad = np.median(
            np.abs(values - median),
            axis=axis,
        )

    return 1.4826 * mad


def find_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if mask.ndim != 1:
        raise ValueError("mask必须是一维数组。")

    padded = np.pad(
        mask.astype(np.int8),
        (1, 1),
        constant_values=0,
    )
    changes = np.diff(padded)

    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)

    return [
        (int(start), int(stop))
        for start, stop in zip(starts, stops, strict=True)
    ]


def expand_mask_by_cm1(
    mask: np.ndarray,
    axis: np.ndarray,
    half_width_cm1: float,
) -> np.ndarray:
    expanded = np.zeros_like(mask, dtype=bool)

    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return expanded

    for index in indices:
        left = np.searchsorted(
            axis,
            axis[index] - half_width_cm1,
            side="left",
        )
        right = np.searchsorted(
            axis,
            axis[index] + half_width_cm1,
            side="right",
        )
        expanded[left:right] = True

    return expanded


def remove_short_background_runs(
    candidate_mask: np.ndarray,
    axis: np.ndarray,
    minimum_width_cm1: float,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    kept = np.zeros_like(candidate_mask, dtype=bool)
    kept_runs: list[tuple[int, int]] = []

    for start, stop in find_true_runs(candidate_mask):
        last = stop - 1

        if last <= start:
            width = 0.0
        else:
            width = float(axis[last] - axis[start])

        if width >= minimum_width_cm1:
            kept[start:stop] = True
            kept_runs.append((start, stop))

    return kept, kept_runs


def detect_training_peaks(
    axis: np.ndarray,
    training_rows: np.ndarray,
    config: AnalysisConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, float | int]],
]:
    """
    返回：
    local_residual_rows
    smoothed_local_rows
    support_count_per_axis_point
    protected_mask
    stable_peak_cluster_records
    """

    spacing = float(np.median(np.diff(axis)))

    baseline_sigma_points = (
        config.baseline_sigma_cm1 / spacing
    )
    smoothing_sigma_points = (
        config.smoothing_sigma_cm1 / spacing
    )

    broad = gaussian_filter1d(
        training_rows,
        sigma=baseline_sigma_points,
        axis=1,
        mode="nearest",
        truncate=4.0,
    )

    local = training_rows - broad

    smooth_local = gaussian_filter1d(
        local,
        sigma=smoothing_sigma_points,
        axis=1,
        mode="nearest",
        truncate=4.0,
    )

    high_frequency = local - smooth_local

    peak_support_matrix = np.zeros(
        training_rows.shape,
        dtype=bool,
    )

    peak_events: list[dict[str, float | int]] = []

    minimum_distance_points = max(
        1,
        int(
            round(
                config.minimum_peak_distance_cm1 / spacing
            )
        ),
    )

    for spectrum_index in range(training_rows.shape[0]):
        smooth = smooth_local[spectrum_index]
        high = high_frequency[spectrum_index]

        high_median = float(np.median(high))
        high_mad = float(
            np.median(
                np.abs(high - high_median)
            )
        )
        noise_sigma = max(
            1.4826 * high_mad,
            1.0e-12,
        )

        positive_scale = max(
            float(
                np.quantile(smooth, 0.99)
                - np.median(smooth)
            ),
            1.0e-12,
        )

        prominence_threshold = max(
            config.noise_prominence_sigma_multiplier
            * noise_sigma,
            config.relative_prominence
            * positive_scale,
        )

        peaks, properties = find_peaks(
            smooth,
            prominence=prominence_threshold,
            distance=minimum_distance_points,
        )

        prominences = properties.get(
            "prominences",
            np.zeros(peaks.size, dtype=np.float64),
        )

        for peak_index, prominence in zip(
            peaks,
            prominences,
            strict=True,
        ):
            position = float(axis[peak_index])

            left = np.searchsorted(
                axis,
                position
                - config.peak_support_half_width_cm1,
                side="left",
            )
            right = np.searchsorted(
                axis,
                position
                + config.peak_support_half_width_cm1,
                side="right",
            )

            peak_support_matrix[
                spectrum_index,
                left:right,
            ] = True

            peak_events.append({
                "spectrum_index": spectrum_index + 1,
                "position_cm-1": position,
                "prominence": float(prominence),
                "prominence_threshold": float(
                    prominence_threshold
                ),
                "noise_sigma": float(noise_sigma),
            })

    support_count = np.sum(
        peak_support_matrix,
        axis=0,
    ).astype(np.int64)

    stable_core = (
        support_count
        >= config.stable_peak_minimum_support
    )

    protected_mask = expand_mask_by_cm1(
        stable_core,
        axis,
        config.stable_peak_protection_half_width_cm1,
    )

    stable_clusters: list[dict[str, float | int]] = []

    for cluster_index, (start, stop) in enumerate(
        find_true_runs(stable_core),
        start=1,
    ):
        cluster_support = support_count[start:stop]

        local_peak = int(
            np.argmax(cluster_support)
        )
        representative_index = start + local_peak
        representative_position = float(
            axis[representative_index]
        )

        nearby_prominences = [
            float(event["prominence"])
            for event in peak_events
            if abs(
                float(event["position_cm-1"])
                - representative_position
            )
            <= config.peak_support_half_width_cm1
        ]

        stable_clusters.append({
            "stable_peak_id": cluster_index,
            "peak_start_cm-1": float(axis[start]),
            "peak_end_cm-1": float(axis[stop - 1]),
            "representative_position_cm-1":
                representative_position,
            "maximum_support_count": int(
                np.max(cluster_support)
            ),
            "maximum_support_fraction": float(
                np.max(cluster_support)
                / config.training_count
            ),
            "median_prominence": (
                float(np.median(nearby_prominences))
                if nearby_prominences
                else np.nan
            ),
            "protected_start_cm-1": max(
                float(axis[0]),
                float(axis[start])
                - config.stable_peak_protection_half_width_cm1,
            ),
            "protected_end_cm-1": min(
                float(axis[-1]),
                float(axis[stop - 1])
                + config.stable_peak_protection_half_width_cm1,
            ),
        })

    return (
        local,
        smooth_local,
        support_count,
        protected_mask,
        stable_clusters,
    )


def calculate_background_mask(
    axis: np.ndarray,
    smooth_local: np.ndarray,
    protected_mask: np.ndarray,
    config: AnalysisConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    list[tuple[int, int]],
]:
    median_smooth = np.median(
        smooth_local,
        axis=0,
    )

    derivative = np.abs(
        np.gradient(
            median_smooth,
            axis,
        )
    )

    per_point_median = np.median(
        smooth_local,
        axis=0,
    )
    per_point_mad = np.median(
        np.abs(
            smooth_local
            - per_point_median[np.newaxis, :]
        ),
        axis=0,
    )
    mapping_variation = (
        1.4826 * per_point_mad
    )

    edge_mask = np.ones(
        axis.size,
        dtype=bool,
    )

    if config.edge_exclusion_cm1 > 0:
        edge_mask &= (
            axis
            >= axis[0]
            + config.edge_exclusion_cm1
        )
        edge_mask &= (
            axis
            <= axis[-1]
            - config.edge_exclusion_cm1
        )

    threshold_reference = (
        (~protected_mask)
        & edge_mask
    )

    if np.sum(threshold_reference) < 10:
        raise RuntimeError(
            "可用于估计低导数/低波动阈值的非峰点过少。"
        )

    derivative_threshold = float(
        np.quantile(
            derivative[
                threshold_reference
            ],
            config.derivative_quantile,
        )
    )

    variation_threshold = float(
        np.quantile(
            mapping_variation[
                threshold_reference
            ],
            config.variation_quantile,
        )
    )

    low_derivative_mask = (
        derivative
        <= derivative_threshold
    )

    low_variation_mask = (
        mapping_variation
        <= variation_threshold
    )

    candidate = (
        (~protected_mask)
        & edge_mask
        & low_derivative_mask
        & low_variation_mask
    )

    background_mask, background_runs = (
        remove_short_background_runs(
            candidate,
            axis,
            config.minimum_background_width_cm1,
        )
    )

    return (
        background_mask,
        low_derivative_mask,
        low_variation_mask,
        derivative,
        derivative_threshold,
        variation_threshold,
        background_runs,
    )


def calculate_noise_statistics(
    training_rows: np.ndarray,
    background_mask: np.ndarray,
    config: AnalysisConfig,
) -> dict[str, float | int]:
    if not np.any(background_mask):
        return {
            "background_value_count": 0,
            "background_negative_value_count": 0,
            "background_negative_fraction": np.nan,
            "background_minimum": np.nan,
            "background_q01": np.nan,
            "background_q05": np.nan,
            "background_median": np.nan,
            "background_mad": np.nan,
            "background_robust_sigma": np.nan,
            "negative_only_median": np.nan,
            "negative_only_q05": np.nan,
            "robust_lower_bound": np.nan,
            "safety_margin": np.nan,
            "strict_negative_floor": np.nan,
            "conservative_negative_floor": np.nan,
            "recommended_negative_floor": np.nan,
        }

    values = training_rows[
        :,
        background_mask,
    ].reshape(-1)

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        raise RuntimeError(
            "背景mask存在，但背景强度值为空。"
        )

    median = float(
        np.median(values)
    )
    mad = float(
        np.median(
            np.abs(
                values - median
            )
        )
    )
    sigma = max(
        1.4826 * mad,
        1.0e-12,
    )

    q01 = float(
        np.quantile(
            values,
            config.noise_lower_quantile,
        )
    )
    q05 = float(
        np.quantile(
            values,
            0.05,
        )
    )

    robust_lower = float(
        median
        - config.robust_lower_sigma_multiplier
        * sigma
    )

    safety_margin = float(
        np.clip(
            config.safety_margin_sigma_fraction
            * sigma,
            config.minimum_safety_margin_intensity,
            config.maximum_safety_margin_intensity,
        )
    )

    # strict:
    # 取 q01 和 robust lower 中较不负者，灵敏度更高。
    strict_floor = float(
        max(
            q01,
            robust_lower,
        )
        - safety_margin
    )

    # conservative:
    # 取两者中更负者，优先避免误伤真实 training 尾部。
    conservative_floor = float(
        min(
            q01,
            robust_lower,
        )
        - safety_margin
    )

    negative_values = values[
        values < 0.0
    ]

    negative_median = (
        float(np.median(negative_values))
        if negative_values.size
        else np.nan
    )

    negative_q05 = (
        float(
            np.quantile(
                negative_values,
                0.05,
            )
        )
        if negative_values.size
        else np.nan
    )

    # 当前只做诊断。初始推荐 conservative floor，
    # 后续经真实 training / RAW generated 验证后再决定是否用于 guard。
    recommended_floor = conservative_floor

    return {
        "background_value_count":
            int(values.size),
        "background_negative_value_count":
            int(negative_values.size),
        "background_negative_fraction":
            float(
                negative_values.size
                / values.size
            ),
        "background_minimum":
            float(np.min(values)),
        "background_q01":
            q01,
        "background_q05":
            q05,
        "background_median":
            median,
        "background_mad":
            mad,
        "background_robust_sigma":
            sigma,
        "negative_only_median":
            negative_median,
        "negative_only_q05":
            negative_q05,
        "robust_lower_bound":
            robust_lower,
        "safety_margin":
            safety_margin,
        "strict_negative_floor":
            strict_floor,
        "conservative_negative_floor":
            conservative_floor,
        "recommended_negative_floor":
            recommended_floor,
    }


def save_condition_plot(
    output_path: Path,
    axis: np.ndarray,
    training_rows: np.ndarray,
    background_mask: np.ndarray,
    protected_mask: np.ndarray,
    stable_clusters: list[dict[str, float | int]],
    recommended_floor: float,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    median_raw = np.median(
        training_rows,
        axis=0,
    )

    fig, ax = plt.subplots(
        figsize=(13, 5),
    )

    ax.plot(
        axis,
        median_raw,
        linewidth=1.2,
        label="Training median",
    )

    for start, stop in find_true_runs(
        background_mask
    ):
        ax.axvspan(
            axis[start],
            axis[stop - 1],
            alpha=0.18,
        )

    for start, stop in find_true_runs(
        protected_mask
    ):
        ax.axvspan(
            axis[start],
            axis[stop - 1],
            alpha=0.08,
        )

    if np.isfinite(recommended_floor):
        ax.axhline(
            recommended_floor,
            linestyle="--",
            linewidth=1.0,
            label=(
                "Recommended negative floor "
                f"{recommended_floor:.2f}"
            ),
        )

    for cluster in stable_clusters:
        ax.axvline(
            float(
                cluster[
                    "representative_position_cm-1"
                ]
            ),
            linewidth=0.7,
            alpha=0.45,
        )

    ax.set_xlabel(
        "Raman shift (cm$^{-1}$)"
    )
    ax.set_ylabel(
        "Intensity"
    )
    ax.set_title(
        output_path.stem
    )
    ax.legend(
        loc="best",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=180,
    )
    plt.close(fig)


def collect_input_files(
    input_directory: Path,
) -> list[Path]:
    return sorted(
        path
        for path in input_directory.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_SUFFIXES
        )
    )


def dataframe_or_empty(
    records: list[dict],
    columns: Iterable[str],
) -> pd.DataFrame:
    if records:
        return pd.DataFrame(records)

    return pd.DataFrame(
        columns=list(columns)
    )


def main() -> None:
    args = parse_arguments()

    config = AnalysisConfig(
        training_count=args.training_count,
        stable_peak_minimum_support=(
            args.stable_peak_minimum_support
        ),
        derivative_quantile=(
            args.derivative_quantile
        ),
        variation_quantile=(
            args.variation_quantile
        ),
        minimum_background_width_cm1=(
            args.minimum_background_width_cm1
        ),
    )

    validate_config(config)

    input_directory = Path(
        args.input_directory
    )

    output_directory = Path(
        args.output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_directory = (
        output_directory
        / "plots"
    )

    if args.save_plots:
        plot_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    files = collect_input_files(
        input_directory
    )

    if not files:
        raise RuntimeError(
            f"{input_directory}中没有找到Excel/CSV。"
        )

    print(
        "===== 自动无峰背景区筛选 ====="
    )
    print(
        "输入目录：",
        input_directory.resolve(),
    )
    print(
        "源文件数量：",
        len(files),
    )
    print(
        "每condition使用training数量：",
        config.training_count,
    )
    print(
        "稳定峰支持阈值：",
        f"{config.stable_peak_minimum_support}/"
        f"{config.training_count}",
    )
    print(
        "低导数quantile：",
        config.derivative_quantile,
    )
    print(
        "低波动quantile：",
        config.variation_quantile,
    )
    print(
        "最小连续背景宽度：",
        f"{config.minimum_background_width_cm1:.1f} cm^-1",
    )

    condition_records: list[dict] = []
    interval_records: list[dict] = []
    stable_peak_records: list[dict] = []
    point_records: list[dict] = []

    warning_conditions: list[str] = []

    for file_index, path in enumerate(
        files,
        start=1,
    ):
        axis, spectra_by_column = (
            read_spectrum_table(path)
        )

        if (
            spectra_by_column.shape[1]
            < config.training_count
        ):
            raise RuntimeError(
                f"{path}: 光谱列数量"
                f"{spectra_by_column.shape[1]} "
                f"< training_count="
                f"{config.training_count}"
            )

        training_rows = (
            spectra_by_column[
                :,
                : config.training_count,
            ].T
        )

        condition = path.stem

        (
            local,
            smooth_local,
            support_count,
            protected_mask,
            stable_clusters,
        ) = detect_training_peaks(
            axis,
            training_rows,
            config,
        )

        (
            background_mask,
            low_derivative_mask,
            low_variation_mask,
            derivative,
            derivative_threshold,
            variation_threshold,
            background_runs,
        ) = calculate_background_mask(
            axis,
            smooth_local,
            protected_mask,
            config,
        )

        noise_statistics = (
            calculate_noise_statistics(
                training_rows,
                background_mask,
                config,
            )
        )

        background_point_count = int(
            np.sum(background_mask)
        )

        background_fraction = float(
            background_point_count
            / axis.size
        )

        protected_fraction = float(
            np.mean(protected_mask)
        )

        total_background_width = float(
            sum(
                max(
                    0.0,
                    float(
                        axis[stop - 1]
                        - axis[start]
                    ),
                )
                for start, stop
                in background_runs
            )
        )

        warning = ""

        if background_point_count == 0:
            warning = (
                "NO_BACKGROUND_REGION"
            )
        elif (
            background_fraction
            < config.minimum_background_fraction_warning
        ):
            warning = (
                "LOW_BACKGROUND_FRACTION"
            )

        if warning:
            warning_conditions.append(
                condition
            )

        condition_record = {
            "condition":
                condition,
            "source_file":
                str(path),
            "axis_start_cm-1":
                float(axis[0]),
            "axis_end_cm-1":
                float(axis[-1]),
            "axis_point_count":
                int(axis.size),
            "training_spectrum_count":
                int(training_rows.shape[0]),
            "stable_peak_cluster_count":
                int(len(stable_clusters)),
            "stable_peak_protected_point_count":
                int(np.sum(protected_mask)),
            "stable_peak_protected_fraction":
                protected_fraction,
            "derivative_threshold":
                derivative_threshold,
            "variation_threshold":
                variation_threshold,
            "background_interval_count":
                int(len(background_runs)),
            "background_point_count":
                background_point_count,
            "background_fraction":
                background_fraction,
            "background_total_width_cm-1":
                total_background_width,
            "warning":
                warning,
            **noise_statistics,
        }

        condition_records.append(
            condition_record
        )

        for interval_index, (
            start,
            stop,
        ) in enumerate(
            background_runs,
            start=1,
        ):
            interval_records.append({
                "condition":
                    condition,
                "background_interval_id":
                    interval_index,
                "start_cm-1":
                    float(axis[start]),
                "end_cm-1":
                    float(axis[stop - 1]),
                "width_cm-1":
                    float(
                        axis[stop - 1]
                        - axis[start]
                    ),
                "point_count":
                    int(stop - start),
            })

        for cluster in stable_clusters:
            stable_peak_records.append({
                "condition":
                    condition,
                **cluster,
            })

        median_smooth = np.median(
            smooth_local,
            axis=0,
        )

        per_point_median = np.median(
            smooth_local,
            axis=0,
        )
        per_point_mad = np.median(
            np.abs(
                smooth_local
                - per_point_median[
                    np.newaxis,
                    :
                ]
            ),
            axis=0,
        )
        mapping_variation = (
            1.4826 * per_point_mad
        )

        for point_index in range(
            axis.size
        ):
            point_records.append({
                "condition":
                    condition,
                "raman_shift_cm-1":
                    float(
                        axis[
                            point_index
                        ]
                    ),
                "stable_peak_support_count":
                    int(
                        support_count[
                            point_index
                        ]
                    ),
                "stable_peak_protected":
                    bool(
                        protected_mask[
                            point_index
                        ]
                    ),
                "median_smoothed_local_residual":
                    float(
                        median_smooth[
                            point_index
                        ]
                    ),
                "absolute_derivative":
                    float(
                        derivative[
                            point_index
                        ]
                    ),
                "low_derivative":
                    bool(
                        low_derivative_mask[
                            point_index
                        ]
                    ),
                "mapping_robust_variation":
                    float(
                        mapping_variation[
                            point_index
                        ]
                    ),
                "low_variation":
                    bool(
                        low_variation_mask[
                            point_index
                        ]
                    ),
                "selected_background":
                    bool(
                        background_mask[
                            point_index
                        ]
                    ),
            })

        if args.save_plots:
            save_condition_plot(
                plot_directory
                / f"{condition}.png",
                axis,
                training_rows,
                background_mask,
                protected_mask,
                stable_clusters,
                float(
                    noise_statistics[
                        "recommended_negative_floor"
                    ]
                ),
            )

        if (
            file_index <= 5
            or file_index == len(files)
            or file_index % 20 == 0
        ):
            print(
                f"[{file_index:3d}/"
                f"{len(files)}] "
                f"{condition}: "
                f"stable peaks="
                f"{len(stable_clusters)}, "
                f"background intervals="
                f"{len(background_runs)}, "
                f"background="
                f"{background_fraction:.2%}, "
                f"floor="
                f"{noise_statistics['recommended_negative_floor']:.3f}"
                if np.isfinite(
                    float(
                        noise_statistics[
                            "recommended_negative_floor"
                        ]
                    )
                )
                else (
                    f"[{file_index:3d}/"
                    f"{len(files)}] "
                    f"{condition}: "
                    "floor=NaN"
                )
            )

    condition_df = pd.DataFrame(
        condition_records
    )

    interval_df = dataframe_or_empty(
        interval_records,
        [
            "condition",
            "background_interval_id",
            "start_cm-1",
            "end_cm-1",
            "width_cm-1",
            "point_count",
        ],
    )

    stable_peak_df = dataframe_or_empty(
        stable_peak_records,
        [
            "condition",
            "stable_peak_id",
            "peak_start_cm-1",
            "peak_end_cm-1",
            "representative_position_cm-1",
            "maximum_support_count",
            "maximum_support_fraction",
            "median_prominence",
            "protected_start_cm-1",
            "protected_end_cm-1",
        ],
    )

    point_df = pd.DataFrame(
        point_records
    )

    condition_csv = (
        output_directory
        / "condition_background_summary.csv"
    )
    interval_csv = (
        output_directory
        / "background_intervals.csv"
    )
    stable_peak_csv = (
        output_directory
        / "stable_peak_regions.csv"
    )
    point_csv = (
        output_directory
        / "background_mask_points.csv"
    )
    excel_path = (
        output_directory
        / "automatic_peak_free_background_report.xlsx"
    )
    report_path = (
        output_directory
        / "summary_report.txt"
    )

    condition_df.to_csv(
        condition_csv,
        index=False,
        encoding="utf-8-sig",
    )
    interval_df.to_csv(
        interval_csv,
        index=False,
        encoding="utf-8-sig",
    )
    stable_peak_df.to_csv(
        stable_peak_csv,
        index=False,
        encoding="utf-8-sig",
    )
    point_df.to_csv(
        point_csv,
        index=False,
        encoding="utf-8-sig",
    )

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl",
    ) as writer:
        condition_df.to_excel(
            writer,
            sheet_name="condition_summary",
            index=False,
        )
        interval_df.to_excel(
            writer,
            sheet_name="background_intervals",
            index=False,
        )
        stable_peak_df.to_excel(
            writer,
            sheet_name="stable_peaks",
            index=False,
        )

    valid_floor = condition_df[
        "recommended_negative_floor"
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    valid_background_fraction = (
        condition_df[
            "background_fraction"
        ].replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()
    )

    report_lines = [
        "===== 自动无峰背景区筛选报告 =====",
        f"source_file_count: {len(files)}",
        (
            "training_spectrum_count_per_condition: "
            f"{config.training_count}"
        ),
        (
            "stable_peak_minimum_support: "
            f"{config.stable_peak_minimum_support}/"
            f"{config.training_count}"
        ),
        (
            "stable_peak_protection_half_width_cm-1: "
            f"{config.stable_peak_protection_half_width_cm1}"
        ),
        (
            "derivative_quantile: "
            f"{config.derivative_quantile}"
        ),
        (
            "variation_quantile: "
            f"{config.variation_quantile}"
        ),
        (
            "minimum_background_width_cm-1: "
            f"{config.minimum_background_width_cm1}"
        ),
        "",
        (
            "condition_with_background: "
            f"{int((condition_df['background_point_count'] > 0).sum())}"
            f"/{len(condition_df)}"
        ),
        (
            "warning_condition_count: "
            f"{len(warning_conditions)}"
        ),
    ]

    if not valid_background_fraction.empty:
        report_lines.extend([
            (
                "background_fraction_median: "
                f"{valid_background_fraction.median():.6f}"
            ),
            (
                "background_fraction_minimum: "
                f"{valid_background_fraction.min():.6f}"
            ),
            (
                "background_fraction_maximum: "
                f"{valid_background_fraction.max():.6f}"
            ),
        ])

    if not valid_floor.empty:
        report_lines.extend([
            "",
            (
                "recommended_negative_floor_median: "
                f"{valid_floor.median():.6f}"
            ),
            (
                "recommended_negative_floor_q05: "
                f"{valid_floor.quantile(0.05):.6f}"
            ),
            (
                "recommended_negative_floor_q95: "
                f"{valid_floor.quantile(0.95):.6f}"
            ),
            (
                "recommended_negative_floor_minimum: "
                f"{valid_floor.min():.6f}"
            ),
            (
                "recommended_negative_floor_maximum: "
                f"{valid_floor.max():.6f}"
            ),
        ])

    if warning_conditions:
        report_lines.extend([
            "",
            "warning_conditions:",
            *[
                f"  - {condition}"
                for condition
                in warning_conditions
            ],
        ])

    report_path.write_text(
        "\n".join(report_lines)
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "===== 完成 ====="
    )
    print(
        "有有效背景区的condition数 =",
        int(
            (
                condition_df[
                    "background_point_count"
                ] > 0
            ).sum()
        ),
        "/",
        len(condition_df),
    )
    print(
        "警告condition数 =",
        len(warning_conditions),
    )

    if not valid_background_fraction.empty:
        print(
            "背景点比例中位数 =",
            f"{valid_background_fraction.median():.2%}",
        )
        print(
            "背景点比例范围 =",
            f"{valid_background_fraction.min():.2%}"
            " ~ "
            f"{valid_background_fraction.max():.2%}",
        )

    if not valid_floor.empty:
        print(
            "推荐负噪声下界中位数 =",
            f"{valid_floor.median():.3f}",
        )
        print(
            "推荐负噪声下界5%~95% =",
            f"{valid_floor.quantile(0.05):.3f}"
            " ~ "
            f"{valid_floor.quantile(0.95):.3f}",
        )
        print(
            "推荐负噪声下界总范围 =",
            f"{valid_floor.min():.3f}"
            " ~ "
            f"{valid_floor.max():.3f}",
        )

    print()
    print(
        "condition汇总：",
        condition_csv,
    )
    print(
        "背景区间：",
        interval_csv,
    )
    print(
        "稳定峰区域：",
        stable_peak_csv,
    )
    print(
        "逐点mask：",
        point_csv,
    )
    print(
        "Excel总表：",
        excel_path,
    )
    print(
        "文本报告：",
        report_path,
    )

    if args.save_plots:
        print(
            "诊断图目录：",
            plot_directory,
        )


if __name__ == "__main__":
    main()
