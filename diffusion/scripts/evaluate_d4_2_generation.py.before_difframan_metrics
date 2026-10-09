"""Evaluate one or all conditional D4 generated-spectrum workbooks.

Generated and real spectra are unpaired. Sample-level MSE, cosine and Pearson
therefore use one consistent match: the same-condition held-out spectrum with
the smallest standardized-shape distance. Mean-spectrum metrics, pointwise
Wasserstein distance, within-condition diversity and QQ data are separate.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import wasserstein_distance

from src.checkpoint_manager import load_checkpoint_file
from src.configuration_loader import load_configuration, resolve_project_path
from src.intensity_normalizer import GlobalMinMaxNormalizer
from src.sers_generation_evaluator import (
    diversity_summary,
    nearest_reference_metrics,
    row_cosine,
    row_pearson,
    summarize_vector,
)
from src.spectrum_conditioning import (
    resolve_source_named_condition_from_metadata,
    source_named_conditions_from_metadata,
)
from src.spectrum_file_reader import read_spectrum_collection


EPSILON = 1.0e-12


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "计算D4条件生成谱的MSE、RMSE、余弦、Pearson、"
            "Wasserstein、QQ拟合、组内多样性和特征峰分布。"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--condition",
        required=True,
        help="输入文件原始条件名，或使用all评价检查点中的全部条件。",
    )
    parser.add_argument(
        "--generated",
        default=None,
        help=(
            "单条件时可填写生成xlsx；否则填写generated根目录。"
            "未填写时使用配置中的generated_spectrum_directory。"
        ),
    )
    parser.add_argument(
        "--reference-split",
        choices=("validation", "test"),
        default="validation",
        help="调参与模型选择使用validation；参数冻结后test只运行一次。",
    )
    parser.add_argument("--output-directory", default=None)
    parser.add_argument("--generated-pair-count", type=int, default=50000)
    parser.add_argument("--qq-quantile-count", type=int, default=101)
    parser.add_argument("--qq-oracle-repeats", type=int, default=256)
    parser.add_argument("--qq-oracle-confidence", type=float, default=0.975)
    parser.add_argument(
        "--plot-mode",
        choices=("all", "none"),
        default="all",
        help=(
            "all输出全部论文图；none只计算并保存数值CSV，适合快速迭代。"
            "该选项不改变任何数值指标。"
        ),
    )
    return parser.parse_args()


def _resolve_path(configuration: dict, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = resolve_project_path(configuration, path)
    return path.resolve()


def _generated_path(
    *,
    generated_argument: str | None,
    configured_root: Path,
    condition: str,
    allow_direct_file: bool,
) -> Path:
    supplied = (
        configured_root
        if generated_argument is None
        else Path(generated_argument).expanduser().resolve()
    )
    if supplied.is_file():
        if not allow_direct_file:
            raise ValueError("condition=all时--generated必须是生成结果根目录。")
        return supplied
    expected = supplied / condition / f"{condition}_generated.xlsx"
    if not expected.is_file():
        raise FileNotFoundError(f"找不到条件生成文件：{expected}")
    return expected


def _read_generated(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    frame = pd.read_excel(path)
    if frame.shape[1] < 3:
        raise ValueError(f"生成文件至少需要Raman轴和2条光谱：{path}")
    axis = frame.iloc[:, 0].to_numpy(dtype=np.float64)
    spectra = frame.iloc[:, 1:].to_numpy(dtype=np.float64).T
    if (
        axis.size < 3
        or not np.isfinite(axis).all()
        or not np.all(np.diff(axis) > 0.0)
        or not np.isfinite(spectra).all()
    ):
        raise ValueError(f"生成文件Raman轴或强度无效：{path}")
    return axis, spectra, [str(value) for value in frame.columns[1:]]


def _interpolate_collection(
    collection: Any,
    indices: np.ndarray,
    target_axis: np.ndarray,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for index in indices:
        source_axis = np.asarray(collection.raman_shifts[index], dtype=np.float64)
        if (
            target_axis[0] < source_axis[0] - 1.0e-6
            or target_axis[-1] > source_axis[-1] + 1.0e-6
        ):
            raise ValueError(
                "生成轴超出同条件真实轴范围，禁止外推："
                f"generated={target_axis[0]:.3f}--{target_axis[-1]:.3f}；"
                f"real={source_axis[0]:.3f}--{source_axis[-1]:.3f}。"
            )
        rows.append(
            np.interp(
                target_axis,
                source_axis,
                np.asarray(collection.spectra[index], dtype=np.float64),
            )
        )
    return np.stack(rows, axis=0)


def _center_metrics(
    reference: np.ndarray,
    generated: np.ndarray,
) -> dict[str, float]:
    real_mean = np.mean(reference, axis=0, keepdims=True)
    generated_mean = np.mean(generated, axis=0, keepdims=True)
    difference = generated_mean - real_mean
    return {
        "mean_spectrum_mse": float(np.mean(np.square(difference))),
        "mean_spectrum_rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "mean_spectrum_cosine": float(row_cosine(generated_mean, real_mean)[0]),
        "mean_spectrum_pearson": float(row_pearson(generated_mean, real_mean)[0]),
    }


def _pointwise_wasserstein(
    *,
    training: np.ndarray,
    reference: np.ndarray,
    generated: np.ndarray,
) -> dict[str, float]:
    distances = np.asarray(
        [
            wasserstein_distance(reference[:, point], generated[:, point])
            for point in range(generated.shape[1])
        ],
        dtype=np.float64,
    )
    training_range = float(
        np.percentile(training, 95.0) - np.percentile(training, 5.0)
    )
    normalized = distances / max(training_range, EPSILON)
    return {
        "wasserstein_raw_pointwise_mean": float(np.mean(distances)),
        "wasserstein_raw_pointwise_median": float(np.median(distances)),
        "wasserstein_raw_pointwise_p95": float(np.percentile(distances, 95.0)),
        "wasserstein_normalized_pointwise_mean": float(np.mean(normalized)),
        "wasserstein_normalized_pointwise_median": float(np.median(normalized)),
        "wasserstein_normalized_pointwise_p95": float(
            np.percentile(normalized, 95.0)
        ),
    }


def _metric_summary_fields(frame: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {}
    for source, target in (
        ("raw_mse", "nearest_reference_mse"),
        ("raw_rmse", "nearest_reference_rmse"),
        (
            "standardized_shape_mse",
            "nearest_reference_standardized_shape_mse",
        ),
        (
            "standardized_shape_rmse",
            "nearest_reference_standardized_shape_rmse",
        ),
        ("cosine", "nearest_reference_cosine"),
        ("pearson", "nearest_reference_pearson"),
    ):
        summary = summarize_vector(frame[source].to_numpy(dtype=np.float64))
        for statistic in ("mean", "std", "p05", "median", "p95"):
            result[f"{target}_{statistic}"] = float(summary[statistic])
    return result


def _comparison_summary_fields(
    frame: pd.DataFrame,
    prefix: str,
) -> dict[str, float]:
    """Keep real-real and generated-real context beside the headline metric."""

    return {
        f"{prefix}_mse_mean": float(
            np.mean(np.square(frame["raw_rmse"].to_numpy(dtype=np.float64)))
        ),
        f"{prefix}_cosine_mean": float(
            frame["cosine"].to_numpy(dtype=np.float64).mean()
        ),
        f"{prefix}_pearson_mean": float(
            frame["pearson"].to_numpy(dtype=np.float64).mean()
        ),
    }


def _training_supported_intensity_metrics(
    *,
    training: np.ndarray,
    generated: np.ndarray,
    configuration: dict,
) -> dict[str, float]:
    """Measure remaining pointwise extremes using training-only support."""

    lower_quantile = float(configuration.get("training_lower_quantile", 0.05))
    upper_quantile = float(configuration.get("training_upper_quantile", 0.95))
    iqr_margin = float(configuration.get("training_iqr_margin", 0.50))
    extrema_margin_fraction = float(
        configuration.get("training_extrema_margin_iqr_fraction", 0.10)
    )
    minimum_extrema_margin = float(
        configuration.get("minimum_extrema_margin_intensity", 2.0)
    )
    minimum_deficit = float(configuration.get("minimum_deficit_intensity", 5.0))
    minimum_excess = float(configuration.get("minimum_excess_intensity", 5.0))
    activation_maximum = float(
        configuration.get("activation_maximum_intensity", -18.0)
    )

    q_low = np.quantile(training, lower_quantile, axis=0)
    q_high = np.quantile(training, upper_quantile, axis=0)
    q25 = np.quantile(training, 0.25, axis=0)
    q75 = np.quantile(training, 0.75, axis=0)
    iqr = np.maximum(q75 - q25, 0.0)
    extrema_margin = np.maximum(
        extrema_margin_fraction * iqr,
        minimum_extrema_margin,
    )
    floor = np.minimum(
        q_low - iqr_margin * iqr,
        np.min(training, axis=0) - extrema_margin,
    )
    ceiling = np.maximum(
        q_high + iqr_margin * iqr,
        np.max(training, axis=0) + extrema_margin,
    )
    floor = np.minimum(floor, activation_maximum)
    lower = np.logical_and(
        generated < activation_maximum,
        generated < floor[np.newaxis, :] - minimum_deficit,
    )
    upper = generated > ceiling[np.newaxis, :] + minimum_excess
    return {
        "training_minimum_intensity": float(np.min(training)),
        "training_maximum_intensity": float(np.max(training)),
        "generated_maximum_intensity": float(np.max(generated)),
        "generated_below_training_envelope_point_fraction": float(np.mean(lower)),
        "generated_below_training_envelope_spectrum_fraction": float(
            np.mean(np.any(lower, axis=1))
        ),
        "generated_above_training_envelope_point_fraction": float(np.mean(upper)),
        "generated_above_training_envelope_spectrum_fraction": float(
            np.mean(np.any(upper, axis=1))
        ),
    }


def _diversity_fields(
    *,
    training: np.ndarray,
    generated: np.ndarray,
    pair_count: int,
    random_seed: int,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    _, training_pairs, generated_pairs = diversity_summary(
        training_spectra=training,
        generated_spectra=generated,
        generated_pair_count=pair_count,
        random_seed=random_seed,
    )
    training_mse = np.square(training_pairs["raw_rmse"].to_numpy(dtype=np.float64))
    generated_mse = np.square(generated_pairs["raw_rmse"].to_numpy(dtype=np.float64))
    training_pearson_distance = training_pairs["pearson_distance"].to_numpy(
        dtype=np.float64
    )
    generated_pearson_distance = generated_pairs["pearson_distance"].to_numpy(
        dtype=np.float64
    )
    train_mse_median = float(np.median(training_mse))
    generated_mse_median = float(np.median(generated_mse))
    train_shape_median = float(np.median(training_pearson_distance))
    generated_shape_median = float(np.median(generated_pearson_distance))
    fields = {
        "training_pairwise_mse_median": train_mse_median,
        "generated_pairwise_mse_median": generated_mse_median,
        "diversity_pairwise_mse_ratio": generated_mse_median
        / max(train_mse_median, EPSILON),
        "training_pairwise_pearson_median": float(
            np.median(training_pairs["pearson"].to_numpy(dtype=np.float64))
        ),
        "generated_pairwise_pearson_median": float(
            np.median(generated_pairs["pearson"].to_numpy(dtype=np.float64))
        ),
        "training_pairwise_pearson_distance_median": train_shape_median,
        "generated_pairwise_pearson_distance_median": generated_shape_median,
        "diversity_pearson_distance_ratio": generated_shape_median
        / max(train_shape_median, EPSILON),
    }
    return fields, training_pairs, generated_pairs


def _qq_agreement_metrics(
    reference_quantiles: np.ndarray,
    generated_quantiles: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    """Return paper-style QQ fit R2 and identity-line RMSE.

    R2 measures the linear fit between both quantile vectors. RMSE measures
    distance from the desired y=x line, so a high R2 cannot hide a wrong
    slope or offset.
    """

    reference_values = np.asarray(reference_quantiles, dtype=np.float64)
    generated_values = np.asarray(generated_quantiles, dtype=np.float64)
    if reference_values.shape != generated_values.shape:
        raise ValueError("QQ参考与生成分位数形状不一致。")
    if reference_values.size < 3:
        raise ValueError("QQ拟合至少需要3个分位点。")
    reference_std = float(np.std(reference_values))
    generated_std = float(np.std(generated_values))
    if reference_std <= EPSILON or generated_std <= EPSILON:
        linear_r2 = np.nan
    else:
        correlation = float(
            np.corrcoef(reference_values, generated_values)[0, 1]
        )
        linear_r2 = correlation * correlation
    identity_rmse = float(
        np.sqrt(np.mean(np.square(generated_values - reference_values)))
    )
    return {
        f"{prefix}_r2": float(linear_r2),
        f"{prefix}_rmse": identity_rmse,
    }


def _qq_piecewise_shape_metrics(
    probabilities: np.ndarray,
    reference_quantiles: np.ndarray,
    generated_quantiles: np.ndarray,
    oracle_lower: np.ndarray,
    oracle_upper: np.ndarray,
) -> dict[str, float]:
    """Measure QQ curvature that a single R2 or 1--99 span can hide."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    reference = np.asarray(reference_quantiles, dtype=np.float64)
    generated = np.asarray(generated_quantiles, dtype=np.float64)
    lower = np.asarray(oracle_lower, dtype=np.float64)
    upper = np.asarray(oracle_upper, dtype=np.float64)
    if not (
        probabilities.shape
        == reference.shape
        == generated.shape
        == lower.shape
        == upper.shape
    ):
        raise ValueError("QQ分段形状数组长度不一致。")

    anchor_indices = [
        int(np.argmin(np.abs(probabilities - value)))
        for value in (0.01, 0.10, 0.50, 0.90, 0.99)
    ]
    reference_anchors = reference[anchor_indices]
    generated_anchors = generated[anchor_indices]
    segment_ratios = np.diff(generated_anchors) / np.maximum(
        np.diff(reference_anchors), EPSILON
    )
    central = np.logical_and(probabilities >= 0.10, probabilities <= 0.90)
    lower_tail = probabilities <= 0.10
    upper_tail = probabilities >= 0.90
    slope, intercept = np.polyfit(reference[central], generated[central], 1)
    continuation = slope * reference + intercept
    curve_residual = generated - continuation
    outside = np.logical_or(generated < lower, generated > upper)

    def fraction(mask: np.ndarray) -> float:
        return float(np.mean(outside[mask])) if np.any(mask) else np.nan

    return {
        "qq_lower_outer_01_10_span_ratio": float(segment_ratios[0]),
        "qq_lower_inner_10_50_span_ratio": float(segment_ratios[1]),
        "qq_upper_inner_50_90_span_ratio": float(segment_ratios[2]),
        "qq_upper_outer_90_99_span_ratio": float(segment_ratios[3]),
        "qq_central_fit_slope": float(slope),
        "qq_lower_tail_signed_curvature": float(
            np.mean(curve_residual[lower_tail])
        ),
        "qq_upper_tail_signed_curvature": float(
            np.mean(curve_residual[upper_tail])
        ),
        "qq_lower_tail_curve_rmse": float(
            np.sqrt(np.mean(np.square(curve_residual[lower_tail])))
        ),
        "qq_upper_tail_curve_rmse": float(
            np.sqrt(np.mean(np.square(curve_residual[upper_tail])))
        ),
        "qq_oracle_outside_fraction": float(np.mean(outside)),
        "qq_oracle_outside_lower_tail_fraction": fraction(lower_tail),
        "qq_oracle_outside_central_fraction": fraction(central),
        "qq_oracle_outside_upper_tail_fraction": fraction(upper_tail),
    }


def _peak_morphology_diversity(
    *,
    condition: str,
    axis: np.ndarray,
    training: np.ndarray,
    generated: np.ndarray,
    configuration: dict,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Separate peak-position, peak-height and peak-width diversity.

    Reliable peaks are detected once from the condition training median.  All
    limits and reference variation therefore remain train-only.  This avoids
    declaring a batch diverse merely because amplitudes differ while every
    peak stays at exactly the same Raman position.
    """

    if not isinstance(configuration, dict):
        raise ValueError("peak_diversity_evaluation配置必须是字典。")
    spacing = float(np.median(np.diff(axis)))
    if spacing <= 0.0:
        raise ValueError("peak diversity要求严格递增Raman轴。")
    baseline_sigma = float(configuration.get("baseline_sigma_cm1", 30.0))
    smoothing_sigma = float(configuration.get("smoothing_sigma_cm1", 2.0))
    relative_prominence = float(
        configuration.get("minimum_relative_prominence", 0.06)
    )
    minimum_distance = float(
        configuration.get("minimum_peak_distance_cm1", 12.0)
    )
    maximum_peak_count = int(configuration.get("maximum_peak_count", 10))
    edge_exclusion = float(configuration.get("edge_exclusion_cm1", 24.0))
    half_width = float(
        configuration.get("measurement_half_width_cm1", 12.0)
    )
    minimum_position_std = float(
        configuration.get("minimum_position_std_cm1", 0.35)
    )
    if baseline_sigma <= smoothing_sigma or smoothing_sigma <= 0.0:
        raise ValueError("峰评价要求baseline sigma大于正的smoothing sigma。")
    if not 0.0 < relative_prominence < 1.0:
        raise ValueError("minimum_relative_prominence必须位于(0,1)。")
    if minimum_distance <= 0.0 or maximum_peak_count < 1 or half_width <= 0.0:
        raise ValueError("峰评价的距离、数量或窗口配置无效。")

    smooth_points = max(smoothing_sigma / spacing, 0.5)
    baseline_points = max(baseline_sigma / spacing, smooth_points + 0.5)

    def corrected_rows(values: np.ndarray) -> np.ndarray:
        smoothed = gaussian_filter1d(
            values, sigma=smooth_points, axis=1, mode="reflect"
        )
        baseline = gaussian_filter1d(
            values, sigma=baseline_points, axis=1, mode="reflect"
        )
        return smoothed - baseline

    training_corrected = corrected_rows(training)
    generated_corrected = corrected_rows(generated)
    reference_curve = np.median(training_corrected, axis=0)
    peak_maximum = float(np.max(reference_curve))
    if peak_maximum <= EPSILON:
        empty = pd.DataFrame()
        return {
            "peak_diversity_peak_count": 0,
            "peak_position_active_count": 0,
            "peak_position_std_ratio_median": np.nan,
            "peak_position_wasserstein_cm1_mean": np.nan,
            "peak_position_frozen_fraction": np.nan,
            "peak_height_iqr_ratio_median": np.nan,
            "peak_width_iqr_ratio_median": np.nan,
        }, empty
    distance_points = max(int(round(minimum_distance / spacing)), 1)
    peaks, properties = find_peaks(
        reference_curve,
        prominence=relative_prominence * peak_maximum,
        distance=distance_points,
    )
    edge_points = int(np.ceil(edge_exclusion / spacing))
    keep = np.logical_and(peaks >= edge_points, peaks < axis.size - edge_points)
    peaks = peaks[keep]
    prominences = properties["prominences"][keep]
    if peaks.size > maximum_peak_count:
        strongest = np.argsort(prominences)[-maximum_peak_count:]
        peaks = peaks[strongest]
        prominences = prominences[strongest]
    order = np.argsort(peaks)
    peaks = peaks[order]
    prominences = prominences[order]

    def measurements(
        corrected: np.ndarray, peak_index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        center = float(axis[peak_index])
        local_indices = np.flatnonzero(np.abs(axis - center) <= half_width)
        local_axis = axis[local_indices]
        local = corrected[:, local_indices]
        maximum_indices = np.argmax(local, axis=1)
        positions = local_axis[maximum_indices]
        heights = local[np.arange(local.shape[0]), maximum_indices]
        weights = np.maximum(local, 0.0)
        weight_sum = np.sum(weights, axis=1)
        weighted_center = np.sum(
            weights * local_axis[np.newaxis, :], axis=1
        ) / np.maximum(weight_sum, EPSILON)
        widths = np.sqrt(
            np.sum(
                weights
                * np.square(local_axis[np.newaxis, :] - weighted_center[:, None]),
                axis=1,
            )
            / np.maximum(weight_sum, EPSILON)
        )
        widths[weight_sum <= EPSILON] = 0.0
        return positions, heights, widths

    rows: list[dict[str, float | int | str | bool]] = []
    position_ratios: list[float] = []
    position_wasserstein: list[float] = []
    frozen: list[float] = []
    height_ratios: list[float] = []
    width_ratios: list[float] = []
    for peak_number, (peak_index, prominence) in enumerate(
        zip(peaks, prominences), start=1
    ):
        train_position, train_height, train_width = measurements(
            training_corrected, int(peak_index)
        )
        gen_position, gen_height, gen_width = measurements(
            generated_corrected, int(peak_index)
        )
        train_position_std = float(np.std(train_position, ddof=1))
        gen_position_std = float(np.std(gen_position, ddof=1))
        position_active = train_position_std >= minimum_position_std
        position_ratio = (
            gen_position_std / max(train_position_std, EPSILON)
            if position_active
            else np.nan
        )
        position_w1 = float(wasserstein_distance(train_position, gen_position))
        train_height_iqr = float(
            np.percentile(train_height, 75.0) - np.percentile(train_height, 25.0)
        )
        gen_height_iqr = float(
            np.percentile(gen_height, 75.0) - np.percentile(gen_height, 25.0)
        )
        height_ratio = gen_height_iqr / max(train_height_iqr, EPSILON)
        train_width_iqr = float(
            np.percentile(train_width, 75.0) - np.percentile(train_width, 25.0)
        )
        gen_width_iqr = float(
            np.percentile(gen_width, 75.0) - np.percentile(gen_width, 25.0)
        )
        width_ratio = gen_width_iqr / max(train_width_iqr, EPSILON)
        if position_active:
            position_ratios.append(float(position_ratio))
            position_wasserstein.append(position_w1)
            frozen.append(float(gen_position_std < 0.5 * train_position_std))
        if train_height_iqr > EPSILON:
            height_ratios.append(height_ratio)
        if train_width_iqr > EPSILON:
            width_ratios.append(width_ratio)
        rows.append(
            {
                "condition": condition,
                "peak_number": peak_number,
                "training_peak_center_cm1": float(axis[peak_index]),
                "training_median_prominence": float(prominence),
                "position_reference_active": bool(position_active),
                "training_position_std_cm1": train_position_std,
                "generated_position_std_cm1": gen_position_std,
                "peak_position_std_ratio": float(position_ratio),
                "peak_position_wasserstein_cm1": position_w1,
                "training_height_iqr": train_height_iqr,
                "generated_height_iqr": gen_height_iqr,
                "peak_height_iqr_ratio": height_ratio,
                "training_width_iqr_cm1": train_width_iqr,
                "generated_width_iqr_cm1": gen_width_iqr,
                "peak_width_iqr_ratio": width_ratio,
            }
        )

    def median_or_nan(values: list[float]) -> float:
        return float(np.median(values)) if values else np.nan

    fields: dict[str, float | int] = {
        "peak_diversity_peak_count": int(len(rows)),
        "peak_position_active_count": int(len(position_ratios)),
        "peak_position_std_ratio_median": median_or_nan(position_ratios),
        "peak_position_wasserstein_cm1_mean": (
            float(np.mean(position_wasserstein))
            if position_wasserstein
            else np.nan
        ),
        "peak_position_frozen_fraction": (
            float(np.mean(frozen)) if frozen else np.nan
        ),
        "peak_height_iqr_ratio_median": median_or_nan(height_ratios),
        "peak_width_iqr_ratio_median": median_or_nan(width_ratios),
    }
    return fields, pd.DataFrame(rows)


def _plot_peak_diversity(details: pd.DataFrame, output: Path) -> None:
    if details.empty:
        return
    figure, axis = plt.subplots(figsize=(9.5, 4.2))
    centers = details["training_peak_center_cm1"].to_numpy(dtype=np.float64)
    for column, label, color, marker in (
        ("peak_position_std_ratio", "position std ratio", "#1565C0", "o"),
        ("peak_height_iqr_ratio", "height IQR ratio", "#EF6C00", "s"),
        ("peak_width_iqr_ratio", "width IQR ratio", "#2E7D32", "^"),
    ):
        values = details[column].to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        axis.scatter(
            centers[finite], values[finite], s=32, label=label,
            color=color, marker=marker, alpha=0.85,
        )
    axis.axhline(1.0, color="#555555", linestyle="--", linewidth=1.0)
    axis.axhspan(0.8, 1.2, color="#A5D6A7", alpha=0.18)
    axis.set_xlabel("Training-defined peak center (cm$^{-1}$)")
    axis.set_ylabel("Generated / training variation")
    axis.set_title(f"{details.iloc[0]['condition']}: peak morphology diversity")
    axis.grid(alpha=0.18)
    axis.legend(fontsize=8, loc="best")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _qq_table(
    *,
    condition: str,
    training: np.ndarray,
    reference: np.ndarray,
    generated: np.ndarray,
    training_model_normalized: np.ndarray | None = None,
    reference_model_normalized: np.ndarray | None = None,
    generated_model_normalized: np.ndarray | None = None,
    quantile_count: int,
    oracle_repeats: int,
    oracle_confidence: float,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, float | bool]]:
    if quantile_count < 21:
        raise ValueError("qq_quantile_count至少为21。")
    probabilities = np.linspace(0.01, 0.99, quantile_count)
    model_arrays = (
        training_model_normalized,
        reference_model_normalized,
        generated_model_normalized,
    )
    if all(value is None for value in model_arrays):
        training_model = training
        reference_model = reference
        generated_model = generated
    elif any(value is None for value in model_arrays):
        raise ValueError("QQ模型归一化数组必须同时提供或同时省略。")
    else:
        training_model = np.asarray(
            training_model_normalized, dtype=np.float64
        )
        reference_model = np.asarray(
            reference_model_normalized, dtype=np.float64
        )
        generated_model = np.asarray(
            generated_model_normalized, dtype=np.float64
        )
    center = np.mean(training, axis=0)
    scale = np.std(training, axis=0, ddof=1)
    positive = scale[scale > EPSILON]
    floor = max(
        float(np.percentile(positive, 10.0)) if positive.size else 1.0,
        EPSILON,
    )
    scale = np.maximum(scale, floor)
    reference_z = ((reference - center[None, :]) / scale[None, :]).reshape(-1)
    generated_z = ((generated - center[None, :]) / scale[None, :]).reshape(-1)
    training_z = (training - center[None, :]) / scale[None, :]
    reference_raw_q = np.quantile(reference, probabilities)
    generated_raw_q = np.quantile(generated, probabilities)
    reference_model_q = np.quantile(reference_model, probabilities)
    generated_model_q = np.quantile(generated_model, probabilities)
    reference_z_q = np.quantile(reference_z, probabilities)
    generated_z_q = np.quantile(generated_z, probabilities)

    repeats = int(oracle_repeats)
    confidence = float(oracle_confidence)
    reference_count = int(reference.shape[0])
    if repeats < 32 or not 0.90 <= confidence < 1.0:
        raise ValueError("QQ oracle要求repeats>=32且confidence位于[0.90,1)。")
    if not 2 <= reference_count < training.shape[0] - 1:
        raise ValueError("QQ oracle真实参考数量与training数量不适合4-vs-8分层。")

    random_generator = np.random.default_rng(int(random_seed))
    indices = np.arange(training.shape[0], dtype=np.int64)
    raw_differences: list[np.ndarray] = []
    model_differences: list[np.ndarray] = []
    standardized_differences: list[np.ndarray] = []
    lower_tail_ratios: list[float] = []
    upper_tail_ratios: list[float] = []
    tail_indices = [
        int(np.argmin(np.abs(probabilities - value)))
        for value in (0.01, 0.50, 0.99)
    ]
    oracle_cache: dict[
        tuple[int, ...],
        tuple[np.ndarray, np.ndarray, np.ndarray, float, float],
    ] = {}
    for _ in range(repeats):
        shuffled = random_generator.permutation(indices)
        reference_subset = tuple(
            sorted(int(value) for value in shuffled[:reference_count])
        )
        cached = oracle_cache.get(reference_subset)
        if cached is not None:
            (
                raw_difference,
                model_difference,
                z_difference,
                low_ratio,
                high_ratio,
            ) = cached
            raw_differences.append(raw_difference)
            model_differences.append(model_difference)
            standardized_differences.append(z_difference)
            lower_tail_ratios.append(low_ratio)
            upper_tail_ratios.append(high_ratio)
            continue
        pseudo_reference_indices = np.asarray(
            reference_subset, dtype=np.int64
        )
        pseudo_population_indices = np.asarray(
            [value for value in indices if int(value) not in reference_subset],
            dtype=np.int64,
        )
        pseudo_reference = training[pseudo_reference_indices]
        pseudo_population = training[pseudo_population_indices]
        pseudo_reference_z = training_z[pseudo_reference_indices]
        pseudo_population_z = training_z[pseudo_population_indices]
        pseudo_reference_model = training_model[pseudo_reference_indices]
        pseudo_population_model = training_model[pseudo_population_indices]
        reference_boot_raw = np.quantile(pseudo_reference, probabilities)
        population_boot_raw = np.quantile(pseudo_population, probabilities)
        reference_boot_model = np.quantile(
            pseudo_reference_model, probabilities
        )
        population_boot_model = np.quantile(
            pseudo_population_model, probabilities
        )
        reference_boot_z = np.quantile(pseudo_reference_z, probabilities)
        population_boot_z = np.quantile(pseudo_population_z, probabilities)
        raw_difference = population_boot_raw - reference_boot_raw
        model_difference = population_boot_model - reference_boot_model
        z_difference = population_boot_z - reference_boot_z
        low, middle, high = tail_indices
        lower_denominator = max(
            float(reference_boot_z[middle] - reference_boot_z[low]), EPSILON
        )
        upper_denominator = max(
            float(reference_boot_z[high] - reference_boot_z[middle]), EPSILON
        )
        low_ratio = float(
            (population_boot_z[middle] - population_boot_z[low])
            / lower_denominator
        )
        high_ratio = float(
            (population_boot_z[high] - population_boot_z[middle])
            / upper_denominator
        )
        oracle_cache[reference_subset] = (
            raw_difference,
            model_difference,
            z_difference,
            low_ratio,
            high_ratio,
        )
        raw_differences.append(raw_difference)
        model_differences.append(model_difference)
        standardized_differences.append(z_difference)
        lower_tail_ratios.append(low_ratio)
        upper_tail_ratios.append(high_ratio)

    alpha = 1.0 - confidence
    raw_difference_array = np.stack(raw_differences, axis=0)
    model_difference_array = np.stack(model_differences, axis=0)
    z_difference_array = np.stack(standardized_differences, axis=0)
    lower_tail_oracle_lower = float(
        np.quantile(lower_tail_ratios, 1.0 - confidence)
    )
    lower_tail_oracle_upper = float(np.quantile(lower_tail_ratios, confidence))
    upper_tail_oracle_lower = float(
        np.quantile(upper_tail_ratios, 1.0 - confidence)
    )
    upper_tail_oracle_upper = float(np.quantile(upper_tail_ratios, confidence))
    table = pd.DataFrame(
        {
            "condition": condition,
            "probability": probabilities,
            "reference_raw_intensity": reference_raw_q,
            "generated_raw_intensity": generated_raw_q,
            "oracle_raw_lower": reference_raw_q
            + np.quantile(raw_difference_array, alpha / 2.0, axis=0),
            "oracle_raw_upper": reference_raw_q
            + np.quantile(raw_difference_array, 1.0 - alpha / 2.0, axis=0),
            "reference_model_normalized": reference_model_q,
            "generated_model_normalized": generated_model_q,
            "oracle_model_normalized_lower": reference_model_q
            + np.quantile(model_difference_array, alpha / 2.0, axis=0),
            "oracle_model_normalized_upper": reference_model_q
            + np.quantile(
                model_difference_array, 1.0 - alpha / 2.0, axis=0
            ),
            "reference_training_standardized": reference_z_q,
            "generated_training_standardized": generated_z_q,
            "oracle_standardized_lower": reference_z_q
            + np.quantile(z_difference_array, alpha / 2.0, axis=0),
            "oracle_standardized_upper": reference_z_q
            + np.quantile(z_difference_array, 1.0 - alpha / 2.0, axis=0),
        }
    )
    low, middle, high = tail_indices
    central_low = int(np.argmin(np.abs(probabilities - 0.10)))
    central_high = int(np.argmin(np.abs(probabilities - 0.90)))
    reference_lower_span = max(
        float(reference_z_q[middle] - reference_z_q[low]), EPSILON
    )
    reference_upper_span = max(
        float(reference_z_q[high] - reference_z_q[middle]), EPSILON
    )
    reference_central_span = max(
        float(reference_z_q[central_high] - reference_z_q[central_low]), EPSILON
    )
    metrics: dict[str, float | bool] = {
        **_qq_agreement_metrics(
            reference_raw_q,
            generated_raw_q,
            "qq_raw",
        ),
        **_qq_agreement_metrics(
            reference_model_q,
            generated_model_q,
            "qq_model_normalized",
        ),
        **_qq_agreement_metrics(
            reference_z_q,
            generated_z_q,
            "qq_training_standardized",
        ),
        "qq_lower_tail_span_ratio": float(
            (generated_z_q[middle] - generated_z_q[low]) / reference_lower_span
        ),
        "qq_upper_tail_span_ratio": float(
            (generated_z_q[high] - generated_z_q[middle]) / reference_upper_span
        ),
        "qq_total_tail_span_ratio": float(
            (generated_z_q[high] - generated_z_q[low])
            / max(float(reference_z_q[high] - reference_z_q[low]), EPSILON)
        ),
        "qq_central_10_90_span_ratio": float(
            (generated_z_q[central_high] - generated_z_q[central_low])
            / reference_central_span
        ),
        "qq_lower_tail_oracle_lower": lower_tail_oracle_lower,
        "qq_lower_tail_oracle_upper": lower_tail_oracle_upper,
        "qq_upper_tail_oracle_lower": upper_tail_oracle_lower,
        "qq_upper_tail_oracle_upper": upper_tail_oracle_upper,
        **_qq_piecewise_shape_metrics(
            probabilities,
            reference_z_q,
            generated_z_q,
            table["oracle_standardized_lower"].to_numpy(dtype=np.float64),
            table["oracle_standardized_upper"].to_numpy(dtype=np.float64),
        ),
    }
    metrics["qq_lower_tail_overrun"] = bool(
        metrics["qq_lower_tail_span_ratio"] > lower_tail_oracle_upper
    )
    metrics["qq_upper_tail_overrun"] = bool(
        metrics["qq_upper_tail_span_ratio"] > upper_tail_oracle_upper
    )
    metrics["qq_lower_tail_underrun"] = bool(
        metrics["qq_lower_tail_span_ratio"] < lower_tail_oracle_lower
    )
    metrics["qq_upper_tail_underrun"] = bool(
        metrics["qq_upper_tail_span_ratio"] < upper_tail_oracle_lower
    )
    metrics["qq_any_tail_overrun"] = bool(
        metrics["qq_lower_tail_overrun"] or metrics["qq_upper_tail_overrun"]
    )
    metrics["qq_any_tail_underrun"] = bool(
        metrics["qq_lower_tail_underrun"] or metrics["qq_upper_tail_underrun"]
    )
    metrics["qq_any_tail_mismatch"] = bool(
        metrics["qq_any_tail_overrun"] or metrics["qq_any_tail_underrun"]
    )
    return table, metrics


def _plot_qq(table: pd.DataFrame, condition: str, output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    panels = (
        (
            "reference_raw_intensity",
            "generated_raw_intensity",
            "oracle_raw_lower",
            "oracle_raw_upper",
            "Raw intensity QQ",
        ),
        (
            "reference_model_normalized",
            "generated_model_normalized",
            "oracle_model_normalized_lower",
            "oracle_model_normalized_upper",
            "Checkpoint-normalized QQ (paper-compatible)",
        ),
        (
            "reference_training_standardized",
            "generated_training_standardized",
            "oracle_standardized_lower",
            "oracle_standardized_upper",
            "Training-standardized QQ",
        ),
    )
    for axis, (
        real_column,
        generated_column,
        oracle_lower_column,
        oracle_upper_column,
        title,
    ) in zip(axes, panels):
        real = table[real_column].to_numpy(dtype=np.float64)
        generated = table[generated_column].to_numpy(dtype=np.float64)
        lower = float(min(np.min(real), np.min(generated)))
        upper = float(max(np.max(real), np.max(generated)))
        axis.plot(
            [lower, upper], [lower, upper], "--", color="#777777", linewidth=1.0
        )
        axis.fill_between(
            real,
            table[oracle_lower_column].to_numpy(dtype=np.float64),
            table[oracle_upper_column].to_numpy(dtype=np.float64),
            color="#B8D8F0",
            alpha=0.45,
            label="real-vs-real oracle band",
        )
        axis.scatter(real, generated, s=13, color="#1769AA", alpha=0.8)
        agreement = _qq_agreement_metrics(real, generated, "panel")
        axis.text(
            0.03,
            0.96,
            (
                f"$R^2$={agreement['panel_r2']:.4f}\n"
                f"identity RMSE={agreement['panel_rmse']:.4g}"
            ),
            transform=axis.transAxes,
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
        )
        axis.set_xlabel("Real reference quantile")
        axis.set_ylabel("Generated quantile")
        axis.set_title(title)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7, loc="best")
    figure.suptitle(condition)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_spectral_distribution(
    *,
    axis: np.ndarray,
    training: np.ndarray,
    reference: np.ndarray,
    generated: np.ndarray,
    condition: str,
    output: Path,
) -> None:
    """Plot robust bands so isolated extremes cannot hide the bulk fit."""

    generated_quantiles = np.quantile(
        generated,
        [0.05, 0.25, 0.50, 0.75, 0.95],
        axis=0,
    )
    q05, q25, _, q75, q95 = generated_quantiles
    generated_mean = np.mean(generated, axis=0)
    training_mean = np.mean(training, axis=0)
    reference_mean = np.mean(reference, axis=0)
    reference_minimum = np.min(reference, axis=0)
    reference_maximum = np.max(reference, axis=0)
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12.0, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.25]},
    )
    upper, lower = axes
    for index, spectrum in enumerate(training):
        upper.plot(
            axis,
            spectrum,
            color="#777777",
            linewidth=0.55,
            alpha=0.22,
            label="12 training spectra" if index == 0 else None,
        )
    upper.fill_between(
        axis,
        reference_minimum,
        reference_maximum,
        color="#EF9A9A",
        alpha=0.24,
        label="held-out min--max band",
    )
    upper.plot(
        axis,
        reference_mean,
        color="#C62828",
        linewidth=1.25,
        label="held-out mean",
    )
    upper.fill_between(
        axis,
        q05,
        q95,
        color="#90CAF9",
        alpha=0.28,
        label="generated 5--95%",
    )
    upper.fill_between(
        axis,
        q25,
        q75,
        color="#42A5F5",
        alpha=0.30,
        label="generated 25--75%",
    )
    upper.plot(
        axis,
        generated_mean,
        color="#0D47A1",
        linewidth=1.25,
        label="generated mean",
    )
    upper.set_ylabel("Intensity")
    upper.set_title(f"{condition}: real spectra and generated distribution")
    upper.grid(alpha=0.18)
    upper.legend(fontsize=8, ncol=3, loc="best")

    lower.axhline(0.0, color="#777777", linestyle="--", linewidth=0.8)
    lower.plot(
        axis,
        generated_mean - reference_mean,
        color="#1565C0",
        linewidth=1.0,
        label="generated mean - held-out mean",
    )
    lower.plot(
        axis,
        training_mean - reference_mean,
        color="#616161",
        linewidth=0.85,
        alpha=0.8,
        label="training mean - held-out mean",
    )
    lower.set_xlabel("Raman shift (cm$^{-1}$)")
    lower.set_ylabel("Mean residual")
    lower.grid(alpha=0.18)
    lower.legend(fontsize=8, loc="best")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_peak_intensity_violins(
    *,
    axis: np.ndarray,
    training: np.ndarray,
    reference: np.ndarray,
    generated: np.ndarray,
    details: pd.DataFrame,
    condition: str,
    output: Path,
) -> None:
    """Compare fixed-position peak intensities at training-defined peaks."""

    if details.empty:
        return
    centers = details["training_peak_center_cm1"].to_numpy(dtype=np.float64)
    column_count = min(3, int(centers.size))
    row_count = int(np.ceil(centers.size / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(3.8 * column_count, 3.25 * row_count),
        squeeze=False,
    )
    for peak_number, (panel, center) in enumerate(
        zip(axes.flat, centers), start=1
    ):
        peak_index = int(np.argmin(np.abs(axis - center)))
        datasets = [training[:, peak_index], generated[:, peak_index]]
        violins = panel.violinplot(
            datasets,
            positions=[1.0, 2.0],
            widths=0.78,
            showmeans=False,
            showmedians=True,
            showextrema=True,
        )
        for body, color in zip(violins["bodies"], ("#90CAF9", "#FFCC80")):
            body.set_facecolor(color)
            body.set_edgecolor("#555555")
            body.set_alpha(0.72)
        reference_x = np.linspace(0.90, 1.10, reference.shape[0])
        panel.scatter(
            reference_x,
            reference[:, peak_index],
            marker="D",
            s=22,
            color="#C62828",
            label="held-out" if peak_number == 1 else None,
            zorder=4,
        )
        panel.set_xticks([1.0, 2.0], ["training", "generated"])
        panel.set_title(f"{axis[peak_index]:.0f} cm$^{{-1}}$")
        panel.set_ylabel("Raw intensity")
        panel.grid(axis="y", alpha=0.18)
    for panel in axes.flat[centers.size:]:
        panel.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper right", fontsize=8)
    figure.suptitle(
        f"{condition}: training-defined characteristic-peak intensities",
        y=1.01,
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_quality_overview(frame: pd.DataFrame, output: Path) -> None:
    """Create an all-condition dashboard of the main quality tradeoffs."""

    figure, axes = plt.subplots(2, 2, figsize=(12.0, 9.0))
    if frame.shape[0] == 1:
        row = frame.iloc[0]
        axes[0, 0].axis("off")
        axes[0, 0].text(
            0.03,
            0.95,
            "\n".join(
                [
                    str(row["condition"]),
                    f"MSE: {row['nearest_reference_mse_mean']:.4g}",
                    f"Cosine: {row['nearest_reference_cosine_mean']:.4f}",
                    f"Pearson: {row['nearest_reference_pearson_mean']:.4f}",
                    f"Wasserstein: {row['wasserstein_raw_pointwise_mean']:.4g}",
                ]
            ),
            va="top",
            fontsize=13,
        )
        diversity_labels = ["pairwise MSE", "Pearson distance", "peak height"]
        diversity_values = [
            row["diversity_pairwise_mse_ratio"],
            row["diversity_pearson_distance_ratio"],
            row["peak_height_iqr_ratio_median"],
        ]
        axes[0, 1].barh(diversity_labels, diversity_values, color="#42A5F5")
        axes[0, 1].axvline(1.0, color="#555555", linestyle="--")
        axes[0, 1].set_title("Diversity ratios")
        axes[0, 1].set_xlabel("Generated / training")
        tail_labels = ["lower 1%", "central 10--90%", "upper 1%"]
        tail_values = [
            row["qq_lower_tail_span_ratio"],
            row["qq_central_10_90_span_ratio"],
            row["qq_upper_tail_span_ratio"],
        ]
        axes[1, 0].barh(tail_labels, tail_values, color="#7E57C2")
        axes[1, 0].axvline(1.0, color="#555555", linestyle="--")
        axes[1, 0].set_title("QQ span ratios")
        axes[1, 0].set_xlabel("Generated / held-out")
        envelope_values = [
            row["generated_below_training_envelope_point_fraction"],
            row["generated_above_training_envelope_point_fraction"],
        ]
        axes[1, 1].barh(
            ["below", "above"], envelope_values, color=["#EF5350", "#FFB74D"]
        )
        axes[1, 1].set_title("Outside training envelope")
        axes[1, 1].set_xlabel("Point fraction")
        for panel in axes.flat:
            panel.grid(alpha=0.18)
        figure.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=220, bbox_inches="tight")
        plt.close(figure)
        return
    sorted_mse = np.sort(
        frame["nearest_reference_mse_mean"].to_numpy(dtype=np.float64)
    )
    axes[0, 0].plot(
        np.arange(1, sorted_mse.size + 1),
        sorted_mse,
        color="#1565C0",
    )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("Condition rank")
    axes[0, 0].set_ylabel("Nearest-reference MSE (log)")
    axes[0, 0].set_title("Fidelity across all conditions")

    mismatch = frame["qq_any_tail_mismatch"].astype(bool).to_numpy()
    colors = np.where(mismatch, "#D32F2F", "#2E7D32")
    axes[0, 1].scatter(
        frame["diversity_pairwise_mse_ratio"],
        frame["nearest_reference_mse_mean"],
        c=colors,
        s=22,
        alpha=0.78,
    )
    axes[0, 1].axvline(1.0, color="#777777", linestyle="--", linewidth=0.9)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("Generated / training pairwise-MSE diversity")
    axes[0, 1].set_ylabel("Nearest-reference MSE (log)")
    axes[0, 1].set_title("Fidelity-diversity tradeoff (red: QQ mismatch)")

    axes[1, 0].scatter(
        frame["qq_lower_tail_span_ratio"],
        frame["qq_upper_tail_span_ratio"],
        c=colors,
        s=22,
        alpha=0.78,
    )
    axes[1, 0].axvline(1.0, color="#777777", linestyle="--", linewidth=0.9)
    axes[1, 0].axhline(1.0, color="#777777", linestyle="--", linewidth=0.9)
    axes[1, 0].set_xlabel("Lower-tail span ratio")
    axes[1, 0].set_ylabel("Upper-tail span ratio")
    axes[1, 0].set_title("Two-sided QQ tail balance")

    lower_violation = np.maximum(
        frame["generated_below_training_envelope_point_fraction"].to_numpy(
            dtype=np.float64
        ),
        EPSILON,
    )
    upper_violation = np.maximum(
        frame["generated_above_training_envelope_point_fraction"].to_numpy(
            dtype=np.float64
        ),
        EPSILON,
    )
    axes[1, 1].scatter(
        lower_violation,
        upper_violation,
        c=colors,
        s=22,
        alpha=0.78,
    )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("Below training envelope fraction")
    axes[1, 1].set_ylabel("Above training envelope fraction")
    axes[1, 1].set_title("Unsupported intensity tails")
    for panel in axes.flat:
        panel.grid(alpha=0.18)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _overall_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    excluded = {"generated_count", "training_count", "reference_count", "point_count"}
    for column in frame.select_dtypes(include=[np.number]).columns:
        if column in excluded:
            continue
        values = frame[column].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        rows.append(
            {
                "metric": column,
                "condition_count": int(values.size),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
                "p05": float(np.percentile(values, 5.0)),
                "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95.0)),
            }
        )
    return pd.DataFrame(rows)


def _qq_stratified_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Pool condition-level QQ evidence without treating Raman points as iid."""

    working = frame.copy()
    working["matrix"] = working["condition"].map(
        lambda value: "water" if str(value).endswith("_water") else "soil"
    )
    working["analyte_count"] = working["condition"].map(
        lambda value: sum(
            f"{analyte}-" in str(value) for analyte in ("DEL", "CHL", "TEB")
        )
    )
    working["axis_group"] = working["point_count"].map(
        lambda value: f"{int(value)}_points"
    )
    grouping_specs: list[tuple[str, list[str]]] = [
        ("overall", []),
        ("matrix", ["matrix"]),
        ("analyte_count", ["analyte_count"]),
        ("matrix_x_analyte_count", ["matrix", "analyte_count"]),
        ("axis_group", ["axis_group"]),
    ]
    rows: list[dict[str, Any]] = []
    for level, columns in grouping_specs:
        groups = [((), working)] if not columns else working.groupby(columns)
        for key, group in groups:
            keys = key if isinstance(key, tuple) else (key,)
            label = "all" if not columns else "|".join(str(value) for value in keys)
            rows.append(
                {
                    "stratum_level": level,
                    "stratum": label,
                    "condition_count": int(group.shape[0]),
                    "lower_tail_ratio_mean": float(
                        group["qq_lower_tail_span_ratio"].mean()
                    ),
                    "lower_tail_ratio_median": float(
                        group["qq_lower_tail_span_ratio"].median()
                    ),
                    "upper_tail_ratio_mean": float(
                        group["qq_upper_tail_span_ratio"].mean()
                    ),
                    "upper_tail_ratio_median": float(
                        group["qq_upper_tail_span_ratio"].median()
                    ),
                    "central_10_90_ratio_mean": float(
                        group["qq_central_10_90_span_ratio"].mean()
                    ),
                    "lower_outer_01_10_ratio_mean": float(
                        group["qq_lower_outer_01_10_span_ratio"].mean()
                    ),
                    "lower_inner_10_50_ratio_mean": float(
                        group["qq_lower_inner_10_50_span_ratio"].mean()
                    ),
                    "upper_inner_50_90_ratio_mean": float(
                        group["qq_upper_inner_50_90_span_ratio"].mean()
                    ),
                    "upper_outer_90_99_ratio_mean": float(
                        group["qq_upper_outer_90_99_span_ratio"].mean()
                    ),
                    "lower_tail_curve_rmse_mean": float(
                        group["qq_lower_tail_curve_rmse"].mean()
                    ),
                    "upper_tail_curve_rmse_mean": float(
                        group["qq_upper_tail_curve_rmse"].mean()
                    ),
                    "oracle_outside_fraction_mean": float(
                        group["qq_oracle_outside_fraction"].mean()
                    ),
                    "raw_r2_mean": float(group["qq_raw_r2"].mean()),
                    "raw_rmse_mean": float(group["qq_raw_rmse"].mean()),
                    "model_normalized_r2_mean": float(
                        group["qq_model_normalized_r2"].mean()
                    ),
                    "model_normalized_rmse_mean": float(
                        group["qq_model_normalized_rmse"].mean()
                    ),
                    "training_standardized_r2_mean": float(
                        group["qq_training_standardized_r2"].mean()
                    ),
                    "training_standardized_rmse_mean": float(
                        group["qq_training_standardized_rmse"].mean()
                    ),
                    "lower_tail_overrun_fraction": float(
                        group["qq_lower_tail_overrun"].astype(float).mean()
                    ),
                    "upper_tail_overrun_fraction": float(
                        group["qq_upper_tail_overrun"].astype(float).mean()
                    ),
                    "any_tail_overrun_fraction": float(
                        group["qq_any_tail_overrun"].astype(float).mean()
                    ),
                    "lower_tail_underrun_fraction": float(
                        group["qq_lower_tail_underrun"].astype(float).mean()
                    ),
                    "upper_tail_underrun_fraction": float(
                        group["qq_upper_tail_underrun"].astype(float).mean()
                    ),
                    "any_tail_underrun_fraction": float(
                        group["qq_any_tail_underrun"].astype(float).mean()
                    ),
                    "any_tail_mismatch_fraction": float(
                        group["qq_any_tail_mismatch"].astype(float).mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _quality_stratified_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize every quality family over scientifically relevant strata."""

    working = frame.copy()
    working["matrix"] = working["condition"].map(
        lambda value: "water" if str(value).endswith("_water") else "soil"
    )
    working["analyte_count"] = working["condition"].map(
        lambda value: sum(
            f"{analyte}-" in str(value) for analyte in ("DEL", "CHL", "TEB")
        )
    )
    working["analyte_signature"] = working["condition"].map(
        lambda value: "+".join(
            analyte
            for analyte in ("DEL", "CHL", "TEB")
            if f"{analyte}-" in str(value)
        )
    )
    working["axis_group"] = working["point_count"].map(
        lambda value: f"{int(value)}_points"
    )
    grouping_specs: list[tuple[str, list[str]]] = [
        ("overall", []),
        ("matrix", ["matrix"]),
        ("analyte_count", ["analyte_count"]),
        ("analyte_signature", ["analyte_signature"]),
        ("matrix_x_analyte_count", ["matrix", "analyte_count"]),
        ("axis_group", ["axis_group"]),
    ]
    metrics = [
        "nearest_reference_mse_mean",
        "nearest_reference_rmse_mean",
        "nearest_reference_standardized_shape_mse_mean",
        "nearest_reference_standardized_shape_rmse_mean",
        "nearest_reference_model_normalized_mse_mean",
        "nearest_reference_model_normalized_rmse_mean",
        "nearest_reference_model_normalized_cosine_mean",
        "nearest_reference_model_normalized_pearson_mean",
        "nearest_reference_cosine_mean",
        "nearest_reference_pearson_mean",
        "mean_spectrum_mse",
        "wasserstein_raw_pointwise_mean",
        "wasserstein_normalized_pointwise_mean",
        "wasserstein_model_normalized_pointwise_mean",
        "wasserstein_model_normalized_global_flattened",
        "generated_to_training_mse_ratio_vs_heldout",
        "diversity_pairwise_mse_ratio",
        "diversity_pearson_distance_ratio",
        "peak_position_std_ratio_median",
        "peak_position_wasserstein_cm1_mean",
        "peak_position_frozen_fraction",
        "peak_height_iqr_ratio_median",
        "peak_width_iqr_ratio_median",
        "qq_lower_tail_span_ratio",
        "qq_central_10_90_span_ratio",
        "qq_upper_tail_span_ratio",
        "qq_lower_outer_01_10_span_ratio",
        "qq_lower_inner_10_50_span_ratio",
        "qq_upper_inner_50_90_span_ratio",
        "qq_upper_outer_90_99_span_ratio",
        "qq_central_fit_slope",
        "qq_lower_tail_signed_curvature",
        "qq_upper_tail_signed_curvature",
        "qq_lower_tail_curve_rmse",
        "qq_upper_tail_curve_rmse",
        "qq_oracle_outside_fraction",
        "qq_oracle_outside_lower_tail_fraction",
        "qq_oracle_outside_central_fraction",
        "qq_oracle_outside_upper_tail_fraction",
        "qq_raw_r2",
        "qq_raw_rmse",
        "qq_model_normalized_r2",
        "qq_model_normalized_rmse",
        "qq_training_standardized_r2",
        "qq_training_standardized_rmse",
        "generated_below_training_envelope_point_fraction",
        "generated_above_training_envelope_point_fraction",
    ]
    rows: list[dict[str, Any]] = []
    for level, columns in grouping_specs:
        groups = [((), working)] if not columns else working.groupby(columns)
        for key, group in groups:
            keys = key if isinstance(key, tuple) else (key,)
            label = "all" if not columns else "|".join(str(value) for value in keys)
            for metric in metrics:
                if metric not in group:
                    continue
                values = group[metric].to_numpy(dtype=np.float64)
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                rows.append(
                    {
                        "stratum_level": level,
                        "stratum": label,
                        "condition_count": int(group.shape[0]),
                        "metric_condition_count": int(values.size),
                        "metric": metric,
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values, ddof=0)),
                        "p05": float(np.percentile(values, 5.0)),
                        "median": float(np.median(values)),
                        "p95": float(np.percentile(values, 95.0)),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    arguments = parse_arguments()
    configuration = load_configuration(arguments.config)
    envelope_configuration = (
        (configuration.get("generation", {}) or {}).get(
            "intensity_envelope_guard", {}
        )
        or {}
    )
    peak_diversity_configuration = (
        (configuration.get("generation", {}) or {}).get(
            "peak_diversity_evaluation", {}
        )
        or {}
    )
    checkpoint_path = _resolve_path(configuration, arguments.checkpoint)
    checkpoint = load_checkpoint_file(checkpoint_path, map_location="cpu")
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("checkpoint缺少metadata。")
    normalization_state = metadata.get("normalization_state")
    if not isinstance(normalization_state, dict):
        raise RuntimeError("checkpoint缺少训练集拟合的normalization_state。")
    normalizer = GlobalMinMaxNormalizer.from_state_dict(normalization_state)

    conditioning_metadata = metadata.get("conditioning_metadata")
    if not isinstance(conditioning_metadata, dict):
        raise RuntimeError("checkpoint缺少conditioning_metadata。")
    requested = str(arguments.condition).strip()
    conditions = (
        [
            record[0]
            for record in source_named_conditions_from_metadata(conditioning_metadata)
        ]
        if requested.lower() == "all"
        else [Path(requested).stem]
    )

    configured_generated_root = resolve_project_path(
        configuration,
        configuration["output"]["generated_spectrum_directory"],
    ).resolve()
    generated_argument = arguments.generated
    if generated_argument is not None:
        supplied = Path(generated_argument).expanduser()
        if not supplied.is_absolute():
            generated_argument = str(resolve_project_path(configuration, supplied))

    if arguments.output_directory is None:
        if generated_argument is None:
            evaluation_parent = configured_generated_root
        else:
            supplied_generated = Path(generated_argument)
            evaluation_parent = (
                supplied_generated.parent
                if supplied_generated.is_file()
                else supplied_generated
            )
        output_root = evaluation_parent / f"quality_metrics_{arguments.reference_split}"
    else:
        output_root = _resolve_path(configuration, arguments.output_directory)
    output_root.mkdir(parents=True, exist_ok=True)

    data = configuration["data"]
    collection = read_spectrum_collection(
        resolve_project_path(configuration, data["input_directory"]), data
    )
    relative = np.asarray(
        [str(value) for value in collection.relative_source_files], dtype=object
    )
    if relative.tolist() != metadata.get("relative_source_files"):
        raise RuntimeError("当前输入文件顺序与checkpoint训练时不一致。")
    training_set = set(int(value) for value in metadata.get("training_indices", []))
    reference_set = set(
        int(value)
        for value in metadata.get(f"{arguments.reference_split}_indices", [])
    )

    summary_rows: list[dict[str, Any]] = []
    nearest_frames: list[pd.DataFrame] = []
    qq_frames: list[pd.DataFrame] = []
    peak_diversity_frames: list[pd.DataFrame] = []
    print("===== D4条件生成核心指标评价 =====")
    print(f"checkpoint：{checkpoint_path}")
    print(f"真实参考子集：{arguments.reference_split}")
    print(f"条件数：{len(conditions)}")

    for condition_index, query in enumerate(conditions):
        source_name, condition_id, _, source_file = (
            resolve_source_named_condition_from_metadata(conditioning_metadata, query)
        )
        generated_path = _generated_path(
            generated_argument=generated_argument,
            configured_root=configured_generated_root,
            condition=source_name,
            allow_direct_file=len(conditions) == 1,
        )
        axis, generated, generated_names = _read_generated(generated_path)
        source_indices = np.flatnonzero(relative == source_file)
        training_indices = np.asarray(
            [index for index in source_indices if int(index) in training_set],
            dtype=np.int64,
        )
        reference_indices = np.asarray(
            [index for index in source_indices if int(index) in reference_set],
            dtype=np.int64,
        )
        if training_indices.size != 12 or reference_indices.size != 4:
            raise RuntimeError(
                f"条件{source_name}不是固定12条训练/4条"
                f"{arguments.reference_split}："
                f"{training_indices.size}/{reference_indices.size}。"
            )
        training = _interpolate_collection(collection, training_indices, axis)
        reference = _interpolate_collection(collection, reference_indices, axis)
        training_model_normalized = normalizer.transform(training)
        reference_model_normalized = normalizer.transform(reference)
        generated_model_normalized = normalizer.transform(generated)

        heldout_to_training = nearest_reference_metrics(
            query_spectra=reference,
            reference_spectra=training,
            query_names=[
                f"{arguments.reference_split}_{index + 1:02d}"
                for index in range(reference.shape[0])
            ],
            reference_names=[
                f"training_{index + 1:02d}"
                for index in range(training.shape[0])
            ],
        )
        generated_to_training = nearest_reference_metrics(
            query_spectra=generated,
            reference_spectra=training,
            query_names=generated_names,
            reference_names=[
                f"training_{index + 1:02d}"
                for index in range(training.shape[0])
            ],
        )
        heldout_context = _comparison_summary_fields(
            heldout_to_training,
            "heldout_to_training",
        )
        generated_training_context = _comparison_summary_fields(
            generated_to_training,
            "generated_to_training",
        )

        nearest = nearest_reference_metrics(
            query_spectra=generated,
            reference_spectra=reference,
            query_names=generated_names,
            reference_names=[
                f"{arguments.reference_split}_{index + 1:02d}"
                for index in range(reference.shape[0])
            ],
        )
        nearest["raw_mse"] = np.square(
            nearest["raw_rmse"].to_numpy(dtype=np.float64)
        )
        nearest["standardized_shape_mse"] = np.square(
            nearest["standardized_shape_rmse"].to_numpy(dtype=np.float64)
        )
        model_normalized_nearest = nearest_reference_metrics(
            query_spectra=generated_model_normalized,
            reference_spectra=reference_model_normalized,
            query_names=generated_names,
            reference_names=[
                f"{arguments.reference_split}_{index + 1:02d}"
                for index in range(reference.shape[0])
            ],
        )
        if not np.array_equal(
            nearest["nearest_reference_index"].to_numpy(dtype=np.int64),
            model_normalized_nearest["nearest_reference_index"].to_numpy(
                dtype=np.int64
            ),
        ):
            raise RuntimeError(
                "原强度与模型归一化空间选择了不同参考谱，禁止混合汇总。"
            )
        nearest["model_normalized_rmse"] = model_normalized_nearest[
            "raw_rmse"
        ].to_numpy(dtype=np.float64)
        nearest["model_normalized_mse"] = np.square(
            nearest["model_normalized_rmse"].to_numpy(dtype=np.float64)
        )
        nearest["model_normalized_cosine"] = model_normalized_nearest[
            "cosine"
        ].to_numpy(dtype=np.float64)
        nearest["model_normalized_pearson"] = model_normalized_nearest[
            "pearson"
        ].to_numpy(dtype=np.float64)
        nearest.insert(0, "condition", source_name)
        nearest.insert(1, "internal_condition_id", condition_id)
        nearest_frames.append(nearest)

        diversity, _, _ = _diversity_fields(
            training=training,
            generated=generated,
            pair_count=arguments.generated_pair_count,
            random_seed=2026 + condition_index,
        )
        peak_diversity, peak_details = _peak_morphology_diversity(
            condition=source_name,
            axis=axis,
            training=training,
            generated=generated,
            configuration=peak_diversity_configuration,
        )
        if not peak_details.empty:
            peak_diversity_frames.append(peak_details)
            if arguments.plot_mode == "all":
                _plot_peak_diversity(
                    peak_details,
                    output_root
                    / "peak_diversity_plots"
                    / f"{source_name}_peak_diversity.png",
                )
                _plot_peak_intensity_violins(
                    axis=axis,
                    training=training,
                    reference=reference,
                    generated=generated,
                    details=peak_details,
                    condition=source_name,
                    output=(
                        output_root
                        / "peak_violin_plots"
                        / f"{source_name}_peak_violin.png"
                    ),
                )
        qq, qq_metrics = _qq_table(
            condition=source_name,
            training=training,
            reference=reference,
            generated=generated,
            training_model_normalized=training_model_normalized,
            reference_model_normalized=reference_model_normalized,
            generated_model_normalized=generated_model_normalized,
            quantile_count=arguments.qq_quantile_count,
            oracle_repeats=arguments.qq_oracle_repeats,
            oracle_confidence=arguments.qq_oracle_confidence,
            random_seed=2026000 + condition_index,
        )
        qq_frames.append(qq)
        if arguments.plot_mode == "all":
            _plot_qq(
                qq,
                source_name,
                output_root / "qq_plots" / f"{source_name}_qq.png",
            )
            _plot_spectral_distribution(
                axis=axis,
                training=training,
                reference=reference,
                generated=generated,
                condition=source_name,
                output=(
                    output_root
                    / "spectral_distribution_plots"
                    / f"{source_name}_distribution.png"
                ),
            )

        generated_to_training_mse = float(
            generated_training_context["generated_to_training_mse_mean"]
        )
        heldout_to_training_mse = float(
            heldout_context["heldout_to_training_mse_mean"]
        )

        model_normalized_wasserstein = _pointwise_wasserstein(
            training=training_model_normalized,
            reference=reference_model_normalized,
            generated=generated_model_normalized,
        )
        row: dict[str, Any] = {
            "condition": source_name,
            "internal_condition_id": condition_id,
            "relative_source_file": source_file,
            "reference_split": arguments.reference_split,
            "training_count": int(training.shape[0]),
            "reference_count": int(reference.shape[0]),
            "generated_count": int(generated.shape[0]),
            "raman_start_cm1": float(axis[0]),
            "raman_end_cm1": float(axis[-1]),
            "point_count": int(axis.size),
            **_metric_summary_fields(nearest),
            **{
                f"nearest_reference_{metric}_{statistic}": float(
                    summarize_vector(
                        nearest[metric].to_numpy(dtype=np.float64)
                    )[statistic]
                )
                for metric in (
                    "model_normalized_mse",
                    "model_normalized_rmse",
                    "model_normalized_cosine",
                    "model_normalized_pearson",
                )
                for statistic in ("mean", "std", "p05", "median", "p95")
            },
            **heldout_context,
            **generated_training_context,
            "generated_to_training_mse_ratio_vs_heldout": (
                generated_to_training_mse
                / max(heldout_to_training_mse, EPSILON)
            ),
            **_center_metrics(reference, generated),
            **_pointwise_wasserstein(
                training=training,
                reference=reference,
                generated=generated,
            ),
            "wasserstein_model_normalized_pointwise_mean": (
                model_normalized_wasserstein[
                    "wasserstein_raw_pointwise_mean"
                ]
            ),
            "wasserstein_model_normalized_pointwise_median": (
                model_normalized_wasserstein[
                    "wasserstein_raw_pointwise_median"
                ]
            ),
            "wasserstein_model_normalized_pointwise_p95": (
                model_normalized_wasserstein[
                    "wasserstein_raw_pointwise_p95"
                ]
            ),
            "wasserstein_model_normalized_global_flattened": float(
                wasserstein_distance(
                    reference_model_normalized.reshape(-1),
                    generated_model_normalized.reshape(-1),
                )
            ),
            **_training_supported_intensity_metrics(
                training=training,
                generated=generated,
                configuration=envelope_configuration,
            ),
            **diversity,
            **peak_diversity,
            **qq_metrics,
            "generated_minimum_intensity": float(np.min(generated)),
            "generated_negative_point_fraction": float(np.mean(generated < 0.0)),
            "generated_below_minus_40_point_fraction": float(
                np.mean(generated < -40.0)
            ),
            "generated_below_minus_40_spectrum_fraction": float(
                np.mean(np.any(generated < -40.0, axis=1))
            ),
        }
        summary_rows.append(row)
        print(
            f"[{condition_index + 1}/{len(conditions)}] {source_name}："
            f"MSE={row['nearest_reference_mse_mean']:.6g}；"
            f"RMSE={row['nearest_reference_rmse_mean']:.6g}；"
            "shape-MSE/RMSE="
            f"{row['nearest_reference_standardized_shape_mse_mean']:.6g}/"
            f"{row['nearest_reference_standardized_shape_rmse_mean']:.6g}；"
            "model-norm MSE/RMSE="
            f"{row['nearest_reference_model_normalized_mse_mean']:.6g}/"
            f"{row['nearest_reference_model_normalized_rmse_mean']:.6g}；"
            f"cos={row['nearest_reference_cosine_mean']:.6f}；"
            f"r={row['nearest_reference_pearson_mean']:.6f}；"
            f"W1raw={row['wasserstein_raw_pointwise_mean']:.6f}；"
            f"W1norm={row['wasserstein_normalized_pointwise_mean']:.6f}"
        )
        print(
            "  平均谱："
            f"MSE={row['mean_spectrum_mse']:.6g}；"
            f"cos={row['mean_spectrum_cosine']:.6f}；"
            f"r={row['mean_spectrum_pearson']:.6f}"
        )
        print(
            "  train分布参照：generated/train MSE="
            f"{row['generated_to_training_mse_mean']:.6g}；"
            "held-out/train MSE="
            f"{row['heldout_to_training_mse_mean']:.6g}；"
            "ratio="
            f"{row['generated_to_training_mse_ratio_vs_heldout']:.4f}"
        )
        print(
            "  多样性MSE："
            f"train={row['training_pairwise_mse_median']:.6g}；"
            f"generated={row['generated_pairwise_mse_median']:.6g}；"
            f"ratio={row['diversity_pairwise_mse_ratio']:.4f}；"
            "Pearson-distance ratio="
            f"{row['diversity_pearson_distance_ratio']:.4f}"
        )
        print(
            "  峰形多样性：position-std ratio="
            f"{row['peak_position_std_ratio_median']:.4f}；"
            "height-IQR ratio="
            f"{row['peak_height_iqr_ratio_median']:.4f}；"
            "width-IQR ratio="
            f"{row['peak_width_iqr_ratio_median']:.4f}；"
            "position frozen="
            f"{row['peak_position_frozen_fraction']:.3%}"
        )
        print(
            "  负值："
            f"min={row['generated_minimum_intensity']:.3f}；"
            f"<-40点={row['generated_below_minus_40_point_fraction']:.6%}；"
            f"<-40光谱={row['generated_below_minus_40_spectrum_fraction']:.3%}"
        )
        print(
            "  training逐点包络外：lower点/光谱="
            f"{row['generated_below_training_envelope_point_fraction']:.6%}/"
            f"{row['generated_below_training_envelope_spectrum_fraction']:.3%}；"
            "upper点/光谱="
            f"{row['generated_above_training_envelope_point_fraction']:.6%}/"
            f"{row['generated_above_training_envelope_spectrum_fraction']:.3%}"
        )
        print(
            "  QQ拟合：raw "
            f"R2={row['qq_raw_r2']:.6f}, RMSE={row['qq_raw_rmse']:.6g}；"
            "model-normalized "
            f"R2={row['qq_model_normalized_r2']:.6f}, "
            f"RMSE={row['qq_model_normalized_rmse']:.6g}；"
            "training-standardized "
            f"R2={row['qq_training_standardized_r2']:.6f}, "
            f"RMSE={row['qq_training_standardized_rmse']:.6g}"
        )
        print(
            "  QQ尾宽/oracle区间：lower="
            f"{row['qq_lower_tail_span_ratio']:.4f}/"
            f"[{row['qq_lower_tail_oracle_lower']:.4f},"
            f"{row['qq_lower_tail_oracle_upper']:.4f}]；upper="
            f"{row['qq_upper_tail_span_ratio']:.4f}/"
            f"[{row['qq_upper_tail_oracle_lower']:.4f},"
            f"{row['qq_upper_tail_oracle_upper']:.4f}]；"
            f"overrun={bool(row['qq_any_tail_overrun'])}；"
            f"underrun={bool(row['qq_any_tail_underrun'])}"
        )
        print(
            "  QQ四段比率[1-10,10-50,50-90,90-99]="
            f"[{row['qq_lower_outer_01_10_span_ratio']:.4f},"
            f"{row['qq_lower_inner_10_50_span_ratio']:.4f},"
            f"{row['qq_upper_inner_50_90_span_ratio']:.4f},"
            f"{row['qq_upper_outer_90_99_span_ratio']:.4f}]；"
            "尾部曲率RMSE(lower/upper)="
            f"{row['qq_lower_tail_curve_rmse']:.4f}/"
            f"{row['qq_upper_tail_curve_rmse']:.4f}；"
            "oracle外分位点="
            f"{row['qq_oracle_outside_fraction']:.2%}"
        )

    condition_summary = pd.DataFrame(summary_rows)
    nearest_details = pd.concat(nearest_frames, ignore_index=True)
    qq_details = pd.concat(qq_frames, ignore_index=True)
    peak_details_all = (
        pd.concat(peak_diversity_frames, ignore_index=True)
        if peak_diversity_frames
        else pd.DataFrame()
    )
    overall = _overall_summary(condition_summary)
    qq_stratified = _qq_stratified_summary(condition_summary)
    quality_stratified = _quality_stratified_summary(condition_summary)
    overview_path = output_root / "quality_overview.png"
    if arguments.plot_mode == "all":
        _plot_quality_overview(condition_summary, overview_path)
    overall_mean = {
        str(record["metric"]): float(record["mean"])
        for record in overall.to_dict(orient="records")
    }
    qq_any_tail_overrun_fraction = float(
        condition_summary["qq_any_tail_overrun"].astype(float).mean()
    )
    qq_any_tail_underrun_fraction = float(
        condition_summary["qq_any_tail_underrun"].astype(float).mean()
    )
    qq_any_tail_mismatch_fraction = float(
        condition_summary["qq_any_tail_mismatch"].astype(float).mean()
    )
    point_weights = (
        condition_summary["generated_count"].to_numpy(dtype=np.float64)
        * condition_summary["point_count"].to_numpy(dtype=np.float64)
    )
    spectrum_weights = condition_summary["generated_count"].to_numpy(
        dtype=np.float64
    )
    generated_global_minimum = float(
        condition_summary["generated_minimum_intensity"].min()
    )
    generated_global_maximum = float(
        condition_summary["generated_maximum_intensity"].max()
    )

    def weighted_fraction(column: str, weights: np.ndarray) -> float:
        return float(
            np.average(
                condition_summary[column].to_numpy(dtype=np.float64),
                weights=weights,
            )
        )

    condition_path = output_root / "condition_metrics.csv"
    overall_path = output_root / "overall_metric_summary.csv"
    nearest_path = output_root / "generated_to_reference_details.csv"
    qq_path = output_root / "qq_quantiles.csv"
    qq_stratified_path = output_root / "qq_stratified_summary.csv"
    quality_stratified_path = output_root / "quality_stratified_summary.csv"
    peak_details_path = output_root / "peak_morphology_details.csv"
    condition_summary.to_csv(condition_path, index=False, encoding="utf-8-sig")
    overall.to_csv(overall_path, index=False, encoding="utf-8-sig")
    nearest_details.to_csv(nearest_path, index=False, encoding="utf-8-sig")
    qq_details.to_csv(qq_path, index=False, encoding="utf-8-sig")
    qq_stratified.to_csv(
        qq_stratified_path, index=False, encoding="utf-8-sig"
    )
    quality_stratified.to_csv(
        quality_stratified_path, index=False, encoding="utf-8-sig"
    )
    peak_details_all.to_csv(
        peak_details_path, index=False, encoding="utf-8-sig"
    )

    report_lines = [
        "D4条件生成核心指标评价",
        f"checkpoint: {checkpoint_path}",
        f"reference_split: {arguments.reference_split}",
        f"condition_count: {condition_summary.shape[0]}",
        f"plot_mode: {arguments.plot_mode}",
        "MSE/RMSE/cosine/Pearson: same Raman positions after selecting the nearest same-condition held-out spectrum; values are never sorted",
        "RMSE: mean per-spectrum raw RMSE; it is not QQ RMSE",
        "Standardized shape MSE/RMSE: per-spectrum z-normalized shape comparison",
        "Model-normalized metrics: final delivered spectra transformed with the checkpoint train-only normalizer",
        "Wasserstein global: flattened normalized intensity distributions; pointwise W1 is retained separately",
        "Diversity: generated pairwise distance relative to 12 training spectra",
        "QQ: independently sorted quantiles; model-normalized QQ is paper-compatible",
        "",
        "Overall original metric values (condition means):",
        f"nearest_reference_mse_mean: {overall_mean['nearest_reference_mse_mean']:.9g}",
        f"nearest_reference_rmse_mean: {overall_mean['nearest_reference_rmse_mean']:.9g}",
        "nearest_reference_standardized_shape_mse_mean: "
        f"{overall_mean['nearest_reference_standardized_shape_mse_mean']:.9g}",
        "nearest_reference_standardized_shape_rmse_mean: "
        f"{overall_mean['nearest_reference_standardized_shape_rmse_mean']:.9g}",
        "nearest_reference_model_normalized_mse_mean: "
        f"{overall_mean['nearest_reference_model_normalized_mse_mean']:.9g}",
        "nearest_reference_model_normalized_rmse_mean: "
        f"{overall_mean['nearest_reference_model_normalized_rmse_mean']:.9g}",
        "nearest_reference_model_normalized_cosine_mean: "
        f"{overall_mean['nearest_reference_model_normalized_cosine_mean']:.9g}",
        "nearest_reference_model_normalized_pearson_mean: "
        f"{overall_mean['nearest_reference_model_normalized_pearson_mean']:.9g}",
        f"nearest_reference_cosine_mean: {overall_mean['nearest_reference_cosine_mean']:.9g}",
        f"nearest_reference_pearson_mean: {overall_mean['nearest_reference_pearson_mean']:.9g}",
        f"heldout_to_training_mse_mean: {overall_mean['heldout_to_training_mse_mean']:.9g}",
        f"generated_to_training_mse_mean: {overall_mean['generated_to_training_mse_mean']:.9g}",
        "generated_to_training_mse_ratio_vs_heldout: "
        f"{overall_mean['generated_to_training_mse_ratio_vs_heldout']:.9g}",
        f"mean_spectrum_mse: {overall_mean['mean_spectrum_mse']:.9g}",
        f"wasserstein_raw_pointwise_mean: {overall_mean['wasserstein_raw_pointwise_mean']:.9g}",
        f"wasserstein_normalized_pointwise_mean: {overall_mean['wasserstein_normalized_pointwise_mean']:.9g}",
        "wasserstein_model_normalized_pointwise_mean: "
        f"{overall_mean['wasserstein_model_normalized_pointwise_mean']:.9g}",
        "wasserstein_model_normalized_global_flattened: "
        f"{overall_mean['wasserstein_model_normalized_global_flattened']:.9g}",
        f"diversity_pairwise_mse_ratio: {overall_mean['diversity_pairwise_mse_ratio']:.9g}",
        f"diversity_pearson_distance_ratio: {overall_mean['diversity_pearson_distance_ratio']:.9g}",
        f"peak_position_std_ratio_median: {overall_mean.get('peak_position_std_ratio_median', float('nan')):.9g}",
        f"peak_height_iqr_ratio_median: {overall_mean.get('peak_height_iqr_ratio_median', float('nan')):.9g}",
        f"peak_width_iqr_ratio_median: {overall_mean.get('peak_width_iqr_ratio_median', float('nan')):.9g}",
        f"peak_position_frozen_fraction: {overall_mean.get('peak_position_frozen_fraction', float('nan')):.9g}",
        f"qq_lower_tail_span_ratio: {overall_mean['qq_lower_tail_span_ratio']:.9g}",
        f"qq_upper_tail_span_ratio: {overall_mean['qq_upper_tail_span_ratio']:.9g}",
        f"qq_central_10_90_span_ratio: {overall_mean['qq_central_10_90_span_ratio']:.9g}",
        "qq_lower_outer_01_10_span_ratio: "
        f"{overall_mean['qq_lower_outer_01_10_span_ratio']:.9g}",
        "qq_lower_inner_10_50_span_ratio: "
        f"{overall_mean['qq_lower_inner_10_50_span_ratio']:.9g}",
        "qq_upper_inner_50_90_span_ratio: "
        f"{overall_mean['qq_upper_inner_50_90_span_ratio']:.9g}",
        "qq_upper_outer_90_99_span_ratio: "
        f"{overall_mean['qq_upper_outer_90_99_span_ratio']:.9g}",
        f"qq_central_fit_slope: {overall_mean['qq_central_fit_slope']:.9g}",
        "qq_lower_tail_curve_rmse: "
        f"{overall_mean['qq_lower_tail_curve_rmse']:.9g}",
        "qq_upper_tail_curve_rmse: "
        f"{overall_mean['qq_upper_tail_curve_rmse']:.9g}",
        "qq_oracle_outside_fraction: "
        f"{overall_mean['qq_oracle_outside_fraction']:.9g}",
        f"qq_raw_r2: {overall_mean['qq_raw_r2']:.9g}",
        f"qq_raw_rmse: {overall_mean['qq_raw_rmse']:.9g}",
        f"qq_model_normalized_r2: {overall_mean['qq_model_normalized_r2']:.9g}",
        "qq_model_normalized_rmse: "
        f"{overall_mean['qq_model_normalized_rmse']:.9g}",
        "qq_training_standardized_r2: "
        f"{overall_mean['qq_training_standardized_r2']:.9g}",
        "qq_training_standardized_rmse: "
        f"{overall_mean['qq_training_standardized_rmse']:.9g}",
        f"qq_any_tail_overrun_fraction: {qq_any_tail_overrun_fraction:.9g}",
        f"qq_any_tail_underrun_fraction: {qq_any_tail_underrun_fraction:.9g}",
        f"qq_any_tail_mismatch_fraction: {qq_any_tail_mismatch_fraction:.9g}",
        f"generated_minimum_intensity_global: {generated_global_minimum:.9g}",
        f"generated_maximum_intensity_global: {generated_global_maximum:.9g}",
        "generated_minimum_intensity_condition_mean: "
        f"{overall_mean['generated_minimum_intensity']:.9g}",
        "generated_below_minus_40_point_fraction_global: "
        f"{weighted_fraction('generated_below_minus_40_point_fraction', point_weights):.9g}",
        "generated_below_minus_40_spectrum_fraction_global: "
        f"{weighted_fraction('generated_below_minus_40_spectrum_fraction', spectrum_weights):.9g}",
        "generated_below_training_envelope_point_fraction_global: "
        f"{weighted_fraction('generated_below_training_envelope_point_fraction', point_weights):.9g}",
        "generated_above_training_envelope_point_fraction_global: "
        f"{weighted_fraction('generated_above_training_envelope_point_fraction', point_weights):.9g}",
        f"condition_metrics: {condition_path}",
        f"overall_summary: {overall_path}",
        "qq_directory: "
        + (
            str(output_root / "qq_plots")
            if arguments.plot_mode == "all"
            else "disabled (--plot-mode none)"
        ),
        "spectral_distribution_directory: "
        + (
            str(output_root / "spectral_distribution_plots")
            if arguments.plot_mode == "all"
            else "disabled (--plot-mode none)"
        ),
        "quality_overview: "
        + (
            str(overview_path)
            if arguments.plot_mode == "all"
            else "disabled (--plot-mode none)"
        ),
        f"qq_stratified_summary: {qq_stratified_path}",
        f"quality_stratified_summary: {quality_stratified_path}",
        f"peak_morphology_details: {peak_details_path}",
        "peak_diversity_plots: "
        + (
            str(output_root / "peak_diversity_plots")
            if arguments.plot_mode == "all"
            else "disabled (--plot-mode none)"
        ),
        "peak_violin_plots: "
        + (
            str(output_root / "peak_violin_plots")
            if arguments.plot_mode == "all"
            else "disabled (--plot-mode none)"
        ),
    ]
    report_path = output_root / "evaluation_report.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("===== 评价完成 =====")
    print(f"逐条件指标：{condition_path}")
    print(f"总体统计：{overall_path}")
    if arguments.plot_mode == "all":
        print(f"QQ图：{output_root / 'qq_plots'}")
        print(f"光谱分布图：{output_root / 'spectral_distribution_plots'}")
        print(f"全条件总览图：{overview_path}")
    else:
        print("图像输出：已跳过（--plot-mode none，数值指标不受影响）")
    print(f"QQ分层汇总：{qq_stratified_path}")
    print(f"全质量分层汇总：{quality_stratified_path}")
    print(f"峰形多样性明细：{peak_details_path}")
    if arguments.plot_mode == "all":
        print(f"特征峰小提琴图：{output_root / 'peak_violin_plots'}")
    print(f"报告：{report_path}")


if __name__ == "__main__":
    main()
