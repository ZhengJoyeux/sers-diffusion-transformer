"""Pointwise robust calibration for broad/background tails in generated SERS.

This script only compresses rare broad-shape deviations that exceed the
distribution learned from the REAL TRAINING spectra.

Preserved:
- spectrum-wide broad/background offset;
- local/fine spectral fluctuation;
- automatically detected SERS peak regions;
- Raman peak positions;
- normal broad/background variation.

This is a validation-stage generation calibration and does not modify the DDPM.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from src.checkpoint_manager import load_checkpoint_file
from src.configuration_loader import load_configuration
from src.dataset_splitter import split_spectrum_collection
from src.spectrum_file_reader import read_spectrum_collection
from src.spectrum_length_adapter import SpectrumLengthAdapter

from scripts.diagnose_real_vs_generated_sers_distribution import (
    detect_peak_regions,
    read_generated_file,
    sigma_samples,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "利用真实训练集逐点robust分布，"
            "软压缩生成SERS异常broad/background尾部。"
        )
    )

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--generated",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--training-tail-quantile",
        type=float,
        default=97.5,
        help=(
            "使用真实训练broad-shape标准化绝对残差"
            "的 pooled 分位数作为软限制起点。"
        ),
    )

    parser.add_argument(
        "--soft-margin",
        type=float,
        default=0.75,
        help=(
            "超过训练阈值后允许的额外标准化软尾部宽度。"
        ),
    )

    parser.add_argument(
        "--scale-floor-quantile",
        type=float,
        default=25.0,
        help=(
            "逐点MAD过小时使用非峰区MAD的该分位数作为下限。"
        ),
    )

    return parser.parse_args()


def build_peak_protection_weight(
    *,
    axis: np.ndarray,
    peak_indices: np.ndarray,
    half_width_cm1: float,
    transition_width_cm1: float,
) -> np.ndarray:
    weight = np.zeros(
        axis.size,
        dtype=np.float64,
    )

    outer_width = (
        float(half_width_cm1)
        + float(transition_width_cm1)
    )

    for peak_index in peak_indices:
        center = float(
            axis[int(peak_index)]
        )

        distance = np.abs(
            axis - center
        )

        current = np.zeros_like(
            distance,
            dtype=np.float64,
        )

        current[
            distance <= half_width_cm1
        ] = 1.0

        transition = (
            (distance > half_width_cm1)
            & (distance < outer_width)
        )

        if np.any(transition):
            phase = (
                distance[transition]
                - half_width_cm1
            ) / transition_width_cm1

            current[transition] = (
                0.5
                * (
                    1.0
                    + np.cos(
                        np.pi * phase
                    )
                )
            )

        weight = np.maximum(
            weight,
            current,
        )

    return weight


def calculate_shape_components(
    *,
    broad: np.ndarray,
    reference_broad: np.ndarray,
    non_peak_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Separate broad deviation into:
      spectrum-wide vertical offset
      broad-shape deviation
    """

    deviation = (
        broad
        - reference_broad[
            np.newaxis,
            :
        ]
    )

    offsets = np.mean(
        deviation[
            :,
            non_peak_mask
        ],
        axis=1,
    )

    shapes = (
        deviation
        - offsets[:, np.newaxis]
    )

    return offsets, shapes


def robust_pointwise_scale(
    *,
    training_shapes: np.ndarray,
    non_peak_mask: np.ndarray,
    floor_quantile: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
]:
    """
    Calculate pointwise median and MAD-based scale from training shapes.
    """

    pointwise_center = np.median(
        training_shapes,
        axis=0,
    )

    absolute_deviation = np.abs(
        training_shapes
        - pointwise_center[
            np.newaxis,
            :
        ]
    )

    mad = np.median(
        absolute_deviation,
        axis=0,
    )

    scale = (
        1.4826
        * mad
    )

    valid_scale = scale[
        non_peak_mask
    ]

    positive_scale = valid_scale[
        valid_scale > 1.0e-12
    ]

    if positive_scale.size == 0:
        raise RuntimeError(
            "训练broad shape的逐点MAD全部接近0。"
        )

    scale_floor = float(
        np.percentile(
            positive_scale,
            float(
                floor_quantile
            ),
        )
    )

    scale = np.maximum(
        scale,
        scale_floor,
    )

    return (
        pointwise_center,
        scale,
        scale_floor,
    )


def soft_clip_standardized(
    *,
    standardized: np.ndarray,
    limit: float,
    margin: float,
) -> np.ndarray:
    """
    Keep normal values unchanged and smoothly compress only tails.

    |z| <= limit:
        unchanged

    |z| > limit:
        approaches limit + margin smoothly
    """

    values = np.asarray(
        standardized,
        dtype=np.float64,
    )

    absolute = np.abs(
        values
    )

    sign = np.sign(
        values
    )

    output = (
        values.copy()
    )

    tail = (
        absolute > limit
    )

    excess = (
        absolute[tail]
        - limit
    )

    compressed_absolute = (
        limit
        + margin
        * np.tanh(
            excess / margin
        )
    )

    output[tail] = (
        sign[tail]
        * compressed_absolute
    )

    return output


def main() -> None:
    args = parse_arguments()

    if not (
        50.0
        < args.training_tail_quantile
        < 100.0
    ):
        raise ValueError(
            "--training-tail-quantile必须位于(50,100)。"
        )

    if (
        not np.isfinite(
            args.soft_margin
        )
        or args.soft_margin <= 0.0
    ):
        raise ValueError(
            "--soft-margin必须为有限正数。"
        )

    if not (
        0.0
        <= args.scale_floor_quantile
        < 100.0
    ):
        raise ValueError(
            "--scale-floor-quantile必须位于[0,100)。"
        )

    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    checkpoint_path = Path(
        args.checkpoint
    ).expanduser()

    if not checkpoint_path.is_absolute():
        checkpoint_path = (
            project_root
            / checkpoint_path
        )

    checkpoint_path = (
        checkpoint_path.resolve()
    )

    generated_path = Path(
        args.generated
    ).expanduser()

    if not generated_path.is_absolute():
        generated_path = (
            project_root
            / generated_path
        )

    generated_path = (
        generated_path.resolve()
    )

    output_path = Path(
        args.output
    ).expanduser()

    if not output_path.is_absolute():
        output_path = (
            project_root
            / output_path
        )

    output_path = (
        output_path.resolve()
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    configuration = load_configuration(
        args.config
    )

    checkpoint = load_checkpoint_file(
        checkpoint_path,
        map_location="cpu",
    )

    checkpoint_configuration = (
        checkpoint.get(
            "configuration"
        )
    )

    metadata = checkpoint.get(
        "metadata"
    )

    if not isinstance(
        checkpoint_configuration,
        dict,
    ):
        raise RuntimeError(
            "checkpoint缺少configuration。"
        )

    if not isinstance(
        metadata,
        dict,
    ):
        raise RuntimeError(
            "checkpoint缺少metadata。"
        )

    data_config = (
        checkpoint_configuration[
            "data"
        ]
    )

    project_config = (
        checkpoint_configuration.get(
            "project",
            {},
        )
    )

    random_config = (
        checkpoint_configuration.get(
            "random",
            {},
        )
    )

    random_seed = int(
        random_config.get(
            "seed",
            project_config.get(
                "random_seed",
                42,
            ),
        )
    )

    input_directory = Path(
        data_config[
            "input_directory"
        ]
    ).expanduser()

    if not input_directory.is_absolute():
        input_directory = (
            project_root
            / input_directory
        )

    input_directory = (
        input_directory.resolve()
    )

    collection = read_spectrum_collection(
        input_directory=input_directory,
        data_config=data_config,
    )

    dataset_split = split_spectrum_collection(
        collection=collection,
        data_config=data_config,
        random_seed=random_seed,
    )

    training_indices = np.asarray(
        dataset_split.train.indices,
        dtype=np.int64,
    )

    length_adapter = (
        SpectrumLengthAdapter.from_metadata(
            metadata
        )
    )

    training_spectra_original = [
        collection.spectra[
            int(index)
        ]
        for index in training_indices
    ]

    training_axes_original = [
        collection.raman_shifts[
            int(index)
        ]
        for index in training_indices
    ]

    training_model_axis = (
        length_adapter
        .interpolate_to_model_axis(
            training_spectra_original,
            training_axes_original,
        )
    )

    (
        generated_axis,
        generated_spectra,
    ) = read_generated_file(
        generated_path
    )

    training_spectra = (
        length_adapter
        .interpolate_from_model_axis(
            training_model_axis,
            generated_axis,
        )
    ).astype(
        np.float64,
        copy=False,
    )

    generated_spectra = np.asarray(
        generated_spectra,
        dtype=np.float64,
    )

    sampling_configuration = (
        configuration
        .get(
            "generation",
            {},
        )
        .get(
            "sampling_calibration",
            {},
        )
        or {}
    )

    non_peak_config = (
        sampling_configuration.get(
            "non_peak_noise",
            {},
        )
        or {}
    )

    reference_smoothing_sigma_cm1 = float(
        non_peak_config.get(
            "reference_smoothing_sigma_cm1",
            30.0,
        )
    )

    minimum_relative_prominence = float(
        non_peak_config.get(
            "minimum_relative_prominence",
            0.06,
        )
    )

    minimum_peak_distance_cm1 = float(
        non_peak_config.get(
            "minimum_peak_distance_cm1",
            12.0,
        )
    )

    maximum_peak_count = int(
        non_peak_config.get(
            "maximum_peak_count",
            20,
        )
    )

    protection_half_width_cm1 = float(
        non_peak_config.get(
            "peak_protection_half_width_cm1",
            24.0,
        )
    )

    transition_width_cm1 = float(
        non_peak_config.get(
            "transition_width_cm1",
            6.0,
        )
    )

    broad_sigma_cm1 = float(
        non_peak_config.get(
            "broad_sigma_cm1",
            7.0,
        )
    )

    (
        peak_indices,
        _,
        _,
    ) = detect_peak_regions(
        axis=generated_axis,
        training_spectra=training_spectra,
        reference_smoothing_sigma_cm1=(
            reference_smoothing_sigma_cm1
        ),
        minimum_relative_prominence=(
            minimum_relative_prominence
        ),
        minimum_peak_distance_cm1=(
            minimum_peak_distance_cm1
        ),
        maximum_peak_count=(
            maximum_peak_count
        ),
        peak_protection_half_width_cm1=(
            protection_half_width_cm1
        ),
    )

    peak_weight = (
        build_peak_protection_weight(
            axis=generated_axis,
            peak_indices=peak_indices,
            half_width_cm1=(
                protection_half_width_cm1
            ),
            transition_width_cm1=(
                transition_width_cm1
            ),
        )
    )

    non_peak_mask = (
        peak_weight <= 0.05
    )

    if int(
        np.count_nonzero(
            non_peak_mask
        )
    ) < 10:
        raise RuntimeError(
            "自动峰区保护后非峰区不足。"
        )

    broad_sigma = sigma_samples(
        broad_sigma_cm1,
        generated_axis,
    )

    real_broad = gaussian_filter1d(
        training_spectra,
        sigma=broad_sigma,
        axis=1,
        mode="nearest",
    )

    generated_broad = gaussian_filter1d(
        generated_spectra,
        sigma=broad_sigma,
        axis=1,
        mode="nearest",
    )

    # ------------------------------------------------------------
    # Training reference
    # ------------------------------------------------------------
    reference_broad = np.median(
        real_broad,
        axis=0,
    )

    (
        real_offsets,
        real_shapes,
    ) = calculate_shape_components(
        broad=real_broad,
        reference_broad=reference_broad,
        non_peak_mask=non_peak_mask,
    )

    (
        generated_offsets,
        generated_shapes,
    ) = calculate_shape_components(
        broad=generated_broad,
        reference_broad=reference_broad,
        non_peak_mask=non_peak_mask,
    )

    (
        pointwise_shape_center,
        pointwise_shape_scale,
        scale_floor,
    ) = robust_pointwise_scale(
        training_shapes=real_shapes,
        non_peak_mask=non_peak_mask,
        floor_quantile=(
            args.scale_floor_quantile
        ),
    )

    real_standardized = (
        (
            real_shapes
            - pointwise_shape_center[
                np.newaxis,
                :
            ]
        )
        / pointwise_shape_scale[
            np.newaxis,
            :
        ]
    )

    real_pooled_absolute = np.abs(
        real_standardized[
            :,
            non_peak_mask
        ]
    ).reshape(-1)

    training_limit = float(
        np.percentile(
            real_pooled_absolute,
            args.training_tail_quantile,
        )
    )

    if (
        not np.isfinite(
            training_limit
        )
        or training_limit <= 0.0
    ):
        raise RuntimeError(
            "训练数据计算出的标准化尾部阈值无效。"
        )

    generated_standardized = (
        (
            generated_shapes
            - pointwise_shape_center[
                np.newaxis,
                :
            ]
        )
        / pointwise_shape_scale[
            np.newaxis,
            :
        ]
    )

    before_tail_mask = (
        np.abs(
            generated_standardized
        )
        > training_limit
    )

    before_tail_mask &= (
        non_peak_mask[
            np.newaxis,
            :
        ]
    )

    calibrated_standardized = (
        generated_standardized.copy()
    )

    calibrated_standardized[
        :,
        non_peak_mask
    ] = soft_clip_standardized(
        standardized=(
            generated_standardized[
                :,
                non_peak_mask
            ]
        ),
        limit=training_limit,
        margin=float(
            args.soft_margin
        ),
    )

    calibrated_shapes = (
        pointwise_shape_center[
            np.newaxis,
            :
        ]
        + calibrated_standardized
        * pointwise_shape_scale[
            np.newaxis,
            :
        ]
    )

    calibrated_broad = (
        reference_broad[
            np.newaxis,
            :
        ]
        + generated_offsets[
            :,
            np.newaxis
        ]
        + calibrated_shapes
    )

    # Preserve original local/fine component.
    generated_local = (
        generated_spectra
        - generated_broad
    )

    reconstructed = (
        calibrated_broad
        + generated_local
    )

    # Preserve peak windows.
    calibrated = (
        peak_weight[
            np.newaxis,
            :
        ]
        * generated_spectra
        + (
            1.0
            - peak_weight[
                np.newaxis,
                :
            ]
        )
        * reconstructed
    )

    if not np.isfinite(
        calibrated
    ).all():
        raise RuntimeError(
            "校准结果包含NaN或无穷值。"
        )

    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------
    generated_broad_after = (
        gaussian_filter1d(
            calibrated,
            sigma=broad_sigma,
            axis=1,
            mode="nearest",
        )
    )

    (
        generated_offsets_after,
        generated_shapes_after,
    ) = calculate_shape_components(
        broad=generated_broad_after,
        reference_broad=reference_broad,
        non_peak_mask=non_peak_mask,
    )

    generated_standardized_after = (
        (
            generated_shapes_after
            - pointwise_shape_center[
                np.newaxis,
                :
            ]
        )
        / pointwise_shape_scale[
            np.newaxis,
            :
        ]
    )

    after_tail_mask = (
        np.abs(
            generated_standardized_after
        )
        > training_limit
    )

    after_tail_mask &= (
        non_peak_mask[
            np.newaxis,
            :
        ]
    )

    changed_points_per_spectrum = np.sum(
        before_tail_mask,
        axis=1,
    )

    changed_spectra = np.flatnonzero(
        changed_points_per_spectrum > 0
    )

    real_abs_z = np.abs(
        real_standardized[
            :,
            non_peak_mask
        ]
    )

    generated_abs_z_before = np.abs(
        generated_standardized[
            :,
            non_peak_mask
        ]
    )

    generated_abs_z_after = np.abs(
        generated_standardized_after[
            :,
            non_peak_mask
        ]
    )

    # ------------------------------------------------------------
    # Export spectra efficiently to avoid fragmented DataFrame warning.
    # ------------------------------------------------------------
    columns = {
        "raman_shift": generated_axis
    }

    columns.update(
        {
            f"generated_{index + 1:04d}":
                calibrated[index]
            for index in range(
                calibrated.shape[0]
            )
        }
    )

    output_frame = pd.DataFrame(
        columns
    )

    output_frame.to_excel(
        output_path,
        index=False,
    )

    report_path = (
        output_path.parent
        / (
            output_path.stem
            + "_report.xlsx"
        )
    )

    report = pd.DataFrame(
        {
            "spectrum_index": np.arange(
                1,
                calibrated.shape[0] + 1,
            ),
            "changed_point_count":
                changed_points_per_spectrum,
            "changed_point_fraction":
                (
                    changed_points_per_spectrum
                    / float(
                        np.count_nonzero(
                            non_peak_mask
                        )
                    )
                ),
            "broad_offset_before":
                generated_offsets,
            "broad_offset_after":
                generated_offsets_after,
        }
    )

    report.to_excel(
        report_path,
        index=False,
    )

    figure_path = (
        output_path.parent
        / (
            output_path.stem
            + "_preview.png"
        )
    )

    plt.figure(
        figsize=(14, 6)
    )

    for spectrum in calibrated:
        plt.plot(
            generated_axis,
            spectrum,
            linewidth=0.45,
            alpha=0.45,
        )

    plt.xlabel(
        "Raman shift (cm$^{-1}$)"
    )

    plt.ylabel(
        "Intensity"
    )

    plt.title(
        "Pointwise robust broad-tail calibrated spectra"
    )

    plt.tight_layout()

    plt.savefig(
        figure_path,
        dpi=200,
    )

    plt.close()

    print(
        "\n===== Pointwise broad-tail校准完成 ====="
    )

    print(
        f"真实训练谱数量："
        f"{training_spectra.shape[0]}"
    )

    print(
        f"生成谱数量："
        f"{generated_spectra.shape[0]}"
    )

    print(
        "自动检测峰位："
        + ", ".join(
            f"{generated_axis[index]:.1f}"
            for index in peak_indices
        )
        + " cm⁻¹"
    )

    print(
        f"pointwise MAD scale floor："
        f"{scale_floor:.6g}"
    )

    print(
        "训练 pooled |z| "
        f"P{args.training_tail_quantile:g}："
        f"{training_limit:.6g}"
    )

    print(
        f"soft margin："
        f"{args.soft_margin:.6g}"
    )

    print(
        f"发生局部broad-tail修正的光谱数量："
        f"{changed_spectra.size}"
    )

    print(
        "修改前超过训练阈值的非峰区点数："
        f"{int(np.count_nonzero(before_tail_mask))}"
    )

    print(
        "重构后仍超过训练阈值的非峰区点数："
        f"{int(np.count_nonzero(after_tail_mask))}"
    )

    print(
        "真实 pooled |z| P95："
        f"{np.percentile(real_abs_z, 95):.6g}"
    )

    print(
        "生成 pooled |z| P95："
        f"{np.percentile(generated_abs_z_before, 95):.6g}"
        " → "
        f"{np.percentile(generated_abs_z_after, 95):.6g}"
    )

    print(
        "生成 pooled |z| P99："
        f"{np.percentile(generated_abs_z_before, 99):.6g}"
        " → "
        f"{np.percentile(generated_abs_z_after, 99):.6g}"
    )

    print(
        "生成 pooled |z| max："
        f"{np.max(generated_abs_z_before):.6g}"
        " → "
        f"{np.max(generated_abs_z_after):.6g}"
    )

    print(
        "Broad offset median："
        f"{np.median(generated_offsets):.6g}"
        " → "
        f"{np.median(generated_offsets_after):.6g}"
    )

    print(
        f"输出光谱：{output_path}"
    )

    print(
        f"报告：{report_path}"
    )

    print(
        f"预览图：{figure_path}"
    )


if __name__ == "__main__":
    main()
