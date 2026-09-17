"""Training-reference-driven soft calibration of broad SERS background tails.

Only abnormally large broad/background SHAPE deviations are compressed.

Preserved:
- broad vertical offset;
- local/fine spectral fluctuation;
- automatically detected peak regions;
- Raman peak positions;
- normal broad variation below the training-derived threshold.

This is a generation-side engineering calibration, not model training.
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
            "使用真实训练集统计量，对生成SERS中少量异常"
            "broad/background shape尾部进行软校准。"
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
        help="校准后的xlsx输出路径。",
    )

    parser.add_argument(
        "--limit-quantile",
        type=float,
        default=95.0,
        help="真实训练集broad-shape RMS软限制起始分位数。",
    )

    parser.add_argument(
        "--soft-margin-fraction",
        type=float,
        default=0.20,
        help=(
            "软尾部额外允许范围相对于训练阈值的比例；"
            "0.20表示极端情况下仍允许约threshold×1.20。"
        ),
    )

    return parser.parse_args()


def build_peak_protection_weight(
    *,
    axis: np.ndarray,
    peak_indices: np.ndarray,
    protection_half_width_cm1: float,
    transition_width_cm1: float,
) -> np.ndarray:
    """
    Build smooth peak protection weights.

    1.0: keep original spectrum completely.
    0.0: broad-tail calibration may operate.
    """

    weight = np.zeros(
        axis.size,
        dtype=np.float64,
    )

    outer_width = (
        float(protection_half_width_cm1)
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
            distance
            <= protection_half_width_cm1
        ] = 1.0

        transition = (
            (distance > protection_half_width_cm1)
            & (distance < outer_width)
        )

        if np.any(transition):
            phase = (
                distance[transition]
                - protection_half_width_cm1
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


def calculate_shape_scores(
    *,
    broad_spectra: np.ndarray,
    reference_broad: np.ndarray,
    non_peak_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return:
    - spectrum-wide broad offset;
    - broad shape RMS after removing that offset.
    """

    deviation = (
        broad_spectra[:, non_peak_mask]
        - reference_broad[
            np.newaxis,
            non_peak_mask,
        ]
    )

    offsets = np.mean(
        deviation,
        axis=1,
    )

    shape_deviation = (
        deviation
        - offsets[:, np.newaxis]
    )

    shape_rms = np.sqrt(
        np.mean(
            shape_deviation ** 2,
            axis=1,
        )
    )

    return offsets, shape_rms


def soft_tail_target(
    *,
    value: float,
    threshold: float,
    margin: float,
) -> float:
    """
    Smoothly compress only values above threshold.

    Below threshold:
        target = value

    Above threshold:
        target approaches threshold + margin smoothly.
    """

    if value <= threshold:
        return float(value)

    excess = (
        float(value)
        - float(threshold)
    )

    return float(
        threshold
        + margin
        * np.tanh(
            excess / margin
        )
    )


def main() -> None:
    args = parse_arguments()

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

    if not (
        0.0
        < args.limit_quantile
        < 100.0
    ):
        raise ValueError(
            "--limit-quantile必须位于(0,100)。"
        )

    if (
        not np.isfinite(
            args.soft_margin_fraction
        )
        or args.soft_margin_fraction <= 0.0
    ):
        raise ValueError(
            "--soft-margin-fraction必须为正数。"
        )

    configuration = (
        load_configuration(
            args.config
        )
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

    collection = (
        read_spectrum_collection(
            input_directory=input_directory,
            data_config=data_config,
        )
    )

    split = split_spectrum_collection(
        collection=collection,
        data_config=data_config,
        random_seed=random_seed,
    )

    training_indices = np.asarray(
        split.train.indices,
        dtype=np.int64,
    )

    length_adapter = (
        SpectrumLengthAdapter
        .from_metadata(
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

    generation_config = (
        configuration.get(
            "generation",
            {},
        )
    )

    sampling_config = (
        generation_config.get(
            "sampling_calibration",
            {},
        )
        or {}
    )

    non_peak_config = (
        sampling_config.get(
            "non_peak_noise",
            {},
        )
        or {}
    )

    reference_sigma_cm1 = float(
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
        peak_mask,
        _,
    ) = detect_peak_regions(
        axis=generated_axis,
        training_spectra=training_spectra,
        reference_smoothing_sigma_cm1=(
            reference_sigma_cm1
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

    protection_weight = (
        build_peak_protection_weight(
            axis=generated_axis,
            peak_indices=peak_indices,
            protection_half_width_cm1=(
                protection_half_width_cm1
            ),
            transition_width_cm1=(
                transition_width_cm1
            ),
        )
    )

    non_peak_mask = (
        protection_weight
        <= 0.05
    )

    if np.count_nonzero(
        non_peak_mask
    ) < 10:
        raise RuntimeError(
            "自动保护峰区后非峰区不足。"
        )

    broad_sigma_samples = (
        sigma_samples(
            broad_sigma_cm1,
            generated_axis,
        )
    )

    real_broad = gaussian_filter1d(
        training_spectra,
        sigma=broad_sigma_samples,
        axis=1,
        mode="nearest",
    )

    generated_broad = (
        gaussian_filter1d(
            generated_spectra,
            sigma=broad_sigma_samples,
            axis=1,
            mode="nearest",
        )
    )

    # 使用真实训练集 broad 中位数作为中心。
    reference_broad = np.median(
        real_broad,
        axis=0,
    )

    (
        real_offsets,
        real_shape_rms,
    ) = calculate_shape_scores(
        broad_spectra=real_broad,
        reference_broad=reference_broad,
        non_peak_mask=non_peak_mask,
    )

    (
        generated_offsets,
        generated_shape_rms_before,
    ) = calculate_shape_scores(
        broad_spectra=generated_broad,
        reference_broad=reference_broad,
        non_peak_mask=non_peak_mask,
    )

    threshold = float(
        np.percentile(
            real_shape_rms,
            args.limit_quantile,
        )
    )

    margin = float(
        threshold
        * args.soft_margin_fraction
    )

    if threshold <= 0.0:
        raise RuntimeError(
            "真实训练集broad-shape阈值无效。"
        )

    calibrated = (
        generated_spectra.copy()
    )

    scales = np.ones(
        generated_spectra.shape[0],
        dtype=np.float64,
    )

    target_scores = (
        generated_shape_rms_before.copy()
    )

    changed_indices = []

    peak_weight_2d = (
        protection_weight[
            np.newaxis,
            :
        ]
    )

    for index in range(
        generated_spectra.shape[0]
    ):
        current_score = float(
            generated_shape_rms_before[
                index
            ]
        )

        target_score = soft_tail_target(
            value=current_score,
            threshold=threshold,
            margin=margin,
        )

        target_scores[index] = (
            target_score
        )

        if (
            current_score
            <= threshold
            or current_score
            <= 1.0e-12
        ):
            continue

        scale = (
            target_score
            / current_score
        )

        scales[index] = scale

        # broad deviation on complete Raman axis
        full_deviation = (
            generated_broad[index]
            - reference_broad
        )

        # Preserve the spectrum-wide vertical broad offset.
        offset = float(
            generated_offsets[
                index
            ]
        )

        full_shape = (
            full_deviation
            - offset
        )

        calibrated_broad = (
            reference_broad
            + offset
            + scale
            * full_shape
        )

        # Preserve all local/fine fluctuation exactly.
        local_component = (
            generated_spectra[index]
            - generated_broad[index]
        )

        reconstructed = (
            calibrated_broad
            + local_component
        )

        # Preserve automatically detected peak windows.
        calibrated[index] = (
            protection_weight
            * generated_spectra[index]
            + (
                1.0
                - protection_weight
            )
            * reconstructed
        )

        changed_indices.append(
            index
        )

    calibrated_broad_after = (
        gaussian_filter1d(
            calibrated,
            sigma=broad_sigma_samples,
            axis=1,
            mode="nearest",
        )
    )

    (
        _,
        generated_shape_rms_after,
    ) = calculate_shape_scores(
        broad_spectra=(
            calibrated_broad_after
        ),
        reference_broad=reference_broad,
        non_peak_mask=non_peak_mask,
    )

    output_frame = pd.DataFrame(
        {
            "raman_shift": generated_axis
        }
    )

    for index in range(
        calibrated.shape[0]
    ):
        output_frame[
            f"generated_{index + 1:04d}"
        ] = calibrated[index]

    output_frame.to_excel(
        output_path,
        index=False,
    )

    report_path = (
        output_path.parent
        / (
            output_path.stem
            + "_tail_report.xlsx"
        )
    )

    report = pd.DataFrame(
        {
            "spectrum_index": np.arange(
                1,
                calibrated.shape[0] + 1,
            ),
            "broad_shape_rms_before":
                generated_shape_rms_before,
            "broad_shape_rms_soft_target":
                target_scores,
            "broad_shape_scale":
                scales,
            "broad_shape_rms_after":
                generated_shape_rms_after,
            "modified":
                scales < 0.999999,
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
        "Broad-shape soft-tail calibrated spectra"
    )
    plt.tight_layout()

    plt.savefig(
        figure_path,
        dpi=200,
    )
    plt.close()

    print(
        "\n===== Broad shape软尾部校准完成 ====="
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
        f"训练broad-shape RMS "
        f"P{args.limit_quantile:g}："
        f"{threshold:.6g}"
    )

    print(
        f"soft margin："
        f"{margin:.6g}"
    )

    print(
        f"被修改光谱数量："
        f"{len(changed_indices)}"
    )

    print(
        "被修改光谱索引："
        + (
            ", ".join(
                str(index + 1)
                for index in changed_indices
            )
            if changed_indices
            else "无"
        )
    )

    print(
        "生成 broad-shape RMS P95："
        f"{np.percentile(generated_shape_rms_before, 95):.6g}"
        " → "
        f"{np.percentile(generated_shape_rms_after, 95):.6g}"
    )

    print(
        "生成 broad-shape RMS max："
        f"{np.max(generated_shape_rms_before):.6g}"
        " → "
        f"{np.max(generated_shape_rms_after):.6g}"
    )

    print(
        f"输出光谱：{output_path}"
    )

    print(
        f"逐谱报告：{report_path}"
    )

    print(
        f"预览图：{figure_path}"
    )


if __name__ == "__main__":
    main()
