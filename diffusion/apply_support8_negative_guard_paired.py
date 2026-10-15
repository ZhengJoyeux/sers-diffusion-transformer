#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


SUMMARY_PATH = Path(
    "outputs/diagnostics/"
    "automatic_peak_free_background_support8/"
    "condition_background_summary.csv"
)

GENERATED_ROOT = Path(
    "outputs/experiments/"
    "d4_3_2_15_warmup_cosine_2h_formal/"
    "generated_raw_diagnostic"
)

CONFIG_PATH = Path(
    "config/"
    "ddpm_training_d4_3_2_15_raw_generation.yaml"
)

OUTPUT_ROOT = Path(
    "outputs/experiments/"
    "d4_3_2_15_warmup_cosine_2h_formal/"
    "generated_support8_guard_paired"
)

SUMMARY_OUTPUT = (
    OUTPUT_ROOT
    / "support8_guard_paired_summary.csv"
)


def read_source_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    if suffix == ".csv":
        return pd.read_csv(path)

    raise RuntimeError(
        f"不支持文件格式：{path}"
    )


def retain_contiguous_runs(
    candidate: np.ndarray,
    minimum_length: int,
) -> np.ndarray:
    retained = np.zeros_like(
        candidate,
        dtype=bool,
    )

    for spectrum_index in range(
        candidate.shape[0]
    ):
        row = candidate[spectrum_index]

        padded = np.pad(
            row.astype(np.int8),
            (1, 1),
            constant_values=0,
        )

        changes = np.diff(padded)

        starts = np.flatnonzero(
            changes == 1
        )

        stops = np.flatnonzero(
            changes == -1
        )

        for start, stop in zip(
            starts,
            stops,
        ):
            if (
                int(stop - start)
                >= minimum_length
            ):
                retained[
                    spectrum_index,
                    start:stop,
                ] = True

    return retained


def count_duplicate_spectra(
    spectra: np.ndarray,
) -> int:
    rounded = np.round(
        spectra,
        decimals=8,
    )

    unique_count = np.unique(
        rounded,
        axis=0,
    ).shape[0]

    return int(
        spectra.shape[0]
        - unique_count
    )


def main() -> None:
    if not SUMMARY_PATH.is_file():
        raise FileNotFoundError(
            SUMMARY_PATH
        )

    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(
            CONFIG_PATH
        )

    if not GENERATED_ROOT.is_dir():
        raise FileNotFoundError(
            GENERATED_ROOT
        )

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    guard = (
        config
        .get("generation", {})
        .get(
            "intensity_envelope_guard",
            {},
        )
        or {}
    )

    training_lower_quantile = float(
        guard.get(
            "training_lower_quantile",
            0.05,
        )
    )

    training_iqr_margin = float(
        guard.get(
            "training_iqr_margin",
            0.50,
        )
    )

    extrema_margin_fraction = float(
        guard.get(
            "training_extrema_margin_iqr_fraction",
            0.10,
        )
    )

    minimum_extrema_margin = float(
        guard.get(
            "minimum_extrema_margin_intensity",
            2.0,
        )
    )

    generated_lower_quantile = float(
        guard.get(
            "generated_lower_quantile",
            0.025,
        )
    )

    local_support_quantile = float(
        guard.get(
            "local_negative_support_quantile",
            0.10,
        )
    )

    activation_maximum = float(
        guard.get(
            "activation_maximum_intensity",
            -18.0,
        )
    )

    minimum_deficit = float(
        guard.get(
            "minimum_deficit_intensity",
            5.0,
        )
    )

    minimum_contiguous_points = int(
        guard.get(
            "minimum_negative_valley_contiguous_points",
            3,
        )
    )

    softness_fraction = float(
        guard.get(
            "softness_iqr_fraction",
            0.05,
        )
    )

    minimum_softness = float(
        guard.get(
            "minimum_softness_intensity",
            0.25,
        )
    )

    maximum_softness = float(
        guard.get(
            "maximum_softness_intensity",
            2.0,
        )
    )

    maximum_modified_fraction = float(
        guard.get(
            "maximum_modified_fraction",
            0.05,
        )
    )

    enforce_global_extrema = bool(
        guard.get(
            "enforce_training_global_extrema",
            True,
        )
    )

    print(
        "===== D4.3.2.15 support=8 paired guard ====="
    )

    print(
        "training_lower_quantile =",
        training_lower_quantile,
    )
    print(
        "generated_lower_quantile =",
        generated_lower_quantile,
    )
    print(
        "local_support_quantile =",
        local_support_quantile,
    )
    print(
        "activation_maximum =",
        activation_maximum,
    )
    print(
        "minimum_deficit =",
        minimum_deficit,
    )
    print(
        "minimum_contiguous_points =",
        minimum_contiguous_points,
    )
    print(
        "softness range =",
        (
            minimum_softness,
            maximum_softness,
        ),
    )
    print(
        "maximum_modified_fraction =",
        maximum_modified_fraction,
    )

    summary = pd.read_csv(
        SUMMARY_PATH
    )

    summary_by_condition = {
        str(row["condition"]): row
        for _, row in summary.iterrows()
    }

    files = sorted(
        GENERATED_ROOT.glob(
            "*/*_generated.xlsx"
        )
    )

    if len(files) != 126:
        raise RuntimeError(
            "RAW生成文件数量不是126："
            f"{len(files)}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    total_spectra = 0
    total_points = 0
    total_modified_spectra = 0
    total_modified_points = 0

    for file_index, generated_path in enumerate(
        files,
        start=1,
    ):
        condition = (
            generated_path.parent.name
        )

        if condition not in summary_by_condition:
            raise RuntimeError(
                "background summary缺少："
                f"{condition}"
            )

        row = summary_by_condition[
            condition
        ]

        source_path = Path(
            str(row["source_file"])
        )

        source_df = read_source_file(
            source_path
        )

        source_axis = (
            pd.to_numeric(
                source_df.iloc[:, 0],
                errors="coerce",
            )
            .to_numpy(dtype=float)
        )

        training = (
            source_df.iloc[:, 1:13]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .to_numpy(dtype=float)
            .T
        )

        generated_df = pd.read_excel(
            generated_path
        )

        generated_axis = (
            pd.to_numeric(
                generated_df.iloc[:, 0],
                errors="coerce",
            )
            .to_numpy(dtype=float)
        )

        generated = (
            generated_df.iloc[:, 1:]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .to_numpy(dtype=float)
            .T
        )

        if training.shape[0] != 12:
            raise RuntimeError(
                f"{condition} training数量不是12。"
            )

        if (
            source_axis.shape
            != generated_axis.shape
            or not np.allclose(
                source_axis,
                generated_axis,
                rtol=0.0,
                atol=1.0e-6,
            )
        ):
            raise RuntimeError(
                f"{condition} Raman轴不一致。"
            )

        if (
            not np.isfinite(
                training
            ).all()
            or not np.isfinite(
                generated
            ).all()
        ):
            raise RuntimeError(
                f"{condition}存在NaN/Inf。"
            )

        # ----------------------------------------------------
        # Pointwise training floor
        # ----------------------------------------------------

        train_q = np.quantile(
            training,
            training_lower_quantile,
            axis=0,
        )

        train_q25 = np.quantile(
            training,
            0.25,
            axis=0,
        )

        train_q75 = np.quantile(
            training,
            0.75,
            axis=0,
        )

        train_iqr = np.maximum(
            train_q75 - train_q25,
            0.0,
        )

        extrema_margin = np.maximum(
            extrema_margin_fraction
            * train_iqr,
            minimum_extrema_margin,
        )

        training_floor = np.minimum(
            train_q
            - training_iqr_margin
            * train_iqr,
            np.min(
                training,
                axis=0,
            )
            - extrema_margin,
        )

        # ----------------------------------------------------
        # support=8 condition-level floor
        # ----------------------------------------------------

        recommended_floor = row[
            "recommended_negative_floor"
        ]

        has_background = not pd.isna(
            recommended_floor
        )

        if has_background:
            condition_floor = min(
                float(
                    recommended_floor
                ),
                activation_maximum,
            )

            local_support = (
                np.quantile(
                    training,
                    local_support_quantile,
                    axis=0,
                )
                <= condition_floor
            )

            allowed_floor = np.where(
                local_support,
                training_floor,
                np.maximum(
                    training_floor,
                    condition_floor,
                ),
            )
        else:
            condition_floor = np.nan
            local_support = np.ones(
                training.shape[1],
                dtype=bool,
            )
            allowed_floor = (
                training_floor.copy()
            )

        # Ordinary small negative noise remains allowed.
        allowed_floor = np.minimum(
            allowed_floor,
            activation_maximum,
        )

        training_global_minimum = float(
            np.min(training)
        )

        if enforce_global_extrema:
            allowed_floor = np.maximum(
                allowed_floor,
                training_global_minimum,
            )

        # ----------------------------------------------------
        # Generated lower-rank gate
        # ----------------------------------------------------

        generated_floor = np.quantile(
            generated,
            generated_lower_quantile,
            axis=0,
        )

        lower_rank_gate = (
            generated
            <= generated_floor[
                np.newaxis,
                :
            ]
        )

        deficit = (
            allowed_floor[
                np.newaxis,
                :
            ]
            - generated
        )

        pointwise_candidate = (
            lower_rank_gate
            & (
                generated
                < allowed_floor[
                    np.newaxis,
                    :
                ]
            )
            & (
                generated
                < activation_maximum
            )
            & (
                deficit
                > minimum_deficit
            )
        )

        final_candidate = (
            retain_contiguous_runs(
                pointwise_candidate,
                minimum_contiguous_points,
            )
        )

        modified_point_count = int(
            np.sum(
                final_candidate
            )
        )

        modified_fraction = (
            modified_point_count
            / float(
                generated.size
            )
        )

        if (
            modified_fraction
            > maximum_modified_fraction
        ):
            raise RuntimeError(
                f"{condition}修正比例"
                f"{modified_fraction:.6%}"
                "超过安全上限"
                f"{maximum_modified_fraction:.6%}。"
            )

        # ----------------------------------------------------
        # Same soft tanh pull-back as production guard.
        # ----------------------------------------------------

        corrected = generated.copy()

        if modified_point_count > 0:
            softness = np.clip(
                softness_fraction
                * train_iqr,
                minimum_softness,
                maximum_softness,
            )

            soft_value = (
                allowed_floor[
                    np.newaxis,
                    :
                ]
                - softness[
                    np.newaxis,
                    :
                ]
                * np.tanh(
                    np.maximum(
                        deficit,
                        0.0,
                    )
                    / softness[
                        np.newaxis,
                        :
                    ]
                )
            )

            corrected[
                final_candidate
            ] = soft_value[
                final_candidate
            ]

            if enforce_global_extrema:
                corrected[
                    final_candidate
                ] = np.maximum(
                    corrected[
                        final_candidate
                    ],
                    training_global_minimum,
                )

        difference = (
            corrected
            - generated
        )

        modified_spectrum_count = int(
            np.sum(
                np.any(
                    np.abs(
                        difference
                    )
                    > 1.0e-12,
                    axis=1,
                )
            )
        )

        maximum_absolute_correction = (
            float(
                np.max(
                    np.abs(
                        difference
                    )
                )
            )
            if difference.size
            else 0.0
        )

        # ----------------------------------------------------
        # Export an exact paired copy.
        # ----------------------------------------------------

        output_directory = (
            OUTPUT_ROOT
            / condition
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            output_directory
            / generated_path.name
        )

        corrected_df = (
            generated_df.copy()
        )

        corrected_df.iloc[
            :,
            1:,
        ] = corrected.T

        corrected_df.to_excel(
            output_path,
            index=False,
        )

        before_minimum = float(
            np.min(
                generated
            )
        )

        after_minimum = float(
            np.min(
                corrected
            )
        )

        before_spectrum_minimum = (
            np.min(
                generated,
                axis=1,
            )
        )

        after_spectrum_minimum = (
            np.min(
                corrected,
                axis=1,
            )
        )

        total_spectra += int(
            generated.shape[0]
        )

        total_points += int(
            generated.size
        )

        total_modified_spectra += (
            modified_spectrum_count
        )

        total_modified_points += (
            modified_point_count
        )

        records.append({
            "condition":
                condition,
            "background_available":
                bool(
                    has_background
                ),
            "background_fraction":
                float(
                    row[
                        "background_fraction"
                    ]
                ),
            "condition_floor":
                (
                    float(
                        condition_floor
                    )
                    if has_background
                    else np.nan
                ),
            "training_global_minimum":
                training_global_minimum,
            "before_global_minimum":
                before_minimum,
            "after_global_minimum":
                after_minimum,
            "modified_spectrum_count":
                modified_spectrum_count,
            "modified_point_count":
                modified_point_count,
            "modified_point_fraction":
                modified_fraction,
            "maximum_absolute_correction":
                maximum_absolute_correction,
            "before_min_lt_-20":
                int(
                    np.sum(
                        before_spectrum_minimum
                        < -20.0
                    )
                ),
            "after_min_lt_-20":
                int(
                    np.sum(
                        after_spectrum_minimum
                        < -20.0
                    )
                ),
            "before_min_lt_-40":
                int(
                    np.sum(
                        before_spectrum_minimum
                        < -40.0
                    )
                ),
            "after_min_lt_-40":
                int(
                    np.sum(
                        after_spectrum_minimum
                        < -40.0
                    )
                ),
            "before_min_lt_-60":
                int(
                    np.sum(
                        before_spectrum_minimum
                        < -60.0
                    )
                ),
            "after_min_lt_-60":
                int(
                    np.sum(
                        after_spectrum_minimum
                        < -60.0
                    )
                ),
            "before_min_lt_-100":
                int(
                    np.sum(
                        before_spectrum_minimum
                        < -100.0
                    )
                ),
            "after_min_lt_-100":
                int(
                    np.sum(
                        after_spectrum_minimum
                        < -100.0
                    )
                ),
            "before_duplicate_count":
                count_duplicate_spectra(
                    generated
                ),
            "after_duplicate_count":
                count_duplicate_spectra(
                    corrected
                ),
        })

        if (
            file_index <= 5
            or file_index % 20 == 0
            or file_index == len(files)
        ):
            print(
                f"[{file_index:3d}/"
                f"{len(files)}] "
                f"{condition}: "
                f"modified spectra="
                f"{modified_spectrum_count}/"
                f"{generated.shape[0]}, "
                f"points="
                f"{modified_point_count}, "
                f"min "
                f"{before_minimum:.3f}"
                f" -> "
                f"{after_minimum:.3f}"
            )

    result = pd.DataFrame(
        records
    )

    result.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print()
    print(
        "===== PAIRED GUARD SUMMARY ====="
    )

    print(
        "condition数 =",
        len(result),
    )

    print(
        "总光谱数 =",
        total_spectra,
    )

    print(
        "实际修改光谱数 =",
        total_modified_spectra,
        "/",
        total_spectra,
        f"({total_modified_spectra / total_spectra:.2%})",
    )

    print(
        "实际修改点数 =",
        total_modified_points,
        "/",
        total_points,
        f"({total_modified_points / total_points:.4%})",
    )

    print()
    print(
        "===== condition层面是否仍存在极端最低值 ====="
    )

    for threshold in (
        -20.0,
        -40.0,
        -60.0,
        -100.0,
    ):
        before_column = (
            f"before_min_lt_{int(threshold)}"
            .replace(
                "--",
                "-",
            )
        )
        after_column = (
            f"after_min_lt_{int(threshold)}"
            .replace(
                "--",
                "-",
            )
        )

        # Column names above are clearer to access explicitly.
        mapping = {
            -20.0: (
                "before_min_lt_-20",
                "after_min_lt_-20",
            ),
            -40.0: (
                "before_min_lt_-40",
                "after_min_lt_-40",
            ),
            -60.0: (
                "before_min_lt_-60",
                "after_min_lt_-60",
            ),
            -100.0: (
                "before_min_lt_-100",
                "after_min_lt_-100",
            ),
        }

        before_column, after_column = (
            mapping[threshold]
        )

        print(
            f"min < {threshold:g}:",
            int(
                (
                    result[
                        before_column
                    ]
                    > 0
                ).sum()
            ),
            "->",
            int(
                (
                    result[
                        after_column
                    ]
                    > 0
                ).sum()
            ),
            "conditions",
        )

    print()
    print(
        "===== spectrum层面最低值计数 ====="
    )

    for threshold in (
        -20,
        -40,
        -60,
        -100,
    ):
        before_column = (
            f"before_min_lt_{threshold}"
        )
        after_column = (
            f"after_min_lt_{threshold}"
        )

        print(
            f"min < {threshold}:",
            int(
                result[
                    before_column
                ].sum()
            ),
            "->",
            int(
                result[
                    after_column
                ].sum()
            ),
            "spectra",
        )

    print()
    print(
        "===== 最严重RAW condition前20 ====="
    )

    print(
        result.sort_values(
            "before_global_minimum"
        )
        .head(20)[
            [
                "condition",
                "condition_floor",
                "training_global_minimum",
                "before_global_minimum",
                "after_global_minimum",
                "modified_spectrum_count",
                "modified_point_count",
                "maximum_absolute_correction",
            ]
        ]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(
        "RAW重复光谱总数 =",
        int(
            result[
                "before_duplicate_count"
            ].sum()
        ),
    )

    print(
        "guard后重复光谱总数 =",
        int(
            result[
                "after_duplicate_count"
            ].sum()
        ),
    )

    print()
    print(
        "输出目录：",
        OUTPUT_ROOT,
    )

    print(
        "汇总CSV：",
        SUMMARY_OUTPUT,
    )


if __name__ == "__main__":
    main()
