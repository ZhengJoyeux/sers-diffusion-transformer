"""
SERSFormer2 T0 dataset adapter for the DEL/CHL/TEB SERS project.

Engineering adaptations only (not paper-method innovations):
- read real spectra from data/real (symlink to ~/project/data/input)
- read D4.24 generated spectra from data/generated
- fixed 12/4/4 real split per source file
- generated spectra are training-only
- use a uniform 600-2500 cm^-1 model axis (1901 points)
- complete missing high-wavenumber tails in memory with reproducible background noise
- preserve negative baseline-corrected SERS values with signed-log1p preprocessing

The output order of pesticide targets is always: [DEL, CHL, TEB].
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd
import scipy.signal as signal
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parent
REAL_DATA_ROOT = PROJECT_ROOT / "data" / "real"
GENERATED_DATA_ROOT = PROJECT_ROOT / "data" / "generated"

PESTICIDES = ("DEL", "CHL", "TEB")
PESTICIDE_TO_INDEX = {name: index for index, name in enumerate(PESTICIDES)}
LEVEL_TO_CODE = {"S": 1.0, "M": 2.0, "H": 3.0}

MODEL_RAMAN_AXIS = np.arange(600.0, 2500.0 + 1.0, 1.0, dtype=np.float32)
MODEL_LENGTH = int(MODEL_RAMAN_AXIS.size)
if MODEL_LENGTH != 1901:
    raise RuntimeError(f"Unexpected full-axis Raman length: {MODEL_LENGTH}")

FULL_AXIS_COVERAGE_TOLERANCE_CM1 = 1.0
DEFAULT_CORE_PEAK_HALF_WIDTH_CM1 = 7.0
DEFAULT_AUXILIARY_PRIOR_MAX_WEIGHT = 0.35
DEFAULT_SHARED_CORE_PEAK_WEIGHT = 0.65

# User-specified chemistry anchors.
# +/-7 cm^-1 is the current initial peak-shift tolerance.
CHEMISTRY_CORE_PEAKS: dict[str, tuple[dict[str, float | str], ...]] = {
    "DEL": (
        {"center_cm-1": 1000.0, "role": "strong"},
        {"center_cm-1": 1600.0, "role": "shared"},
    ),
    "CHL": (
        {"center_cm-1": 2230.0, "role": "strong"},
    ),
    "TEB": (
        {"center_cm-1": 1090.0, "role": "strong"},
        {"center_cm-1": 1597.0, "role": "shared"},
    ),
}

REAL_TRAIN_INDICES = tuple(range(0, 12))
REAL_VALIDATION_INDICES = tuple(range(12, 16))
REAL_TEST_INDICES = tuple(range(16, 20))


@dataclass(frozen=True)
class ConditionInfo:
    condition: str
    presence: np.ndarray
    level_codes: np.ndarray
    concentration_target: np.ndarray
    matrix_name: str
    matrix_code: int


@dataclass(frozen=True)
class SampleRef:
    source: str
    condition: str
    spectrum_index: int


def load_concentration_map(path: str | Path | None) -> dict[str, dict[str, float]] | None:
    """
    Optional physical-concentration map.

    Expected JSON structure:
    {
        "DEL": {"S": 1.0, "M": 10.0, "H": 100.0},
        "CHL": {"S": ..., "M": ..., "H": ...},
        "TEB": {"S": ..., "M": ..., "H": ...}
    }

    If path is None, concentration_target uses level codes 0/1/2/3 only.
    Those level codes must NOT be reported as physical concentration results.
    """
    if path is None:
        return None

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    result: dict[str, dict[str, float]] = {}
    for pesticide in PESTICIDES:
        if pesticide not in raw:
            raise ValueError(f"Concentration map missing pesticide: {pesticide}")
        result[pesticide] = {}
        for level in ("S", "M", "H"):
            if level not in raw[pesticide]:
                raise ValueError(f"Concentration map missing {pesticide}-{level}")
            result[pesticide][level] = float(raw[pesticide][level])
    return result


def _collect_table_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {root}")

    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".xlsx", ".xls", ".csv"}
    )


def _read_table(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    elif suffix == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported table format: {path}")

    if frame.shape[1] < 2:
        raise ValueError(f"No spectrum columns found: {path}")

    axis = pd.to_numeric(frame.iloc[:, 0], errors="coerce").to_numpy(dtype=np.float64)
    spectrum_frame = frame.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    spectra = spectrum_frame.to_numpy(dtype=np.float32)

    valid_rows = np.isfinite(axis) & np.all(np.isfinite(spectra), axis=1)
    axis = axis[valid_rows]
    spectra = spectra[valid_rows, :]

    if axis.size < 2:
        raise ValueError(f"Too few valid Raman points: {path}")
    if not np.all(np.diff(axis) > 0.0):
        raise ValueError(f"Raman axis is not strictly increasing: {path}")

    names = [str(name) for name in spectrum_frame.columns]
    return axis, spectra, names


def _condition_from_real_path(path: Path) -> str:
    return path.stem


def _condition_from_generated_path(path: Path) -> str:
    name = path.stem
    suffix = "_generated"
    return name[: -len(suffix)] if name.endswith(suffix) else name


def _parse_condition(
    condition: str,
    concentration_map: Mapping[str, Mapping[str, float]] | None,
) -> ConditionInfo:
    if condition.endswith("_water"):
        matrix_name = "water"
        matrix_code = 0
    elif condition.endswith("_soil"):
        matrix_name = "soil"
        matrix_code = 1
    else:
        raise ValueError(f"Condition must end with _water or _soil: {condition}")

    presence = np.zeros(len(PESTICIDES), dtype=np.float32)
    level_codes = np.zeros(len(PESTICIDES), dtype=np.float32)
    concentration_target = np.zeros(len(PESTICIDES), dtype=np.float32)

    matches = re.findall(r"(DEL|CHL|TEB)-(S|M|H)", condition)
    if not matches:
        raise ValueError(f"No pesticide label found in condition: {condition}")

    seen: set[str] = set()
    for pesticide, level in matches:
        if pesticide in seen:
            raise ValueError(f"Duplicate pesticide in condition: {condition}")
        seen.add(pesticide)

        index = PESTICIDE_TO_INDEX[pesticide]
        presence[index] = 1.0
        level_codes[index] = LEVEL_TO_CODE[level]
        if concentration_map is None:
            concentration_target[index] = LEVEL_TO_CODE[level]
        else:
            concentration_target[index] = float(concentration_map[pesticide][level])

    return ConditionInfo(
        condition=condition,
        presence=presence,
        level_codes=level_codes,
        concentration_target=concentration_target,
        matrix_name=matrix_name,
        matrix_code=matrix_code,
    )


def _mask_for_source_axis(axis: np.ndarray) -> np.ndarray:
    lower = float(axis[0])
    upper = float(axis[-1])
    mask = (MODEL_RAMAN_AXIS >= lower) & (MODEL_RAMAN_AXIS <= upper)
    if int(mask.sum()) < 2:
        raise ValueError(
            f"Raman range {lower}-{upper} has insufficient overlap with model axis"
        )
    return mask.astype(np.bool_)


def _adapt_spectrum_to_model_axis(
    axis: np.ndarray,
    spectrum: np.ndarray,
    allowed_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    axis = np.asarray(axis, dtype=np.float64)
    spectrum = np.asarray(spectrum, dtype=np.float64)

    if axis.ndim != 1 or spectrum.ndim != 1:
        raise ValueError("axis and spectrum must be 1D")
    if axis.size != spectrum.size:
        raise ValueError("axis and spectrum lengths differ")

    # Direct single-spectrum callers use the same runtime completion as
    # repository loading. No source file is opened for writing.
    if _classify_raw_axis_coverage(axis) in {"short_2000", "short_tail"}:
        source_digest = hashlib.sha256(
            axis.tobytes() + spectrum.tobytes()
        ).hexdigest()
        axis, completed, _ = _impute_missing_tail_to_2500_with_noise(
            axis, spectrum[:, np.newaxis],
            source_key=f"single-spectrum|{source_digest}",
        )
        spectrum = completed[:, 0]

    valid_mask = _mask_for_source_axis(axis)
    if allowed_mask is not None:
        allowed_mask = np.asarray(allowed_mask, dtype=np.bool_)
        if allowed_mask.shape != valid_mask.shape:
            raise ValueError("allowed_mask shape mismatch")
        valid_mask &= allowed_mask

    adapted = np.zeros(MODEL_LENGTH, dtype=np.float32)
    adapted[valid_mask] = np.interp(
        MODEL_RAMAN_AXIS[valid_mask], axis, spectrum
    ).astype(np.float32)
    return adapted, valid_mask


def _signed_log1p(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.sign(values) * np.log1p(np.abs(values))


def _preprocess_spectrum(
    spectrum: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Produce the three SERSFormer2 input branches:
      raw        : [1, 1901], valid-region min-max after signed-log1p
      percentile : [4, 1901], 95/85/75/50 percentile feature channels
      smoothed   : [1, 1901], Hann-smoothed signed-log1p signal

    signed-log1p is used instead of the upstream "x>1 else 0.001" rule because
    these baseline-corrected SERS spectra legitimately contain negative values.
    """
    valid_mask = np.asarray(valid_mask, dtype=np.bool_)
    transformed = np.zeros_like(spectrum, dtype=np.float32)
    transformed[valid_mask] = _signed_log1p(spectrum[valid_mask])

    valid_values = transformed[valid_mask]
    raw = np.zeros_like(transformed, dtype=np.float32)
    if valid_values.size:
        minimum = float(valid_values.min())
        maximum = float(valid_values.max())
        if maximum > minimum:
            raw[valid_mask] = (valid_values - minimum) / (maximum - minimum)
        else:
            raw[valid_mask] = 0.0

    percentile_channels = np.zeros((4, MODEL_LENGTH), dtype=np.float32)
    for channel, percentile in enumerate((95.0, 85.0, 75.0, 50.0)):
        threshold = float(np.percentile(valid_values, percentile))
        keep = valid_mask & (transformed >= threshold)
        percentile_channels[channel, keep] = transformed[keep]

    smoothed = np.zeros_like(transformed, dtype=np.float32)
    valid_indices = np.flatnonzero(valid_mask)
    if valid_indices.size:
        values = transformed[valid_indices]
        if values.size >= 3:
            window_length = min(32, int(values.size))
            window = signal.windows.hann(window_length)
            window_sum = float(window.sum())
            if window_sum > 0.0:
                filtered = signal.convolve(
                    values, window, mode="same", method="direct"
                ) / window_sum
                smoothed[valid_indices] = filtered.astype(np.float32)
            else:
                smoothed[valid_indices] = values
        else:
            smoothed[valid_indices] = values

    return raw[np.newaxis, :], percentile_channels, smoothed[np.newaxis, :]


def _stable_noise_seed(
    source_key: str,
    spectrum_index: int,
) -> int:
    """Return a deterministic seed for one real mapping spectrum."""
    payload = (
        f"sersformer-noise-tail-v1|{source_key}|{int(spectrum_index)}"
    ).encode("utf-8")

    digest = hashlib.sha256(payload).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    )


def _estimate_background_noise_sigma(
    axis: np.ndarray,
    spectrum: np.ndarray,
) -> tuple[float, float]:
    """
    Estimate background level and noise scale from the measured spectrum.

    The noise scale is estimated from residuals after Savitzky-Golay smoothing.
    Large residuals and the manually specified chemistry-core regions are
    excluded so obvious Raman peaks do not inflate the synthetic-tail noise.
    """
    axis = np.asarray(axis, dtype=np.float64)
    spectrum = np.asarray(spectrum, dtype=np.float64)

    if spectrum.size < 31:
        raise ValueError(
            "At least 31 measured points are required for "
            "background-noise estimation."
        )

    # Smooth only to estimate local background/noise.
    window = min(51, int(spectrum.size))

    if window % 2 == 0:
        window -= 1

    window = max(window, 7)

    smooth = signal.savgol_filter(
        spectrum,
        window_length=window,
        polyorder=2,
        mode="interp",
    )

    residual = spectrum - smooth

    candidate = np.ones(
        spectrum.shape,
        dtype=np.bool_,
    )

    # Do not use known chemistry-core regions when estimating background noise.
    for pesticide in PESTICIDES:
        for item in CHEMISTRY_CORE_PEAKS[pesticide]:
            center = float(item["center_cm-1"])

            candidate &= ~(
                (axis >= center - 20.0)
                & (axis <= center + 20.0)
            )

    candidate_residual = residual[candidate]

    if candidate_residual.size < 20:
        candidate_residual = residual

    # Remove strong residual excursions: retain the quieter 60% of points.
    absolute_residual = np.abs(
        candidate_residual
        - np.median(candidate_residual)
    )

    cutoff = float(
        np.quantile(
            absolute_residual,
            0.60,
        )
    )

    quiet = candidate_residual[
        absolute_residual <= cutoff
    ]

    if quiet.size < 20:
        quiet = candidate_residual

    median_residual = float(
        np.median(quiet)
    )

    mad = float(
        np.median(
            np.abs(
                quiet - median_residual
            )
        )
    )

    sigma = 1.4826 * mad

    # Numerical fallback only.
    if not np.isfinite(sigma) or sigma <= 1.0e-8:
        sigma = float(
            np.std(quiet)
        )

    if not np.isfinite(sigma) or sigma <= 1.0e-8:
        sigma = 1.0e-6

    # Tail background level comes from the final measured region.
    # This avoids forcing the synthetic tail to zero.
    trailing_points = min(
        100,
        int(smooth.size),
    )

    background_level = float(
        np.median(
            smooth[-trailing_points:]
        )
    )

    return background_level, sigma


def _impute_missing_tail_to_2500_with_noise(
    axis: np.ndarray,
    spectra: np.ndarray,
    *,
    source_key: str,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Complete any valid spectrum starting near 600 and ending before 2500.

    Only model-axis positions beyond the actual measured endpoint are filled.
    Existing 600-2500 spectra are returned unchanged.

    Missing tail points are filled with deterministic correlated
    background noise whose scale is estimated independently from each
    measured spectrum.

    Returns:
        axis_out,
        spectra_out,
        tail_imputed
    """
    axis = np.asarray(
        axis,
        dtype=np.float64,
    )

    spectra = np.asarray(
        spectra,
        dtype=np.float64,
    )

    if spectra.ndim != 2:
        raise ValueError(
            f"Expected spectra matrix [points, spectra], got {spectra.shape}"
        )

    if (
        axis.ndim != 1 or axis.size < 2
        or spectra.shape[0] != axis.size or spectra.shape[1] < 1
        or not np.isfinite(axis).all() or not np.isfinite(spectra).all()
        or not np.all(np.diff(axis) > 0)
    ):
        raise ValueError("Expected a finite increasing axis and matching finite spectra")

    # --------------------------------------------------------
    # Case 1:
    # Already contains genuine 600-2500 data -> DO NOTHING.
    # --------------------------------------------------------
    if (
        abs(float(axis[0]) - 600.0) <= FULL_AXIS_COVERAGE_TOLERANCE_CM1
        and float(axis[-1]) >= 2500.0
    ):
        return (
            axis.astype(np.float32, copy=False),
            spectra.astype(np.float32, copy=False),
            False,
        )

    # --------------------------------------------------------
    # Case 2:
    # General missing high-wavenumber tail, based on the actual endpoint.
    # --------------------------------------------------------
    if not (
        abs(float(axis[0]) - 600.0) <= FULL_AXIS_COVERAGE_TOLERANCE_CM1
        and axis.size >= 31
        and 600.0 < float(axis[-1]) < 2500.0
    ):
        raise RuntimeError(
            "Runtime tail completion requires at least 31 finite measured "
            "points, an increasing axis starting near 600 cm^-1, "
            "and an endpoint below 2500 cm^-1. "
            f"Received axis {float(axis[0]):.1f}-"
            f"{float(axis[-1]):.1f} cm^-1 "
            f"({int(axis.size)} points)."
        )

    measured_axis = MODEL_RAMAN_AXIS[
        MODEL_RAMAN_AXIS <= float(axis[-1])
    ]

    tail_axis = MODEL_RAMAN_AXIS[
        MODEL_RAMAN_AXIS > float(axis[-1])
    ]

    number_of_spectra = int(
        spectra.shape[1]
    )

    output = np.empty(
        (
            MODEL_LENGTH,
            number_of_spectra,
        ),
        dtype=np.float32,
    )

    # Preserve/interpolate only the genuinely measured range.
    for spectrum_index in range(number_of_spectra):
        measured = np.interp(
            measured_axis.astype(np.float64),
            axis,
            spectra[:, spectrum_index],
        )

        output[
            : measured_axis.size,
            spectrum_index,
        ] = measured.astype(
            np.float32
        )

        background_level, sigma = (
            _estimate_background_noise_sigma(
                axis,
                spectra[:, spectrum_index],
            )
        )

        seed = _stable_noise_seed(
            source_key,
            spectrum_index,
        )

        rng = np.random.default_rng(
            seed
        )

        # Estimate short-range correlation from measured background residuals.
        window = min(
            51,
            int(axis.size),
        )

        if window % 2 == 0:
            window -= 1

        window = max(
            window,
            7,
        )

        smooth = signal.savgol_filter(
            spectra[:, spectrum_index].astype(
                np.float64
            ),
            window_length=window,
            polyorder=2,
            mode="interp",
        )

        residual = (
            spectra[:, spectrum_index].astype(
                np.float64
            )
            - smooth
        )

        if residual.size >= 3:
            x = residual[:-1]
            y = residual[1:]

            denominator = float(
                np.sqrt(
                    np.sum(x * x)
                    * np.sum(y * y)
                )
            )

            if denominator > 1.0e-12:
                rho = float(
                    np.sum(x * y)
                    / denominator
                )
            else:
                rho = 0.0
        else:
            rho = 0.0

        # Avoid producing unrealistic long-memory random walks.
        rho = float(
            np.clip(
                rho,
                -0.60,
                0.60,
            )
        )

        white_sigma = float(
            sigma
            * np.sqrt(
                max(
                    1.0 - rho * rho,
                    1.0e-6,
                )
            )
        )

        tail_noise = np.zeros(
            tail_axis.size,
            dtype=np.float64,
        )

        tail_noise[0] = rng.normal(
            0.0,
            sigma,
        )

        for point_index in range(
            1,
            tail_noise.size,
        ):
            tail_noise[point_index] = (
                rho
                * tail_noise[
                    point_index - 1
                ]
                + rng.normal(
                    0.0,
                    white_sigma,
                )
            )

        # Blend the baseline from the last measured model-axis point
        # toward the estimated local background. This avoids a sharp
        # discontinuity at the measured/synthetic boundary.
        measured_end = float(
            measured[-1]
        )

        blend_length = min(
            30,
            int(tail_axis.size),
        )

        baseline = np.full(
            tail_axis.size,
            background_level,
            dtype=np.float64,
        )

        if blend_length > 0:
            baseline[:blend_length] = np.linspace(
                measured_end,
                background_level,
                blend_length,
                endpoint=True,
            )

        synthetic_tail = (
            baseline
            + tail_noise
        )

        output[
            measured_axis.size:,
            spectrum_index,
        ] = synthetic_tail.astype(
            np.float32
        )

    return (
        MODEL_RAMAN_AXIS.copy(),
        output,
        True,
    )




def _impute_missing_2000_2500_tail_with_noise(
    axis: np.ndarray,
    spectra: np.ndarray,
    *,
    source_key: str,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Compatibility alias; now accepts any valid missing tail before 2500."""
    return _impute_missing_tail_to_2500_with_noise(
        axis, spectra, source_key=source_key,
    )


def _classify_raw_axis_coverage(
    axis: np.ndarray,
) -> str:
    """
    Classify the ORIGINAL file Raman coverage before runtime imputation.

    Returns:
        "full_2500"  : approximately 600-2500 cm^-1
        "short_2000" : approximately 600-2000 cm^-1 (legacy category)
        "short_tail" : another valid endpoint below 2500 cm^-1
        "other"      : unexpected range; never silently imputed
    """
    axis = np.asarray(
        axis,
        dtype=np.float64,
    )

    if (
        axis.ndim != 1 or axis.size < 2
        or not np.isfinite(axis).all() or not np.all(np.diff(axis) > 0)
    ):
        return "other"

    start = float(axis[0])
    end = float(axis[-1])

    start_ok = (
        abs(
            start
            - float(MODEL_RAMAN_AXIS[0])
        )
        <= FULL_AXIS_COVERAGE_TOLERANCE_CM1
    )

    if (
        start_ok
        and end
        >= float(MODEL_RAMAN_AXIS[-1])
    ):
        return "full_2500"

    if (
        start_ok
        and axis.size >= 31
        and 1999.0
        <= end
        <= 2001.0
    ):
        return "short_2000"

    if start_ok and axis.size >= 31 and 600.0 < end < 2500.0:
        return "short_tail"

    return "other"


def _audit_raw_input_root(
    root: Path,
    source_name: str,
) -> dict[str, Any]:
    """
    Scan every ORIGINAL input table before runtime completion.

    Important:
    - every spectrum column is counted;
    - nothing is written back to disk;
    - this audit happens before any missing tail is synthesized.
    """
    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(
            f"{source_name} data root does not exist: {root}"
        )

    files = _collect_table_files(root)

    file_count = 0
    spectrum_count = 0

    short_file_count = 0
    short_spectrum_count = 0

    full_file_count = 0
    full_spectrum_count = 0

    point_count_distribution: dict[int, int] = {}
    short_files: list[str] = []
    unexpected_files: list[str] = []

    for file_path in files:
        axis, spectra, _ = _read_table(
            file_path
        )

        number_of_spectra = int(
            spectra.shape[1]
        )

        file_count += 1
        spectrum_count += number_of_spectra

        points = int(axis.size)

        point_count_distribution[points] = (
            point_count_distribution.get(
                points,
                0,
            )
            + 1
        )

        coverage = (
            _classify_raw_axis_coverage(
                axis
            )
        )

        if coverage in {"short_2000", "short_tail"}:
            short_file_count += 1
            short_spectrum_count += (
                number_of_spectra
            )

            short_files.append(
                f"{file_path}: "
                f"{float(axis[0]):.1f}-"
                f"{float(axis[-1]):.1f} cm^-1, "
                f"{points} points, "
                f"{number_of_spectra} spectra"
            )

        elif coverage == "full_2500":
            full_file_count += 1
            full_spectrum_count += (
                number_of_spectra
            )

        else:
            unexpected_files.append(
                f"{file_path}: "
                f"{float(axis[0]):.1f}-"
                f"{float(axis[-1]):.1f} cm^-1, "
                f"{points} points, "
                f"{number_of_spectra} spectra"
            )

    return {
        "source": source_name,
        "root": str(root),
        "file_count": int(file_count),
        "spectrum_count": int(
            spectrum_count
        ),
        "short_axis_file_count": int(
            short_file_count
        ),
        "short_axis_spectrum_count": int(
            short_spectrum_count
        ),
        "full_axis_file_count": int(
            full_file_count
        ),
        "full_axis_spectrum_count": int(
            full_spectrum_count
        ),
        "point_count_distribution": {
            str(key): int(value)
            for key, value
            in sorted(
                point_count_distribution.items()
            )
        },
        "short_files": short_files,
        "unexpected_files": (
            unexpected_files
        ),
    }


def audit_input_axis_coverage(
    real_root: str | Path = REAL_DATA_ROOT,
    generated_root: str | Path = GENERATED_DATA_ROOT,
) -> dict[str, Any]:
    """
    Audit ALL real and generated RAW spectra before training.

    Short 600-2000 spectra are only reported here.
    Actual completion happens later during Dataset loading.
    """
    real_summary = (
        _audit_raw_input_root(
            Path(real_root),
            "real",
        )
    )

    generated_summary = (
        _audit_raw_input_root(
            Path(generated_root),
            "generated",
        )
    )

    print()
    print(
        "========================================"
    )
    print(
        "RAW INPUT AXIS AUDIT BEFORE "
        "RUNTIME TAIL IMPUTATION"
    )
    print(
        "========================================"
    )

    for summary in (
        real_summary,
        generated_summary,
    ):
        print()
        print(
            f"[{summary['source'].upper()}]"
        )

        print(
            "files                 =",
            summary["file_count"],
        )

        print(
            "spectra               =",
            summary["spectrum_count"],
        )

        print(
            "raw point counts      =",
            summary[
                "point_count_distribution"
            ],
        )

        print(
            "full-axis files       =",
            summary[
                "full_axis_file_count"
            ],
        )

        print(
            "full-axis spectra     =",
            summary[
                "full_axis_spectrum_count"
            ],
        )

        print(
            "short-axis files      =",
            summary[
                "short_axis_file_count"
            ],
        )

        print(
            "short-axis spectra    =",
            summary[
                "short_axis_spectrum_count"
            ],
        )

        print(
            "spectra needing tail  =",
            summary[
                "short_axis_spectrum_count"
            ],
        )

    unexpected = (
        real_summary["unexpected_files"]
        + generated_summary[
            "unexpected_files"
        ]
    )

    if unexpected:
        preview = "\n".join(
            f"  - {item}"
            for item in unexpected[:50]
        )

        raise RuntimeError(
            "Found invalid axes or axes not starting near 600 cm^-1. "
            "Missing-tail sources must have at least 31 measured points. "
            "These files will NOT be silently completed:\n"
            + preview
        )

    print()
    print(
        "RAW INPUT AXIS AUDIT: PASS"
    )
    print(
        "No Excel/CSV file has been modified."
    )
    print("Runtime model axis: 600-2500 cm^-1, 1901 points.")
    print("Raw point counts describe stored files BEFORE in-memory completion.")
    print(
        "========================================"
    )

    return {
        "real": real_summary,
        "generated": generated_summary,
    }

class SERSDataRepository:
    """Load and validate real/generated condition tables once."""

    def __init__(
        self,
        real_root: str | Path = REAL_DATA_ROOT,
        generated_root: str | Path = GENERATED_DATA_ROOT,
        concentration_map: Mapping[str, Mapping[str, float]] | None = None,
    ) -> None:
        self.real_root = Path(real_root)
        self.generated_root = Path(generated_root)
        self.concentration_map = concentration_map
        self.target_mode = "physical" if concentration_map is not None else "level_code"

        # ----------------------------------------------------
        # RAW audit BEFORE any runtime tail imputation.
        # This runs every time training constructs a repository.
        # ----------------------------------------------------
        self.input_axis_audit = audit_input_axis_coverage(
            real_root=self.real_root,
            generated_root=self.generated_root,
        )

        self.real_data: dict[str, dict[str, Any]] = {}
        self.generated_data: dict[str, dict[str, Any]] | None = None

        self.generated_tail_imputation_summary: dict[str, Any] = {
            "method": "deterministic_background_noise_tail",
            "range_policy": "model-axis positions beyond each source endpoint up to 2500 cm^-1",
            "target_end_cm-1": 2500.0,
            "imputed_file_count": 0,
            "imputed_spectrum_count": 0,
            "note": (
                "Generated short-axis spectra are completed "
                "in memory during dataset loading. "
                "Original files remain unchanged."
            ),
        }

        self._load_real_data()

    def _load_real_data(self) -> None:
        files = _collect_table_files(self.real_root)
        if len(files) != 126:
            raise RuntimeError(f"Expected 126 real source files, found {len(files)}")

        imputed_full_axis: list[str] = []

        for path in files:
            condition = _condition_from_real_path(path)
            if condition in self.real_data:
                raise RuntimeError(f"Duplicate real condition: {condition}")

            axis, spectra, names = _read_table(path)

            original_axis_start = float(axis[0])
            original_axis_end = float(axis[-1])
            original_axis_points = int(axis.size)

            axis, spectra, tail_imputed = (
                _impute_missing_tail_to_2500_with_noise(
                    axis,
                    spectra,
                    source_key=str(path),
                )
            )

            if tail_imputed:
                imputed_full_axis.append(
                    f"{path}: "
                    f"{original_axis_start:.1f}-"
                    f"{original_axis_end:.1f} cm^-1 "
                    f"({original_axis_points} points) "
                    f"-> 600.0-2500.0 cm^-1 "
                    f"({MODEL_LENGTH} points)"
                )

            if spectra.shape[1] != 20:
                raise RuntimeError(
                    f"Each real source file must contain 20 spectra: {path}; "
                    f"found {spectra.shape[1]}"
                )

            info = _parse_condition(condition, self.concentration_map)
            self.real_data[condition] = {
                "path": path,
                "axis": axis,
                "spectra": spectra,
                "names": names,
                "original_axis_start_cm-1": original_axis_start,
                "original_axis_end_cm-1": original_axis_end,
                "original_axis_points": original_axis_points,
                "tail_imputed": bool(tail_imputed),
                "valid_mask": _mask_for_source_axis(axis),
                "condition_info": info,
            }

        self.tail_imputation_summary = {
            "method": "deterministic_background_noise_tail",
            "range_policy": "model-axis positions beyond each source endpoint up to 2500 cm^-1",
            "target_end_cm-1": 2500.0,
            "imputed_file_count": int(len(imputed_full_axis)),
            "imputed_files": list(imputed_full_axis),
            "note": (
                "All valid real source files ending before 2500 cm^-1 are completed in memory. "
                "Existing genuine 600-2500 cm^-1 files are unchanged."
            ),
        }

        if imputed_full_axis:
            print(
                "===== Real-spectrum noise-tail imputation ====="
            )
            print(
                f"imputed files: {len(imputed_full_axis)}"
            )
            print(
                "range: beyond each measured endpoint to 2500 cm^-1"
            )
            print(
                "method: deterministic background-matched noise"
            )

    def load_generated_data(self, require_complete: bool = True) -> None:
        if self.generated_data is not None:
            if require_complete and len(self.generated_data) != 126:
                raise RuntimeError(
                    f"Generated set incomplete: {len(self.generated_data)}/126 conditions"
                )
            return

        files = _collect_table_files(self.generated_root)
        generated: dict[str, dict[str, Any]] = {}

        imputed_generated_files = 0
        imputed_generated_spectra = 0

        for path in files:
            condition = _condition_from_generated_path(path)
            if condition not in self.real_data:
                raise RuntimeError(
                    f"Generated condition has no matching real condition: {condition}"
                )
            if condition in generated:
                raise RuntimeError(f"Duplicate generated condition: {condition}")

            axis, spectra, names = _read_table(path)

            original_coverage = (
                _classify_raw_axis_coverage(
                    axis
                )
            )

            original_number_of_spectra = int(
                spectra.shape[1]
            )
            original_axis_start = float(axis[0])
            original_axis_end = float(axis[-1])
            original_axis_points = int(axis.size)

            axis, spectra, tail_imputed = (
                _impute_missing_tail_to_2500_with_noise(
                    axis,
                    spectra,
                    source_key=f"generated|{path}",
                )
            )

            if tail_imputed:
                imputed_generated_files += 1
                imputed_generated_spectra += (
                    original_number_of_spectra
                )

            generated[condition] = {
                "path": path,
                "axis": axis,
                "spectra": spectra,
                "names": names,
                "original_axis_coverage": original_coverage,
                "original_axis_start_cm-1": original_axis_start,
                "original_axis_end_cm-1": original_axis_end,
                "original_axis_points": original_axis_points,
                "tail_imputed": bool(
                    tail_imputed
                ),
            }

        if require_complete and len(generated) != 126:
            raise RuntimeError(f"Generated set incomplete: {len(generated)}/126 conditions")

        self.generated_tail_imputation_summary = {
            "method": "deterministic_background_noise_tail",
            "range_policy": "model-axis positions beyond each source endpoint up to 2500 cm^-1",
            "target_end_cm-1": 2500.0,
            "imputed_file_count": int(
                imputed_generated_files
            ),
            "imputed_spectrum_count": int(
                imputed_generated_spectra
            ),
            "note": (
                "All valid generated source files ending before "
                "2500 cm^-1 were completed in memory. "
                "Existing 600-2500 cm^-1 generated spectra "
                "were left unchanged."
            ),
        }

        print()
        print(
            "===== Generated-spectrum runtime "
            "noise-tail imputation ====="
        )
        print(
            "imputed files   =",
            imputed_generated_files,
        )
        print(
            "imputed spectra =",
            imputed_generated_spectra,
        )
        print(
            "range           = beyond each measured endpoint to 2500 cm^-1"
        )
        print(
            "disk files      = unchanged"
        )

        self.generated_data = generated



def build_axis_label_audit(repository: SERSDataRepository) -> dict[str, Any]:
    """Describe source-axis length versus pesticide presence before common-axis cropping.

    This audit is intentionally based on the original source axes. It documents
    acquisition/label confounding without exposing that information to the model.
    """
    counts: dict[str, dict[str, int]] = {pesticide: {} for pesticide in PESTICIDES}
    axis_lengths: dict[int, int] = {}

    for record in repository.real_data.values():
        source_length = int(record.get("original_axis_points", len(record["axis"])))
        axis_lengths[source_length] = axis_lengths.get(source_length, 0) + 1
        info: ConditionInfo = record["condition_info"]
        for pesticide_index, pesticide in enumerate(PESTICIDES):
            presence = int(info.presence[pesticide_index] > 0.5)
            key = f"presence={presence}|points={source_length}"
            counts[pesticide][key] = counts[pesticide].get(key, 0) + 1

    return {
        "source_file_count": int(len(repository.real_data)),
        "source_axis_point_counts": {str(k): int(v) for k, v in sorted(axis_lengths.items())},
        "pesticide_by_source_axis": counts,
        "model_axis_start_cm-1": float(MODEL_RAMAN_AXIS[0]),
        "model_axis_end_cm-1": float(MODEL_RAMAN_AXIS[-1]),
        "model_axis_points": int(MODEL_LENGTH),
        "mitigation": (
            "Before training, all raw real/generated source axes are audited. "
            "All valid files ending below 2500 cm^-1 are completed only in memory "
            "beyond their measured endpoints using deterministic background-matched noise. "
            "Existing 600-2500 cm^-1 spectra are left unchanged."
        ),
    }


def build_training_peak_priors(
    repository: SERSDataRepository,
    *,
    top_k: int = 8,
    min_peak_distance_cm1: float = 20.0,
    peak_prominence: float = 0.02,
    peak_half_width_cm1: float = 15.0,
    min_support_fraction: float = 0.25,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Build DEL/CHL/TEB soft peak priors from *real training spectra only*.

    Important leakage rule:
    - only REAL_TRAIN_INDICES (0..11) are used;
    - validation/test spectra are never read for prior fitting;
    - generated spectra are never used for prior fitting.

    The prior is not a hard spectral mask. It is a [3, 1901] non-negative
    guidance map later added to attention logits. Consequently a pesticide
    query can still use every valid non-peak region.

    Peak candidates are detected on the mean normalized spectrum of all
    training spectra containing the pesticide and are ranked by a combination
    of peak prominence, positive-vs-negative contrast and valid-axis support.
    This makes the prior data-driven instead of hard-coding literature peak
    positions.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if min_peak_distance_cm1 <= 0.0:
        raise ValueError("min_peak_distance_cm1 must be positive")
    if peak_prominence < 0.0:
        raise ValueError("peak_prominence must be non-negative")
    if peak_half_width_cm1 <= 0.0:
        raise ValueError("peak_half_width_cm1 must be positive")
    if not (0.0 < min_support_fraction <= 1.0):
        raise ValueError("min_support_fraction must be in (0, 1]")

    signals: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for condition in sorted(repository.real_data):
        record = repository.real_data[condition]
        info: ConditionInfo = record["condition_info"]
        for spectrum_index in REAL_TRAIN_INDICES:
            spectrum = record["spectra"][:, spectrum_index]
            adapted, valid_mask = _adapt_spectrum_to_model_axis(
                record["axis"], spectrum
            )
            raw, _, _ = _preprocess_spectrum(adapted, valid_mask)
            signals.append(raw[0].astype(np.float32, copy=False))
            masks.append(valid_mask.astype(np.bool_, copy=False))
            labels.append(info.presence.astype(np.float32, copy=False))

    signal_matrix = np.stack(signals, axis=0)
    valid_matrix = np.stack(masks, axis=0)
    label_matrix = np.stack(labels, axis=0)

    expected_samples = len(repository.real_data) * len(REAL_TRAIN_INDICES)
    if signal_matrix.shape != (expected_samples, MODEL_LENGTH):
        raise RuntimeError(
            "Unexpected training matrix shape while fitting peak priors: "
            f"{signal_matrix.shape}"
        )

    axis_step = float(np.median(np.diff(MODEL_RAMAN_AXIS)))
    distance_points = max(
        1, int(round(float(min_peak_distance_cm1) / max(axis_step, 1e-8)))
    )
    gaussian_sigma = max(
        float(peak_half_width_cm1) / 2.0 / max(axis_step, 1e-8), 1.0
    )

    priors = np.zeros((len(PESTICIDES), MODEL_LENGTH), dtype=np.float32)
    summary: dict[str, Any] = {
        "fit_source": "real training spectra only",
        "real_training_indices": list(REAL_TRAIN_INDICES),
        "number_of_training_spectra": int(expected_samples),
        "parameters": {
            "top_k": int(top_k),
            "min_peak_distance_cm-1": float(min_peak_distance_cm1),
            "peak_prominence": float(peak_prominence),
            "peak_half_width_cm-1": float(peak_half_width_cm1),
            "min_support_fraction": float(min_support_fraction),
        },
        "pesticides": {},
    }

    point_index = np.arange(MODEL_LENGTH, dtype=np.float64)

    for pesticide_index, pesticide in enumerate(PESTICIDES):
        positive_rows = label_matrix[:, pesticide_index] > 0.5
        negative_rows = ~positive_rows
        positive_total = int(positive_rows.sum())
        negative_total = int(negative_rows.sum())
        if positive_total == 0:
            raise RuntimeError(f"No positive training spectra found for {pesticide}")

        positive_mask = valid_matrix[positive_rows]
        positive_signal = signal_matrix[positive_rows]
        positive_count = positive_mask.sum(axis=0).astype(np.float64)
        positive_sum = (positive_signal * positive_mask).sum(axis=0, dtype=np.float64)
        positive_mean = np.divide(
            positive_sum,
            positive_count,
            out=np.zeros(MODEL_LENGTH, dtype=np.float64),
            where=positive_count > 0,
        )

        negative_mask = valid_matrix[negative_rows]
        negative_signal = signal_matrix[negative_rows]
        negative_count = negative_mask.sum(axis=0).astype(np.float64)
        negative_sum = (negative_signal * negative_mask).sum(axis=0, dtype=np.float64)
        negative_mean = np.divide(
            negative_sum,
            negative_count,
            out=np.zeros(MODEL_LENGTH, dtype=np.float64),
            where=negative_count > 0,
        )

        support_fraction = positive_count / float(max(positive_total, 1))
        supported = support_fraction >= float(min_support_fraction)

        # Smooth only for robust peak discovery. The model still receives the
        # original preprocessed spectra, not this smoothed average.
        gaussian_radius = max(2, int(round(3.0 * gaussian_sigma)))
        kernel_x = np.arange(-gaussian_radius, gaussian_radius + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (kernel_x / gaussian_sigma) ** 2)
        kernel /= kernel.sum()
        positive_smooth = np.convolve(positive_mean, kernel, mode="same")
        positive_smooth[~supported] = 0.0

        candidate_indices, properties = signal.find_peaks(
            positive_smooth,
            distance=distance_points,
            prominence=float(peak_prominence),
        )
        if candidate_indices.size == 0:
            candidate_indices, properties = signal.find_peaks(
                positive_smooth,
                distance=distance_points,
            )

        candidate_indices = candidate_indices[supported[candidate_indices]]
        if candidate_indices.size == 0:
            supported_indices = np.flatnonzero(supported)
            if supported_indices.size == 0:
                raise RuntimeError(
                    f"No Raman position has sufficient training support for {pesticide}"
                )
            candidate_indices = np.asarray(
                [supported_indices[np.argmax(positive_smooth[supported_indices])]],
                dtype=np.int64,
            )

        prominences = signal.peak_prominences(
            positive_smooth, candidate_indices
        )[0]
        contrast = np.maximum(positive_mean - negative_mean, 0.0)
        contrast_max = float(contrast.max())
        if contrast_max > 0.0:
            contrast_norm = contrast / contrast_max
        else:
            contrast_norm = np.zeros_like(contrast)

        # Prominence keeps the prior focused on real spectral peaks; contrast
        # makes pesticide-specific evidence rank above peaks shared by all
        # spectra; a non-zero base term retains useful shared peaks.
        ranking = prominences * (
            0.25 + 0.75 * contrast_norm[candidate_indices]
        ) * support_fraction[candidate_indices]
        order = np.argsort(ranking)[::-1][: int(top_k)]
        selected = candidate_indices[order]
        selected_scores = ranking[order]

        score_max = float(selected_scores.max()) if selected_scores.size else 0.0
        if score_max > 0.0:
            selected_weights = selected_scores / score_max
        else:
            selected_weights = np.ones_like(selected_scores, dtype=np.float64)

        prior = np.zeros(MODEL_LENGTH, dtype=np.float64)
        for center_index, weight in zip(selected, selected_weights, strict=True):
            gaussian = np.exp(
                -0.5 * ((point_index - float(center_index)) / gaussian_sigma) ** 2
            )
            prior = np.maximum(prior, float(weight) * gaussian)

        prior_max = float(prior.max())
        if prior_max > 0.0:
            prior /= prior_max
        priors[pesticide_index] = prior.astype(np.float32)

        summary["pesticides"][pesticide] = {
            "positive_training_spectra": positive_total,
            "negative_training_spectra": negative_total,
            "peak_centers_cm-1": [
                float(MODEL_RAMAN_AXIS[index]) for index in selected.tolist()
            ],
            "peak_weights": [float(value) for value in selected_weights.tolist()],
            "support_fractions": [
                float(support_fraction[index]) for index in selected.tolist()
            ],
        }

    return priors, summary


def _training_condition_representative(
    record: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Build one robust representative from the 12 real training mappings.

    Mapping spectra from the same source condition are aggregated before any
    cross-condition comparison. This avoids treating highly related mappings as
    independent evidence when constructing a pesticide-specific spectral prior.
    """
    signals: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    for spectrum_index in REAL_TRAIN_INDICES:
        adapted, valid_mask = _adapt_spectrum_to_model_axis(
            record["axis"],
            record["spectra"][:, spectrum_index],
        )
        raw, _, _ = _preprocess_spectrum(adapted, valid_mask)
        signals.append(raw[0].astype(np.float32, copy=False))
        masks.append(valid_mask.astype(np.bool_, copy=False))

    signal_matrix = np.stack(signals, axis=0).astype(np.float64)
    valid_matrix = np.stack(masks, axis=0)
    masked = np.where(valid_matrix, signal_matrix, np.nan)
    with np.errstate(invalid="ignore"):
        representative = np.nanmedian(masked, axis=0)
    representative = np.nan_to_num(representative, nan=0.0).astype(np.float32)
    valid = valid_matrix.any(axis=0)
    return representative, valid


def build_matched_condition_peak_priors(
    repository: SERSDataRepository,
    *,
    top_k: int = 8,
    min_peak_distance_cm1: float = 20.0,
    peak_prominence: float = 0.02,
    peak_half_width_cm1: float = 15.0,
    min_matched_pairs: int = 4,
    shared_peak_floor: float = 0.35,
    excluded_intervals_cm1: tuple[tuple[float, float], ...] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Build shared-aware soft priors from matched training conditions.

    For pesticide p, a present condition is compared only with an available
    condition having the same matrix and the same concentrations of the other
    pesticides, but p=absent. Thus the spectral difference is much less
    confounded by co-occurring pesticides than an all-positive vs all-negative
    average.

    The result is deliberately *not* an exclusive peak assignment. A Raman
    region can be informative for DEL and TEB simultaneously. Specificity only
    attenuates heavily shared regions; ``shared_peak_floor`` guarantees that a
    genuinely shared/overlapping region can still guide multiple queries.

    Leakage rules:
    - only the 12 real training mappings are used;
    - each source condition is first collapsed to one median representative;
    - validation, test and generated spectra are never used.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if min_peak_distance_cm1 <= 0.0:
        raise ValueError("min_peak_distance_cm1 must be positive")
    if peak_prominence < 0.0:
        raise ValueError("peak_prominence must be non-negative")
    if peak_half_width_cm1 <= 0.0:
        raise ValueError("peak_half_width_cm1 must be positive")
    if min_matched_pairs <= 0:
        raise ValueError("min_matched_pairs must be positive")
    if not (0.0 <= shared_peak_floor <= 1.0):
        raise ValueError("shared_peak_floor must be in [0, 1]")

    excluded_intervals_cm1 = tuple(excluded_intervals_cm1 or ())

    for lower, upper in excluded_intervals_cm1:
        if not (
            np.isfinite(lower)
            and np.isfinite(upper)
            and lower <= upper
        ):
            raise ValueError(
                f"Invalid excluded Raman interval: {(lower, upper)}"
            )

    # One representative per condition, built from training mappings only.
    representatives: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    condition_lookup: dict[tuple[str, tuple[int, ...]], str] = {}
    for condition, record in repository.real_data.items():
        info: ConditionInfo = record["condition_info"]
        levels = tuple(int(round(float(v))) for v in info.level_codes.tolist())
        key = (info.matrix_name, levels)
        if key in condition_lookup:
            raise RuntimeError(
                "Matched-condition prior requires unique matrix/level conditions; "
                f"duplicate key={key}"
            )
        condition_lookup[key] = condition
        representatives[condition] = _training_condition_representative(record)

    delta_profiles: list[np.ndarray] = []
    valid_profiles: list[np.ndarray] = []
    pair_descriptions: list[list[dict[str, Any]]] = []

    for pesticide_index, pesticide in enumerate(PESTICIDES):
        pesticide_deltas: list[np.ndarray] = []
        pesticide_valids: list[np.ndarray] = []
        pesticide_pairs: list[dict[str, Any]] = []

        for present_condition, present_record in sorted(repository.real_data.items()):
            present_info: ConditionInfo = present_record["condition_info"]
            present_levels = np.asarray(present_info.level_codes, dtype=np.int64)
            target_level = int(present_levels[pesticide_index])
            if target_level <= 0:
                continue

            absent_levels = present_levels.copy()
            absent_levels[pesticide_index] = 0

            # The current 126-condition dataset has no all-absent blank. A
            # single-pesticide condition therefore cannot form a matched pair
            # and is intentionally omitted instead of using an unmatched blank.
            absent_key = (
                present_info.matrix_name,
                tuple(int(v) for v in absent_levels.tolist()),
            )
            absent_condition = condition_lookup.get(absent_key)
            if absent_condition is None:
                continue

            present_rep, present_valid = representatives[present_condition]
            absent_rep, absent_valid = representatives[absent_condition]
            common_valid = present_valid & absent_valid
            if int(common_valid.sum()) < 2:
                continue

            delta = np.zeros(MODEL_LENGTH, dtype=np.float32)
            delta[common_valid] = (
                present_rep[common_valid] - absent_rep[common_valid]
            )
            pesticide_deltas.append(delta)
            pesticide_valids.append(common_valid)
            pesticide_pairs.append(
                {
                    "present_condition": present_condition,
                    "matched_absent_condition": absent_condition,
                    "target_level_code": target_level,
                    "matrix": present_info.matrix_name,
                }
            )

        if len(pesticide_deltas) < int(min_matched_pairs):
            raise RuntimeError(
                f"Only {len(pesticide_deltas)} matched pairs are available for "
                f"{pesticide}; require at least {min_matched_pairs}."
            )

        delta_profiles.append(np.stack(pesticide_deltas, axis=0))
        valid_profiles.append(np.stack(pesticide_valids, axis=0))
        pair_descriptions.append(pesticide_pairs)

    # First pass: robust target-sensitive effect and direction consistency.
    effect_profiles = np.zeros((len(PESTICIDES), MODEL_LENGTH), dtype=np.float64)
    consistency_profiles = np.zeros_like(effect_profiles)
    support_profiles = np.zeros_like(effect_profiles)

    for pesticide_index in range(len(PESTICIDES)):
        deltas = delta_profiles[pesticide_index].astype(np.float64)
        valids = valid_profiles[pesticide_index]
        masked_abs = np.where(valids, np.abs(deltas), np.nan)
        with np.errstate(invalid="ignore"):
            effect = np.nanmedian(masked_abs, axis=0)
        effect = np.nan_to_num(effect, nan=0.0)

        valid_count = valids.sum(axis=0).astype(np.float64)
        support = valid_count / float(max(deltas.shape[0], 1))

        signs = np.sign(deltas)
        signed_sum = (signs * valids).sum(axis=0, dtype=np.float64)
        direction_consistency = np.divide(
            np.abs(signed_sum),
            valid_count,
            out=np.zeros(MODEL_LENGTH, dtype=np.float64),
            where=valid_count > 0,
        )

        effect_profiles[pesticide_index] = effect
        consistency_profiles[pesticide_index] = direction_consistency
        support_profiles[pesticide_index] = support

    # Normalize each pesticide effect independently before estimating whether a
    # region is pesticide-specific or shared. This prevents one globally
    # stronger pesticide from suppressing all other priors.
    effect_norm = np.zeros_like(effect_profiles)
    for pesticide_index in range(len(PESTICIDES)):
        maximum = float(effect_profiles[pesticide_index].max())
        if maximum > 0.0:
            effect_norm[pesticide_index] = effect_profiles[pesticide_index] / maximum

    total_effect = effect_norm.sum(axis=0)
    specificity = np.divide(
        effect_norm,
        total_effect[np.newaxis, :],
        out=np.zeros_like(effect_norm),
        where=total_effect[np.newaxis, :] > 1e-12,
    )

    axis_step = float(np.median(np.diff(MODEL_RAMAN_AXIS)))
    distance_points = max(
        1, int(round(float(min_peak_distance_cm1) / max(axis_step, 1e-8)))
    )
    gaussian_sigma = max(
        float(peak_half_width_cm1) / 2.0 / max(axis_step, 1e-8), 1.0
    )
    gaussian_radius = max(2, int(round(3.0 * gaussian_sigma)))
    kernel_x = np.arange(-gaussian_radius, gaussian_radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (kernel_x / gaussian_sigma) ** 2)
    kernel /= kernel.sum()
    point_index = np.arange(MODEL_LENGTH, dtype=np.float64)

    priors = np.zeros((len(PESTICIDES), MODEL_LENGTH), dtype=np.float32)
    summary: dict[str, Any] = {
        "method": "matched_condition_shared_aware",
        "fit_source": "real training mappings only; one median representative per condition",
        "real_training_indices": list(REAL_TRAIN_INDICES),
        "parameters": {
            "top_k": int(top_k),
            "min_peak_distance_cm-1": float(min_peak_distance_cm1),
            "peak_prominence": float(peak_prominence),
            "peak_half_width_cm-1": float(peak_half_width_cm1),
            "min_matched_pairs": int(min_matched_pairs),
            "shared_peak_floor": float(shared_peak_floor),
            "excluded_intervals_cm-1": [
                [float(lower), float(upper)]
                for lower, upper in excluded_intervals_cm1
            ],
        },
        "interpretation": (
            "Selected regions are pesticide-sensitive spectral evidence, not "
            "chemically exclusive peak assignments. Shared regions remain usable "
            "by multiple pesticide queries."
        ),
        "pesticides": {},
    }

    for pesticide_index, pesticide in enumerate(PESTICIDES):
        smooth_effect = np.convolve(
            effect_norm[pesticide_index], kernel, mode="same"
        )
        candidates, _ = signal.find_peaks(
            smooth_effect,
            distance=distance_points,
            prominence=float(peak_prominence),
        )
        if candidates.size == 0:
            candidates, _ = signal.find_peaks(
                smooth_effect,
                distance=distance_points,
            )
        if candidates.size == 0:
            candidates = np.asarray(
                [int(np.argmax(smooth_effect))], dtype=np.int64
            )

        if excluded_intervals_cm1:
            candidate_shifts = MODEL_RAMAN_AXIS[candidates].astype(
                np.float64
            )
            keep = np.ones(
                candidates.shape,
                dtype=np.bool_,
            )

            for lower, upper in excluded_intervals_cm1:
                keep &= ~(
                    (candidate_shifts >= float(lower))
                    & (candidate_shifts <= float(upper))
                )

            candidates = candidates[keep]

            if candidates.size == 0:
                allowed = np.ones(
                    MODEL_LENGTH,
                    dtype=np.bool_,
                )

                for lower, upper in excluded_intervals_cm1:
                    allowed &= ~(
                        (MODEL_RAMAN_AXIS >= float(lower))
                        & (MODEL_RAMAN_AXIS <= float(upper))
                    )

                if not bool(allowed.any()):
                    raise RuntimeError(
                        "All Raman positions were excluded "
                        "from auxiliary peak search"
                    )

                masked_effect = np.where(
                    allowed,
                    smooth_effect,
                    -np.inf,
                )

                candidates = np.asarray(
                    [int(np.argmax(masked_effect))],
                    dtype=np.int64,
                )

        prominences = signal.peak_prominences(
            smooth_effect,
            candidates,
        )[0]
        consistency_factor = 0.5 + 0.5 * consistency_profiles[
            pesticide_index, candidates
        ]
        specificity_factor = (
            float(shared_peak_floor)
            + (1.0 - float(shared_peak_floor))
            * specificity[pesticide_index, candidates]
        )
        ranking = (
            prominences
            * effect_norm[pesticide_index, candidates]
            * consistency_factor
            * specificity_factor
            * support_profiles[pesticide_index, candidates]
        )

        order = np.argsort(ranking)[::-1][: int(top_k)]
        selected = candidates[order]
        selected_scores = ranking[order]
        score_max = float(selected_scores.max()) if selected_scores.size else 0.0
        if score_max > 0.0:
            selected_weights = selected_scores / score_max
        else:
            selected_weights = np.ones_like(selected_scores, dtype=np.float64)

        prior = np.zeros(MODEL_LENGTH, dtype=np.float64)
        selected_evidence: list[dict[str, Any]] = []
        for rank, (center_index, weight) in enumerate(
            zip(selected, selected_weights, strict=True), start=1
        ):
            gaussian = np.exp(
                -0.5 * ((point_index - float(center_index)) / gaussian_sigma) ** 2
            )
            prior = np.maximum(prior, float(weight) * gaussian)
            selected_evidence.append(
                {
                    "rank": int(rank),
                    "raman_shift_cm-1": float(MODEL_RAMAN_AXIS[center_index]),
                    "prior_weight": float(weight),
                    "matched_effect": float(effect_norm[pesticide_index, center_index]),
                    "direction_consistency": float(
                        consistency_profiles[pesticide_index, center_index]
                    ),
                    "specificity": float(specificity[pesticide_index, center_index]),
                    "sharedness": float(1.0 - specificity[pesticide_index, center_index]),
                    "support_fraction": float(
                        support_profiles[pesticide_index, center_index]
                    ),
                }
            )

        prior_max = float(prior.max())
        if prior_max > 0.0:
            prior /= prior_max
        priors[pesticide_index] = prior.astype(np.float32)

        summary["pesticides"][pesticide] = {
            "matched_pair_count": int(delta_profiles[pesticide_index].shape[0]),
            "matched_pairs": pair_descriptions[pesticide_index],
            "peak_centers_cm-1": [
                float(MODEL_RAMAN_AXIS[index]) for index in selected.tolist()
            ],
            "peak_weights": [float(value) for value in selected_weights.tolist()],
            "selected_evidence": selected_evidence,
        }

    return priors, summary


def _chemistry_core_intervals(
    half_width_cm1: float,
) -> tuple[tuple[float, float], ...]:
    intervals: list[tuple[float, float]] = []

    for pesticide in PESTICIDES:
        for item in CHEMISTRY_CORE_PEAKS[pesticide]:
            center = float(item["center_cm-1"])
            intervals.append(
                (
                    center - float(half_width_cm1),
                    center + float(half_width_cm1),
                )
            )

    return tuple(intervals)


def _truncated_core_gaussian(
    center_cm1: float,
    half_width_cm1: float,
    weight: float,
) -> np.ndarray:
    if half_width_cm1 <= 0.0:
        raise ValueError("core half-width must be positive")

    sigma = max(
        float(half_width_cm1) / 2.0,
        1e-6,
    )

    distance = (
        MODEL_RAMAN_AXIS.astype(np.float64)
        - float(center_cm1)
    )

    inside = (
        np.abs(distance)
        <= float(half_width_cm1)
    )

    values = np.zeros(
        MODEL_LENGTH,
        dtype=np.float64,
    )

    values[inside] = np.exp(
        -0.5
        * np.square(
            distance[inside] / sigma
        )
    )

    maximum = float(values.max())

    if maximum > 0.0:
        values /= maximum

    return (
        float(weight) * values
    ).astype(np.float32)


def build_chemistry_hybrid_peak_priors(
    repository: SERSDataRepository,
    *,
    core_half_width_cm1: float = DEFAULT_CORE_PEAK_HALF_WIDTH_CM1,
    auxiliary_max_weight: float = DEFAULT_AUXILIARY_PRIOR_MAX_WEIGHT,
    shared_core_weight: float = DEFAULT_SHARED_CORE_PEAK_WEIGHT,
    top_k: int = 4,
    min_peak_distance_cm1: float = 20.0,
    peak_prominence: float = 0.02,
    auxiliary_peak_half_width_cm1: float = 15.0,
    min_matched_pairs: int = 4,
    shared_peak_floor: float = 0.35,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Combine manually specified chemistry core peaks with
    training-derived matched-condition auxiliary regions.

    Core peaks:
      DEL: 1000 and 1600 cm^-1
      CHL: 2230 cm^-1
      TEB: 1090 and 1597 cm^-1

    The current permitted peak displacement is +/-7 cm^-1.

    DEL ~1600 and TEB ~1597 are overlapping/shared evidence,
    so their core prior uses shared_core_weight rather than 1.0.

    matched_shared regions are only secondary data-driven
    discriminative regions and are not treated as chemically
    assigned characteristic peaks.
    """
    if core_half_width_cm1 <= 0.0:
        raise ValueError(
            "core_half_width_cm1 must be positive"
        )

    if not (
        0.0
        <= auxiliary_max_weight
        <= 1.0
    ):
        raise ValueError(
            "auxiliary_max_weight must be in [0, 1]"
        )

    if not (
        0.0
        <= shared_core_weight
        <= 1.0
    ):
        raise ValueError(
            "shared_core_weight must be in [0, 1]"
        )

    excluded_intervals = (
        tuple(_chemistry_core_intervals(core_half_width_cm1))
        + ((2000.000001, float(MODEL_RAMAN_AXIS[-1])),)
    )

    auxiliary_prior, auxiliary_summary = (
        build_matched_condition_peak_priors(
            repository,
            top_k=top_k,
            min_peak_distance_cm1=min_peak_distance_cm1,
            peak_prominence=peak_prominence,
            peak_half_width_cm1=auxiliary_peak_half_width_cm1,
            min_matched_pairs=min_matched_pairs,
            shared_peak_floor=shared_peak_floor,
            excluded_intervals_cm1=excluded_intervals,
        )
    )

    core_prior = np.zeros(
        (len(PESTICIDES), MODEL_LENGTH),
        dtype=np.float32,
    )

    hybrid_prior = np.zeros_like(
        core_prior
    )

    pesticide_summary: dict[str, Any] = {}

    for pesticide_index, pesticide in enumerate(
        PESTICIDES
    ):
        core_entries: list[dict[str, Any]] = []

        for item in CHEMISTRY_CORE_PEAKS[pesticide]:
            center = float(
                item["center_cm-1"]
            )

            role = str(
                item["role"]
            )

            weight = float(
                shared_core_weight
                if role == "shared"
                else 1.0
            )

            component = _truncated_core_gaussian(
                center_cm1=center,
                half_width_cm1=core_half_width_cm1,
                weight=weight,
            )

            core_prior[pesticide_index] = np.maximum(
                core_prior[pesticide_index],
                component,
            )

            core_entries.append(
                {
                    "center_cm-1": center,
                    "window_cm-1": [
                        center
                        - float(core_half_width_cm1),
                        center
                        + float(core_half_width_cm1),
                    ],
                    "role": role,
                    "prior_weight": weight,
                }
            )

        scaled_auxiliary = (
            float(auxiliary_max_weight)
            * auxiliary_prior[pesticide_index]
        ).astype(np.float32)

        hybrid_prior[pesticide_index] = np.maximum(
            core_prior[pesticide_index],
            scaled_auxiliary,
        )

        aux_info = auxiliary_summary[
            "pesticides"
        ][pesticide]

        auxiliary_centers = [
            float(value)
            for value in aux_info[
                "peak_centers_cm-1"
            ]
        ]

        core_centers = [
            float(item["center_cm-1"])
            for item in core_entries
        ]

        pesticide_summary[pesticide] = {
            "core_peaks": core_entries,
            "core_peak_centers_cm-1": (
                core_centers
            ),
            "auxiliary_peak_centers_cm-1": (
                auxiliary_centers
            ),
            "peak_centers_cm-1": (
                core_centers
                + auxiliary_centers
            ),
            "matched_pair_count": int(
                aux_info[
                    "matched_pair_count"
                ]
            ),
            "matched_pairs": aux_info[
                "matched_pairs"
            ],
            "auxiliary_selected_evidence": (
                aux_info[
                    "selected_evidence"
                ]
            ),
        }

    summary: dict[str, Any] = {
        "method": (
            "chemistry_core_plus_"
            "matched_auxiliary"
        ),
        "fit_source": (
            "chemistry core centers are user-specified; "
            "auxiliary regions use real training mappings "
            "only; validation/test/generated excluded"
        ),
        "model_axis_cm-1": [
            float(MODEL_RAMAN_AXIS[0]),
            float(MODEL_RAMAN_AXIS[-1]),
        ],
        "parameters": {
            "core_half_width_cm-1": float(
                core_half_width_cm1
            ),
            "auxiliary_max_weight": float(
                auxiliary_max_weight
            ),
            "shared_core_weight": float(
                shared_core_weight
            ),
            "auxiliary_top_k": int(
                top_k
            ),
            "auxiliary_peak_half_width_cm-1": float(
                auxiliary_peak_half_width_cm1
            ),
        },
        "interpretation": (
            "Chemically specified core peaks are primary "
            "soft anchors. Matched-condition peaks are "
            "secondary data-driven discriminative regions "
            "and are not asserted to be chemically assigned "
            "characteristic peaks."
        ),
        "pesticides": pesticide_summary,
        "auxiliary_summary": auxiliary_summary,
    }

    return hybrid_prior, summary


# Fixed per-condition generated subset, seed=2026 (random24-v1).
def _generated_sample_indices(condition, number, maximum, seed=2026):
    if maximum is None:
        return list(range(number))
    maximum = int(maximum)
    if maximum < 0:
        raise ValueError("maximum_generated_per_condition must be non-negative")
    if maximum >= number:
        return list(range(number))
    # A separate stable stream for each full condition; independent of file order.
    digest = hashlib.sha256(f"{seed}\0{condition}".encode("utf-8")).digest()
    condition_seed = int.from_bytes(digest[:16], "big")
    rng = np.random.Generator(np.random.PCG64(condition_seed))
    # Prefixes of the same permutation support nested 12/24/48 subset comparisons.
    selected = rng.permutation(number)[:maximum]
    return sorted(int(index) for index in selected)


class SERSDataset(Dataset):
    """
    Real-only validation/test; generated spectra may be added only to training.
    All samples returned to the model use the full 600-2500 cm^-1 axis (1901 points).
    """

    def __init__(
        self,
        repository: SERSDataRepository,
        split: str,
        include_generated: bool = False,
        maximum_generated_per_condition: int | None = None,
        require_complete_generated: bool = True,
    ) -> None:
        super().__init__()

        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split: {split}")
        if include_generated and split != "train":
            raise ValueError("Generated spectra are allowed in training only")

        self.repository = repository
        self.split = split
        self.include_generated = include_generated
        self.maximum_generated_per_condition = maximum_generated_per_condition
        self.samples: list[SampleRef] = []

        split_indices = {
            "train": REAL_TRAIN_INDICES,
            "validation": REAL_VALIDATION_INDICES,
            "test": REAL_TEST_INDICES,
        }[split]

        for condition in sorted(repository.real_data):
            for spectrum_index in split_indices:
                self.samples.append(
                    SampleRef(
                        source="real",
                        condition=condition,
                        spectrum_index=int(spectrum_index),
                    )
                )

        if include_generated:
            repository.load_generated_data(require_complete=require_complete_generated)
            assert repository.generated_data is not None
            for condition in sorted(repository.generated_data):
                number = int(repository.generated_data[condition]["spectra"].shape[1])
                selected_indices = _generated_sample_indices(
                    condition, number, maximum_generated_per_condition, seed=2026
                )
                for spectrum_index in selected_indices:
                    self.samples.append(
                        SampleRef(
                            source="generated",
                            condition=condition,
                            spectrum_index=spectrum_index,
                        )
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        ref = self.samples[index]
        real_record = self.repository.real_data[ref.condition]
        condition_info: ConditionInfo = real_record["condition_info"]

        if ref.source == "real":
            axis = real_record["axis"]
            spectrum = real_record["spectra"][:, ref.spectrum_index]
            source_path = real_record["path"]
            spectrum_name = real_record["names"][ref.spectrum_index]
            adapted, valid_mask = _adapt_spectrum_to_model_axis(axis, spectrum)
        else:
            assert self.repository.generated_data is not None
            generated_record = self.repository.generated_data[ref.condition]
            axis = generated_record["axis"]
            spectrum = generated_record["spectra"][:, ref.spectrum_index]
            source_path = generated_record["path"]
            spectrum_name = generated_record["names"][ref.spectrum_index]

            # Both sources are completed in memory to the same model range.
            # Respect any repository-level allowed_mask without discarding
            # measured data merely because another source was shorter.
            adapted, valid_mask = _adapt_spectrum_to_model_axis(
                axis,
                spectrum,
                allowed_mask=real_record["valid_mask"],
            )

        raw, percentile, smoothed = _preprocess_spectrum(adapted, valid_mask)

        return {
            "raw": torch.from_numpy(raw).float(),
            "raw_intensity": torch.from_numpy(adapted[np.newaxis, :].copy()).float(),
            "percentile": torch.from_numpy(percentile).float(),
            "smoothed": torch.from_numpy(smoothed).float(),
            "valid_mask": torch.from_numpy(valid_mask.copy()).bool(),
            "raman_axis": torch.from_numpy(MODEL_RAMAN_AXIS.copy()).float(),
            "class_target": torch.from_numpy(condition_info.presence.copy()).float(),
            "concentration_target": torch.from_numpy(
                condition_info.concentration_target.copy()
            ).float(),
            "concentration_level": torch.from_numpy(
                condition_info.level_codes.copy()
            ).float(),
            "matrix_target": torch.tensor(condition_info.matrix_code, dtype=torch.long),
            "condition": ref.condition,
            "matrix_name": condition_info.matrix_name,
            "source": ref.source,
            "source_file": str(source_path),
            "spectrum_name": spectrum_name,
            "spectrum_index": int(ref.spectrum_index),
            "target_mode": self.repository.target_mode,
        }


def collect_real_training_intensities(dataset: SERSDataset) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """T3.13: normalization input from fixed real training rows, never generated.

    Source-axis adaptation follows the existing dataset path; this helper does
    not alter the split, file contents, tail policy or the three legacy inputs.
    """
    if dataset.split != "train":
        raise ValueError("Local quantitative normalization accepts the training dataset only")
    rows: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    per_condition: dict[str, int] = {}
    seen: set[tuple[str, int]] = set()
    for ref in dataset.samples:
        if ref.source != "real":
            continue
        key = (ref.condition, int(ref.spectrum_index))
        if ref.spectrum_index not in REAL_TRAIN_INDICES or key in seen:
            raise RuntimeError("Local normalization received a non-training or duplicate real row")
        seen.add(key)
        record = dataset.repository.real_data[ref.condition]
        values, valid = _adapt_spectrum_to_model_axis(record["axis"], record["spectra"][:, ref.spectrum_index])
        rows.append(values[np.newaxis, :])
        masks.append(valid)
        per_condition[ref.condition] = per_condition.get(ref.condition, 0) + 1
    if (not rows or set(per_condition) != set(dataset.repository.real_data)
            or any(count != len(REAL_TRAIN_INDICES) for count in per_condition.values())):
        raise RuntimeError("Local normalization requires exactly 12 real training rows per condition")
    return np.stack(rows).astype(np.float32), np.stack(masks), {
        "real_training_rows": len(rows), "conditions": len(per_condition),
        "spectra_per_condition": len(REAL_TRAIN_INDICES),
        "generated_rows_used": 0, "validation_rows_used": 0, "test_rows_used": 0,
    }


def build_datasets(
    include_generated_train: bool = False,
    maximum_generated_per_condition: int | None = None,
    require_complete_generated: bool = True,
    concentration_map: Mapping[str, Mapping[str, float]] | None = None,
    real_root: str | Path = REAL_DATA_ROOT,
    generated_root: str | Path = GENERATED_DATA_ROOT,
) -> tuple[SERSDataset, SERSDataset, SERSDataset]:
    repository = SERSDataRepository(
        real_root=real_root,
        generated_root=generated_root,
        concentration_map=concentration_map,
    )

    train = SERSDataset(
        repository,
        split="train",
        include_generated=include_generated_train,
        maximum_generated_per_condition=maximum_generated_per_condition,
        require_complete_generated=require_complete_generated,
    )
    validation = SERSDataset(repository, split="validation", include_generated=False)
    test = SERSDataset(repository, split="test", include_generated=False)
    return train, validation, test
