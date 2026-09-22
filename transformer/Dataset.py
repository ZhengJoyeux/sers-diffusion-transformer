"""
SERSFormer2 T0 dataset adapter for the DEL/CHL/TEB SERS project.

Engineering adaptations only (not paper-method innovations):
- read real spectra from data/real (symlink to ~/project/data/input)
- read D4.24 generated spectra from data/generated
- fixed 12/4/4 real split per source file
- generated spectra are training-only
- unify all model inputs to the common measured range 600-2000 cm^-1 (1401 points)
- keep valid_mask support for safety, while preventing Raman-length/CHL label leakage
- preserve negative baseline-corrected SERS values with signed-log1p preprocessing

The output order of pesticide targets is always: [DEL, CHL, TEB].
"""

from __future__ import annotations

from dataclasses import dataclass
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

MODEL_RAMAN_AXIS = np.arange(600.0, 2000.0 + 1.0, 1.0, dtype=np.float32)
MODEL_LENGTH = int(MODEL_RAMAN_AXIS.size)
if MODEL_LENGTH != 1401:
    raise RuntimeError(f"Unexpected common-axis Raman length: {MODEL_LENGTH}")

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
      raw        : [1, 1401], valid-region min-max after signed-log1p
      percentile : [4, 1401], 95/85/75/50 percentile feature channels
      smoothed   : [1, 1401], Hann-smoothed signed-log1p signal

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

        self.real_data: dict[str, dict[str, Any]] = {}
        self.generated_data: dict[str, dict[str, Any]] | None = None
        self._load_real_data()

    def _load_real_data(self) -> None:
        files = _collect_table_files(self.real_root)
        if len(files) != 126:
            raise RuntimeError(f"Expected 126 real source files, found {len(files)}")

        for path in files:
            condition = _condition_from_real_path(path)
            if condition in self.real_data:
                raise RuntimeError(f"Duplicate real condition: {condition}")

            axis, spectra, names = _read_table(path)
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
                "valid_mask": _mask_for_source_axis(axis),
                "condition_info": info,
            }

    def load_generated_data(self, require_complete: bool = True) -> None:
        if self.generated_data is not None:
            if require_complete and len(self.generated_data) != 126:
                raise RuntimeError(
                    f"Generated set incomplete: {len(self.generated_data)}/126 conditions"
                )
            return

        files = _collect_table_files(self.generated_root)
        generated: dict[str, dict[str, Any]] = {}

        for path in files:
            condition = _condition_from_generated_path(path)
            if condition not in self.real_data:
                raise RuntimeError(
                    f"Generated condition has no matching real condition: {condition}"
                )
            if condition in generated:
                raise RuntimeError(f"Duplicate generated condition: {condition}")

            axis, spectra, names = _read_table(path)
            generated[condition] = {
                "path": path,
                "axis": axis,
                "spectra": spectra,
                "names": names,
            }

        if require_complete and len(generated) != 126:
            raise RuntimeError(f"Generated set incomplete: {len(generated)}/126 conditions")

        self.generated_data = generated



def build_axis_label_audit(repository: SERSDataRepository) -> dict[str, Any]:
    """Describe source-axis length versus pesticide presence before common-axis cropping.

    This audit is intentionally based on the original source axes. It documents
    acquisition/label confounding without exposing that information to the model.
    """
    counts: dict[str, dict[str, int]] = {pesticide: {} for pesticide in PESTICIDES}
    axis_lengths: dict[int, int] = {}

    for record in repository.real_data.values():
        source_length = int(len(record["axis"]))
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
            "All model inputs are cropped/interpolated to the common 600-2000 cm^-1 "
            "range, so source-axis length and the 2000-2500 cm^-1 acquisition tail "
            "cannot be used as a pesticide-class cue."
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

    The prior is not a hard spectral mask. It is a [3, 1401] non-negative
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


class SERSDataset(Dataset):
    """
    Real-only validation/test; generated spectra may be added only to training.
    All samples returned to the model use the common 600-2000 cm^-1 axis (1401 points).
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
                if maximum_generated_per_condition is not None:
                    number = min(number, int(maximum_generated_per_condition))
                for spectrum_index in range(number):
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

            # Generated files may extend to 2500 cm^-1, but the downstream
            # model deliberately uses only the common 600-2000 cm^-1 range so
            # Raman-axis length cannot leak the CHL label.
            adapted, valid_mask = _adapt_spectrum_to_model_axis(
                axis,
                spectrum,
                allowed_mask=real_record["valid_mask"],
            )

        raw, percentile, smoothed = _preprocess_spectrum(adapted, valid_mask)

        return {
            "raw": torch.from_numpy(raw).float(),
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
