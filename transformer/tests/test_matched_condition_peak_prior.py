from types import SimpleNamespace

import numpy as np

from Dataset import (
    MODEL_LENGTH,
    MODEL_RAMAN_AXIS,
    ConditionInfo,
    build_matched_condition_peak_priors,
)


def _gaussian(center: float, width: float = 5.0) -> np.ndarray:
    return np.exp(-0.5 * ((MODEL_RAMAN_AXIS - center) / width) ** 2).astype(np.float32)


def _condition_record(condition: str, levels: tuple[int, int, int]) -> dict:
    base = 0.05 + 0.00001 * (MODEL_RAMAN_AXIS - 600.0)
    spectrum = base.astype(np.float32)

    # Pesticide-specific regions plus a deliberately shared region at 1200 cm^-1.
    if levels[0] > 0:
        spectrum = spectrum + (1.5 + 0.2 * levels[0]) * _gaussian(1000.0)
        spectrum = spectrum + 0.55 * _gaussian(1200.0)
    if levels[1] > 0:
        spectrum = spectrum + (1.4 + 0.2 * levels[1]) * _gaussian(1300.0)
        spectrum = spectrum + 0.55 * _gaussian(1200.0)
    if levels[2] > 0:
        spectrum = spectrum + (1.6 + 0.2 * levels[2]) * _gaussian(1090.0)
        spectrum = spectrum + 0.55 * _gaussian(1200.0)

    # 20 deterministic mapping spectra from the same source condition.
    columns = []
    for index in range(20):
        scale = 1.0 + 0.002 * (index - 9.5)
        columns.append((spectrum * scale).astype(np.float32))
    spectra = np.stack(columns, axis=1)

    presence = (np.asarray(levels) > 0).astype(np.float32)
    level_codes = np.asarray(levels, dtype=np.float32)
    info = ConditionInfo(
        condition=condition,
        presence=presence,
        level_codes=level_codes,
        concentration_target=level_codes.copy(),
        matrix_name="water",
        matrix_code=0,
    )
    return {
        "axis": MODEL_RAMAN_AXIS.astype(np.float64),
        "spectra": spectra,
        "condition_info": info,
    }


def test_matched_condition_prior_finds_specific_and_retains_shared_regions():
    # Six pairwise conditions are sufficient to build matched comparisons for
    # each target pesticide without requiring an all-absent blank.
    records = {
        "DEL-M_water": _condition_record("DEL-M_water", (2, 0, 0)),
        "CHL-M_water": _condition_record("CHL-M_water", (0, 2, 0)),
        "TEB-M_water": _condition_record("TEB-M_water", (0, 0, 2)),
        "DEL-M_CHL-M_water": _condition_record("DEL-M_CHL-M_water", (2, 2, 0)),
        "DEL-M_TEB-M_water": _condition_record("DEL-M_TEB-M_water", (2, 0, 2)),
        "CHL-M_TEB-M_water": _condition_record("CHL-M_TEB-M_water", (0, 2, 2)),
    }
    repository = SimpleNamespace(real_data=records)

    priors, summary = build_matched_condition_peak_priors(
        repository,
        top_k=4,
        peak_prominence=0.005,
        min_matched_pairs=2,
        shared_peak_floor=0.35,
    )

    assert priors.shape == (3, MODEL_LENGTH)
    assert np.isfinite(priors).all()
    assert (priors >= 0.0).all()
    assert summary["method"] == "matched_condition_shared_aware"

    expected = {"DEL": 1000.0, "CHL": 1300.0, "TEB": 1090.0}
    for pesticide_index, pesticide in enumerate(("DEL", "CHL", "TEB")):
        centers = np.asarray(
            summary["pesticides"][pesticide]["peak_centers_cm-1"],
            dtype=float,
        )
        assert np.min(np.abs(centers - expected[pesticide])) <= 20.0
        assert summary["pesticides"][pesticide]["matched_pair_count"] >= 2

        shared_index = int(np.argmin(np.abs(MODEL_RAMAN_AXIS - 1200.0)))
        assert priors[pesticide_index, shared_index] > 0.0


def test_shared_peak_floor_validation():
    repository = SimpleNamespace(real_data={})
    try:
        build_matched_condition_peak_priors(repository, shared_peak_floor=1.1)
    except ValueError as exc:
        assert "shared_peak_floor" in str(exc)
    else:
        raise AssertionError("Expected shared_peak_floor validation error")
