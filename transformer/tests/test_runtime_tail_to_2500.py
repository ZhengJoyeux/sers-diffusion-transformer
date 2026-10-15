import hashlib
from itertools import product

import numpy as np
import pandas as pd
import pytest

import Dataset


def _signal(axis, columns=2):
    return np.stack([
        2.0 + 0.002 * (axis - 600.0) + 0.3 * np.sin(axis / 3.0 + i)
        + 20.0 * np.exp(-0.5 * ((axis - 1090.0) / 4.0) ** 2)
        for i in range(columns)
    ], axis=1)


@pytest.mark.parametrize("end", [1800.0, 2000.0, 2100.0, 2300.0, 2499.0, 2499.5])
def test_any_missing_tail_preserves_measured_values_and_is_reproducible(end):
    axis = np.arange(600.0, np.floor(end) + 1)
    if end != np.floor(end):
        axis = np.append(axis, end)
    spectra = _signal(axis)
    axis_before, spectra_before = axis.copy(), spectra.copy()
    completed_axis, completed, imputed = Dataset._impute_missing_tail_to_2500_with_noise(
        axis, spectra, source_key="runtime-test",
    )
    assert imputed
    np.testing.assert_array_equal(axis, axis_before)
    np.testing.assert_array_equal(spectra, spectra_before)
    np.testing.assert_array_equal(completed_axis, Dataset.MODEL_RAMAN_AXIS)
    assert completed.shape == (1901, 2)
    measured = Dataset.MODEL_RAMAN_AXIS <= end
    for i in range(2):
        np.testing.assert_array_equal(
            completed[measured, i],
            np.interp(Dataset.MODEL_RAMAN_AXIS[measured], axis, spectra[:, i]).astype(np.float32),
        )
    assert np.isfinite(completed).all()
    again = Dataset._impute_missing_tail_to_2500_with_noise(axis, spectra, source_key="runtime-test")[1]
    np.testing.assert_array_equal(completed, again)


def test_full_axis_is_unchanged_and_completion_is_idempotent():
    short_axis = np.arange(600.0, 2301.0)
    axis, spectra, _ = Dataset._impute_missing_tail_to_2500_with_noise(
        short_axis, _signal(short_axis), source_key="idempotence",
    )
    actual_axis, actual_spectra, imputed = Dataset._impute_missing_tail_to_2500_with_noise(
        axis, spectra, source_key="different-key",
    )
    assert not imputed
    np.testing.assert_array_equal(actual_axis, axis)
    np.testing.assert_array_equal(actual_spectra, spectra)


def test_old_function_name_is_still_supported():
    axis = np.arange(600.0, 2001.0)
    old = Dataset._impute_missing_2000_2500_tail_with_noise(axis, _signal(axis), source_key="compat")
    new = Dataset._impute_missing_tail_to_2500_with_noise(axis, _signal(axis), source_key="compat")
    np.testing.assert_array_equal(old[1], new[1])


def test_direct_single_spectrum_adapter_fills_tail_for_inference():
    axis = np.arange(600.0, 2301.0)
    signal = _signal(axis, 1)[:, 0]
    adapted, valid = Dataset._adapt_spectrum_to_model_axis(axis, signal)
    assert valid.all()
    assert adapted.shape == (1901,)
    np.testing.assert_array_equal(adapted[:1701], signal.astype(np.float32))
    assert np.isfinite(adapted).all()
    assert np.std(adapted[1701:]) > 0
    np.testing.assert_array_equal(adapted, Dataset._adapt_spectrum_to_model_axis(axis, signal)[0])


@pytest.mark.parametrize("kind", ["descending", "nonfinite", "wrong_start", "too_short"])
def test_invalid_input_is_rejected(kind):
    axis = np.arange(600.0, 2101.0)
    if kind == "descending":
        axis = axis[::-1]
    elif kind == "nonfinite":
        axis[10] = np.nan
    elif kind == "wrong_start":
        axis = np.arange(700.0, 2101.0)
    else:
        axis = np.arange(600.0, 620.0)
    assert Dataset._classify_raw_axis_coverage(axis) == "other"
    with pytest.raises((ValueError, RuntimeError)):
        Dataset._impute_missing_tail_to_2500_with_noise(axis, _signal(axis), source_key="invalid")


def test_csv_audit_and_runtime_completion_do_not_write_source(tmp_path):
    axis = np.arange(600.0, 2201.0)
    path = tmp_path / "TEB-M_water.csv"
    pd.DataFrame(np.column_stack([axis, _signal(axis)]), columns=["Raman", "a", "b"]).to_csv(path, index=False)
    before = path.read_bytes()
    stamp = path.stat().st_mtime_ns
    audit = Dataset._audit_raw_input_root(tmp_path, "REAL")
    assert audit["short_axis_spectrum_count"] == 2
    assert audit["unexpected_files"] == []
    raw_axis, raw_spectra, _ = Dataset._read_table(path)
    Dataset._impute_missing_tail_to_2500_with_noise(raw_axis, raw_spectra, source_key=str(path))
    assert hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(before).digest()
    assert path.stat().st_mtime_ns == stamp


def test_real_and_generated_repository_and_all_splits_use_completed_axis(tmp_path, monkeypatch):
    real_root, gen_root = tmp_path / "real", tmp_path / "generated"
    real_root.mkdir()
    gen_root.mkdir()
    names = []
    for levels in product((None, "S", "M", "H"), repeat=3):
        if all(level is None for level in levels):
            continue
        condition = "_".join(f"{p}-{level}" for p, level in zip(Dataset.PESTICIDES, levels) if level)
        names.extend(condition + "_" + matrix for matrix in ("water", "soil"))
    assert len(names) == 126
    files = {
        real_root: [real_root / (n + ".csv") for n in names],
        gen_root: [gen_root / (n + "_generated.csv") for n in names],
    }
    original_arrays = {}
    for root, paths in files.items():
        for index, path in enumerate(paths):
            end = (2000.0, 2100.0, 2300.0, 2500.0)[index % 4]
            axis = np.arange(600.0, end + 1)
            columns = 20 if root == real_root else 2
            original_arrays[path] = (axis, _signal(axis, columns), [f"s{i}" for i in range(columns)])
    before = {p: (a.copy(), s.copy()) for p, (a, s, _) in original_arrays.items()}
    monkeypatch.setattr(Dataset, "_collect_table_files", lambda root: files[root])
    monkeypatch.setattr(Dataset, "_read_table", lambda path: original_arrays[path])
    repository = Dataset.SERSDataRepository(real_root, gen_root)
    repository.load_generated_data()
    assert repository.input_axis_audit["real"]["short_axis_file_count"] == 95
    for records in (repository.real_data, repository.generated_data):
        for record in records.values():
            assert record["spectra"].shape[0] == 1901
            assert record["axis"][-1] == 2500
    train = Dataset.SERSDataset(repository, "train", include_generated=True)
    validation = Dataset.SERSDataset(repository, "validation")
    test = Dataset.SERSDataset(repository, "test")
    assert len(train) == 1512 + 252
    assert len(validation) == len(test) == 504
    for dataset, index in ((train, 0), (train, 1512), (validation, 0), (test, 0)):
        item = dataset[index]
        assert item["raw"].shape == (1, 1901)
        assert item["valid_mask"].all()
    audit = Dataset.build_axis_label_audit(repository)
    assert len(audit["source_axis_point_counts"]) == 4
    for path, (axis, spectra, _) in original_arrays.items():
        np.testing.assert_array_equal(axis, before[path][0])
        np.testing.assert_array_equal(spectra, before[path][1])
