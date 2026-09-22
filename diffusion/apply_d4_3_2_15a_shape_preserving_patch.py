#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
from pathlib import Path
import sys
import yaml

ROOT = Path.cwd()
GENERATOR = ROOT / "src/spectrum_generator.py"
GEN_SCRIPT = ROOT / "scripts/generate_conditional_spectra.py"
BASE_CONFIG = ROOT / "config/ddpm_training_d4_3_2_15_raw_generation.yaml"
NEW_CONFIG = ROOT / "config/ddpm_training_d4_3_2_15a_shape_preserving_generation.yaml"
TEST_FILE = ROOT / "tests/test_d4_3_2_15a_shape_preserving_guard.py"
PATCH_FILE = ROOT / "D4_3_2_15a_shape_preserving_negative_valley.patch"


def require_text(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f"缺少文件：{path}")
    return path.read_text(encoding="utf-8")


def insert_before_once(text: str, anchor: str, insertion: str, label: str) -> str:
    count = text.count(anchor)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected 1 anchor, got {count}. 为避免误改，尚未写入任何文件。"
        )
    return text.replace(anchor, insertion + anchor, 1)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected 1 match, got {count}. 为避免误改，尚未写入任何文件。"
        )
    return text.replace(old, new, 1)


HELPER = r'''def _apply_condition_pointwise_shape_preserving_negative_valley_guard(
    generated_spectra: np.ndarray,
    training_spectra: np.ndarray,
    *,
    configuration: dict,
    raman_shift: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    # Repair coherent unsupported negative valleys while preserving shape.

    generated, training = _condition_arrays(
        generated_spectra,
        training_spectra,
    )

    if generated.shape[1] < 3:
        raise ValueError("保形负谷保护要求至少3个Raman点。")

    if raman_shift is not None:
        axis = np.asarray(raman_shift, dtype=np.float64).reshape(-1)
        if axis.size != generated.shape[1]:
            raise ValueError("raman_shift长度与生成光谱长度不一致。")
        if not np.isfinite(axis).all() or not np.all(np.diff(axis) > 0.0):
            raise ValueError("raman_shift必须是严格递增有限数轴。")

    lower_quantile = float(
        configuration.get("pointwise_negative_lower_quantile", 0.10)
    )
    reference_quantile = float(
        configuration.get("pointwise_negative_reference_quantile", 0.25)
    )
    minimum_margin = float(
        configuration.get("pointwise_negative_minimum_margin_intensity", 2.0)
    )
    maximum_margin = float(
        configuration.get("pointwise_negative_maximum_margin_intensity", 8.0)
    )
    activation_maximum = float(
        configuration.get("activation_maximum_intensity", -18.0)
    )
    minimum_deficit = float(
        configuration.get("minimum_deficit_intensity", 5.0)
    )
    minimum_contiguous_points = int(
        configuration.get("minimum_negative_valley_contiguous_points", 3)
    )
    raw_retention = float(
        configuration.get("shape_preserving_raw_retention", 0.20)
    )
    maximum_modified_fraction = float(
        configuration.get("maximum_modified_fraction", 0.05)
    )

    if not 0.0 <= lower_quantile < reference_quantile <= 0.50:
        raise ValueError(
            "pointwise_negative_lower_quantile和"
            "pointwise_negative_reference_quantile配置无效。"
        )
    if (
        not np.isfinite(minimum_margin)
        or not np.isfinite(maximum_margin)
        or minimum_margin < 0.0
        or maximum_margin < minimum_margin
    ):
        raise ValueError("逐Raman负尾margin配置无效。")
    if not np.isfinite(activation_maximum):
        raise ValueError("activation_maximum_intensity必须是有限数。")
    if not np.isfinite(minimum_deficit) or minimum_deficit < 0.0:
        raise ValueError("minimum_deficit_intensity必须是非负有限数。")
    if not 1 <= minimum_contiguous_points <= 31:
        raise ValueError(
            "minimum_negative_valley_contiguous_points必须位于[1,31]。"
        )
    if not np.isfinite(raw_retention) or not 0.0 < raw_retention < 1.0:
        raise ValueError("shape_preserving_raw_retention必须位于(0,1)。")
    if not 0.0 < maximum_modified_fraction <= 0.20:
        raise ValueError("maximum_modified_fraction必须位于(0,0.20]。")

    train_lower = np.quantile(training, lower_quantile, axis=0)
    train_reference = np.quantile(training, reference_quantile, axis=0)
    train_minimum = np.min(training, axis=0)

    lower_tail_spread = np.maximum(train_reference - train_lower, 0.0)
    pointwise_margin = np.clip(
        lower_tail_spread,
        minimum_margin,
        maximum_margin,
    )

    # 逐Raman训练下边界；如果training全为正值，也至少允许到-18附近的普通负噪声。
    pointwise_floor = np.minimum(
        train_minimum - pointwise_margin,
        activation_maximum,
    )

    deficit = pointwise_floor[np.newaxis, :] - generated
    candidate = (
        (generated < activation_maximum)
        & (deficit > minimum_deficit)
    )

    # 只保留连续负谷；单点极端噪声不在本模块处理。
    retained = np.zeros_like(candidate, dtype=bool)
    for spectrum_index in range(candidate.shape[0]):
        row = candidate[spectrum_index]
        padded = np.pad(row.astype(np.int8), (1, 1), constant_values=0)
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        stops = np.flatnonzero(changes == -1)
        for start, stop in zip(starts, stops, strict=True):
            if int(stop - start) >= minimum_contiguous_points:
                retained[spectrum_index, start:stop] = True

    modified_point_count = int(np.sum(retained))
    modified_fraction = modified_point_count / float(generated.size)
    if modified_fraction > maximum_modified_fraction:
        raise RuntimeError(
            "保形负谷修正点比例超过安全上限："
            f"{modified_fraction:.6%} > {maximum_modified_fraction:.6%}。"
            "这说明异常不是少数局部深谷，应停止导出并检查模型。"
        )

    corrected = generated.copy()
    if modified_point_count:
        # 20%保留RAW谷形，80%向training逐点下边界靠近；不会tanh饱和成水平地板。
        blended = (
            raw_retention * generated
            + (1.0 - raw_retention) * pointwise_floor[np.newaxis, :]
        )
        corrected[retained] = blended[retained]

    difference = corrected - generated
    diagnostics: dict[str, float | int | bool | str] = {
        "enabled": True,
        "mode": "pointwise_shape_preserving_negative_valley",
        "training_count": int(training.shape[0]),
        "generated_count": int(generated.shape[0]),
        "pointwise_negative_lower_quantile": lower_quantile,
        "pointwise_negative_reference_quantile": reference_quantile,
        "pointwise_negative_minimum_margin_intensity": minimum_margin,
        "pointwise_negative_maximum_margin_intensity": maximum_margin,
        "activation_maximum_intensity": activation_maximum,
        "minimum_deficit_intensity": minimum_deficit,
        "minimum_negative_valley_contiguous_points": int(minimum_contiguous_points),
        "shape_preserving_raw_retention": raw_retention,
        "negative_valley_candidate_point_count": int(np.sum(candidate)),
        "negative_valley_retained_point_count": modified_point_count,
        "modified_point_count": modified_point_count,
        "modified_point_fraction": float(modified_fraction),
        "modified_spectrum_count": int(np.sum(np.any(retained, axis=1))),
        "minimum_before": float(np.min(generated)),
        "minimum_after": float(np.min(corrected)),
        "pointwise_floor_minimum": float(np.min(pointwise_floor)),
        "pointwise_floor_median": float(np.median(pointwise_floor)),
        "correction_rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "maximum_absolute_correction": float(np.max(np.abs(difference))),
    }
    return corrected.astype(np.float32, copy=False), diagnostics


'''

DISPATCH_OLD = '''    if mode == "local_residual_negative_valley":
        return _apply_condition_local_residual_negative_valley_guard(
            generated,
            training,
            configuration=configuration,
            raman_shift=raman_shift,
        )
    if mode != "legacy_intensity_envelope":
        raise ValueError(
            "intensity_envelope_guard.mode必须为"
            "legacy_intensity_envelope或"
            "local_residual_negative_valley。"
        )
'''

DISPATCH_NEW = '''    if mode == "local_residual_negative_valley":
        return _apply_condition_local_residual_negative_valley_guard(
            generated,
            training,
            configuration=configuration,
            raman_shift=raman_shift,
        )
    if mode == "pointwise_shape_preserving_negative_valley":
        return _apply_condition_pointwise_shape_preserving_negative_valley_guard(
            generated,
            training,
            configuration=configuration,
            raman_shift=raman_shift,
        )
    if mode != "legacy_intensity_envelope":
        raise ValueError(
            "intensity_envelope_guard.mode必须为"
            "legacy_intensity_envelope、"
            "local_residual_negative_valley或"
            "pointwise_shape_preserving_negative_valley。"
        )
'''

TEST_TEXT = '''import numpy as np

from src.spectrum_generator import apply_condition_intensity_envelope_guard


def _configuration() -> dict:
    return {
        "mode": "pointwise_shape_preserving_negative_valley",
        "pointwise_negative_lower_quantile": 0.10,
        "pointwise_negative_reference_quantile": 0.25,
        "pointwise_negative_minimum_margin_intensity": 2.0,
        "pointwise_negative_maximum_margin_intensity": 8.0,
        "activation_maximum_intensity": -18.0,
        "minimum_deficit_intensity": 5.0,
        "minimum_negative_valley_contiguous_points": 3,
        "shape_preserving_raw_retention": 0.20,
        "maximum_modified_fraction": 0.05,
    }


def test_shape_preserving_guard_repairs_coherent_valley_without_flat_floor():
    rng = np.random.default_rng(2026)
    training = rng.normal(loc=2.0, scale=1.0, size=(12, 80)).astype(np.float64)
    generated = np.repeat(
        np.median(training, axis=0, keepdims=True),
        20,
        axis=0,
    )
    generated[3, 30:37] = np.asarray(
        [-35.0, -45.0, -60.0, -75.0, -60.0, -45.0, -35.0],
        dtype=np.float64,
    )
    before = generated.copy()

    corrected, diagnostics = apply_condition_intensity_envelope_guard(
        generated,
        training,
        configuration=_configuration(),
        raman_shift=np.arange(80, dtype=np.float64),
    )

    assert diagnostics["mode"] == "pointwise_shape_preserving_negative_valley"
    assert diagnostics["modified_spectrum_count"] == 1
    assert diagnostics["negative_valley_retained_point_count"] == 7
    assert np.min(corrected[3, 30:37]) > np.min(before[3, 30:37])
    corrected_valley = corrected[3, 30:37]
    assert float(np.ptp(corrected_valley)) > 1.0
    assert np.unique(np.round(corrected_valley, decimals=5)).size >= 5
    np.testing.assert_allclose(corrected[0], before[0], rtol=0.0, atol=0.0)


def test_shape_preserving_guard_preserves_small_noise_and_isolated_outlier():
    training = np.zeros((12, 60), dtype=np.float64)
    generated = np.zeros((20, 60), dtype=np.float64)
    generated[2, 20] = -80.0
    generated[3, 30:33] = np.asarray([-19.0, -20.0, -19.0], dtype=np.float64)
    before = generated.copy()

    corrected, diagnostics = apply_condition_intensity_envelope_guard(
        generated,
        training,
        configuration=_configuration(),
        raman_shift=np.arange(60, dtype=np.float64),
    )

    assert diagnostics["modified_point_count"] == 0
    np.testing.assert_allclose(corrected, before, rtol=0.0, atol=0.0)


def test_shape_preserving_guard_preserves_training_supported_negative_valley():
    training = np.zeros((12, 70), dtype=np.float64)
    for row_index in range(12):
        training[row_index, 40:46] = (
            -34.0
            - 0.25 * row_index
            + np.asarray([0.0, -2.0, -4.0, -3.0, -1.0, 0.0], dtype=np.float64)
        )

    generated = np.repeat(training[[5]], 20, axis=0)
    before = generated.copy()

    corrected, diagnostics = apply_condition_intensity_envelope_guard(
        generated,
        training,
        configuration=_configuration(),
        raman_shift=np.arange(70, dtype=np.float64),
    )

    assert diagnostics["modified_point_count"] == 0
    np.testing.assert_allclose(corrected, before, rtol=0.0, atol=0.0)
'''


def patch_generator(text: str) -> str:
    marker = "def _apply_condition_pointwise_shape_preserving_negative_valley_guard("
    if marker in text:
        raise RuntimeError("D4.3.2.15a helper已经存在，请不要重复应用补丁。")
    text = insert_before_once(
        text,
        "def apply_condition_intensity_envelope_guard(\n",
        HELPER,
        "insert shape-preserving helper",
    )
    return replace_once(
        text,
        DISPATCH_OLD,
        DISPATCH_NEW,
        "extend intensity guard dispatcher",
    )


def build_config(text: str) -> str:
    config = yaml.safe_load(text)
    if not isinstance(config, dict):
        raise RuntimeError("raw generation config不是字典。")

    config.setdefault("project", {})["name"] = "d4_3_2_15a_shape_preserving_generation"
    generation = config.setdefault("generation", {})
    generation["number_of_spectra"] = 50
    generation["model_source"] = "ema"
    generation["require_training_cross_fit_checkpoint"] = True

    for key in (
        "mean_fidelity_calibration",
        "pca_spread_calibration",
        "tail_calibration",
        "sampling_calibration",
    ):
        section = generation.get(key)
        if isinstance(section, dict):
            section["enabled"] = False

    generation["intensity_envelope_guard"] = {
        "enabled": True,
        "mode": "pointwise_shape_preserving_negative_valley",
        "pointwise_negative_lower_quantile": 0.10,
        "pointwise_negative_reference_quantile": 0.25,
        "pointwise_negative_minimum_margin_intensity": 2.0,
        "pointwise_negative_maximum_margin_intensity": 8.0,
        "activation_maximum_intensity": -18.0,
        "minimum_deficit_intensity": 5.0,
        "minimum_negative_valley_contiguous_points": 3,
        "shape_preserving_raw_retention": 0.20,
        "maximum_modified_fraction": 0.05,
    }

    output = config.setdefault("output", {})
    output["generated_spectrum_directory"] = (
        "outputs/experiments/"
        "d4_3_2_15_warmup_cosine_2h_formal/"
        "generated_d4_3_2_15a_shape_preserving"
    )
    if "generated_plot_directory" in output:
        output["generated_plot_directory"] = (
            "outputs/experiments/"
            "d4_3_2_15_warmup_cosine_2h_formal/"
            "plots_d4_3_2_15a_shape_preserving"
        )

    header = (
        "# D4.3.2.15a generation-only shape-preserving negative-valley guard\n"
        "# Training/checkpoint unchanged. Uses fixed training spectra only.\n"
    )
    return header + yaml.safe_dump(config, allow_unicode=True, sort_keys=False)


def unified(path: Path, old: str, new: str) -> str:
    rel = path.relative_to(ROOT).as_posix()
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true")
    group.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    generator_old = require_text(GENERATOR)
    generation_script = require_text(GEN_SCRIPT)
    base_config = require_text(BASE_CONFIG)

    if "raman_shift=output_axis" not in generation_script:
        raise RuntimeError(
            "generate_conditional_spectra.py尚未把output_axis传给guard；"
            "为避免基于旧代码误改，停止。"
        )

    generator_new = patch_generator(generator_old)
    config_new = build_config(base_config)

    if TEST_FILE.exists():
        raise RuntimeError(f"测试文件已存在：{TEST_FILE}")
    if NEW_CONFIG.exists():
        raise RuntimeError(f"新配置已存在：{NEW_CONFIG}")

    diff = unified(GENERATOR, generator_old, generator_new)
    diff += "".join(
        difflib.unified_diff(
            [],
            TEST_TEXT.splitlines(keepends=True),
            fromfile="/dev/null",
            tofile="b/" + TEST_FILE.relative_to(ROOT).as_posix(),
        )
    )
    diff += "".join(
        difflib.unified_diff(
            [],
            config_new.splitlines(keepends=True),
            fromfile="/dev/null",
            tofile="b/" + NEW_CONFIG.relative_to(ROOT).as_posix(),
        )
    )

    if args.check:
        print("CHECK PASSED")
        print("将修改：", GENERATOR.relative_to(ROOT))
        print("将新增：", TEST_FILE.relative_to(ROOT))
        print("将新增：", NEW_CONFIG.relative_to(ROOT))
        print("不会修改训练代码、checkpoint或RAW输出。")
        return 0

    GENERATOR.write_text(generator_new, encoding="utf-8")
    TEST_FILE.write_text(TEST_TEXT, encoding="utf-8")
    NEW_CONFIG.write_text(config_new, encoding="utf-8")
    PATCH_FILE.write_text(diff, encoding="utf-8")

    print("APPLY PASSED")
    print("修改：", GENERATOR.relative_to(ROOT))
    print("新增：", TEST_FILE.relative_to(ROOT))
    print("新增：", NEW_CONFIG.relative_to(ROOT))
    print("补丁记录：", PATCH_FILE.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
