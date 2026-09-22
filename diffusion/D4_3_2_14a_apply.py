#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
from pathlib import Path
import sys
import yaml

ROOT = Path.cwd()
GENERATOR = ROOT / 'src/spectrum_generator.py'
GEN_SCRIPT = ROOT / 'scripts/generate_conditional_spectra.py'
TEST_FILE = ROOT / 'tests/test_spectrum_generation.py'
BASE_CONFIG = ROOT / 'config/ddpm_training_d4_3_2_14_raw_generation.yaml'
NEW_CONFIG = ROOT / 'config/ddpm_training_d4_3_2_14_negative_guard_generation.yaml'
PATCH_FILE = ROOT / 'D4_3_2_14a_negative_valley_guard.patch'


def require(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f'缺少文件：{path}')
    return path.read_text(encoding='utf-8')


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f'{label}: expected 1 match, got {count}; '
            '为避免误改，尚未写入任何文件。'
        )
    return text.replace(old, new, 1)


def insert_before_once(text: str, anchor: str, insertion: str, label: str) -> str:
    count = text.count(anchor)
    if count != 1:
        raise RuntimeError(
            f'{label}: anchor expected 1 match, got {count}; '
            '为避免误改，尚未写入任何文件。'
        )
    return text.replace(anchor, insertion + anchor, 1)


HELPER = r'''def _apply_condition_local_residual_negative_valley_guard(
    generated_spectra: np.ndarray,
    training_spectra: np.ndarray,
    *,
    configuration: dict,
    raman_shift: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    """Repair only unsupported coherent negative valleys in local-residual space."""

    from scipy.ndimage import gaussian_filter1d

    generated, training = _condition_arrays(
        generated_spectra,
        training_spectra,
    )

    if generated.shape[1] < 3:
        raise ValueError("局部负谷保护要求至少3个Raman点。")

    if raman_shift is None:
        axis = np.arange(generated.shape[1], dtype=np.float64)
    else:
        axis = np.asarray(raman_shift, dtype=np.float64).reshape(-1)
        if axis.size != generated.shape[1]:
            raise ValueError("raman_shift长度与生成光谱长度不一致。")
        if not np.isfinite(axis).all() or not np.all(np.diff(axis) > 0.0):
            raise ValueError("raman_shift必须是严格递增有限数轴。")

    spacing = float(np.median(np.diff(axis)))
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("Raman轴间隔无效。")

    baseline_sigma_cm1 = float(
        configuration.get("local_residual_baseline_sigma_cm1", 30.0)
    )
    training_lower_quantile = float(
        configuration.get("local_residual_training_lower_quantile", 0.10)
    )
    training_iqr_margin = float(
        configuration.get("local_residual_training_iqr_margin", 0.50)
    )
    minimum_scale = float(
        configuration.get("local_residual_minimum_scale_intensity", 2.0)
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
    softness_fraction = float(
        configuration.get("local_residual_softness_scale_fraction", 0.20)
    )
    minimum_softness = float(
        configuration.get("minimum_softness_intensity", 0.25)
    )
    maximum_softness = float(
        configuration.get("maximum_softness_intensity", 2.0)
    )
    maximum_modified_fraction = float(
        configuration.get("maximum_modified_fraction", 0.05)
    )

    if not np.isfinite(baseline_sigma_cm1) or baseline_sigma_cm1 <= 0.0:
        raise ValueError("local_residual_baseline_sigma_cm1必须大于0。")
    if not 0.0 <= training_lower_quantile <= 0.25:
        raise ValueError("local_residual_training_lower_quantile必须位于[0,0.25]。")
    if not np.isfinite(training_iqr_margin) or training_iqr_margin < 0.0:
        raise ValueError("local_residual_training_iqr_margin必须是非负有限数。")
    if not np.isfinite(minimum_scale) or minimum_scale <= 0.0:
        raise ValueError("local_residual_minimum_scale_intensity必须大于0。")
    if not np.isfinite(activation_maximum):
        raise ValueError("activation_maximum_intensity必须是有限数。")
    if not np.isfinite(minimum_deficit) or minimum_deficit < 0.0:
        raise ValueError("minimum_deficit_intensity必须是非负有限数。")
    if not 1 <= minimum_contiguous_points <= 31:
        raise ValueError("minimum_negative_valley_contiguous_points必须位于[1,31]。")
    if not np.isfinite(softness_fraction) or softness_fraction < 0.0:
        raise ValueError("local_residual_softness_scale_fraction必须是非负有限数。")
    if (
        not np.isfinite(minimum_softness)
        or not np.isfinite(maximum_softness)
        or minimum_softness <= 0.0
        or maximum_softness < minimum_softness
    ):
        raise ValueError("负谷softness配置无效。")
    if not 0.0 < maximum_modified_fraction <= 0.20:
        raise ValueError("maximum_modified_fraction必须位于(0,0.20]。")

    sigma_samples = baseline_sigma_cm1 / spacing
    training_baseline = gaussian_filter1d(
        training,
        sigma=sigma_samples,
        axis=1,
        mode="nearest",
        truncate=4.0,
    )
    generated_baseline = gaussian_filter1d(
        generated,
        sigma=sigma_samples,
        axis=1,
        mode="nearest",
        truncate=4.0,
    )

    training_local = training - training_baseline
    generated_local = generated - generated_baseline

    train_q = np.quantile(training_local, training_lower_quantile, axis=0)
    train_q25 = np.quantile(training_local, 0.25, axis=0)
    train_q75 = np.quantile(training_local, 0.75, axis=0)
    train_iqr = np.maximum(train_q75 - train_q25, minimum_scale)

    local_floor = train_q - training_iqr_margin * train_iqr
    local_floor = np.minimum(local_floor, 0.0)

    local_deficit = local_floor[np.newaxis, :] - generated_local
    candidate = (
        (generated_local < local_floor[np.newaxis, :])
        & (generated < activation_maximum)
        & (local_deficit > minimum_deficit)
    )

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
            "局部负谷修正点比例超过安全上限："
            f"{modified_fraction:.6%} > {maximum_modified_fraction:.6%}。"
            "这说明异常不是少数局部深谷，应停止导出并检查模型。"
        )

    corrected = generated.copy()
    if modified_point_count:
        softness = np.clip(
            softness_fraction * train_iqr,
            minimum_softness,
            maximum_softness,
        )
        soft_local = (
            local_floor[np.newaxis, :]
            - softness[np.newaxis, :]
            * np.tanh(
                np.maximum(local_deficit, 0.0)
                / softness[np.newaxis, :]
            )
        )
        corrected_local = generated_local.copy()
        corrected_local[retained] = soft_local[retained]
        candidate_corrected = generated_baseline + corrected_local
        corrected[retained] = candidate_corrected[retained]

    difference = corrected - generated
    diagnostics: dict[str, float | int | bool | str] = {
        "enabled": True,
        "mode": "local_residual_negative_valley",
        "baseline_sigma_cm1": baseline_sigma_cm1,
        "raman_spacing_cm1": spacing,
        "training_lower_quantile": training_lower_quantile,
        "training_iqr_margin": training_iqr_margin,
        "activation_maximum_intensity": activation_maximum,
        "minimum_deficit_intensity": minimum_deficit,
        "minimum_negative_valley_contiguous_points": minimum_contiguous_points,
        "negative_valley_candidate_point_count": int(np.sum(candidate)),
        "negative_valley_retained_point_count": modified_point_count,
        "modified_point_count": modified_point_count,
        "modified_point_fraction": float(modified_fraction),
        "modified_spectrum_count": int(np.sum(np.any(retained, axis=1))),
        "minimum_before": float(np.min(generated)),
        "minimum_after": float(np.min(corrected)),
        "correction_rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "maximum_absolute_correction": float(np.max(np.abs(difference))),
        "local_floor_minimum": float(np.min(local_floor)),
        "local_floor_median": float(np.median(local_floor)),
    }
    return corrected.astype(np.float32, copy=False), diagnostics


'''


def patch_generator(text: str) -> str:
    text = insert_before_once(
        text,
        'def apply_condition_intensity_envelope_guard(\n',
        HELPER,
        'insert local-residual helper',
    )

    old_sig = '''def apply_condition_intensity_envelope_guard(
    generated_spectra: np.ndarray,
    training_spectra: np.ndarray,
    *,
    configuration: dict,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
'''
    new_sig = '''def apply_condition_intensity_envelope_guard(
    generated_spectra: np.ndarray,
    training_spectra: np.ndarray,
    *,
    configuration: dict,
    raman_shift: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
'''
    text = replace_once(text, old_sig, new_sig, 'extend guard signature')

    old_start = '''    generated, training = _condition_arrays(generated_spectra, training_spectra)
    if not isinstance(configuration, dict):
        raise ValueError("intensity_envelope_guard配置必须是字典。")

'''
    new_start = '''    generated, training = _condition_arrays(generated_spectra, training_spectra)
    if not isinstance(configuration, dict):
        raise ValueError("intensity_envelope_guard配置必须是字典。")

    mode = str(
        configuration.get("mode", "legacy_intensity_envelope")
    ).strip().lower()
    if mode == "local_residual_negative_valley":
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
    return replace_once(text, old_start, new_start, 'dispatch guard mode')


def patch_generation_script(text: str) -> str:
    old = '''                apply_condition_intensity_envelope_guard(
                    spectra,
                    condition_training,
                    configuration=envelope_configuration,
                )
'''
    new = '''                apply_condition_intensity_envelope_guard(
                    spectra,
                    condition_training,
                    configuration=envelope_configuration,
                    raman_shift=output_axis,
                )
'''
    return replace_once(text, old, new, 'pass output_axis to guard')


TESTS = r'''


def test_d4_3_2_14a_local_residual_guard_repairs_unsupported_coherent_valley():
    import numpy as np
    from src.spectrum_generator import apply_condition_intensity_envelope_guard

    axis = np.arange(1000.0, 1100.0, 1.0)
    x = np.arange(axis.size, dtype=np.float64)
    rows = []
    for index in range(12):
        broad = 120.0 + 18.0 * index + 0.015 * (x - 50.0) ** 2
        local = 1.5 * np.sin(2.0 * np.pi * x / 17.0 + 0.2 * index)
        rows.append(broad + local)
    training = np.asarray(rows, dtype=np.float64)
    generated = np.repeat(training[[5]], 20, axis=0)
    generated[3, 45:52] -= 85.0
    generated[4, 20] -= 8.0

    configuration = {
        "mode": "local_residual_negative_valley",
        "local_residual_baseline_sigma_cm1": 30.0,
        "local_residual_training_lower_quantile": 0.10,
        "local_residual_training_iqr_margin": 0.50,
        "local_residual_minimum_scale_intensity": 2.0,
        "activation_maximum_intensity": -18.0,
        "minimum_deficit_intensity": 5.0,
        "minimum_negative_valley_contiguous_points": 3,
        "local_residual_softness_scale_fraction": 0.20,
        "minimum_softness_intensity": 0.25,
        "maximum_softness_intensity": 2.0,
        "maximum_modified_fraction": 0.05,
    }
    corrected, diagnostics = apply_condition_intensity_envelope_guard(
        generated,
        training,
        configuration=configuration,
        raman_shift=axis,
    )
    assert diagnostics["mode"] == "local_residual_negative_valley"
    assert diagnostics["negative_valley_retained_point_count"] >= 3
    assert diagnostics["modified_spectrum_count"] == 1
    assert float(np.min(corrected[3, 45:52])) > float(np.min(generated[3, 45:52])) + 20.0
    np.testing.assert_allclose(corrected[0], generated[0], rtol=0.0, atol=0.0)
    assert corrected[4, 20] == generated[4, 20]
    np.testing.assert_allclose(corrected[3, :40], generated[3, :40], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(corrected[3, 60:], generated[3, 60:], rtol=0.0, atol=0.0)


def test_d4_3_2_14a_local_residual_guard_preserves_training_supported_valley():
    import numpy as np
    from src.spectrum_generator import apply_condition_intensity_envelope_guard

    axis = np.arange(1000.0, 1100.0, 1.0)
    x = np.arange(axis.size, dtype=np.float64)
    rows = []
    for index in range(12):
        broad = 80.0 + 4.0 * index + 0.01 * (x - 50.0) ** 2
        row = broad.copy()
        row[45:52] -= 62.0 + 0.5 * index
        rows.append(row)
    training = np.asarray(rows, dtype=np.float64)
    generated = np.repeat(training[[6]], 20, axis=0)
    before = generated.copy()
    configuration = {
        "mode": "local_residual_negative_valley",
        "local_residual_baseline_sigma_cm1": 30.0,
        "local_residual_training_lower_quantile": 0.10,
        "local_residual_training_iqr_margin": 0.50,
        "local_residual_minimum_scale_intensity": 2.0,
        "activation_maximum_intensity": -18.0,
        "minimum_deficit_intensity": 5.0,
        "minimum_negative_valley_contiguous_points": 3,
        "local_residual_softness_scale_fraction": 0.20,
        "minimum_softness_intensity": 0.25,
        "maximum_softness_intensity": 2.0,
        "maximum_modified_fraction": 0.05,
    }
    corrected, diagnostics = apply_condition_intensity_envelope_guard(
        generated,
        training,
        configuration=configuration,
        raman_shift=axis,
    )
    assert diagnostics["modified_point_count"] == 0
    np.testing.assert_allclose(corrected, before, rtol=0.0, atol=0.0)
'''


def patch_tests(text: str) -> str:
    marker = 'test_d4_3_2_14a_local_residual_guard_repairs_unsupported_coherent_valley'
    if marker in text:
        raise RuntimeError('D4.3.2.14a tests already present.')
    return text.rstrip() + TESTS + '\n'


def build_config(text: str) -> str:
    config = yaml.safe_load(text)
    if not isinstance(config, dict):
        raise RuntimeError('raw generation config is invalid.')
    generation = config['generation']
    generation['mean_fidelity_calibration']['enabled'] = False
    generation['pca_spread_calibration']['enabled'] = False
    generation['tail_calibration']['enabled'] = False
    generation['sampling_calibration']['enabled'] = False
    guard = generation.setdefault('intensity_envelope_guard', {})
    guard.update({
        'enabled': True,
        'mode': 'local_residual_negative_valley',
        'local_residual_baseline_sigma_cm1': 30.0,
        'local_residual_training_lower_quantile': 0.10,
        'local_residual_training_iqr_margin': 0.50,
        'local_residual_minimum_scale_intensity': 2.0,
        'activation_maximum_intensity': -18.0,
        'minimum_deficit_intensity': 5.0,
        'minimum_negative_valley_contiguous_points': 3,
        'local_residual_softness_scale_fraction': 0.20,
        'minimum_softness_intensity': 0.25,
        'maximum_softness_intensity': 2.0,
        'maximum_modified_fraction': 0.05,
    })
    generation['require_training_cross_fit_checkpoint'] = True
    root = (
        'outputs/experiments/'
        'd4_3_2_14_cross_fitted_pca_medium/'
        'generated_negative_guard_diagnostic'
    )
    config['output']['generated_spectrum_directory'] = root
    header = (
        '# D4.3.2.14a generation-only safety guard\n'
        '# checkpoint unchanged; local-residual negative valleys only\n'
    )
    return header + yaml.safe_dump(config, allow_unicode=True, sort_keys=False)


def unified(path: Path, old: str, new: str) -> str:
    rel = path.relative_to(ROOT).as_posix()
    return ''.join(difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f'a/{rel}',
        tofile=f'b/{rel}',
    ))


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true')
    group.add_argument('--apply', action='store_true')
    args = parser.parse_args()

    generator_old = require(GENERATOR)
    script_old = require(GEN_SCRIPT)
    tests_old = require(TEST_FILE)
    config_old = require(BASE_CONFIG)

    generator_new = patch_generator(generator_old)
    script_new = patch_generation_script(script_old)
    tests_new = patch_tests(tests_old)
    config_new = build_config(config_old)

    if NEW_CONFIG.exists() and NEW_CONFIG.read_text(encoding='utf-8') != config_new:
        raise RuntimeError(f'{NEW_CONFIG}已存在且内容不同；为避免覆盖，停止。')

    patch_text = (
        unified(GENERATOR, generator_old, generator_new)
        + unified(GEN_SCRIPT, script_old, script_new)
        + unified(TEST_FILE, tests_old, tests_new)
    )
    rel = NEW_CONFIG.relative_to(ROOT).as_posix()
    patch_text += ''.join(difflib.unified_diff(
        [],
        config_new.splitlines(keepends=True),
        fromfile='/dev/null',
        tofile=f'b/{rel}',
    ))

    print('===== D4.3.2.14a negative-valley guard =====')
    print('修改:')
    print(' -', GENERATOR)
    print(' -', GEN_SCRIPT)
    print(' -', TEST_FILE)
    print('创建:')
    print(' -', NEW_CONFIG)
    print(' -', PATCH_FILE)
    print('旧intensity-envelope默认行为保持不变；新配置启用local_residual_negative_valley。')
    print('训练与checkpoint不修改。')

    if args.check:
        print('CHECK PASSED; no files written.')
        return 0

    GENERATOR.write_text(generator_new, encoding='utf-8')
    GEN_SCRIPT.write_text(script_new, encoding='utf-8')
    TEST_FILE.write_text(tests_new, encoding='utf-8')
    NEW_CONFIG.write_text(config_new, encoding='utf-8')
    PATCH_FILE.write_text(patch_text, encoding='utf-8')
    print('APPLY PASSED.')
    print('config:', NEW_CONFIG)
    print('audit patch:', PATCH_FILE)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f'ERROR: {type(error).__name__}: {error}', file=sys.stderr)
        raise
