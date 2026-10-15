import numpy as np
import pytest

from src.spectrum_generator import (
    _protect_pairwise_diversity_after_calibration,
)


def _make_training():
    # 非退化的确定性测试数据。
    base = np.linspace(
        -1.0,
        1.0,
        12,
        dtype=np.float64,
    )[:, None]

    feature_scale = np.linspace(
        0.5,
        1.5,
        8,
        dtype=np.float64,
    )[None, :]

    return base * feature_scale


def _run_case(ratio_before):
    training = _make_training()

    # 所有pairwise MSE都会按幅值平方缩放，
    # 因此可以精确构造指定的pre-calibration ratio。
    original = (
        np.sqrt(ratio_before)
        * training
    )

    # 模拟一个会明显压缩多样性的QQ校准。
    calibrated = 0.20 * original

    protected, diagnostics = (
        _protect_pairwise_diversity_after_calibration(
            original,
            calibrated,
            training,
            minimum_ratio=0.86,
            search_iterations=24,
            minimum_calibration_blend_factor=0.0,
            minimum_precalibration_diversity_retention=0.95,
            pair_count=10000,
            random_seed=2026,
        )
    )

    assert protected.shape == original.shape

    return diagnostics


def test_when_precalibration_is_below_absolute_floor_use_relative_retention():
    diagnostics = _run_case(
        ratio_before=0.64,
    )

    # 0.64本来就低于0.86：
    # 不应该强制QQ后达到0.86，
    # 而应该保留原多样性的95%。
    expected = 0.64 * 0.95

    assert diagnostics[
        "pairwise_mse_ratio_before"
    ] == pytest.approx(
        0.64,
        abs=1.0e-10,
    )

    assert diagnostics[
        "pairwise_mse_ratio_required"
    ] == pytest.approx(
        expected,
        abs=1.0e-10,
    )

    assert diagnostics[
        "pairwise_mse_ratio_after"
    ] >= expected - 1.0e-6

    assert diagnostics[
        "pairwise_mse_ratio_after"
    ] <= 0.64 + 1.0e-6


def test_when_precalibration_is_above_absolute_floor_enforce_95_percent_retention():
    diagnostics = _run_case(
        ratio_before=1.0,
    )

    # 1.0已经高于0.86：
    # 应同时满足绝对floor和95% retention，
    # 因而required应为0.95，而不能被0.86截断。
    expected = 0.95

    assert diagnostics[
        "pairwise_mse_ratio_before"
    ] == pytest.approx(
        1.0,
        abs=1.0e-10,
    )

    assert diagnostics[
        "pairwise_mse_ratio_required"
    ] == pytest.approx(
        expected,
        abs=1.0e-10,
    )

    assert diagnostics[
        "pairwise_mse_ratio_after"
    ] >= expected - 1.0e-6

    assert diagnostics[
        "pairwise_mse_ratio_after"
    ] <= 1.0 + 1.0e-6
