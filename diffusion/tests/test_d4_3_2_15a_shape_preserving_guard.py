import numpy as np

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
    # 原始测试谷是对称的，因此修正后只有4个唯一强度值是正常的。
    # 真正需要验证的是：
    # 1. 没有被压成水平平底；
    # 2. 中心仍然是最低点；
    # 3. 左侧逐步下降、右侧逐步回升。
    assert (
        np.unique(
            np.round(
                corrected_valley,
                decimals=5,
            )
        ).size
        >= 4
    )

    assert int(
        np.argmin(corrected_valley)
    ) == 3

    assert np.all(
        np.diff(
            corrected_valley[:4]
        )
        < 0.0
    )

    assert np.all(
        np.diff(
            corrected_valley[3:]
        )
        > 0.0
    )
    # guard输出统一转换为float32。
    # 因此这里验证：
    # 未触发修正的光谱与原始光谱做同样的float32转换后完全一致，
    # 从而排除guard对非异常光谱的额外修改。
    np.testing.assert_array_equal(
        corrected[0],
        before[0].astype(
            np.float32,
            copy=False,
        ),
    )


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
