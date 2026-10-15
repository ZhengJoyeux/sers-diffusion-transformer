import numpy as np

from Dataset import (
    MODEL_LENGTH,
    MODEL_RAMAN_AXIS,
    _adapt_spectrum_to_model_axis,
    _impute_missing_2000_2500_tail_with_noise,
)
from Inference import _classification_metrics, _regression_metrics


def test_runtime_tail_completion_removes_1401_1901_tensor_length_cue():
    """
    Current full-axis design:

    - original 600-2000 cm^-1 spectra are completed in memory for
      2001-2500 cm^-1 using deterministic background-matched noise;
    - original 600-2500 cm^-1 spectra remain unchanged;
    - both finally enter the model as 1901-point tensors.

    This test intentionally follows the runtime data-loading path rather
    than the old 1401-point common-axis cropping design.
    """

    short_axis = np.arange(
        600.0,
        2001.0,
        1.0,
        dtype=np.float64,
    )

    long_axis = np.arange(
        600.0,
        2501.0,
        1.0,
        dtype=np.float64,
    )

    short_signal = np.sin(
        short_axis / 100.0
    )

    long_signal = np.sin(
        long_axis / 100.0
    )

    # --------------------------------------------------------
    # 1. Simulate actual runtime completion of a short spectrum.
    #    Helper expects [raman_points, spectra].
    # --------------------------------------------------------
    (
        completed_axis,
        completed_spectra,
        was_imputed,
    ) = _impute_missing_2000_2500_tail_with_noise(
        short_axis,
        short_signal[:, np.newaxis],
        source_key="unit-test-short-spectrum",
    )

    assert was_imputed is True

    assert completed_axis.shape == (
        1901,
    )

    assert completed_spectra.shape == (
        1901,
        1,
    )

    np.testing.assert_allclose(
        completed_axis,
        MODEL_RAMAN_AXIS,
        atol=1e-6,
    )

    # The genuinely measured 600-2000 cm^-1 section must not change.
    np.testing.assert_allclose(
        completed_spectra[:1401, 0],
        short_signal,
        atol=1e-6,
    )

    # Runtime-added tail must be finite and must not simply be zero padding.
    completed_tail = completed_spectra[
        1401:,
        0,
    ]

    assert np.isfinite(
        completed_tail
    ).all()

    assert not np.allclose(
        completed_tail,
        0.0,
    )

    # --------------------------------------------------------
    # 2. The same short source must receive the same deterministic
    #    runtime tail for the same source_key.
    # --------------------------------------------------------
    (
        completed_axis_again,
        completed_spectra_again,
        was_imputed_again,
    ) = _impute_missing_2000_2500_tail_with_noise(
        short_axis,
        short_signal[:, np.newaxis],
        source_key="unit-test-short-spectrum",
    )

    assert was_imputed_again is True

    np.testing.assert_allclose(
        completed_axis_again,
        completed_axis,
        atol=0.0,
    )

    np.testing.assert_allclose(
        completed_spectra_again,
        completed_spectra,
        atol=0.0,
    )

    # --------------------------------------------------------
    # 3. A genuine full 600-2500 spectrum must remain unchanged.
    # --------------------------------------------------------
    (
        unchanged_axis,
        unchanged_spectra,
        long_was_imputed,
    ) = _impute_missing_2000_2500_tail_with_noise(
        long_axis,
        long_signal[:, np.newaxis],
        source_key="unit-test-long-spectrum",
    )

    assert long_was_imputed is False

    np.testing.assert_allclose(
        unchanged_axis,
        long_axis,
        atol=0.0,
    )

    np.testing.assert_allclose(
        unchanged_spectra[:, 0],
        long_signal,
        atol=0.0,
    )

    # --------------------------------------------------------
    # 4. Both paths must finally produce the same model tensor length.
    # --------------------------------------------------------
    short_adapted, short_mask = (
        _adapt_spectrum_to_model_axis(
            completed_axis,
            completed_spectra[:, 0],
        )
    )

    long_adapted, long_mask = (
        _adapt_spectrum_to_model_axis(
            unchanged_axis,
            unchanged_spectra[:, 0],
        )
    )

    assert MODEL_LENGTH == 1901
    assert MODEL_RAMAN_AXIS[0] == 600.0
    assert MODEL_RAMAN_AXIS[-1] == 2500.0

    assert short_adapted.shape == (
        1901,
    )

    assert long_adapted.shape == (
        1901,
    )

    assert short_mask.shape == (
        1901,
    )

    assert long_mask.shape == (
        1901,
    )

    assert short_mask.all()
    assert long_mask.all()

    # Their genuinely measured common region must still agree.
    np.testing.assert_allclose(
        short_adapted[:1401],
        long_adapted[:1401],
        atol=1e-6,
    )


def test_multilabel_metrics_and_regression_scopes_are_finite():
    y_true = np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.int64,
    )
    probabilities = np.array(
        [
            [0.9, 0.2, 0.1],
            [0.8, 0.7, 0.2],
            [0.2, 0.8, 0.9],
            [0.8, 0.9, 0.7],
        ],
        dtype=np.float64,
    )
    classification, per_pesticide, y_pred = _classification_metrics(
        y_true, probabilities, 0.5
    )

    assert classification["multilabel_exact_match_accuracy"] == 1.0
    assert classification["f1_macro"] == 1.0
    assert len(per_pesticide) == 3

    y_reg = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [3.0, 3.0, 3.0],
        ]
    )
    p_reg = y_reg + np.array(
        [
            [0.1, 0.1, 0.0],
            [-0.1, 0.1, 0.1],
            [0.0, -0.1, 0.1],
            [0.1, -0.1, 0.1],
        ]
    )
    regression, per_reg = _regression_metrics(
        y_reg,
        p_reg,
        y_true,
        y_pred,
    )

    assert regression["present_targets_only"]["n"] == int(
        y_true.sum()
    )
    assert regression[
        "sersformer2_conditional_correctly_detected_present"
    ]["n"] == int(
        y_true.sum()
    )
    assert np.isfinite(
        regression[
            "present_targets_only"
        ]["mae"]
    )
    assert len(per_reg) == 9
