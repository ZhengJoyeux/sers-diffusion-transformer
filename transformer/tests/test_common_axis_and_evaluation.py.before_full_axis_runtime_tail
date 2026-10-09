import numpy as np

from Dataset import (
    MODEL_LENGTH,
    MODEL_RAMAN_AXIS,
    _adapt_spectrum_to_model_axis,
)
from Inference import _classification_metrics, _regression_metrics


def test_common_axis_removes_1401_1901_length_cue():
    short_axis = np.arange(600.0, 2001.0, 1.0)
    long_axis = np.arange(600.0, 2501.0, 1.0)
    short_signal = np.sin(short_axis / 100.0)
    long_signal = np.sin(long_axis / 100.0)

    short_adapted, short_mask = _adapt_spectrum_to_model_axis(
        short_axis, short_signal
    )
    long_adapted, long_mask = _adapt_spectrum_to_model_axis(
        long_axis, long_signal
    )

    assert MODEL_LENGTH == 1401
    assert MODEL_RAMAN_AXIS[0] == 600.0
    assert MODEL_RAMAN_AXIS[-1] == 2000.0
    assert short_adapted.shape == (1401,)
    assert long_adapted.shape == (1401,)
    assert short_mask.all()
    assert long_mask.all()
    np.testing.assert_allclose(short_adapted, long_adapted, atol=1e-6)


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
    regression, per_reg = _regression_metrics(y_reg, p_reg, y_true, y_pred)

    assert regression["present_targets_only"]["n"] == int(y_true.sum())
    assert regression["sersformer2_conditional_correctly_detected_present"]["n"] == int(
        y_true.sum()
    )
    assert np.isfinite(regression["present_targets_only"]["mae"])
    assert len(per_reg) == 9
