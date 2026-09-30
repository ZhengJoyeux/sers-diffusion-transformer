from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from Inference import _ordinal_level_metrics
from Model_v2 import (
    TransformerClassifyRegress_sep,
    level_codes_to_ordinal_targets,
    present_only_ordinal_bce,
)


def _batch(batch_size: int = 2, length: int = 1401):
    return {
        "raw": torch.rand(batch_size, 1, length),
        "percentile": torch.rand(batch_size, 4, length),
        "smoothed": torch.rand(batch_size, 1, length),
        "valid_mask": torch.ones(batch_size, length, dtype=torch.bool),
    }


def test_ordinal_head_shapes_monotonicity_and_default_interface():
    torch.manual_seed(2026)
    model = TransformerClassifyRegress_sep(
        dim_model=8,
        attn_head=2,
        dim_ff=32,
        encoder_layers=1,
        n_labels=3,
        model_length=1401,
        concentration_head_mode="ordinal",
    )
    model.eval()

    with torch.no_grad():
        class_pred, level_expectation, ordinal_probability = model(
            _batch(), return_ordinal=True
        )

    assert class_pred.shape == (2, 3)
    assert level_expectation.shape == (2, 3)
    assert ordinal_probability.shape == (2, 3, 2)
    assert torch.all(ordinal_probability[..., 1] <= ordinal_probability[..., 0])
    assert torch.all(level_expectation >= 1.0)
    assert torch.all(level_expectation <= 3.0)

    # Existing callers still receive exactly two outputs by default.
    default_outputs = model(_batch())
    assert len(default_outputs) == 2


def test_present_only_ordinal_bce_ignores_absent_pesticides():
    probability = torch.tensor(
        [
            [[0.2, 0.1], [0.8, 0.2], [0.9, 0.8]],
            [[0.4, 0.2], [0.7, 0.3], [0.6, 0.2]],
        ],
        dtype=torch.float32,
    )
    concentration = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    presence = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    target = level_codes_to_ordinal_targets(concentration)
    assert target[0].tolist() == [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]

    loss_a = present_only_ordinal_bce(probability, concentration, presence)

    changed_absent = probability.clone()
    changed_absent[1, 0] = torch.tensor([0.99, 0.99])
    changed_absent[1, 2] = torch.tensor([0.01, 0.01])
    loss_b = present_only_ordinal_bce(changed_absent, concentration, presence)

    torch.testing.assert_close(loss_a, loss_b)


def test_ordinal_evaluation_reports_present_and_profile_accuracy():
    y_class = np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=np.int64,
    )
    y_reg = np.array(
        [
            [1, 0, 0],
            [2, 3, 0],
            [3, 2, 1],
        ],
        dtype=np.float64,
    )
    # All six present pesticide-level targets are decoded correctly.
    ordinal_probability = np.array(
        [
            [[0.10, 0.02], [0.10, 0.02], [0.10, 0.02]],
            [[0.90, 0.10], [0.95, 0.80], [0.10, 0.02]],
            [[0.95, 0.80], [0.90, 0.10], [0.10, 0.02]],
        ],
        dtype=np.float64,
    )
    y_pred_class = y_class.copy()
    frame = pd.DataFrame({"matrix": ["water", "water", "soil"]})

    overall, per_pesticide, by_mixture, by_matrix, confusion = (
        _ordinal_level_metrics(
            y_reg,
            ordinal_probability,
            y_class,
            y_pred_class,
            frame,
        )
    )

    assert overall["present_target_accuracy"] == 1.0
    assert overall["full_0_S_M_H_profile_exact_accuracy"] == 1.0
    assert set(per_pesticide["pesticide"]) == {"DEL", "CHL", "TEB"}
    assert set(by_mixture["group"]) == {"single", "binary", "ternary"}
    assert set(by_matrix["matrix"]) == {"water", "soil"}
    assert set(confusion) == {"DEL", "CHL", "TEB"}


def test_continuous_mode_remains_t2_compatible():
    torch.manual_seed(2026)
    model = TransformerClassifyRegress_sep(
        dim_model=8,
        attn_head=2,
        dim_ff=32,
        encoder_layers=1,
        n_labels=3,
        model_length=1401,
        concentration_head_mode="continuous",
    )
    model.eval()
    with torch.no_grad():
        class_pred, reg_pred = model(_batch())
    assert class_pred.shape == (2, 3)
    assert reg_pred.shape == (2, 3)
    assert hasattr(model, "regression_heads")
    assert not hasattr(model, "ordinal_heads")
