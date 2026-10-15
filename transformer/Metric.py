"""Metrics for the adapted SERSFormer2 T0 baseline.

No torchmetrics dependency is required. Classification/regression summaries use
scikit-learn and NumPy, while training itself remains pure PyTorch.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


PESTICIDES = ("DEL", "CHL", "TEB")


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def multilabel_classification_metrics(
    targets,
    probabilities,
    threshold: float = 0.5,
) -> dict[str, float]:
    targets = _as_numpy(targets).astype(np.int64)
    probabilities = _as_numpy(probabilities).astype(np.float64)
    predictions = (probabilities >= threshold).astype(np.int64)

    if targets.ndim != 2 or targets.shape[1] != len(PESTICIDES):
        raise ValueError(f"Expected targets [N,3], got {targets.shape}")
    if predictions.shape != targets.shape:
        raise ValueError("Prediction/target shape mismatch")

    result = {
        "class_subset_accuracy": float(accuracy_score(targets, predictions)),
        "class_hamming_accuracy": float(1.0 - hamming_loss(targets, predictions)),
        "class_precision_micro": float(
            precision_score(targets, predictions, average="micro", zero_division=0)
        ),
        "class_recall_micro": float(
            recall_score(targets, predictions, average="micro", zero_division=0)
        ),
        "class_f1_micro": float(
            f1_score(targets, predictions, average="micro", zero_division=0)
        ),
        "class_precision_macro": float(
            precision_score(targets, predictions, average="macro", zero_division=0)
        ),
        "class_recall_macro": float(
            recall_score(targets, predictions, average="macro", zero_division=0)
        ),
        "class_f1_macro": float(
            f1_score(targets, predictions, average="macro", zero_division=0)
        ),
    }

    for index, pesticide in enumerate(PESTICIDES):
        result[f"class_f1_{pesticide}"] = float(
            f1_score(targets[:, index], predictions[:, index], zero_division=0)
        )
    return result


def regression_metrics(
    targets,
    predictions,
    presence,
) -> dict[str, float]:
    targets = _as_numpy(targets).astype(np.float64)
    predictions = _as_numpy(predictions).astype(np.float64)
    presence = _as_numpy(presence).astype(bool)

    if targets.shape != predictions.shape or targets.shape != presence.shape:
        raise ValueError("Regression target/prediction/presence shape mismatch")
    if targets.ndim != 2 or targets.shape[1] != len(PESTICIDES):
        raise ValueError(f"Expected regression arrays [N,3], got {targets.shape}")

    result: dict[str, float] = {}

    mse_all = mean_squared_error(targets, predictions)
    result["reg_mse_all"] = float(mse_all)
    result["reg_rmse_all"] = float(np.sqrt(mse_all))
    result["reg_mae_all"] = float(mean_absolute_error(targets, predictions))

    if targets.shape[0] >= 2:
        try:
            result["reg_r2_all"] = float(
                r2_score(targets, predictions, multioutput="uniform_average")
            )
        except ValueError:
            result["reg_r2_all"] = float("nan")
    else:
        result["reg_r2_all"] = float("nan")

    present_targets = targets[presence]
    present_predictions = predictions[presence]
    if present_targets.size:
        mse_present = mean_squared_error(present_targets, present_predictions)
        result["reg_mse_present"] = float(mse_present)
        result["reg_rmse_present"] = float(np.sqrt(mse_present))
        result["reg_mae_present"] = float(
            mean_absolute_error(present_targets, present_predictions)
        )
        if present_targets.size >= 2 and np.unique(present_targets).size >= 2:
            result["reg_r2_present"] = float(
                r2_score(present_targets, present_predictions)
            )
        else:
            result["reg_r2_present"] = float("nan")
    else:
        result["reg_mse_present"] = float("nan")
        result["reg_rmse_present"] = float("nan")
        result["reg_mae_present"] = float("nan")
        result["reg_r2_present"] = float("nan")

    for index, pesticide in enumerate(PESTICIDES):
        mask = presence[:, index]
        if not np.any(mask):
            continue
        y_true = targets[mask, index]
        y_pred = predictions[mask, index]
        mse = mean_squared_error(y_true, y_pred)
        result[f"reg_rmse_{pesticide}"] = float(np.sqrt(mse))
        result[f"reg_mae_{pesticide}"] = float(mean_absolute_error(y_true, y_pred))
        if y_true.size >= 2 and np.unique(y_true).size >= 2:
            result[f"reg_r2_{pesticide}"] = float(r2_score(y_true, y_pred))
        else:
            result[f"reg_r2_{pesticide}"] = float("nan")

    return result


def merge_metrics(*metric_dicts: Mapping[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for metrics in metric_dicts:
        merged.update(metrics)
    return merged


def compute_dwa_weights(
    classification_history: list[float],
    regression_history: list[float],
    temperature: float = 2.0,
) -> tuple[float, float]:
    """Epoch-level Dynamic Weight Averaging with weights summing to 1."""
    if len(classification_history) < 2 or len(regression_history) < 2:
        return 0.5, 0.5

    eps = 1e-12
    ratio_class = classification_history[-1] / max(classification_history[-2], eps)
    ratio_reg = regression_history[-1] / max(regression_history[-2], eps)
    scaled = np.asarray([ratio_class, ratio_reg], dtype=np.float64) / temperature
    scaled -= float(scaled.max())
    weights = np.exp(scaled)
    weights /= weights.sum()
    return float(weights[0]), float(weights[1])
