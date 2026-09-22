"""Comprehensive post-training evaluation for DEL/CHL/TEB SERSFormer.

The evaluation follows the SERSFormer-2.0 paper where applicable:
- multilabel accuracy / precision / recall / F1
- per-pesticide confusion matrices
- pesticide co-occurrence confusion matrices
- ROC curves and AUROC
- multiregression MSE and R2
- violin plots for concentration prediction

For a more transparent quantitative assessment, this project also reports MAE,
RMSE, per-pesticide metrics, end-to-end gated regression, and stratified results
for single/binary/ternary mixtures and water/soil matrices.
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    jaccard_score,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from Dataset import MODEL_RAMAN_AXIS, PESTICIDES, build_datasets
from Model_v2 import TransformerClassifyRegress_sep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation of peak-guided SERSFormer checkpoint"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "validation", "test", "all"), default="test"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-directory", type=Path, default=Path("outputs/inference"))
    parser.add_argument("--classification-threshold", type=float, default=0.5)
    parser.add_argument("--training-history", type=Path, default=None)
    return parser.parse_args()


def _resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2 or np.allclose(y_true, y_true[0]):
        return float("nan")
    return float(r2_score(y_true, y_pred))


def _regression_metric_block(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[finite]
    y_pred = y_pred[finite]
    if y_true.size == 0:
        return {
            "n": 0,
            "mae": float("nan"),
            "mse": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
        }
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "n": int(y_true.size),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(math.sqrt(max(mse, 0.0))),
        "r2": _safe_r2(y_true, y_pred),
    }


def _classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    y_pred = (probabilities >= float(threshold)).astype(np.int64)

    result: dict[str, Any] = {
        "threshold": float(threshold),
        # Exact-match / subset accuracy: all three labels must be correct.
        "multilabel_exact_match_accuracy": float(accuracy_score(y_true, y_pred)),
        # Elementwise accuracy is useful because exact-match becomes strict for mixtures.
        "multilabel_hamming_accuracy": float(np.mean(y_true == y_pred)),
        "multilabel_jaccard_accuracy_samples": float(
            jaccard_score(y_true, y_pred, average="samples", zero_division=0)
        ),
    }

    for average in ("micro", "macro"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average=average,
            zero_division=0,
        )
        result[f"precision_{average}"] = float(precision)
        result[f"recall_{average}"] = float(recall)
        result[f"f1_{average}"] = float(f1)

    try:
        result["auroc_macro"] = float(
            roc_auc_score(y_true, probabilities, average="macro")
        )
    except ValueError:
        result["auroc_macro"] = float("nan")
    try:
        result["auroc_micro"] = float(
            roc_auc_score(y_true, probabilities, average="micro")
        )
    except ValueError:
        result["auroc_micro"] = float("nan")

    per_pesticide_rows: list[dict[str, Any]] = []
    for index, pesticide in enumerate(PESTICIDES):
        truth = y_true[:, index]
        pred = y_pred[:, index]
        prob = probabilities[:, index]
        tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth,
            pred,
            average="binary",
            zero_division=0,
        )
        try:
            label_auc = float(roc_auc_score(truth, prob))
        except ValueError:
            label_auc = float("nan")
        specificity = float(tn / (tn + fp)) if (tn + fp) else float("nan")
        per_pesticide_rows.append(
            {
                "pesticide": pesticide,
                "n": int(truth.size),
                "positive_n": int(truth.sum()),
                "negative_n": int((1 - truth).sum()),
                "accuracy": float(np.mean(truth == pred)),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "specificity": specificity,
                "auroc": label_auc,
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
            }
        )
    return result, pd.DataFrame(per_pesticide_rows), y_pred


def _regression_metrics(
    y_reg: np.ndarray,
    p_reg: np.ndarray,
    y_class: np.ndarray,
    y_pred_class: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame]:
    y_reg = np.asarray(y_reg, dtype=np.float64)
    p_reg = np.asarray(p_reg, dtype=np.float64)
    y_class = np.asarray(y_class, dtype=np.int64)
    y_pred_class = np.asarray(y_pred_class, dtype=np.int64)

    present = y_class > 0
    correctly_detected_present = present & (y_pred_class > 0)
    gated_prediction = np.where(y_pred_class > 0, p_reg, 0.0)

    overall = {
        # All outputs including absent=0. Useful as true end-to-end multioutput error,
        # but can be helped by many zeros, so it is never reported alone.
        "all_outputs_raw_regression": _regression_metric_block(y_reg, p_reg),
        # Classification-gated regression includes FN/FP impact in the final pipeline.
        "end_to_end_classification_gated": _regression_metric_block(
            y_reg, gated_prediction
        ),
        # Quantification accuracy on every truly present pesticide, even if classification
        # missed it. This avoids selecting only easy correctly classified cases.
        "present_targets_only": _regression_metric_block(
            y_reg[present], p_reg[present]
        ),
        # SERSFormer-2.0-style conditional evaluation: quantify concentrations after the
        # corresponding pesticide has been positively identified. Reported explicitly as
        # conditional because it can be more optimistic than end-to-end evaluation.
        "sersformer2_conditional_correctly_detected_present": _regression_metric_block(
            y_reg[correctly_detected_present], p_reg[correctly_detected_present]
        ),
        "absent_targets_prediction_error": _regression_metric_block(
            y_reg[~present], p_reg[~present]
        ),
    }

    rows: list[dict[str, Any]] = []
    for index, pesticide in enumerate(PESTICIDES):
        pesticide_present = present[:, index]
        correct_present = correctly_detected_present[:, index]
        for scope, mask, prediction in (
            ("present_targets_only", pesticide_present, p_reg[:, index]),
            (
                "sersformer2_conditional_correctly_detected_present",
                correct_present,
                p_reg[:, index],
            ),
            (
                "end_to_end_classification_gated",
                np.ones(y_reg.shape[0], dtype=bool),
                gated_prediction[:, index],
            ),
        ):
            metrics = _regression_metric_block(y_reg[mask, index], prediction[mask])
            rows.append({"pesticide": pesticide, "scope": scope, **metrics})
    return overall, pd.DataFrame(rows)


def _plot_matrix(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    path: Path,
    value_format: str = "d",
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    threshold = float(np.nanmax(matrix)) / 2.0 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            text = format(int(value), value_format) if value_format == "d" else format(float(value), value_format)
            ax.text(col, row, text, ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_directory: Path,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    for index, pesticide in enumerate(PESTICIDES):
        matrix = confusion_matrix(y_true[:, index], y_pred[:, index], labels=[0, 1])
        pd.DataFrame(
            matrix,
            index=["true_absent", "true_present"],
            columns=["pred_absent", "pred_present"],
        ).to_csv(output_directory / f"{pesticide}_counts.csv")
        _plot_matrix(
            matrix,
            ["Absent", "Present"],
            f"{pesticide} confusion matrix (counts)",
            output_directory / f"{pesticide}_counts.png",
        )

        row_sum = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(
            matrix,
            row_sum,
            out=np.zeros_like(matrix, dtype=np.float64),
            where=row_sum > 0,
        )
        pd.DataFrame(
            normalized,
            index=["true_absent", "true_present"],
            columns=["pred_absent", "pred_present"],
        ).to_csv(output_directory / f"{pesticide}_row_normalized.csv")
        _plot_matrix(
            normalized,
            ["Absent", "Present"],
            f"{pesticide} confusion matrix (row normalized)",
            output_directory / f"{pesticide}_row_normalized.png",
            value_format=".3f",
        )


def _save_cooccurrence_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_directory: Path,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    for size in (2, 3):
        for indices in combinations(range(len(PESTICIDES)), size):
            names = [PESTICIDES[index] for index in indices]
            true_joint = np.all(y_true[:, indices] > 0, axis=1).astype(np.int64)
            pred_joint = np.all(y_pred[:, indices] > 0, axis=1).astype(np.int64)
            matrix = confusion_matrix(true_joint, pred_joint, labels=[0, 1])
            stem = "_".join(names)
            pd.DataFrame(
                matrix,
                index=["true_not_joint", "true_joint"],
                columns=["pred_not_joint", "pred_joint"],
            ).to_csv(output_directory / f"{stem}.csv")
            _plot_matrix(
                matrix,
                ["Not joint", "Joint"],
                f"{' + '.join(names)} co-occurrence",
                output_directory / f"{stem}.png",
            )


def _save_roc_plot(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    valid_curves: list[tuple[np.ndarray, np.ndarray]] = []
    for index, pesticide in enumerate(PESTICIDES):
        truth = y_true[:, index]
        if np.unique(truth).size < 2:
            continue
        fpr, tpr, _ = roc_curve(truth, probabilities[:, index])
        label_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{pesticide} (AUC={label_auc:.3f})")
        valid_curves.append((fpr, tpr))

    if valid_curves:
        grid = np.linspace(0.0, 1.0, 201)
        interpolated = [np.interp(grid, fpr, tpr) for fpr, tpr in valid_curves]
        macro_tpr = np.mean(interpolated, axis=0)
        macro_auc = auc(grid, macro_tpr)
        ax.plot(grid, macro_tpr, linestyle="--", label=f"Macro ROC (AUC={macro_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle=":", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Multilabel ROC curves")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_regression_plots(
    frame: pd.DataFrame,
    output_directory: Path,
    target_mode: str,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    y_label = "Concentration" if target_mode == "physical" else "Concentration level code"

    for pesticide in PESTICIDES:
        truth_class = frame[f"true_class_{pesticide}"].to_numpy(dtype=int)
        present = truth_class > 0
        truth = frame[f"true_concentration_{pesticide}"].to_numpy(dtype=float)
        pred = frame[f"pred_concentration_{pesticide}"].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(truth[present], pred[present], s=18, alpha=0.7)
        if present.any():
            lo = float(min(np.min(truth[present]), np.min(pred[present])))
            hi = float(max(np.max(truth[present]), np.max(pred[present])))
            ax.plot([lo, hi], [lo, hi], linestyle="--")
        ax.set_xlabel(f"True {y_label}")
        ax.set_ylabel(f"Predicted {y_label}")
        ax.set_title(f"{pesticide}: predicted vs true")
        fig.tight_layout()
        fig.savefig(
            output_directory / f"{pesticide}_predicted_vs_true.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        # Article-style concentration violin: prediction distributions grouped by
        # the corresponding true concentration/level.
        unique_targets = np.unique(truth[present]) if present.any() else np.array([])
        groups = [pred[present & np.isclose(truth, value)] for value in unique_targets]
        groups = [group for group in groups if group.size > 0]
        if groups:
            fig, ax = plt.subplots(figsize=(6, 5))
            positions = np.arange(1, len(groups) + 1)
            ax.violinplot(groups, positions=positions, showmeans=True, showmedians=True)
            ax.set_xticks(positions)
            ax.set_xticklabels([f"{value:g}" for value in unique_targets])
            ax.set_xlabel(f"True {y_label}")
            ax.set_ylabel(f"Predicted {y_label}")
            ax.set_title(f"{pesticide}: prediction distribution by true level")
            fig.tight_layout()
            fig.savefig(
                output_directory / f"{pesticide}_violin.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)


def _stratified_metrics(
    frame: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    class_true = frame[[f"true_class_{p}" for p in PESTICIDES]].to_numpy(dtype=int)
    class_prob = frame[[f"prob_class_{p}" for p in PESTICIDES]].to_numpy(dtype=float)
    complexity = class_true.sum(axis=1)

    def compute_subset(mask: np.ndarray, label: str, category: str) -> dict[str, Any]:
        y = class_true[mask]
        p = class_prob[mask]
        if y.shape[0] == 0:
            return {"category": category, "group": label, "n": 0}
        metrics, _, _ = _classification_metrics(y, p, threshold)
        return {"category": category, "group": label, "n": int(y.shape[0]), **metrics}

    complexity_rows = []
    names = {1: "single", 2: "binary", 3: "ternary"}
    for value in (1, 2, 3):
        complexity_rows.append(
            compute_subset(complexity == value, names[value], "mixture_complexity")
        )

    matrix_rows = []
    matrix_values = frame["matrix"].astype(str).to_numpy()
    for matrix_name in sorted(set(matrix_values.tolist())):
        matrix_rows.append(
            compute_subset(matrix_values == matrix_name, matrix_name, "matrix")
        )

    combination_rows = []
    pattern_names = []
    for row in class_true:
        names_present = [p for p, value in zip(PESTICIDES, row) if value > 0]
        pattern_names.append("+".join(names_present) if names_present else "none")
    pattern_names = np.asarray(pattern_names, dtype=object)
    for pattern in sorted(set(pattern_names.tolist())):
        combination_rows.append(
            compute_subset(pattern_names == pattern, pattern, "pesticide_combination")
        )

    return (
        pd.DataFrame(complexity_rows),
        pd.DataFrame(matrix_rows),
        pd.DataFrame(combination_rows),
    )


def _plot_training_history(history_path: Path, output_directory: Path) -> None:
    if not history_path.exists():
        return
    frame = pd.read_csv(history_path)
    if frame.empty or "epoch" not in frame.columns:
        return
    output_directory.mkdir(parents=True, exist_ok=True)

    groups = [
        (
            "loss_curves.png",
            ["train_loss_total", "validation_loss_total", "train_loss_class", "validation_loss_class", "train_loss_reg", "validation_loss_reg"],
            "Training and validation losses",
            "Loss",
        ),
        (
            "classification_curves.png",
            ["train_class_f1_micro", "validation_class_f1_micro", "train_class_f1_macro", "validation_class_f1_macro", "train_class_auroc_macro", "validation_class_auroc_macro"],
            "Classification metrics during training",
            "Score",
        ),
        (
            "regression_curves.png",
            ["train_reg_rmse_present", "validation_reg_rmse_present", "train_reg_mae_present", "validation_reg_mae_present", "train_reg_r2_present", "validation_reg_r2_present"],
            "Regression metrics during training",
            "Metric",
        ),
    ]
    for filename, candidates, title, ylabel in groups:
        available = [column for column in candidates if column in frame.columns]
        if not available:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        for column in available:
            ax.plot(frame["epoch"], frame[column], label=column)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_directory / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)


def _query_token_raman_axis(token_length: int) -> np.ndarray:
    """Approximate Raman center of each token after Conv3/Pool3/Conv3/Pool3.

    With stride-1 valid Conv1d(kernel=3) and MaxPool1d(kernel=3, stride=3),
    token k has receptive-field center at original point index 8 + 9*k.
    """
    centers = 8 + 9 * np.arange(int(token_length), dtype=np.int64)
    if centers.size == 0 or centers[-1] >= len(MODEL_RAMAN_AXIS):
        raise ValueError(
            f"Cannot map token_length={token_length} to model axis length "
            f"{len(MODEL_RAMAN_AXIS)}"
        )
    return np.asarray(MODEL_RAMAN_AXIS, dtype=np.float64)[centers]


def _save_query_attention_profiles(
    attention: np.ndarray,
    y_true: np.ndarray,
    output_directory: Path,
) -> None:
    """Export interpretable pesticide-query attention without changing predictions."""
    attention = np.asarray(attention, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    if attention.ndim != 3 or attention.shape[1] != len(PESTICIDES):
        raise ValueError(
            "attention must be [samples, pesticides, tokens], got "
            f"{attention.shape}"
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    token_axis = _query_token_raman_axis(attention.shape[-1])
    frame = pd.DataFrame({"Raman_shift_cm-1": token_axis})
    top_rows: list[dict[str, Any]] = []

    for pesticide_index, pesticide in enumerate(PESTICIDES):
        all_mean = attention[:, pesticide_index, :].mean(axis=0)
        present_mask = y_true[:, pesticide_index] > 0
        absent_mask = ~present_mask
        present_mean = (
            attention[present_mask, pesticide_index, :].mean(axis=0)
            if present_mask.any()
            else np.full_like(all_mean, np.nan)
        )
        absent_mean = (
            attention[absent_mask, pesticide_index, :].mean(axis=0)
            if absent_mask.any()
            else np.full_like(all_mean, np.nan)
        )
        difference = present_mean - absent_mean

        frame[f"{pesticide}_all_mean"] = all_mean
        frame[f"{pesticide}_present_mean"] = present_mean
        frame[f"{pesticide}_absent_mean"] = absent_mean
        frame[f"{pesticide}_present_minus_absent"] = difference

        ranking_signal = np.nan_to_num(difference, nan=-np.inf)
        top_indices = np.argsort(ranking_signal)[::-1][:10]
        for rank, token_index in enumerate(top_indices, start=1):
            top_rows.append(
                {
                    "pesticide": pesticide,
                    "rank": rank,
                    "token_index": int(token_index),
                    "Raman_shift_cm-1": float(token_axis[token_index]),
                    "present_mean_attention": float(present_mean[token_index]),
                    "absent_mean_attention": float(absent_mean[token_index]),
                    "present_minus_absent": float(difference[token_index]),
                }
            )

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(token_axis, present_mean, label="Target present")
        ax.plot(token_axis, absent_mean, label="Target absent")
        ax.set_xlabel("Raman shift (cm$^{-1}$)")
        ax.set_ylabel("Mean query attention")
        ax.set_title(f"{pesticide} query attention profile")
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            output_directory / f"{pesticide}_query_attention.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    frame.to_csv(output_directory / "query_attention_profiles.csv", index=False)
    pd.DataFrame(top_rows).to_csv(
        output_directory / "query_attention_top_regions.csv", index=False
    )


def _collect_predictions(
    checkpoint: dict[str, Any],
    dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    model = TransformerClassifyRegress_sep(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    class_targets: list[np.ndarray] = []
    class_probabilities: list[np.ndarray] = []
    concentration_targets: list[np.ndarray] = []
    concentration_predictions: list[np.ndarray] = []
    query_attentions: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            model_batch = {
                "raw": batch["raw"].to(device, non_blocking=True),
                "percentile": batch["percentile"].to(device, non_blocking=True),
                "smoothed": batch["smoothed"].to(device, non_blocking=True),
                "valid_mask": batch["valid_mask"].to(device, non_blocking=True),
            }
            class_pred, reg_pred, attention = model(
                model_batch, return_attention=True
            )
            class_true_np = batch["class_target"].numpy()
            class_prob_np = class_pred.cpu().numpy()
            reg_true_np = batch["concentration_target"].numpy()
            reg_pred_np = reg_pred.cpu().numpy()

            class_targets.append(class_true_np)
            class_probabilities.append(class_prob_np)
            concentration_targets.append(reg_true_np)
            concentration_predictions.append(reg_pred_np)
            query_attentions.append(attention.cpu().numpy())

            for row_index in range(class_true_np.shape[0]):
                row: dict[str, Any] = {
                    "condition": batch["condition"][row_index],
                    "matrix": batch["matrix_name"][row_index],
                    "source": batch["source"][row_index],
                    "source_file": batch["source_file"][row_index],
                    "spectrum_name": batch["spectrum_name"][row_index],
                    "spectrum_index": int(batch["spectrum_index"][row_index]),
                    "target_mode": batch["target_mode"][row_index],
                }
                for pesticide_index, pesticide in enumerate(PESTICIDES):
                    row[f"true_class_{pesticide}"] = int(
                        class_true_np[row_index, pesticide_index]
                    )
                    row[f"prob_class_{pesticide}"] = float(
                        class_prob_np[row_index, pesticide_index]
                    )
                    row[f"true_concentration_{pesticide}"] = float(
                        reg_true_np[row_index, pesticide_index]
                    )
                    row[f"pred_concentration_{pesticide}"] = float(
                        reg_pred_np[row_index, pesticide_index]
                    )
                rows.append(row)

    return (
        np.concatenate(class_targets, axis=0),
        np.concatenate(class_probabilities, axis=0),
        np.concatenate(concentration_targets, axis=0),
        np.concatenate(concentration_predictions, axis=0),
        np.concatenate(query_attentions, axis=0),
        pd.DataFrame(rows),
    )


def evaluate_split(
    *,
    checkpoint: dict[str, Any],
    dataset,
    split: str,
    output_directory: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    classification_threshold: float,
) -> dict[str, Any]:
    split_dir = output_directory / split
    split_dir.mkdir(parents=True, exist_ok=True)

    y_class, p_class, y_reg, p_reg, query_attention, frame = _collect_predictions(
        checkpoint,
        dataset,
        device,
        batch_size,
        num_workers,
    )
    classification, per_pesticide, y_pred = _classification_metrics(
        y_class,
        p_class,
        classification_threshold,
    )
    regression, regression_per_pesticide = _regression_metrics(
        y_reg,
        p_reg,
        y_class,
        y_pred,
    )

    for index, pesticide in enumerate(PESTICIDES):
        frame[f"pred_class_{pesticide}"] = y_pred[:, index]
    frame.to_csv(split_dir / "predictions.csv", index=False)
    per_pesticide.to_csv(split_dir / "classification_per_pesticide.csv", index=False)
    regression_per_pesticide.to_csv(
        split_dir / "regression_per_pesticide.csv", index=False
    )

    complexity, matrix_metrics, combination_metrics = _stratified_metrics(
        frame, classification_threshold
    )
    complexity.to_csv(split_dir / "metrics_by_mixture_complexity.csv", index=False)
    matrix_metrics.to_csv(split_dir / "metrics_by_matrix.csv", index=False)
    combination_metrics.to_csv(split_dir / "metrics_by_pesticide_combination.csv", index=False)

    _save_confusion_matrices(
        y_class,
        y_pred,
        split_dir / "confusion_matrices",
    )
    _save_cooccurrence_matrices(
        y_class,
        y_pred,
        split_dir / "cooccurrence_matrices",
    )
    _save_roc_plot(y_class, p_class, split_dir / "roc_curves.png")
    _save_query_attention_profiles(
        query_attention,
        y_class,
        split_dir / "query_attention",
    )
    target_mode = str(frame["target_mode"].iloc[0]) if not frame.empty else "unknown"
    _save_regression_plots(
        frame,
        split_dir / "regression_plots",
        target_mode,
    )

    result = {
        "split": split,
        "samples": int(len(dataset)),
        "target_mode": target_mode,
        "query_attention_mode": (
            "peak_guided"
            if checkpoint.get("model_config", {}).get("use_peak_guidance", True)
            else "query_only"
        ),
        "classification": classification,
        "regression": regression,
    }
    with (split_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=True)

    lines = [
        f"split: {split}",
        f"samples: {len(dataset)}",
        f"target_mode: {target_mode}",
        "",
        "[Multilabel classification]",
    ]
    for key, value in classification.items():
        lines.append(f"{key}: {value}")
    lines.extend(["", "[Regression]"])
    for scope, values in regression.items():
        lines.append(f"{scope}: {values}")
    if target_mode != "physical":
        lines.extend(
            [
                "",
                "WARNING: concentration targets are ordinal level codes 0/1/2/3,",
                "not physical concentration values. Do not report these regression",
                "numbers as final quantitative concentration performance.",
            ]
        )
    (split_dir / "evaluation_report.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"===== {split.upper()} EVALUATION =====")
    print(f"samples: {len(dataset)}")
    print(
        "classification: "
        f"exact_acc={classification['multilabel_exact_match_accuracy']:.4f}, "
        f"F1_macro={classification['f1_macro']:.4f}, "
        f"F1_micro={classification['f1_micro']:.4f}, "
        f"AUROC_macro={classification['auroc_macro']:.4f}"
    )
    present_metrics = regression["present_targets_only"]
    print(
        "regression present-only: "
        f"MAE={present_metrics['mae']}, RMSE={present_metrics['rmse']}, "
        f"R2={present_metrics['r2']}"
    )
    print(f"saved: {split_dir}")
    return result


def evaluate_checkpoint(
    *,
    checkpoint_path: str | Path,
    output_directory: str | Path,
    device_name: str = "cuda",
    batch_size: int = 64,
    num_workers: int = 4,
    classification_threshold: float = 0.5,
    splits: Iterable[str] = ("test",),
    training_history_path: str | Path | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(device_name)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    concentration_map = checkpoint.get("concentration_map")
    train_dataset, validation_dataset, test_dataset = build_datasets(
        include_generated_train=False,
        concentration_map=concentration_map,
    )
    datasets = {
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset,
    }

    split_results: dict[str, Any] = {}
    for split in splits:
        if split not in datasets:
            raise ValueError(f"Unknown split for evaluation: {split}")
        split_results[split] = evaluate_split(
            checkpoint=checkpoint,
            dataset=datasets[split],
            split=split,
            output_directory=output_directory,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            classification_threshold=classification_threshold,
        )

    if training_history_path is not None:
        _plot_training_history(Path(training_history_path), output_directory / "training_curves")

    split_description = str(checkpoint.get("data_config", {}).get("split", "unknown"))
    split_warning = None
    if "within each source file" in split_description.lower() or "12/4/4" in split_description.lower():
        split_warning = (
            "Current 12/4/4 split places mapping spectra from the same source file in "
            "train/validation/test. If one source file represents one independent sample, "
            "these metrics are not an independent-sample generalization estimate. A final "
            "paper experiment should split by source_file/sample_id."
        )

    summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "classification_threshold": float(classification_threshold),
        "splits": split_results,
        "evaluation_notes": {
            "threshold_policy": (
                "Fixed threshold supplied before test evaluation; test set is not used "
                "for threshold tuning."
            ),
            "sersformer2_metrics": (
                "multilabel accuracy/precision/recall/F1, per-label confusion matrices, "
                "co-occurrence matrices, ROC/AUROC, MSE/R2 and concentration violins"
            ),
            "additional_metrics": (
                "MAE/RMSE, per-pesticide regression, classification-gated end-to-end "
                "regression, and single/binary/ternary + matrix stratification"
            ),
            "data_split": split_description,
            "data_split_warning": split_warning,
        },
    }
    with (output_directory / "evaluation_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=True)
    return summary


def main() -> None:
    args = parse_args()
    if args.split == "all":
        splits = ("train", "validation", "test")
    else:
        splits = (args.split,)
    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        output_directory=args.output_directory,
        device_name=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        classification_threshold=args.classification_threshold,
        splits=splits,
        training_history_path=args.training_history,
    )


if __name__ == "__main__":
    main()
