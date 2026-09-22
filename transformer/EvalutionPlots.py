"""Matplotlib-only plotting utilities for adapted SERSFormer2 outputs.

The historical filename is retained for repository compatibility.
No seaborn or torchmetrics dependency is required.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


PESTICIDES = ("DEL", "CHL", "TEB")


def plot_multilabel_confusion_matrices(
    predictions: pd.DataFrame,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for axis, pesticide in zip(axes, PESTICIDES):
        y_true = predictions[f"true_class_{pesticide}"].to_numpy(dtype=int)
        y_pred = predictions[f"pred_class_{pesticide}"].to_numpy(dtype=int)
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

        image = axis.imshow(matrix)
        axis.set_title(pesticide)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_xticks([0, 1])
        axis.set_yticks([0, 1])
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")

    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8)
    fig.suptitle("Multi-label confusion matrices")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_regression_scatter(
    predictions: pd.DataFrame,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for axis, pesticide in zip(axes, PESTICIDES):
        present = predictions[f"true_class_{pesticide}"].to_numpy(dtype=int) == 1
        y_true = predictions.loc[present, f"true_concentration_{pesticide}"].to_numpy(
            dtype=float
        )
        y_pred = predictions.loc[present, f"pred_concentration_{pesticide}"].to_numpy(
            dtype=float
        )

        axis.scatter(y_true, y_pred, s=18, alpha=0.7)
        if y_true.size:
            lower = float(min(y_true.min(), y_pred.min()))
            upper = float(max(y_true.max(), y_pred.max()))
            axis.plot([lower, upper], [lower, upper], linestyle="--")
        axis.set_title(pesticide)
        axis.set_xlabel("True concentration")
        axis.set_ylabel("Predicted concentration")

    fig.suptitle("Concentration regression on present pesticides")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_training_history(
    history: pd.DataFrame,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(history["epoch"], history["train_loss_total"], label="train")
    axis.plot(history["epoch"], history["validation_loss_total"], label="validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Total loss")
    axis.legend()
    axis.set_title("SERSFormer2 training history")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot adapted SERSFormer2 results")
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--history", type=Path, default=None)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)

    if args.predictions is not None:
        predictions = pd.read_csv(args.predictions)
        plot_multilabel_confusion_matrices(
            predictions, args.output_directory / "multilabel_confusion_matrices.png"
        )
        plot_regression_scatter(
            predictions, args.output_directory / "concentration_regression_scatter.png"
        )

    if args.history is not None:
        history = pd.read_csv(args.history)
        plot_training_history(history, args.output_directory / "training_history.png")


if __name__ == "__main__":
    main()
