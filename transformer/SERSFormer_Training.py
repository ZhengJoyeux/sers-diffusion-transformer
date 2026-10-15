"""Training entry for pesticide-query DEL/CHL/TEB SERSFormer variants.

The default next-step ablation is query-only attention: three pesticide-specific
learnable queries attend to the full valid spectrum without a peak prior. The same
entry can later enable training-only soft peak guidance for a controlled ablation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Dataset import (
    CHEMISTRY_CORE_PEAKS,
    MODEL_LENGTH,
    MODEL_RAMAN_AXIS,
    PESTICIDES,
    build_axis_label_audit,
    build_chemistry_hybrid_peak_priors,
    build_datasets,
    build_matched_condition_peak_priors,
    build_training_peak_priors,
    collect_real_training_intensities,
    load_concentration_map,
)
from Metric import (
    compute_dwa_weights,
    merge_metrics,
    multilabel_classification_metrics,
    regression_metrics,
)
from Model_v2 import (
    TransformerClassifyRegress_sep,
    decode_ordinal_numpy,
    present_only_corn_ordinal_loss,
    present_only_ordinal_bce,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train pesticide-query SERSFormer on DEL/CHL/TEB SERS spectra."
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/t1_query_only_sersformer_real_only"),
    )
    parser.add_argument("--include-generated", action="store_true")
    parser.add_argument("--maximum-generated-per-condition", type=int, default=None)
    parser.add_argument("--allow-incomplete-generated", action="store_true")
    parser.add_argument("--concentration-map", type=Path, default=None)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--early-stopping-patience", type=int, default=12)

    parser.add_argument("--dim-model", type=int, default=32)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--dim-ff", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--local-quantitative-branch", action="store_true",
                        help="T3.13: add native-resolution peak-window evidence to ordinal concentration heads")
    parser.add_argument("--local-half-width-cm1", type=int, default=24)
    parser.add_argument("--local-hidden", type=int, default=16)
    parser.add_argument("--local-strength", type=float, default=0.25)
    parser.add_argument("--ordinal-decoding", choices=("median", "map"), default="median",
                        help="Median preserves legacy decoding; MAP selects the most probable S/M/H level")
    parser.add_argument("--checkpoint-selection", choices=("loss", "ordinal"), default="loss",
                        help="Ordinal uses validation present-level accuracy, then profile exact accuracy, for selection and early stopping")
    parser.add_argument(
        "--concentration-head-mode",
        choices=("continuous", "ordinal"),
        default="continuous",
        help=(
            "continuous keeps the T1/T2 scalar level-code regression head; "
            "ordinal enables T3.1 pesticide-specific present-only ordered S/M/H heads."
        ),
    )

    parser.add_argument(
        "--mixture-aware-query-fusion",
        action="store_true",
        help=(
            "T3.3: enable lightweight DEL/CHL/TEB cross-query fusion "
            "for the ordinal concentration branch."
        ),
    )

    parser.add_argument(
        "--adaptive-mixture-gate",
        action="store_true",
        help=(
            "T3.11: enable sample-adaptive, pesticide-specific "
            "mixture fusion gates. Requires "
            "--mixture-aware-query-fusion."
        ),
    )

    parser.add_argument(
        "--boundary-specific-mixture-gate",
        action="store_true",
        help=(
            "T3.12: use separate adaptive mixture gates for "
            "the S/M and M/H ordinal boundaries. Requires "
            "--adaptive-mixture-gate."
        ),
    )

    parser.add_argument(
        "--ordinal-head-hidden",
        type=int,
        default=None,
        help=(
            "Optional hidden dimension used only by the ordinal "
            "S/M/H concentration heads. "
            "None preserves the legacy dimension. "
            "T3.6 uses 64."
        ),
    )

    parser.add_argument(
        "--ordinal-loss-mode",
        choices=("cumulative", "corn"),
        default="cumulative",
        help=(
            "Ordinal base-loss formulation. "
            "cumulative preserves T3.1-T3.6 behavior; "
            "corn enables T3.7 conditional S/M and M/H training."
        ),
    )

    parser.add_argument(
        "--query-attention-mode",
        choices=("query_only", "peak_guided"),
        default="query_only",
        help=(
            "query_only: three pesticide queries learn from the complete valid spectrum "
            "without peak bias; peak_guided: add the training-only soft peak prior. "
            "Use query_only first as the clean ablation baseline."
        ),
    )

    # Peak prior is fitted from the fixed real training split only. These
    # arguments control prior extraction and its initial soft-attention bias.
    parser.add_argument(
        "--peak-prior-mode",
        choices=("global_contrast", "matched_shared", "chemistry_hybrid"),
        default="chemistry_hybrid",
        help=(
            "global_contrast keeps the earlier all-positive vs all-negative prior; "
            "matched_shared finds training-data discriminative regions; "
            "chemistry_hybrid uses DEL 1000/1600, CHL 2230 and TEB 1090/1597 "
            "as primary chemistry anchors and matched_shared only for auxiliaries."
        ),
    )
    parser.add_argument("--peak-top-k", type=int, default=8)
    parser.add_argument("--peak-min-distance-cm1", type=float, default=20.0)
    parser.add_argument("--peak-prominence", type=float, default=0.02)
    parser.add_argument("--peak-half-width-cm1", type=float, default=15.0)
    parser.add_argument("--peak-min-support-fraction", type=float, default=0.25)
    parser.add_argument("--peak-min-matched-pairs", type=int, default=4)
    parser.add_argument("--peak-shared-floor", type=float, default=0.35)
    parser.add_argument("--peak-guidance-strength", type=float, default=1.5)
    parser.add_argument(
        "--chemistry-core-half-width-cm1",
        type=float,
        default=7.0,
        help="Initial +/- Raman-shift tolerance for user-specified chemistry core peaks.",
    )
    parser.add_argument(
        "--chemistry-auxiliary-max-weight",
        type=float,
        default=0.35,
        help="Maximum prior amplitude assigned to matched_shared auxiliary regions.",
    )
    parser.add_argument(
        "--chemistry-shared-core-weight",
        type=float,
        default=0.65,
        help="Prior weight for the overlapping DEL~1600 / TEB~1597 core region.",
    )

    parser.add_argument("--loss-weighting", choices=("fixed", "dwa"), default="dwa")
    parser.add_argument("--classification-weight", type=float, default=0.5)
    parser.add_argument("--regression-weight", type=float, default=0.5)
    parser.add_argument("--dwa-temperature", type=float, default=2.0)

    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-validation-batches", type=int, default=None)
    parser.add_argument(
        "--skip-final-evaluation",
        action="store_true",
        help="Skip automatic validation/test evaluation after training (useful for smoke tests).",
    )
    parser.add_argument(
        "--final-test-evaluation",
        action="store_true",
        help=(
            "After training, also open the held-out test set for final reporting. "
            "Leave this off during model development/tuning."
        ),
    )
    parser.add_argument(
        "--evaluation-threshold",
        type=float,
        default=0.5,
        help="Fixed multilabel probability threshold. Test data are never used to tune it.",
    )
    parser.add_argument(
        "--ordinal-boundary-weight",
        type=float,
        default=0.0,
        help=(
            "T3.2-A: weight of the boundary-aware ordinal margin loss. "
            "0.0 preserves the original T3.1 objective."
        ),
    )
    parser.add_argument(
        "--ordinal-boundary-margin",
        type=float,
        default=0.15,
        help=(
            "T3.2-A: confidence margin around the 0.5 ordinal decision boundary. "
            "For margin=0.15, low/high targets are 0.35/0.65."
        ),
    )

    parser.add_argument(
        "--ternary-boundary-multiplier",
        type=float,
        default=1.0,
        help=(
            "T3.5: multiply only the ordinal boundary-loss contribution "
            "of true ternary pesticide spectra. "
            "1.0 disables ternary reweighting."
        ),
    )

    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, device: torch.device):
    generator = torch.Generator()
    generator.manual_seed(2026)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        drop_last=(shuffle and len(dataset) % batch_size == 1),
        generator=generator if shuffle else None,
    )


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    output = dict(batch)
    for key in (
        "raw",
        "percentile",
        "smoothed",
        "valid_mask",
        "class_target",
        "concentration_target",
    ):
        output[key] = batch[key].to(device, non_blocking=True)
    if "raw_intensity" in batch:
        output["raw_intensity"] = batch["raw_intensity"].to(device, non_blocking=True)
    return output



def present_only_boundary_margin_loss(
    ordinal_probability: torch.Tensor,
    concentration_target: torch.Tensor,
    presence: torch.Tensor,
    *,
    margin: float = 0.15,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    # T3.2-A boundary-aware ordinal margin loss.
    # ordinal_probability[..., 0] = P(level >= M)
    # ordinal_probability[..., 1] = P(level >= H)
    # Only true-present pesticide targets contribute.
    if ordinal_probability.ndim != 3:
        raise ValueError(
            "ordinal_probability must have shape [B, P, 2], "
            f"got {tuple(ordinal_probability.shape)}"
        )
    if ordinal_probability.shape[-1] != 2:
        raise ValueError(
            "ordinal_probability last dimension must be 2 "
            "for [P(level>=M), P(level>=H)]."
        )
    if concentration_target.shape != ordinal_probability.shape[:2]:
        raise ValueError(
            "concentration_target shape must match ordinal_probability[:2]."
        )
    if presence.shape != ordinal_probability.shape[:2]:
        raise ValueError(
            "presence shape must match ordinal_probability[:2]."
        )

    margin = float(margin)
    if not 0.0 <= margin < 0.5:
        raise ValueError(
            f"margin must satisfy 0 <= margin < 0.5, got {margin}"
        )

    present_mask = presence > 0.5
    if not bool(present_mask.any()):
        return ordinal_probability.sum() * 0.0

    batch_size = ordinal_probability.shape[0]

    if sample_weight is None:
        sample_weight = torch.ones(
            batch_size,
            dtype=ordinal_probability.dtype,
            device=ordinal_probability.device,
        )
    else:
        if sample_weight.ndim != 1 or sample_weight.shape[0] != batch_size:
            raise ValueError(
                "sample_weight must have shape [B], "
                f"got {tuple(sample_weight.shape)}"
            )

        sample_weight = sample_weight.to(
            dtype=ordinal_probability.dtype,
            device=ordinal_probability.device,
        )

        if not torch.isfinite(sample_weight).all():
            raise ValueError(
                "sample_weight contains NaN or Inf"
            )

        if bool((sample_weight <= 0.0).any()):
            raise ValueError(
                "sample_weight values must all be > 0"
            )

    weight_matrix = (
        sample_weight.unsqueeze(1)
        .expand_as(presence)
    )

    level = concentration_target.round().long()
    p_ge_m = ordinal_probability[..., 0]
    p_ge_h = ordinal_probability[..., 1]

    low = 0.5 - margin
    high = 0.5 + margin

    penalties: list[torch.Tensor] = []
    penalty_weights: list[torch.Tensor] = []

    low_mask = present_mask & level.eq(1)
    middle_mask = present_mask & level.eq(2)
    high_mask = present_mask & level.eq(3)

    # S should stay clearly below the S/M decision boundary.
    if bool(low_mask.any()):
        penalties.append(
            torch.relu(p_ge_m[low_mask] - low).square()
        )
        penalty_weights.append(
            weight_matrix[low_mask]
        )

    # M should stay above the S/M boundary and below the M/H boundary.
    if bool(middle_mask.any()):
        penalties.append(
            torch.relu(high - p_ge_m[middle_mask]).square()
        )
        penalty_weights.append(
            weight_matrix[middle_mask]
        )

        penalties.append(
            torch.relu(p_ge_h[middle_mask] - low).square()
        )
        penalty_weights.append(
            weight_matrix[middle_mask]
        )

    # H should stay clearly above the M/H decision boundary.
    if bool(high_mask.any()):
        penalties.append(
            torch.relu(high - p_ge_h[high_mask]).square()
        )
        penalty_weights.append(
            weight_matrix[high_mask]
        )

    if not penalties:
        return ordinal_probability.sum() * 0.0

    penalty_vector = torch.cat(
        [item.reshape(-1) for item in penalties],
        dim=0,
    )

    weight_vector = torch.cat(
        [item.reshape(-1) for item in penalty_weights],
        dim=0,
    )

    return (
        penalty_vector * weight_vector
    ).sum() / weight_vector.sum().clamp_min(1e-12)


def _ordinal_metrics_from_probabilities(
    concentration_target: np.ndarray,
    class_target: np.ndarray,
    class_probability: np.ndarray,
    ordinal_probability: np.ndarray,
    threshold: float = 0.5,
    decoding: str = "median",
) -> dict[str, float]:
    """Training-time level metrics; final detailed tables are produced by Inference.py."""
    concentration_target = np.asarray(concentration_target)
    class_target = np.asarray(class_target)
    class_probability = np.asarray(class_probability)
    ordinal_probability = np.asarray(ordinal_probability)

    present = class_target > 0.5
    predicted_present_level = decode_ordinal_numpy(ordinal_probability, mode=decoding, threshold=threshold)
    true_level = np.rint(concentration_target).astype(np.int64)
    present_accuracy = float(
        np.mean(predicted_present_level[present] == true_level[present])
    ) if present.any() else float("nan")

    predicted_presence = class_probability >= 0.5
    final_level = np.where(predicted_presence, predicted_present_level, 0)
    exact_profile = float(np.mean(np.all(final_level == true_level, axis=1)))

    metrics = {
        "ordinal_accuracy_present": present_accuracy,
        "ordinal_exact_profile_accuracy": exact_profile,
    }
    for index, pesticide in enumerate(PESTICIDES):
        mask = present[:, index]
        value = float(
            np.mean(predicted_present_level[mask, index] == true_level[mask, index])
        ) if mask.any() else float("nan")
        metrics[f"ordinal_accuracy_{pesticide}"] = value
        ternary_pesticide = mask & (present.sum(axis=1) == 3)
        metrics[f"ordinal_accuracy_{pesticide}_ternary"] = float(
            np.mean(predicted_present_level[ternary_pesticide, index] == true_level[ternary_pesticide, index])
        ) if ternary_pesticide.any() else float("nan")
    error = np.abs(predicted_present_level - true_level)[present]
    metrics["ordinal_mae_present"] = float(error.mean()) if error.size else float("nan")
    metrics["ordinal_severe_error_rate"] = float(np.mean(error >= 2)) if error.size else float("nan")
    ternary = present.sum(axis=1) == 3
    metrics["ordinal_exact_ternary"] = float(np.mean(np.all(final_level[ternary] == true_level[ternary], axis=1))) if ternary.any() else float("nan")
    return metrics


def run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    classification_loss_fn: nn.Module,
    regression_loss_fn: nn.Module,
    concentration_head_mode: str,
    ordinal_loss_mode: str,
    ordinal_boundary_weight: float,
    ordinal_boundary_margin: float,
    ternary_boundary_multiplier: float,
    classification_weight: float,
    regression_weight: float,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
    ordinal_decoding: str = "median",
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    training = optimizer is not None
    model.train(training)

    total_loss_sum = 0.0
    class_loss_sum = 0.0
    reg_loss_sum = 0.0
    ordinal_base_loss_sum = 0.0
    ordinal_boundary_loss_sum = 0.0
    sample_count = 0

    class_targets: list[np.ndarray] = []
    class_probabilities: list[np.ndarray] = []
    concentration_targets: list[np.ndarray] = []
    concentration_predictions: list[np.ndarray] = []
    ordinal_probabilities: list[np.ndarray] = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            batch = _to_device(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)

            class_target = batch["class_target"]
            reg_target = batch["concentration_target"]

            if concentration_head_mode == "ordinal":
                if ordinal_loss_mode == "cumulative":
                    class_pred, reg_pred, ordinal_probability = model(
                        batch,
                        return_ordinal=True,
                    )

                    ordinal_base_loss = present_only_ordinal_bce(
                        ordinal_probability,
                        reg_target,
                        class_target,
                    )

                elif ordinal_loss_mode == "corn":
                    (
                        class_pred,
                        reg_pred,
                        ordinal_probability,
                        ordinal_logits,
                    ) = model(
                        batch,
                        return_ordinal=True,
                        return_ordinal_logits=True,
                    )

                    ordinal_base_loss = present_only_corn_ordinal_loss(
                        ordinal_logits,
                        reg_target,
                        class_target,
                    )

                else:
                    raise ValueError(
                        f"Unknown ordinal_loss_mode: {ordinal_loss_mode}"
                    )

                mixture_count = class_target.sum(
                    dim=1
                )

                ternary_sample = (
                    mixture_count >= 2.5
                )

                boundary_sample_weight = torch.ones(
                    class_target.shape[0],
                    dtype=ordinal_probability.dtype,
                    device=ordinal_probability.device,
                )

                boundary_sample_weight = torch.where(
                    ternary_sample,
                    torch.full_like(
                        boundary_sample_weight,
                        float(ternary_boundary_multiplier),
                    ),
                    boundary_sample_weight,
                )

                ordinal_boundary_loss = present_only_boundary_margin_loss(
                    ordinal_probability,
                    reg_target,
                    class_target,
                    margin=ordinal_boundary_margin,
                    sample_weight=boundary_sample_weight,
                )
                reg_loss = (
                    ordinal_base_loss
                    + ordinal_boundary_weight
                    * ordinal_boundary_loss
                )
            elif concentration_head_mode == "continuous":
                class_pred, reg_pred = model(batch)
                ordinal_probability = None
                ordinal_base_loss = reg_target.sum() * 0.0
                ordinal_boundary_loss = reg_target.sum() * 0.0
                # Preserve the exact T1/T2 objective for checkpoint/ablation compatibility.
                reg_loss = regression_loss_fn(reg_pred, reg_target)
            else:
                raise ValueError(
                    f"Unknown concentration_head_mode: {concentration_head_mode}"
                )

            class_loss = classification_loss_fn(class_pred, class_target)
            total_loss = classification_weight * class_loss + regression_weight * reg_loss

            if training:
                total_loss.backward()
                optimizer.step()

            batch_size = int(class_target.shape[0])
            sample_count += batch_size
            total_loss_sum += float(total_loss.detach().cpu()) * batch_size
            class_loss_sum += float(class_loss.detach().cpu()) * batch_size
            reg_loss_sum += float(reg_loss.detach().cpu()) * batch_size
            if concentration_head_mode == "ordinal":
                ordinal_base_loss_sum += (
                    float(ordinal_base_loss.detach().cpu()) * batch_size
                )
                ordinal_boundary_loss_sum += (
                    float(ordinal_boundary_loss.detach().cpu()) * batch_size
                )

            class_targets.append(class_target.detach().cpu().numpy())
            class_probabilities.append(class_pred.detach().cpu().numpy())
            concentration_targets.append(reg_target.detach().cpu().numpy())
            concentration_predictions.append(reg_pred.detach().cpu().numpy())
            if ordinal_probability is not None:
                ordinal_probabilities.append(
                    ordinal_probability.detach().cpu().numpy()
                )

    if sample_count == 0:
        raise RuntimeError("No samples were processed in the epoch")

    y_class = np.concatenate(class_targets, axis=0)
    p_class = np.concatenate(class_probabilities, axis=0)
    y_reg = np.concatenate(concentration_targets, axis=0)
    p_reg = np.concatenate(concentration_predictions, axis=0)

    loss_metrics = {
        "loss_total": total_loss_sum / sample_count,
        "loss_class": class_loss_sum / sample_count,
        "loss_reg": reg_loss_sum / sample_count,
    }
    if concentration_head_mode == "ordinal":
        loss_metrics["loss_ordinal_base"] = (
            ordinal_base_loss_sum / sample_count
        )
        loss_metrics["loss_ordinal_boundary"] = (
            ordinal_boundary_loss_sum / sample_count
        )
    metrics = merge_metrics(
        loss_metrics,
        multilabel_classification_metrics(y_class, p_class),
        regression_metrics(y_reg, p_reg, y_class),
    )
    arrays = {
        "class_target": y_class,
        "class_probability": p_class,
        "concentration_target": y_reg,
        "concentration_prediction": p_reg,
    }
    if concentration_head_mode == "ordinal":
        if not ordinal_probabilities:
            raise RuntimeError("Ordinal mode produced no ordinal probabilities")
        p_ordinal = np.concatenate(ordinal_probabilities, axis=0)
        metrics.update(
            _ordinal_metrics_from_probabilities(
                y_reg,
                y_class,
                p_class,
                p_ordinal,
                decoding=ordinal_decoding,
            )
        )
        arrays["ordinal_probability"] = p_ordinal
    return metrics, arrays


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    epoch: int,
    best_validation_loss: float,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    data_config: dict[str, Any],
    concentration_map: dict[str, dict[str, float]] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_version": 4 if model_config.get("use_local_quantitative_branch", False) else 3,
            "epoch": int(epoch),
            "best_validation_loss": float(best_validation_loss),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_config": model_config,
            "training_config": training_config,
            "data_config": data_config,
            "concentration_map": concentration_map,
            "pesticides": list(PESTICIDES),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.concentration_head_mode != "ordinal" and (
        args.local_quantitative_branch or args.checkpoint_selection == "ordinal" or args.ordinal_decoding != "median"
    ):
        raise ValueError("Local quantitative branch / ordinal selection / MAP decoding require --concentration-head-mode ordinal")
    if args.local_quantitative_branch and args.boundary_specific_mixture_gate:
        raise ValueError("T3.13 first ablation requires --boundary-specific-mixture-gate to remain disabled")

    if (
        args.concentration_head_mode != "ordinal"
        and args.ordinal_loss_mode != "cumulative"
    ):
        raise ValueError(
            "--ordinal-loss-mode corn requires "
            "--concentration-head-mode ordinal"
        )

    seed_everything(args.seed)
    device = resolve_device(args.device)
    output_directory = args.output_directory.resolve()
    checkpoint_directory = output_directory / "checkpoints"
    if checkpoint_directory.exists() and any(checkpoint_directory.glob("*.pt")):
        raise FileExistsError(f"Output already contains checkpoints; choose a new --output-directory: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)

    if (
        args.adaptive_mixture_gate
        and not args.mixture_aware_query_fusion
    ):
        raise ValueError(
            "--adaptive-mixture-gate requires "
            "--mixture-aware-query-fusion"
        )

    if (
        args.boundary_specific_mixture_gate
        and not args.adaptive_mixture_gate
    ):
        raise ValueError(
            "--boundary-specific-mixture-gate requires "
            "--adaptive-mixture-gate"
        )

    concentration_map = load_concentration_map(args.concentration_map)
    train_dataset, validation_dataset, test_dataset = build_datasets(
        include_generated_train=args.include_generated,
        maximum_generated_per_condition=args.maximum_generated_per_condition,
        require_complete_generated=not args.allow_incomplete_generated,
        concentration_map=concentration_map,
    )

    axis_audit = build_axis_label_audit(train_dataset.repository)
    with (output_directory / "axis_label_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(axis_audit, handle, ensure_ascii=False, indent=2)

    print("===== Source-axis label audit =====")
    print(json.dumps(axis_audit["pesticide_by_source_axis"], ensure_ascii=False, indent=2))
    print(
        f"Model input uses the full Raman range "
        f"{MODEL_RAMAN_AXIS[0]:.0f}-{MODEL_RAMAN_AXIS[-1]:.0f} cm^-1 "
        f"({MODEL_LENGTH} points); existing short-axis sources follow the repository's in-memory noise-tail policy."
    )

    print("===== Dataset =====")
    print(f"train      : {len(train_dataset)}")
    print(f"validation : {len(validation_dataset)}")
    print(f"test       : {len(test_dataset)}")
    print(f"target mode: {train_dataset.repository.target_mode}")
    if train_dataset.repository.target_mode == "level_code":
        print(
            "WARNING: concentration_target currently uses level codes 0/1/2/3; "
            "do not interpret regression metrics as physical concentration metrics."
        )
    if (
        args.concentration_head_mode == "ordinal"
        and train_dataset.repository.target_mode != "level_code"
    ):
        raise ValueError(
            "T3.1 ordinal concentration head requires level_code targets S/M/H. "
            "Use continuous mode when physical concentration targets are enabled."
        )
    print(f"concentration head: {args.concentration_head_mode}")
    if args.concentration_head_mode == "ordinal":
        if args.ordinal_boundary_weight < 0.0:
            raise ValueError("--ordinal-boundary-weight must be >= 0")
        if not 0.0 <= args.ordinal_boundary_margin < 0.5:
            raise ValueError(
                "--ordinal-boundary-margin must satisfy 0 <= margin < 0.5"
            )

        if args.ternary_boundary_multiplier < 1.0:
            raise ValueError(
                "--ternary-boundary-multiplier must be >= 1.0"
            )

        print(
            "ordinal boundary-aware loss: "
            f"weight={args.ordinal_boundary_weight:.4f}, "
            f"margin={args.ordinal_boundary_margin:.4f}, "
            f"ternary_multiplier="
            f"{args.ternary_boundary_multiplier:.4f}"
        )

    train_loader = make_loader(
        train_dataset, args.batch_size, True, args.num_workers, device
    )
    validation_loader = make_loader(
        validation_dataset, args.batch_size, False, args.num_workers, device
    )

    use_peak_guidance = args.query_attention_mode == "peak_guided"
    if use_peak_guidance:
        # Peak guidance is fitted only from the fixed real training split.
        # Generated spectra, validation spectra and test spectra are excluded.
        if args.peak_prior_mode == "chemistry_hybrid":
            peak_prior, peak_prior_summary = build_chemistry_hybrid_peak_priors(
                train_dataset.repository,
                core_half_width_cm1=args.chemistry_core_half_width_cm1,
                auxiliary_max_weight=args.chemistry_auxiliary_max_weight,
                shared_core_weight=args.chemistry_shared_core_weight,
                top_k=args.peak_top_k,
                min_peak_distance_cm1=args.peak_min_distance_cm1,
                peak_prominence=args.peak_prominence,
                auxiliary_peak_half_width_cm1=args.peak_half_width_cm1,
                min_matched_pairs=args.peak_min_matched_pairs,
                shared_peak_floor=args.peak_shared_floor,
            )
        elif args.peak_prior_mode == "matched_shared":
            peak_prior, peak_prior_summary = build_matched_condition_peak_priors(
                train_dataset.repository,
                top_k=args.peak_top_k,
                min_peak_distance_cm1=args.peak_min_distance_cm1,
                peak_prominence=args.peak_prominence,
                peak_half_width_cm1=args.peak_half_width_cm1,
                min_matched_pairs=args.peak_min_matched_pairs,
                shared_peak_floor=args.peak_shared_floor,
            )
        else:
            peak_prior, peak_prior_summary = build_training_peak_priors(
                train_dataset.repository,
                top_k=args.peak_top_k,
                min_peak_distance_cm1=args.peak_min_distance_cm1,
                peak_prominence=args.peak_prominence,
                peak_half_width_cm1=args.peak_half_width_cm1,
                min_support_fraction=args.peak_min_support_fraction,
            )

        peak_prior_frame = pd.DataFrame({"Raman_shift_cm-1": MODEL_RAMAN_AXIS})
        for pesticide_index, pesticide in enumerate(PESTICIDES):
            peak_prior_frame[f"peak_prior_{pesticide}"] = peak_prior[pesticide_index]
        peak_prior_frame.to_csv(
            output_directory / "training_peak_priors.csv", index=False
        )
        with (output_directory / "training_peak_prior_summary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(peak_prior_summary, handle, ensure_ascii=False, indent=2)

        print("===== Soft spectral priors =====")
        print(f"prior mode: {args.peak_prior_mode}")
        for pesticide in PESTICIDES:
            info = peak_prior_summary["pesticides"][pesticide]
            if args.peak_prior_mode == "chemistry_hybrid":
                core = ", ".join(
                    f"{value:.1f}" for value in info["core_peak_centers_cm-1"]
                )
                auxiliary = ", ".join(
                    f"{value:.1f}"
                    for value in info["auxiliary_peak_centers_cm-1"]
                )
                print(
                    f"{pesticide}: core=[{core}] cm^-1; "
                    f"auxiliary=[{auxiliary}] cm^-1; "
                    f"matched_pairs={info['matched_pair_count']}"
                )
            else:
                centers = ", ".join(
                    f"{value:.1f}" for value in info["peak_centers_cm-1"]
                )
                pair_text = (
                    f" | matched_pairs={info.get('matched_pair_count')}"
                    if "matched_pair_count" in info
                    else ""
                )
                print(f"{pesticide}: {centers} cm^-1{pair_text}")
    else:
        peak_prior = np.zeros((len(PESTICIDES), MODEL_LENGTH), dtype=np.float32)
        peak_prior_summary = {
            "mode": "query_only",
            "description": (
                "Peak bias disabled. DEL/CHL/TEB learnable queries attend to the "
                "entire valid 600-2500 cm^-1 spectrum."
            ),
        }
        print("===== Query attention mode =====")
        print(
            "query_only: peak prior is disabled; pesticide-specific queries learn "
            "attention directly from the full valid spectrum."
        )

    model_config = {
        "dim_model": args.dim_model,
        "attn_head": args.attention_heads,
        "dim_ff": args.dim_ff,
        "drop": args.dropout,
        "batch_f": True,
        "encoder_layers": args.encoder_layers,
        "n_labels": len(PESTICIDES),
        "model_length": MODEL_LENGTH,
        "peak_guidance_strength_init": args.peak_guidance_strength,
        "use_peak_guidance": use_peak_guidance,
        "concentration_head_mode": args.concentration_head_mode,
        "ordinal_head_hidden": args.ordinal_head_hidden,
        "use_mixture_aware_query_fusion": bool(
            args.mixture_aware_query_fusion
        ),
        "use_adaptive_mixture_gate": bool(
            args.adaptive_mixture_gate
        ),
        "use_boundary_specific_mixture_gate": bool(
            args.boundary_specific_mixture_gate
        ),
        "use_local_quantitative_branch": bool(args.local_quantitative_branch),
        "local_window_centers": [
            [float(peak["center_cm-1"]) for peak in CHEMISTRY_CORE_PEAKS[pesticide]]
            for pesticide in PESTICIDES
        ] if args.local_quantitative_branch else None,
        "local_half_width": args.local_half_width_cm1,
        "local_hidden": args.local_hidden,
        "local_strength": args.local_strength,
    }
    if not args.local_quantitative_branch:
        for key in ("use_local_quantitative_branch", "local_window_centers", "local_half_width", "local_hidden", "local_strength"):
            model_config.pop(key)
    model = TransformerClassifyRegress_sep(**model_config)
    model.set_peak_prior(torch.from_numpy(peak_prior))
    local_normalization_summary = None
    if model.local_quantitative_branch is not None:
        train_intensity, train_masks, fit_audit = collect_real_training_intensities(train_dataset)
        local_normalization_summary = model.local_quantitative_branch.fit_normalization(
            torch.from_numpy(train_intensity), torch.from_numpy(train_masks)
        )
        local_normalization_summary["audit"] = fit_audit
        with (output_directory / "local_quantitative_state.json").open("w", encoding="utf-8") as handle:
            json.dump(local_normalization_summary, handle, ensure_ascii=False, indent=2)
        print("===== T3.13 local quantitative normalization =====")
        print(json.dumps(fit_audit, ensure_ascii=False))
        print(f"native windows +/-{args.local_half_width_cm1} cm^-1; hidden={args.local_hidden}; strength={args.local_strength}")
        del train_intensity, train_masks
    model = model.to(device)

    if args.concentration_head_mode == "ordinal":
        print(
            "ordinal head hidden dimension: "
            f"{model.ordinal_head_hidden}"
        )
        print(
            "ordinal loss mode:",
            args.ordinal_loss_mode,
        )

    print(
        "mixture-aware query fusion:",
        (
            "enabled for ordinal concentration branch"
            if args.mixture_aware_query_fusion
            else "disabled"
        ),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-7,
    )
    classification_loss_fn = nn.BCELoss()
    regression_loss_fn = nn.MSELoss()

    class_history: list[float] = []
    reg_history: list[float] = []
    rows: list[dict[str, Any]] = []
    best_validation_loss = math.inf
    best_ordinal_accuracy = -math.inf
    best_ordinal_exact = -math.inf
    ordinal_epochs_without_improvement = 0
    epochs_without_improvement = 0

    training_config = vars(args).copy()
    for key, value in list(training_config.items()):
        if isinstance(value, Path):
            training_config[key] = str(value)

    data_config = {
        "real_root": str(train_dataset.repository.real_root),
        "generated_root": str(train_dataset.repository.generated_root),
        "include_generated": bool(args.include_generated),
        "maximum_generated_per_condition": args.maximum_generated_per_condition,
        "split": "12/4/4 within each source file",
        "model_axis": "600-2500 cm^-1 model axis, 1901 points",
        "source_axis_handling": (
            "genuine full-range sources retained; short-axis tails completed in memory "
            "using deterministic background-matched noise (existing Dataset policy)"
        ),
        "axis_label_audit": axis_audit,
        "query_attention_mode": args.query_attention_mode,
        "concentration_head_mode": args.concentration_head_mode,
        "ordinal_head_hidden": (
            model.ordinal_head_hidden
            if args.concentration_head_mode == "ordinal"
            else None
        ),
        "mixture_aware_query_fusion": bool(
            args.mixture_aware_query_fusion
        ),
        "ordinal_boundary_weight": args.ordinal_boundary_weight,
        "ordinal_boundary_margin": args.ordinal_boundary_margin,
        "ternary_boundary_multiplier": args.ternary_boundary_multiplier,
        "concentration_training_scope": (
            "present-only cumulative ordinal BCE"
            if args.concentration_head_mode == "ordinal"
            else "legacy T1/T2 continuous MSE"
        ),
        "peak_prior_fit": (
            "real training spectra only; validation/test/generated excluded"
            if use_peak_guidance
            else "disabled for query-only ablation"
        ),
        "peak_prior_summary": peak_prior_summary,
        "target_mode": train_dataset.repository.target_mode,
        "local_quantitative_normalization": local_normalization_summary,
    }

    training_config["ordinal_loss_mode"] = (
        args.ordinal_loss_mode
    )
    training_config["ordinal_decoding"] = args.ordinal_decoding
    training_config["checkpoint_selection"] = args.checkpoint_selection

    for epoch in range(1, args.epochs + 1):
        if args.loss_weighting == "dwa":
            class_weight, reg_weight = compute_dwa_weights(
                class_history, reg_history, temperature=args.dwa_temperature
            )
        else:
            weight_sum = args.classification_weight + args.regression_weight
            if weight_sum <= 0.0:
                raise ValueError("Classification + regression weights must be positive")
            class_weight = args.classification_weight / weight_sum
            reg_weight = args.regression_weight / weight_sum

        train_metrics, _ = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            classification_loss_fn=classification_loss_fn,
            regression_loss_fn=regression_loss_fn,
            concentration_head_mode=args.concentration_head_mode,
            ordinal_loss_mode=args.ordinal_loss_mode,
            ordinal_boundary_weight=args.ordinal_boundary_weight,
            ordinal_boundary_margin=args.ordinal_boundary_margin,
            ternary_boundary_multiplier=args.ternary_boundary_multiplier,
            classification_weight=class_weight,
            regression_weight=reg_weight,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
            ordinal_decoding=args.ordinal_decoding,
        )
        class_history.append(train_metrics["loss_class"])
        reg_history.append(train_metrics["loss_reg"])

        validation_metrics, _ = run_epoch(
            model=model,
            loader=validation_loader,
            device=device,
            classification_loss_fn=classification_loss_fn,
            regression_loss_fn=regression_loss_fn,
            concentration_head_mode=args.concentration_head_mode,
            ordinal_loss_mode=args.ordinal_loss_mode,
            ordinal_boundary_weight=args.ordinal_boundary_weight,
            ordinal_boundary_margin=args.ordinal_boundary_margin,
            ternary_boundary_multiplier=args.ternary_boundary_multiplier,
            classification_weight=class_weight,
            regression_weight=reg_weight,
            optimizer=None,
            max_batches=args.max_validation_batches,
            ordinal_decoding=args.ordinal_decoding,
        )

        validation_loss = validation_metrics["loss_total"]
        scheduler.step(validation_loss)
        learning_rate = float(optimizer.param_groups[0]["lr"])

        row: dict[str, Any] = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "classification_weight": class_weight,
            "regression_weight": reg_weight,
        }
        with torch.no_grad():
            peak_strength = (
                model.peak_guided_attention.effective_guidance_strength
                .detach().cpu().numpy()
            )
        for pesticide_index, pesticide in enumerate(PESTICIDES):
            row[f"peak_guidance_strength_{pesticide}"] = float(
                peak_strength[pesticide_index]
            )
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"validation_{k}": v for k, v in validation_metrics.items()})
        rows.append(row)
        pd.DataFrame(rows).to_csv(output_directory / "training_history.csv", index=False)

        if args.concentration_head_mode == "ordinal":
            concentration_text = (
                f"val_level_acc_present="
                f"{validation_metrics['ordinal_accuracy_present']:.4f} "
                f"val_level_exact="
                f"{validation_metrics['ordinal_exact_profile_accuracy']:.4f}"
            )
        else:
            concentration_text = (
                f"val_RMSE_present="
                f"{validation_metrics['reg_rmse_present']:.6f}"
            )
        print(
            f"epoch={epoch:03d} "
            f"train={train_metrics['loss_total']:.6f} "
            f"val={validation_loss:.6f} "
            f"val_F1_micro={validation_metrics['class_f1_micro']:.4f} "
            f"{concentration_text} "
            f"weights=({class_weight:.3f},{reg_weight:.3f}) "
            f"lr={learning_rate:.3e}"
        )

        save_checkpoint(
            checkpoint_directory / "latest.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_validation_loss=min(best_validation_loss, validation_loss),
            model_config=model_config,
            training_config=training_config,
            data_config=data_config,
            concentration_map=concentration_map,
        )

        loss_improved = validation_loss < best_validation_loss
        if loss_improved:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_directory / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_validation_loss=best_validation_loss,
                model_config=model_config,
                training_config=training_config,
                data_config=data_config,
                concentration_map=concentration_map,
            )
        else:
            epochs_without_improvement += 1

        ordinal_selected_improved = False
        if args.concentration_head_mode == "ordinal":
            current_ordinal_accuracy = float(
                validation_metrics[
                    "ordinal_accuracy_present"
                ]
            )

            current_ordinal_exact = float(
                validation_metrics[
                    "ordinal_exact_profile_accuracy"
                ]
            )

            ordinal_improved = (
                current_ordinal_accuracy
                > best_ordinal_accuracy
                + 1e-12
            )

            ordinal_tied_but_exact_improved = (
                abs(
                    current_ordinal_accuracy
                    - best_ordinal_accuracy
                )
                <= 1e-12
                and current_ordinal_exact
                > best_ordinal_exact
                + 1e-12
            )

            if (
                ordinal_improved
                or ordinal_tied_but_exact_improved
            ):
                ordinal_selected_improved = True
                best_ordinal_accuracy = (
                    current_ordinal_accuracy
                )
                best_ordinal_exact = (
                    current_ordinal_exact
                )

                save_checkpoint(
                    checkpoint_directory
                    / "best_ordinal.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_validation_loss=min(
                        best_validation_loss,
                        validation_loss,
                    ),
                    model_config=model_config,
                    training_config=training_config,
                    data_config=data_config,
                    concentration_map=concentration_map,
                )

        if args.checkpoint_selection == "ordinal":
            # The loss policy above remains legacy-compatible; ordinal selection
            # has its own counter and must not inherit a loss-based reset.
            if ordinal_selected_improved:
                ordinal_epochs_without_improvement = 0
            else:
                ordinal_epochs_without_improvement += 1
            epochs_without_improvement = ordinal_epochs_without_improvement

        if epochs_without_improvement >= args.early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch}: no validation improvement for "
                f"{args.early_stopping_patience} epochs."
            )
            break

    with (output_directory / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "best_validation_loss": best_validation_loss,
                "model_config": model_config,
                "training_config": training_config,
                "data_config": data_config,
                "checkpoint_selection": args.checkpoint_selection,
                "ordinal_decoding": args.ordinal_decoding,
                "best_ordinal_accuracy": best_ordinal_accuracy if args.concentration_head_mode == "ordinal" else None,
                "best_ordinal_exact": best_ordinal_exact if args.concentration_head_mode == "ordinal" else None,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    best_checkpoint = checkpoint_directory / ("best_ordinal.pt" if args.checkpoint_selection == "ordinal" else "best.pt")
    print(f"Training complete. Best checkpoint: {best_checkpoint}")

    if not args.skip_final_evaluation:
        print("===== Automatic comprehensive evaluation =====")
        from Inference import evaluate_checkpoint

        evaluation_splits = ("validation", "test") if args.final_test_evaluation else ("validation",)
        if not args.final_test_evaluation:
            print(
                "Held-out test set remains sealed. Use --final-test-evaluation only "
                "for the final frozen model."
            )

        evaluate_checkpoint(
            checkpoint_path=best_checkpoint,
            output_directory=output_directory / "evaluation",
            device_name=args.device,
            batch_size=max(args.batch_size, 64),
            num_workers=args.num_workers,
            classification_threshold=args.evaluation_threshold,
            splits=evaluation_splits,
            training_history_path=output_directory / "training_history.csv",
        )


if __name__ == "__main__":
    main()
