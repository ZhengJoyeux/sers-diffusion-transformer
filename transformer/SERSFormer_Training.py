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
    MODEL_LENGTH,
    MODEL_RAMAN_AXIS,
    PESTICIDES,
    build_axis_label_audit,
    build_datasets,
    build_training_peak_priors,
    load_concentration_map,
)
from Metric import (
    compute_dwa_weights,
    merge_metrics,
    multilabel_classification_metrics,
    regression_metrics,
)
from Model_v2 import TransformerClassifyRegress_sep


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
    parser.add_argument("--peak-top-k", type=int, default=8)
    parser.add_argument("--peak-min-distance-cm1", type=float, default=20.0)
    parser.add_argument("--peak-prominence", type=float, default=0.02)
    parser.add_argument("--peak-half-width-cm1", type=float, default=15.0)
    parser.add_argument("--peak-min-support-fraction", type=float, default=0.25)
    parser.add_argument("--peak-guidance-strength", type=float, default=1.5)

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
    return output


def run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    classification_loss_fn: nn.Module,
    regression_loss_fn: nn.Module,
    classification_weight: float,
    regression_weight: float,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    training = optimizer is not None
    model.train(training)

    total_loss_sum = 0.0
    class_loss_sum = 0.0
    reg_loss_sum = 0.0
    sample_count = 0

    class_targets: list[np.ndarray] = []
    class_probabilities: list[np.ndarray] = []
    concentration_targets: list[np.ndarray] = []
    concentration_predictions: list[np.ndarray] = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            batch = _to_device(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)

            class_pred, reg_pred = model(batch)
            class_target = batch["class_target"]
            reg_target = batch["concentration_target"]

            class_loss = classification_loss_fn(class_pred, class_target)
            reg_loss = regression_loss_fn(reg_pred, reg_target)
            total_loss = classification_weight * class_loss + regression_weight * reg_loss

            if training:
                total_loss.backward()
                optimizer.step()

            batch_size = int(class_target.shape[0])
            sample_count += batch_size
            total_loss_sum += float(total_loss.detach().cpu()) * batch_size
            class_loss_sum += float(class_loss.detach().cpu()) * batch_size
            reg_loss_sum += float(reg_loss.detach().cpu()) * batch_size

            class_targets.append(class_target.detach().cpu().numpy())
            class_probabilities.append(class_pred.detach().cpu().numpy())
            concentration_targets.append(reg_target.detach().cpu().numpy())
            concentration_predictions.append(reg_pred.detach().cpu().numpy())

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
            "checkpoint_version": 1,
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
    seed_everything(args.seed)
    device = resolve_device(args.device)
    output_directory = args.output_directory.resolve()
    checkpoint_directory = output_directory / "checkpoints"
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)

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
        f"Model input is restricted to common Raman range "
        f"{MODEL_RAMAN_AXIS[0]:.0f}-{MODEL_RAMAN_AXIS[-1]:.0f} cm^-1 "
        f"({MODEL_LENGTH} points) to prevent axis-length label leakage."
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

        print("===== Training-only peak priors =====")
        for pesticide in PESTICIDES:
            info = peak_prior_summary["pesticides"][pesticide]
            centers = ", ".join(
                f"{value:.1f}" for value in info["peak_centers_cm-1"]
            )
            print(f"{pesticide}: {centers} cm^-1")
    else:
        peak_prior = np.zeros((len(PESTICIDES), MODEL_LENGTH), dtype=np.float32)
        peak_prior_summary = {
            "mode": "query_only",
            "description": (
                "Peak bias disabled. DEL/CHL/TEB learnable queries attend to the "
                "entire valid 600-2000 cm^-1 spectrum."
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
    }
    model = TransformerClassifyRegress_sep(**model_config)
    model.set_peak_prior(torch.from_numpy(peak_prior))
    model = model.to(device)

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
        "model_axis": "600-2000 cm^-1 common range, 1401 points",
        "source_axis_handling": (
            "original 1401/1901-point acquisitions are both restricted to the common "
            "600-2000 cm^-1 model range; valid_mask remains supported"
        ),
        "axis_label_audit": axis_audit,
        "query_attention_mode": args.query_attention_mode,
        "peak_prior_fit": (
            "real training spectra only; validation/test/generated excluded"
            if use_peak_guidance
            else "disabled for query-only ablation"
        ),
        "peak_prior_summary": peak_prior_summary,
        "target_mode": train_dataset.repository.target_mode,
    }

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
            classification_weight=class_weight,
            regression_weight=reg_weight,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
        )
        class_history.append(train_metrics["loss_class"])
        reg_history.append(train_metrics["loss_reg"])

        validation_metrics, _ = run_epoch(
            model=model,
            loader=validation_loader,
            device=device,
            classification_loss_fn=classification_loss_fn,
            regression_loss_fn=regression_loss_fn,
            classification_weight=class_weight,
            regression_weight=reg_weight,
            optimizer=None,
            max_batches=args.max_validation_batches,
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

        print(
            f"epoch={epoch:03d} "
            f"train={train_metrics['loss_total']:.6f} "
            f"val={validation_loss:.6f} "
            f"val_F1_micro={validation_metrics['class_f1_micro']:.4f} "
            f"val_RMSE_present={validation_metrics['reg_rmse_present']:.6f} "
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

        if validation_loss < best_validation_loss:
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
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    best_checkpoint = checkpoint_directory / "best.pt"
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
