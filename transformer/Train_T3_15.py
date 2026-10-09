"""Fine-tune only T3.10's concentration-specific fusion and ordinal heads.

The architecture is unchanged. Original feature/classification paths and their
buffers remain fixed. All real/generated training rows use equal weight.
Test is counted but never evaluated by this entry point.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from Dataset import build_datasets, PESTICIDES
from Metric import multilabel_classification_metrics, regression_metrics
from Model_v2 import TransformerClassifyRegress_sep
from SERSFormer_Training import seed_everything, make_loader, _to_device, _ordinal_metrics_from_probabilities
from T3_14_Calibration import weighted_corn_loss, anchor_kl_loss, guard_candidate, selection_score, warmup_cosine_factor
from Train_T3_14 import validation_strata, write_json, save_checkpoint


TRAINABLE_PREFIXES = ("mixture_aware_query_fusion.", "ordinal_heads.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--include-generated", action="store_true", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--warmup-start-ratio", type=float, default=0.1)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-6)
    parser.add_argument("--anchor-penalty", type=float, default=0.1)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-validation-batches", type=int, default=None)
    parser.add_argument("--skip-final-evaluation", action="store_true")
    parser.add_argument("--check-only", action="store_true",
                        help="Load/check inputs and full validation baseline without writing or training")
    return parser.parse_args()


def validate_args(args):
    if args.epochs < 1 or args.batch_size < 2 or args.num_workers < 0 or args.patience < 1:
        raise ValueError("Positive epochs/patience; batch-size>=2; num-workers>=0 required")
    if not 0 <= args.warmup_epochs < args.epochs:
        raise ValueError("warmup-epochs must be nonnegative and less than epochs")
    for value in (args.max_train_batches, args.max_validation_batches):
        if value is not None and value < 1:
            raise ValueError("Batch limits must be positive")
    if (not math.isfinite(args.anchor_penalty) or args.anchor_penalty < 0
            or not math.isfinite(args.weight_decay) or args.weight_decay < 0
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not math.isfinite(args.minimum_learning_rate)
            or not 0 < args.minimum_learning_rate < args.learning_rate
            or not math.isfinite(args.warmup_start_ratio)
            or not 0 < args.warmup_start_ratio <= 1):
        raise ValueError("Invalid loss / optimizer / warmup parameters")


def anchor_config(checkpoint):
    config = dict(checkpoint.get("model_config", {}))
    if (config.get("concentration_head_mode") != "ordinal"
            or not config.get("use_mixture_aware_query_fusion", False)
            or any(config.get(key, False) for key in (
                "use_adaptive_mixture_gate", "use_boundary_specific_mixture_gate",
                "use_local_quantitative_branch", "use_anchored_local_calibration"))
            or checkpoint.get("training_config", {}).get("ordinal_loss_mode") != "corn"
            or checkpoint.get("training_config", {}).get("training_phase") == "t315_quantitative_branch_finetune"
            or checkpoint.get("concentration_map") is not None):
        raise ValueError("Use the original verified CORN ordinal T3.10 anchor")
    return config


def is_quantitative_state(name):
    return name.startswith(TRAINABLE_PREFIXES)


def configure_finetuning(model):
    if (model.concentration_head_mode != "ordinal"
            or model.mixture_aware_query_fusion is None
            or model.local_quantitative_branch is not None
            or model.anchored_local_calibration is not None
            or model.use_adaptive_mixture_gate or model.use_boundary_specific_mixture_gate):
        raise ValueError("Quantitative fine-tuning requires the original T3.10 path")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(is_quantitative_state(name))
    set_finetuning_mode(model, False)
    counts = {name: sum(parameter.numel() for parameter in module.parameters())
              for name, module in (("mixture_aware_query_fusion", model.mixture_aware_query_fusion),
                                   ("ordinal_heads", model.ordinal_heads))}
    if any(value == 0 for value in counts.values()):
        raise ValueError("Missing trainable quantitative module")
    frozen = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
              if not is_quantitative_state(name)}
    return counts, frozen


def set_finetuning_mode(model, training):
    # Never call model.train(True): frozen BatchNorm and dropout must stay fixed.
    model.eval()
    model.mixture_aware_query_fusion.train(training)
    model.ordinal_heads.train(training)
    for name, module in model.named_modules():
        if name and not (is_quantitative_state(name) or name in ("mixture_aware_query_fusion", "ordinal_heads")):
            if module.training:
                raise RuntimeError("Frozen module is in training mode: " + name)


def check_frozen_state(model, frozen):
    state = model.state_dict()
    for name, value in frozen.items():
        if not torch.equal(state[name].detach().cpu(), value):
            raise RuntimeError("Original feature/classification state changed: " + name)


def check_classification(class_prediction, anchor_prediction, tolerance=1e-6):
    error = float((class_prediction.detach() - anchor_prediction.detach()).abs().max().cpu())
    if not math.isfinite(error) or error > tolerance:
        raise RuntimeError(f"Classification output drifted from anchor: max_abs_error={error}")
    return error


def run_finetuning_epoch(model, anchor, loader, device, *, optimizer=None, scheduler=None,
                         max_batches=None, anchor_penalty=0.1):
    training = optimizer is not None
    set_finetuning_mode(model, training)
    anchor.eval()
    totals = {name: 0.0 for name in (
        "loss_total", "loss_corn", "loss_anchor_kl", "mean_absolute_logit_change")}
    arrays = {name: [] for name in ("class_target", "class_probability", "concentration_target",
                                   "concentration_prediction", "ordinal_probability")}
    count = real_rows = generated_rows = 0
    max_class_error = 0.0
    with torch.enable_grad() if training else torch.no_grad():
        for index, batch in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            sources = batch["source"]
            if any(source not in ("real", "generated") for source in sources):
                raise ValueError("Unexpected data source")
            batch = _to_device(batch, device)
            with torch.no_grad():
                base_class, _, base_probability, base_logits = anchor(
                    batch, return_ordinal=True, return_ordinal_logits=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            class_prediction, concentration, probability, logits = model(
                batch, return_ordinal=True, return_ordinal_logits=True)
            max_class_error = max(max_class_error, check_classification(class_prediction, base_class))
            weights = torch.ones(len(sources), dtype=logits.dtype, device=device)
            corn = weighted_corn_loss(logits, batch["concentration_target"], batch["class_target"], weights)
            kl = anchor_kl_loss(base_probability, probability, batch["class_target"], weights)
            loss = corn + anchor_penalty * kl
            if not torch.isfinite(loss):
                raise RuntimeError("Fine-tuning loss is NaN/Inf")
            change = (logits.detach() - base_logits).abs().mean()
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                               1.0, error_if_nonfinite=True)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            size = len(sources)
            count += size
            real_rows += sum(source == "real" for source in sources)
            generated_rows += sum(source == "generated" for source in sources)
            for key, value in {"loss_total": loss, "loss_corn": corn, "loss_anchor_kl": kl,
                               "mean_absolute_logit_change": change}.items():
                totals[key] += float(value.detach().cpu()) * size
            for key, value in zip(arrays, (batch["class_target"], class_prediction,
                                           batch["concentration_target"], concentration, probability)):
                arrays[key].append(value.detach().cpu().numpy())
    if count == 0:
        raise RuntimeError("No batches processed")
    merged = {name: np.concatenate(values, axis=0) for name, values in arrays.items()}
    metrics = {name: value / count for name, value in totals.items()}
    metrics.update(multilabel_classification_metrics(merged["class_target"], merged["class_probability"]))
    metrics.update(regression_metrics(merged["concentration_target"], merged["concentration_prediction"], merged["class_target"]))
    metrics.update(_ordinal_metrics_from_probabilities(merged["concentration_target"], merged["class_target"],
                    merged["class_probability"], merged["ordinal_probability"], decoding="median"))
    metrics.update(processed_rows=count, real_rows_processed=real_rows, generated_rows_processed=generated_rows,
                   classification_max_abs_error_to_anchor=max_class_error)
    if loader.dataset.split == "validation":
        validation_strata(metrics, merged, loader.dataset)
    return metrics, merged


def main():
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu explicitly for artificial tests")
    device = torch.device(args.device)
    anchor_path = args.anchor_checkpoint.expanduser().resolve()
    checkpoint = torch.load(anchor_path, map_location="cpu", weights_only=False)
    config = anchor_config(checkpoint)
    output = args.output_directory.expanduser().resolve()
    if not args.check_only and output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a fresh/empty output directory: " + str(output))
    model = TransformerClassifyRegress_sep(**config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    counts, frozen = configure_finetuning(model)
    anchor = TransformerClassifyRegress_sep(**config)
    anchor.load_state_dict(checkpoint["model_state_dict"], strict=True)
    anchor.requires_grad_(False).eval()
    model.to(device)
    anchor.to(device)
    train, validation, test = build_datasets(include_generated_train=True,
            maximum_generated_per_condition=None, require_complete_generated=True)
    if train.repository.target_mode != "level_code":
        raise ValueError("T3.15 requires S/M/H level-code targets")
    sources = {source: sum(ref.source == source for ref in train.samples) for source in ("real", "generated")}
    if sources["generated"] == 0:
        raise ValueError("No generated spectra loaded")
    print("===== T3.15 QUANTITATIVE BRANCH FINE-TUNING =====", flush=True)
    print(f"train={len(train)} = real {sources['real']} + generated {sources['generated']}; validation={len(validation)}; test={len(test)}", flush=True)
    print("All generated spectra included with equal per-spectrum weight. Test is not evaluated.", flush=True)
    print("Trainable modules:", json.dumps(counts), flush=True)
    print("Trainable quantitative parameters:", sum(counts.values()), flush=True)
    print("Frozen original parameters:", sum(p.numel() for p in model.parameters() if not p.requires_grad), flush=True)
    print("No local calibration module, no uncertainty mask, no new boundary gate; original architecture retained.", flush=True)
    train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
            drop_last=False, generator=torch.Generator().manual_seed(args.seed))
    validation_loader = make_loader(validation, args.batch_size, False, args.num_workers, device)
    steps = min(len(train_loader), args.max_train_batches or len(train_loader))
    total_steps, warmup_steps = steps * args.epochs, steps * args.warmup_epochs
    print(f"LR schedule: per-update linear warmup {args.warmup_epochs} epochs; start={args.learning_rate*args.warmup_start_ratio:.3e}; peak={args.learning_rate:.3e}; cosine minimum={args.minimum_learning_rate:.3e}; total updates={total_steps}", flush=True)
    metrics, _ = run_finetuning_epoch(model, anchor, validation_loader, device,
            max_batches=args.max_validation_batches, anchor_penalty=args.anchor_penalty)
    allowed, failures = guard_candidate(metrics, metrics)
    if not allowed:
        raise ValueError("Validation lacks guard strata: " + ";".join(failures))
    check_frozen_state(model, frozen)
    print(f"epoch=000 anchor_present={metrics['ordinal_accuracy_present']:.4f} ternary={metrics['ordinal_exact_ternary']:.4f} water={metrics['ordinal_exact_ternary_water']:.4f} soil={metrics['ordinal_exact_ternary_soil']:.4f}; class_drift={metrics['classification_max_abs_error_to_anchor']:.3e}", flush=True)
    if args.check_only:
        print("PRETRAIN CHECK: PASS. No updates, output directories or checkpoints written.", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: warmup_cosine_factor(
        step, total_steps=total_steps, warmup_steps=warmup_steps,
        minimum_ratio=args.minimum_learning_rate / args.learning_rate, start_ratio=args.warmup_start_ratio))
    training_config = {**vars(args), "ordinal_loss_mode": "corn", "ordinal_decoding": "median",
        "checkpoint_selection": "guarded", "include_generated": True, "maximum_generated_per_condition": None,
        "training_phase": "t315_quantitative_branch_finetune", "trainable_prefixes": list(TRAINABLE_PREFIXES),
        "lr_schedule": "per_update_linear_warmup_cosine", "planned_optimizer_updates": total_steps,
        "warmup_optimizer_updates": warmup_steps, "real_weight": 1.0, "generated_weight": 1.0,
        "anchor_sha256": hashlib.sha256(anchor_path.read_bytes()).hexdigest()}
    data_config = dict(checkpoint.get("data_config", {}))
    data_config.update(include_generated=True, maximum_generated_per_condition=None,
        real_root=str(train.repository.real_root), generated_root=str(train.repository.generated_root),
        train_source_counts=sources, split="12/4/4 within each source file", matrix_input_required=False,
        runtime_tail_policy="missing tail to 2500 in memory; original files unchanged")
    write_json(output / "anchor_validation.json", metrics)
    write_json(output / "training_config.json", training_config)
    for name in ("anchor_zero.pt", "best.pt", "best_ordinal.pt"):
        save_checkpoint(checkpoints / name, model, optimizer, scheduler, 0, config, training_config, data_config, metrics)
    best_metrics, best_epoch, waiting = dict(metrics), 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics, _ = run_finetuning_epoch(model, anchor, train_loader, device,
            optimizer=optimizer, scheduler=scheduler, max_batches=args.max_train_batches, anchor_penalty=args.anchor_penalty)
        if args.max_train_batches is None and any(train_metrics[source + "_rows_processed"] != sources[source]
                                                for source in ("real", "generated")):
            raise RuntimeError("A formal epoch did not process every real/generated training row")
        current, _ = run_finetuning_epoch(model, anchor, validation_loader, device,
            max_batches=args.max_validation_batches, anchor_penalty=args.anchor_penalty)
        check_frozen_state(model, frozen)
        allowed, failures = guard_candidate(current, metrics)
        better = allowed and selection_score(current) > selection_score(best_metrics)
        if better:
            best_metrics, best_epoch, waiting = dict(current), epoch, 0
            for name in ("best.pt", "best_ordinal.pt"):
                save_checkpoint(checkpoints / name, model, optimizer, scheduler, epoch, config, training_config, data_config, current)
        else:
            waiting += 1
        save_checkpoint(checkpoints / "latest.pt", model, optimizer, scheduler, epoch, config, training_config, data_config, current)
        history.append({"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"],
            "guard_pass": allowed, "guard_failures": ";".join(failures), "selected": better,
            **{"train_" + key: value for key, value in train_metrics.items()},
            **{"validation_" + key: value for key, value in current.items()}})
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
        print(f"epoch={epoch:03d} loss={train_metrics['loss_total']:.6f} val={current['loss_total']:.6f} present={current['ordinal_accuracy_present']:.4f} ternary={current['ordinal_exact_ternary']:.4f} water={current['ordinal_exact_ternary_water']:.4f} soil={current['ordinal_exact_ternary_soil']:.4f} CHL={current['ordinal_accuracy_CHL_ternary']:.4f} class_drift={current['classification_max_abs_error_to_anchor']:.3e} guard={'PASS' if allowed else 'REJECT'} selected={better} lr={optimizer.param_groups[0]['lr']:.3e}", flush=True)
        if waiting >= args.patience:
            print(f"Early stopping: {waiting} epochs without qualifying improvement.", flush=True)
            break
    write_json(output / "run_summary.json", {"best_epoch": best_epoch, "anchor_validation": metrics,
        "best_validation": best_metrics, "model_config": config, "training_config": training_config,
        "data_config": data_config, "trainable_module_counts": counts, "frozen_state_verified": True,
        "selection_note": "Epoch 0 means no qualifying improvement; retained original T3.10."})
    print(f"Training complete. Best epoch: {best_epoch}; checkpoint: {checkpoints / 'best_ordinal.pt'}", flush=True)
    if best_epoch == 0:
        print("No qualifying validation gain. The selected model retains original T3.10 weights.", flush=True)
    if not args.skip_final_evaluation:
        from Inference import evaluate_checkpoint
        evaluate_checkpoint(checkpoint_path=checkpoints / "best_ordinal.pt", output_directory=output / "evaluation",
            device_name=args.device, batch_size=args.batch_size, num_workers=args.num_workers,
            splits=("validation",), ordinal_decoding="median", training_history_path=output / "training_history.csv")


if __name__ == "__main__":
    main()
