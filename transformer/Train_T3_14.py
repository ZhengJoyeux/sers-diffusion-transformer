"""Calibrate a frozen trusted T3.10 using real + ALL generated training spectra.

This is a calibration-stage experiment, not another from-scratch 50-epoch
architecture ablation. Test is never evaluated by this training entry point.
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

from Dataset import build_datasets, collect_real_training_intensities, PESTICIDES
from Metric import multilabel_classification_metrics, regression_metrics
from Model_v2 import TransformerClassifyRegress_sep, decode_ordinal_numpy
from SERSFormer_Training import seed_everything, resolve_device, make_loader, _to_device, _ordinal_metrics_from_probabilities
from T3_14_Calibration import anchor_kl_loss, weighted_corn_loss, guard_candidate, selection_score, warmup_cosine_factor


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--include-generated", action="store_true", required=True,
                        help="Required explicitly: this experiment trains on real + all generated spectra")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--warmup-start-ratio", type=float, default=0.1)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-6)
    parser.add_argument("--logit-bound", type=float, default=0.5)
    parser.add_argument("--uncertainty-band", type=float, default=0.15)
    parser.add_argument("--local-half-width", type=int, default=24)
    parser.add_argument("--local-hidden", type=int, default=16)
    parser.add_argument("--anchor-penalty", type=float, default=0.1)
    parser.add_argument("--real-weight", type=float, default=1.0,
                        help="Default 1: equal per-spectrum weights; generated spectra always remain in the loader")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-validation-batches", type=int, default=None)
    parser.add_argument("--skip-final-evaluation", action="store_true")
    return parser.parse_args()


def anchor_model_config(checkpoint: dict, args) -> dict:
    config = dict(checkpoint.get("model_config", {}))
    if (config.get("concentration_head_mode") != "ordinal"
            or not config.get("use_mixture_aware_query_fusion", False)
            or config.get("use_adaptive_mixture_gate", False)
            or config.get("use_boundary_specific_mixture_gate", False)
            or config.get("use_local_quantitative_branch", False)
            or config.get("use_anchored_local_calibration", False)):
        raise ValueError("Anchor must be original ordinal T3.10, not T3.11/T3.12/T3.13/T3.14")
    if checkpoint.get("concentration_map") is not None:
        raise ValueError("T3.14 uses S/M/H level codes, not a physical concentration map")
    if checkpoint.get("training_config", {}).get("ordinal_loss_mode") != "corn":
        raise ValueError("Select the verified T3.10 CORN checkpoint")
    config.update(use_anchored_local_calibration=True,
                  calibration_logit_bound=args.logit_bound,
                  calibration_uncertainty_band=args.uncertainty_band,
                  calibration_hidden=args.local_hidden,
                  calibration_half_width=args.local_half_width)
    return config


def load_anchor_weights(model, checkpoint):
    state = checkpoint["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not name.startswith("anchored_local_calibration.") for name in missing):
        raise ValueError(f"Anchor state is incompatible. Missing={missing}; unexpected={unexpected}")
    frozen = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
              if not name.startswith("anchored_local_calibration.")}
    return frozen


def check_frozen_state(model, frozen):
    # Check parameters AND buffers: catches accidental BatchNorm updates.
    state = model.state_dict()
    for name, expected in frozen.items():
        if not torch.equal(state[name].detach().cpu(), expected):
            raise RuntimeError(f"Frozen T3.10 state changed: {name}")


def validation_strata(metrics, arrays, dataset):
    count = len(arrays["class_target"])
    if dataset.split != "validation" or count > len(dataset.samples):
        raise ValueError("Guard metrics require ordered real validation rows")
    matrix = np.array([dataset.repository.real_data[ref.condition]["condition_info"].matrix_name
                       for ref in dataset.samples[:count]])
    present = arrays["class_target"] > 0.5
    truth = np.rint(arrays["concentration_target"]).astype(int)
    decoded = decode_ordinal_numpy(arrays["ordinal_probability"], mode="median")
    gated = np.where(arrays["class_probability"] >= 0.5, decoded, 0)
    exact = np.all(gated == truth, axis=1)
    ternary = present.sum(axis=1) == 3
    for name in ("water", "soil"):
        rows = ternary & (matrix == name)
        metrics["ordinal_exact_ternary_" + name] = float(exact[rows].mean()) if rows.any() else float("nan")
        for index, pesticide in enumerate(PESTICIDES):
            metrics[f"ordinal_accuracy_{pesticide}_ternary_{name}"] = float(
                (decoded[rows,index] == truth[rows,index]).mean()) if rows.any() else float("nan")
    return metrics


def run_calibration_epoch(model, loader, device, *, optimizer=None, max_batches=None,
                          anchor_penalty=0.1, real_weight=1.0, scheduler=None):
    training = optimizer is not None
    model.train(training)
    total = {"loss_total": 0.0, "loss_corn": 0.0, "loss_anchor_kl": 0.0,
             "eligible_fraction": 0.0, "mean_absolute_logit_shift": 0.0}
    arrays = {name: [] for name in ("class_target", "class_probability", "concentration_target",
                                   "concentration_prediction", "ordinal_probability")}
    number, real_rows, generated_rows = 0, 0, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for index, batch in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            sources = batch["source"]
            if any(source not in ("real", "generated") for source in sources):
                raise ValueError("Unexpected training source")
            batch = _to_device(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            class_pred, reg_pred, probability, logits = model(batch, return_ordinal=True, return_ordinal_logits=True)
            weights = torch.tensor([real_weight if source == "real" else 1.0 for source in sources],
                                   dtype=logits.dtype, device=device)
            corn = weighted_corn_loss(logits, batch["concentration_target"], batch["class_target"], weights)
            module = model.anchored_local_calibration
            kl = anchor_kl_loss(module.last_base_probabilities, probability, batch["class_target"], weights)
            loss = corn + anchor_penalty * kl
            if not torch.isfinite(loss):
                raise RuntimeError("Calibration loss is NaN/Inf")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0,
                                               error_if_nonfinite=True)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            size = len(sources)
            number += size
            real_rows += sum(source == "real" for source in sources)
            generated_rows += sum(source == "generated" for source in sources)
            values = {"loss_total": loss, "loss_corn": corn, "loss_anchor_kl": kl,
                      "eligible_fraction": module.last_eligible.float().mean(),
                      "mean_absolute_logit_shift": module.last_shift.detach().abs().mean()}
            for key, value in values.items():
                total[key] += float(value.detach().cpu()) * size
            for key, value in zip(arrays, (batch["class_target"], class_pred, batch["concentration_target"], reg_pred, probability)):
                arrays[key].append(value.detach().cpu().numpy())
    if number == 0:
        raise RuntimeError("No batches processed")
    merged = {key: np.concatenate(value, axis=0) for key, value in arrays.items()}
    metrics = {key: value / number for key, value in total.items()}
    metrics.update(multilabel_classification_metrics(merged["class_target"], merged["class_probability"]))
    metrics.update(regression_metrics(merged["concentration_target"], merged["concentration_prediction"], merged["class_target"]))
    metrics.update(_ordinal_metrics_from_probabilities(merged["concentration_target"], merged["class_target"],
                    merged["class_probability"], merged["ordinal_probability"], decoding="median"))
    metrics.update(processed_rows=number, real_rows_processed=real_rows, generated_rows_processed=generated_rows)
    if loader.dataset.split == "validation":
        validation_strata(metrics, merged, loader.dataset)
    return metrics, merged


def write_json(path, content):
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def save_checkpoint(path, model, optimizer, scheduler, epoch, model_config, training_config, data_config, metrics):
    torch.save({"checkpoint_version": 5, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                "epoch": int(epoch), "model_config": model_config, "training_config": training_config,
                "data_config": data_config, "concentration_map": None, "pesticides": list(PESTICIDES),
                "best_validation_loss": float(metrics["loss_total"]), "validation_metrics": metrics}, path)


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.num_workers < 0 or args.patience < 1:
        raise ValueError("epochs/patience positive, batch-size>=2, num-workers>=0")
    if (not math.isfinite(args.anchor_penalty) or args.anchor_penalty < 0
            or not math.isfinite(args.real_weight) or args.real_weight <= 0
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not math.isfinite(args.weight_decay) or args.weight_decay < 0):
        raise ValueError("Invalid loss weights / learning rate / weight decay")
    for value in (args.max_train_batches, args.max_validation_batches):
        if value is not None and value < 1:
            raise ValueError("Batch limits must be positive")
    if not 0 <= args.warmup_epochs < args.epochs:
        raise ValueError("warmup-epochs must be >=0 and less than epochs; smoke uses --warmup-epochs 1")
    if (not math.isfinite(args.minimum_learning_rate) or not 0 < args.minimum_learning_rate < args.learning_rate
            or not math.isfinite(args.warmup_start_ratio) or not 0 < args.warmup_start_ratio <= 1):
        raise ValueError("Require 0 < minimum-learning-rate < learning-rate, and 0 < warmup-start-ratio <=1")
    seed_everything(args.seed)
    anchor_path = args.anchor_checkpoint.expanduser().resolve()
    # Explicitly chosen trusted project checkpoint, not an untrusted download.
    checkpoint = torch.load(anchor_path, map_location="cpu", weights_only=False)
    model_config = anchor_model_config(checkpoint, args)
    output = args.output_directory.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty/fresh output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir()
    device = resolve_device(args.device)
    train, validation, test = build_datasets(include_generated_train=True,
                         maximum_generated_per_condition=None, require_complete_generated=True)
    if train.repository.target_mode != "level_code":
        raise ValueError("T3.14 requires level-code targets")
    sources = {source: sum(ref.source == source for ref in train.samples) for source in ("real", "generated")}
    if sources["generated"] == 0:
        raise ValueError("No generated training spectra found")
    print("===== T3.14 anchored calibration =====", flush=True)
    print(f"train={len(train)} = real {sources['real']} + generated {sources['generated']}; validation={len(validation)}; test={len(test)}", flush=True)
    print("All generated spectra included; test is not evaluated. No matrix labels are required as model input.", flush=True)
    model = TransformerClassifyRegress_sep(**model_config)
    frozen = load_anchor_weights(model, checkpoint)
    intensity, mask, audit = collect_real_training_intensities(train)
    normalization = model.anchored_local_calibration.local.fit_normalization(torch.from_numpy(intensity), torch.from_numpy(mask))
    normalization["audit"] = audit
    del intensity, mask
    write_json(output / "local_quantitative_state.json", normalization)
    print("Local normalization:", json.dumps(audit), flush=True)
    model.to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    print("Trainable calibration parameters:", sum(p.numel() for p in parameters), flush=True)
    print("Frozen anchor parameters:", sum(p.numel() for p in model.parameters() if not p.requires_grad), flush=True)
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=device.type == "cuda",
                       persistent_workers=args.num_workers > 0, drop_last=False,
                       generator=torch.Generator().manual_seed(args.seed))
    validation_loader = make_loader(validation, args.batch_size, False, args.num_workers, device)
    steps_per_epoch = min(len(train_loader), args.max_train_batches or len(train_loader))
    total_steps, warmup_steps = steps_per_epoch * args.epochs, steps_per_epoch * args.warmup_epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: warmup_cosine_factor(
                step, total_steps=total_steps, warmup_steps=warmup_steps,
                minimum_ratio=args.minimum_learning_rate / args.learning_rate,
                start_ratio=args.warmup_start_ratio))
    print(f"LR schedule: per-update linear warmup {args.warmup_epochs} epochs, start={args.learning_rate*args.warmup_start_ratio:.3e}, peak={args.learning_rate:.3e}, cosine minimum={args.minimum_learning_rate:.3e}; total updates={total_steps}", flush=True)
    training_config = {**vars(args), "ordinal_loss_mode": "corn", "ordinal_decoding": "median",
                       "checkpoint_selection": "guarded", "include_generated": True,
                       "maximum_generated_per_condition": None, "frozen_anchor": True,
                       "lr_schedule": "per_update_linear_warmup_cosine", "planned_optimizer_updates": total_steps,
                       "warmup_optimizer_updates": warmup_steps,
                       "anchor_sha256": hashlib.sha256(anchor_path.read_bytes()).hexdigest()}
    data_config = dict(checkpoint.get("data_config", {}))
    data_config.update(include_generated=True, maximum_generated_per_condition=None,
                       real_root=str(train.repository.real_root), generated_root=str(train.repository.generated_root),
                       split="12/4/4 within each source file", train_source_counts=sources,
                       local_normalization_audit=audit, matrix_input_required=False)
    anchor_metrics, _ = run_calibration_epoch(model, validation_loader, device,
                        max_batches=args.max_validation_batches, anchor_penalty=args.anchor_penalty)
    allowed, reasons = guard_candidate(anchor_metrics, anchor_metrics)
    if not allowed:
        raise ValueError("Validation lacks guard strata: " + ", ".join(reasons))
    check_frozen_state(model, frozen)
    write_json(output / "anchor_validation.json", anchor_metrics)
    write_json(output / "training_config.json", training_config)
    for filename in ("anchor_zero.pt", "best.pt", "best_ordinal.pt"):
        save_checkpoint(checkpoints / filename, model, optimizer, scheduler, 0, model_config, training_config, data_config, anchor_metrics)
    best_metrics, best_epoch, waiting = dict(anchor_metrics), 0, 0
    print(f"epoch=000 anchor_present={anchor_metrics['ordinal_accuracy_present']:.4f} ternary={anchor_metrics['ordinal_exact_ternary']:.4f} water={anchor_metrics['ordinal_exact_ternary_water']:.4f} soil={anchor_metrics['ordinal_exact_ternary_soil']:.4f}", flush=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics, _ = run_calibration_epoch(model, train_loader, device, optimizer=optimizer,
                          max_batches=args.max_train_batches, anchor_penalty=args.anchor_penalty, real_weight=args.real_weight,
                          scheduler=scheduler)
        metrics, _ = run_calibration_epoch(model, validation_loader, device,
                          max_batches=args.max_validation_batches, anchor_penalty=args.anchor_penalty)
        check_frozen_state(model, frozen)
        allowed, failures = guard_candidate(metrics, anchor_metrics)
        better = allowed and selection_score(metrics) > selection_score(best_metrics)
        if better:
            best_metrics, best_epoch, waiting = dict(metrics), epoch, 0
            for filename in ("best.pt", "best_ordinal.pt"):
                save_checkpoint(checkpoints / filename, model, optimizer, scheduler, epoch, model_config, training_config, data_config, metrics)
        else:
            waiting += 1
        save_checkpoint(checkpoints / "latest.pt", model, optimizer, scheduler, epoch, model_config, training_config, data_config, metrics)
        history.append({"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"],
                        "guard_pass": allowed, "selected": better, "guard_failures": ";".join(failures),
                        **{"train_" + key: value for key, value in train_metrics.items()},
                        **{"validation_" + key: value for key, value in metrics.items()}})
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
        print(f"epoch={epoch:03d} loss={train_metrics['loss_total']:.6f} val={metrics['loss_total']:.6f} present={metrics['ordinal_accuracy_present']:.4f} ternary={metrics['ordinal_exact_ternary']:.4f} water={metrics['ordinal_exact_ternary_water']:.4f} soil={metrics['ordinal_exact_ternary_soil']:.4f} CHL={metrics['ordinal_accuracy_CHL_ternary']:.4f} eligible={metrics['eligible_fraction']:.3f} guard={'PASS' if allowed else 'REJECT'} selected={better} lr={optimizer.param_groups[0]['lr']:.3e}", flush=True)
        if waiting >= args.patience:
            print(f"Early stopping: {waiting} epochs without a selected guarded improvement.", flush=True)
            break
    write_json(output / "run_summary.json", {"best_epoch": best_epoch, "anchor_validation": anchor_metrics,
               "best_validation": best_metrics, "model_config": model_config, "training_config": training_config,
               "data_config": data_config, "frozen_state_verified": True,
               "selection_note": "Epoch 0 means no qualifying validation improvement; no gain is claimed."})
    print(f"Training complete. Best epoch: {best_epoch}; checkpoint: {checkpoints / 'best_ordinal.pt'}", flush=True)
    if best_epoch == 0:
        print("No qualifying guarded improvement. The selected zero-correction model retains the anchor.", flush=True)
    if not args.skip_final_evaluation:
        from Inference import evaluate_checkpoint
        evaluate_checkpoint(checkpoint_path=checkpoints / "best_ordinal.pt", output_directory=output / "evaluation",
                            device_name=args.device, batch_size=args.batch_size, num_workers=args.num_workers,
                            splits=("validation",), ordinal_decoding="median", training_history_path=output / "training_history.csv")


if __name__ == "__main__":
    main()
