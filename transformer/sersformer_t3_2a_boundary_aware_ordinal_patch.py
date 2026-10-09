#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

TRAIN = Path("SERSFormer_Training.py")
TEST = Path("tests/test_boundary_aware_ordinal_loss.py")

if not TRAIN.exists():
    raise FileNotFoundError(
        "SERSFormer_Training.py not found. Run this script from ~/project_transformer."
    )

text = TRAIN.read_text(encoding="utf-8")

MARKER = "def present_only_boundary_margin_loss("
if MARKER in text:
    raise RuntimeError(
        "T3.2-A boundary-aware ordinal patch already appears to be applied."
    )

# 1. CLI arguments. Defaults preserve T3.1 behavior exactly.
parse_anchor = "    return parser.parse_args()\n"
if parse_anchor not in text:
    raise RuntimeError("Cannot find parse_args() return anchor.")

parser_block = """    parser.add_argument(
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
"""
text = text.replace(parse_anchor, parser_block + parse_anchor, 1)

# 2. Boundary-aware loss.
helper_anchor = "\ndef _ordinal_metrics_from_probabilities("
if helper_anchor not in text:
    raise RuntimeError(
        "Cannot find _ordinal_metrics_from_probabilities() anchor."
    )

helper = '''

def present_only_boundary_margin_loss(
    ordinal_probability: torch.Tensor,
    concentration_target: torch.Tensor,
    presence: torch.Tensor,
    *,
    margin: float = 0.15,
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

    level = concentration_target.round().long()
    p_ge_m = ordinal_probability[..., 0]
    p_ge_h = ordinal_probability[..., 1]

    low = 0.5 - margin
    high = 0.5 + margin

    penalties: list[torch.Tensor] = []

    low_mask = present_mask & level.eq(1)
    middle_mask = present_mask & level.eq(2)
    high_mask = present_mask & level.eq(3)

    # S should stay clearly below the S/M decision boundary.
    if bool(low_mask.any()):
        penalties.append(
            torch.relu(p_ge_m[low_mask] - low).square()
        )

    # M should stay above the S/M boundary and below the M/H boundary.
    if bool(middle_mask.any()):
        penalties.append(
            torch.relu(high - p_ge_m[middle_mask]).square()
        )
        penalties.append(
            torch.relu(p_ge_h[middle_mask] - low).square()
        )

    # H should stay clearly above the M/H decision boundary.
    if bool(high_mask.any()):
        penalties.append(
            torch.relu(high - p_ge_h[high_mask]).square()
        )

    if not penalties:
        return ordinal_probability.sum() * 0.0

    return torch.cat(
        [item.reshape(-1) for item in penalties],
        dim=0,
    ).mean()

'''
text = text.replace(helper_anchor, helper + helper_anchor, 1)

# 3. run_epoch() signature.
signature_old = """    concentration_head_mode: str,
    classification_weight: float,
    regression_weight: float,
"""
signature_new = """    concentration_head_mode: str,
    ordinal_boundary_weight: float,
    ordinal_boundary_margin: float,
    classification_weight: float,
    regression_weight: float,
"""
if signature_old not in text:
    raise RuntimeError("Cannot find run_epoch() concentration signature.")
text = text.replace(signature_old, signature_new, 1)

# 4. Loss accumulators.
acc_old = """    class_loss_sum = 0.0
    reg_loss_sum = 0.0
    sample_count = 0
"""
acc_new = """    class_loss_sum = 0.0
    reg_loss_sum = 0.0
    ordinal_base_loss_sum = 0.0
    ordinal_boundary_loss_sum = 0.0
    sample_count = 0
"""
if acc_old not in text:
    raise RuntimeError("Cannot find loss accumulator block.")
text = text.replace(acc_old, acc_new, 1)

# 5. Replace ordinal/continuous loss branch.
branch_old = """            if concentration_head_mode == "ordinal":
                class_pred, reg_pred, ordinal_probability = model(
                    batch, return_ordinal=True
                )
                reg_loss = present_only_ordinal_bce(
                    ordinal_probability,
                    reg_target,
                    class_target,
                )
            elif concentration_head_mode == "continuous":
                class_pred, reg_pred = model(batch)
                ordinal_probability = None
                # Preserve the exact T1/T2 objective for checkpoint/ablation compatibility.
                reg_loss = regression_loss_fn(reg_pred, reg_target)
"""
branch_new = """            if concentration_head_mode == "ordinal":
                class_pred, reg_pred, ordinal_probability = model(
                    batch, return_ordinal=True
                )
                ordinal_base_loss = present_only_ordinal_bce(
                    ordinal_probability,
                    reg_target,
                    class_target,
                )
                ordinal_boundary_loss = present_only_boundary_margin_loss(
                    ordinal_probability,
                    reg_target,
                    class_target,
                    margin=ordinal_boundary_margin,
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
"""
if branch_old not in text:
    raise RuntimeError(
        "Cannot find current T3.1 ordinal/continuous loss branch."
    )
text = text.replace(branch_old, branch_new, 1)

# 6. Accumulate ordinal loss components.
sum_old = """            class_loss_sum += float(class_loss.detach().cpu()) * batch_size
            reg_loss_sum += float(reg_loss.detach().cpu()) * batch_size

            class_targets.append(class_target.detach().cpu().numpy())
"""
sum_new = """            class_loss_sum += float(class_loss.detach().cpu()) * batch_size
            reg_loss_sum += float(reg_loss.detach().cpu()) * batch_size
            if concentration_head_mode == "ordinal":
                ordinal_base_loss_sum += (
                    float(ordinal_base_loss.detach().cpu()) * batch_size
                )
                ordinal_boundary_loss_sum += (
                    float(ordinal_boundary_loss.detach().cpu()) * batch_size
                )

            class_targets.append(class_target.detach().cpu().numpy())
"""
if sum_old not in text:
    raise RuntimeError("Cannot find batch loss accumulation anchor.")
text = text.replace(sum_old, sum_new, 1)

# 7. Expose loss components in training_history.csv.
metrics_old = """    loss_metrics = {
        "loss_total": total_loss_sum / sample_count,
        "loss_class": class_loss_sum / sample_count,
        "loss_reg": reg_loss_sum / sample_count,
    }
"""
metrics_new = """    loss_metrics = {
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
"""
if metrics_old not in text:
    raise RuntimeError("Cannot find loss_metrics block.")
text = text.replace(metrics_old, metrics_new, 1)

# 8. Validate and print T3.2-A settings.
main_anchor = """    print(f"concentration head: {args.concentration_head_mode}")

    train_loader = make_loader(
"""
main_new = """    print(f"concentration head: {args.concentration_head_mode}")
    if args.concentration_head_mode == "ordinal":
        if args.ordinal_boundary_weight < 0.0:
            raise ValueError("--ordinal-boundary-weight must be >= 0")
        if not 0.0 <= args.ordinal_boundary_margin < 0.5:
            raise ValueError(
                "--ordinal-boundary-margin must satisfy 0 <= margin < 0.5"
            )
        print(
            "ordinal boundary-aware loss: "
            f"weight={args.ordinal_boundary_weight:.4f}, "
            f"margin={args.ordinal_boundary_margin:.4f}"
        )

    train_loader = make_loader(
"""
if main_anchor not in text:
    raise RuntimeError("Cannot find concentration-head print anchor.")
text = text.replace(main_anchor, main_new, 1)

# 9. Save settings in metadata.
metadata_old = """        "concentration_head_mode": args.concentration_head_mode,
        "concentration_training_scope": (
"""
metadata_new = """        "concentration_head_mode": args.concentration_head_mode,
        "ordinal_boundary_weight": args.ordinal_boundary_weight,
        "ordinal_boundary_margin": args.ordinal_boundary_margin,
        "concentration_training_scope": (
"""
if metadata_old not in text:
    raise RuntimeError("Cannot find concentration metadata anchor.")
text = text.replace(metadata_old, metadata_new, 1)

# 10. Pass settings to both train and validation run_epoch calls.
call_old = """            concentration_head_mode=args.concentration_head_mode,
            classification_weight=class_weight,
            regression_weight=reg_weight,
"""
call_new = """            concentration_head_mode=args.concentration_head_mode,
            ordinal_boundary_weight=args.ordinal_boundary_weight,
            ordinal_boundary_margin=args.ordinal_boundary_margin,
            classification_weight=class_weight,
            regression_weight=reg_weight,
"""
count = text.count(call_old)
if count != 2:
    raise RuntimeError(
        "Expected exactly 2 run_epoch call anchors, "
        f"found {count}."
    )
text = text.replace(call_old, call_new)

TRAIN.write_text(text, encoding="utf-8")

# 11. Add unit tests.
TEST.parent.mkdir(parents=True, exist_ok=True)
TEST.write_text(
    '''import torch

from SERSFormer_Training import present_only_boundary_margin_loss


def test_boundary_loss_zero_for_well_separated_levels():
    probability = torch.tensor(
        [[
            [0.20, 0.05],
            [0.80, 0.20],
            [0.90, 0.80],
        ]],
        dtype=torch.float32,
    )
    target = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    presence = torch.ones_like(target)

    loss = present_only_boundary_margin_loss(
        probability,
        target,
        presence,
        margin=0.15,
    )

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-8)


def test_boundary_loss_penalizes_middle_collapse():
    probability = torch.tensor(
        [[
            [0.60, 0.20],
            [0.55, 0.50],
            [0.80, 0.40],
        ]],
        dtype=torch.float32,
        requires_grad=True,
    )
    target = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    presence = torch.ones_like(target)

    loss = present_only_boundary_margin_loss(
        probability,
        target,
        presence,
        margin=0.15,
    )

    assert float(loss.detach()) > 0.0
    loss.backward()
    assert probability.grad is not None
    assert torch.isfinite(probability.grad).all()


def test_boundary_loss_ignores_absent_targets():
    probability = torch.tensor(
        [[[0.99, 0.99]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    target = torch.tensor([[0.0]], dtype=torch.float32)
    presence = torch.tensor([[0.0]], dtype=torch.float32)

    loss = present_only_boundary_margin_loss(
        probability,
        target,
        presence,
        margin=0.15,
    )

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-8)


def test_boundary_loss_rejects_invalid_margin():
    probability = torch.zeros(1, 1, 2, dtype=torch.float32)
    target = torch.ones(1, 1, dtype=torch.float32)
    presence = torch.ones_like(target)

    try:
        present_only_boundary_margin_loss(
            probability,
            target,
            presence,
            margin=0.5,
        )
    except ValueError:
        return

    raise AssertionError("Expected ValueError for margin >= 0.5")
''',
    encoding="utf-8",
)

print("T3.2-A PATCH APPLIED")
print("Modified:", TRAIN)
print("Created :", TEST)
print()
print("Compatibility:")
print("  --ordinal-boundary-weight 0.0 -> original T3.1 objective")
print("  --ordinal-boundary-weight >0  -> T3.2-A boundary-aware objective")
