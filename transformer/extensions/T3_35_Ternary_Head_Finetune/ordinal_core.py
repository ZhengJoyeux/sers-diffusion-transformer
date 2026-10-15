"""Existing S/M/H decoding and conditional CORN loss, unchanged."""
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

def ordinal_class_probabilities_numpy(probabilities: np.ndarray) -> np.ndarray:
    """Convert ordered cumulative probabilities to [S, M, H] probabilities."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape[-1:] != (2,) or not np.isfinite(values).all():
        raise ValueError("Expected finite cumulative probabilities ending in dimension 2")
    if (values < -1e-7).any() or (values > 1.0 + 1e-7).any():
        raise ValueError("Cumulative probabilities must be in [0, 1]")
    if (values[..., 1] > values[..., 0] + 1e-7).any():
        raise ValueError("P(level>=H) must not exceed P(level>=M)")
    values = np.clip(values, 0.0, 1.0)
    return np.maximum(np.stack(
        (1.0 - values[..., 0], values[..., 0] - values[..., 1], values[..., 1]),
        axis=-1,
    ), 0.0)

def decode_ordinal_numpy(
    probabilities: np.ndarray, *, mode: str = "median", threshold: float = 0.5,
) -> np.ndarray:
    """One shared decoder for training metrics, inference tables and CSV rows.

    median preserves the historical cumulative-threshold policy. map chooses
    the most probable S/M/H class; exact ties choose the lower level.
    """
    classes = ordinal_class_probabilities_numpy(probabilities)
    if mode == "map":
        return 1 + classes.argmax(axis=-1)
    if mode != "median":
        raise ValueError("ordinal decoding must be median or map")
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("ordinal threshold must be in (0, 1)")
    return 1 + (np.asarray(probabilities) >= float(threshold)).sum(axis=-1)

def present_only_corn_ordinal_loss(
    ordinal_logits: Tensor,
    concentration_target: Tensor,
    presence_target: Tensor,
) -> Tensor:
    """CORN-style conditional ordinal loss for present pesticides only.

    For S/M/H level codes 1/2/3:

    task 0:
        P(level >= M)
        trained on all present S/M/H targets.

    task 1:
        P(level >= H | level >= M)
        trained only on present M/H targets.

    The loss is normalized by the total number of valid conditional
    training entries, following the CORN conditional-training-set
    formulation.
    """
    if ordinal_logits.ndim != 3 or ordinal_logits.shape[-1] != 2:
        raise ValueError(
            "ordinal_logits must be [B,P,2], got "
            f"{tuple(ordinal_logits.shape)}"
        )

    if concentration_target.shape != ordinal_logits.shape[:2]:
        raise ValueError(
            "concentration_target shape does not match ordinal logits"
        )

    if presence_target.shape != ordinal_logits.shape[:2]:
        raise ValueError(
            "presence_target shape does not match ordinal logits"
        )

    present = presence_target > 0.5

    if not bool(present.any()):
        return ordinal_logits.sum() * 0.0

    # --------------------------------------------------------
    # Conditional task 0:
    # S versus {M,H}
    #
    # All truly present pesticides participate.
    # --------------------------------------------------------
    target_ge_m = (
        concentration_target >= 2.0
    ).to(ordinal_logits.dtype)

    first_logits = ordinal_logits[..., 0][present]
    first_target = target_ge_m[present]

    loss_sum = F.binary_cross_entropy_with_logits(
        first_logits,
        first_target,
        reduction="sum",
    )

    valid_count = int(first_logits.numel())

    # --------------------------------------------------------
    # Conditional task 1:
    # M versus H
    #
    # S samples must NOT train this conditional boundary.
    # --------------------------------------------------------
    second_mask = (
        present
        & (concentration_target >= 2.0)
    )

    if bool(second_mask.any()):
        target_h_given_ge_m = (
            concentration_target >= 3.0
        ).to(ordinal_logits.dtype)

        second_logits = ordinal_logits[..., 1][second_mask]
        second_target = target_h_given_ge_m[second_mask]

        loss_sum = (
            loss_sum
            + F.binary_cross_entropy_with_logits(
                second_logits,
                second_target,
                reduction="sum",
            )
        )

        valid_count += int(second_logits.numel())

    return loss_sum / float(valid_count)
