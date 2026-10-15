"""Spectrum-only, bounded, shared-shift ordinal calibration for a frozen T3.10.

No matrix/pesticide-presence/concentration metadata is fed to the correction.
The two CORN conditional logits share one shift per pesticide. This is not
equivalent to two learned independent boundary gates or physical peak fitting.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from Model_v2 import LocalQuantitativeBranch


def cumulative_probabilities(logits: Tensor) -> Tensor:
    conditional = logits.sigmoid()
    return torch.stack((conditional[..., 0], conditional[..., 0] * conditional[..., 1]), dim=-1)


def level_probabilities(cumulative: Tensor) -> Tensor:
    return torch.stack((1.0 - cumulative[..., 0], cumulative[..., 0] - cumulative[..., 1], cumulative[..., 1]), dim=-1)


class AnchoredLocalCalibrator(nn.Module):
    def __init__(self, centers: Sequence[Sequence[float]], *, logit_bound: float = 0.5,
                 uncertainty_band: float = 0.15, hidden: int = 16, half_width: int = 24):
        super().__init__()
        if not math.isfinite(logit_bound) or not 0 < logit_bound <= 1.0:
            raise ValueError("calibration logit bound must be finite and in (0,1]")
        if not math.isfinite(uncertainty_band) or not 0 < uncertainty_band < 0.5:
            raise ValueError("calibration uncertainty band must be in (0,0.5)")
        self.logit_bound = float(logit_bound)
        self.uncertainty_band = float(uncertainty_band)
        # Reuse the tested native-window extraction/real-train normalization.
        # Output dim is ONE: the same scalar shifts both CORN conditional logits.
        self.local = LocalQuantitativeBranch(1, 1901, centers, half_width=half_width,
                                             hidden=hidden, strength=1.0, projection_hidden=hidden)
        self.gate_logits = nn.Parameter(torch.zeros(3))
        self.last_shift: Tensor | None = None
        self.last_base_probabilities: Tensor | None = None
        self.last_eligible: Tensor | None = None
        self.last_reliability: Tensor | None = None

    def forward(self, raw_intensity: Tensor, valid_mask: Tensor, base_logits: Tensor) -> Tensor:
        if base_logits.shape != (raw_intensity.shape[0], 3, 2) or not torch.isfinite(base_logits).all():
            raise ValueError("Anchor logits must be finite [B,3,2]")
        base = cumulative_probabilities(base_logits.detach())
        # The margin is measured on cumulative probabilities, including the
        # actual M/H cumulative boundary, NOT sigmoid(conditional logit 1).
        nearest_margin = (base - 0.5).abs().amin(dim=-1, keepdim=True)
        eligible = nearest_margin <= self.uncertainty_band
        raw_shift = self.local(raw_intensity, valid_mask)
        _, features, available = self.local._measure(raw_intensity, valid_mask)
        height = torch.expm1(features[..., 0]).clamp_min(0.0)
        noise = torch.expm1(features[..., 7]).clamp_min(0.0)
        # Effective window height/noise only; it does not certify a pure chemical peak.
        quality = height / (height + 3.0 * noise + 1e-8)
        count = available.sum(dim=-1, keepdim=True).clamp_min(1)
        reliability = (quality * available).sum(dim=-1, keepdim=True) / count
        reliability = reliability.detach().clamp(0.0, 1.0)
        gate = self.gate_logits.sigmoid().view(1, 3, 1)
        shift = self.logit_bound * gate * reliability * raw_shift.tanh() * eligible
        self.last_shift = shift
        self.last_base_probabilities = base
        self.last_eligible = eligible
        self.last_reliability = reliability
        return base_logits + shift


def weighted_corn_loss(logits: Tensor, targets: Tensor, presence: Tensor, sample_weight: Tensor) -> Tensor:
    if logits.ndim != 3 or logits.shape[-1] != 2 or targets.shape != logits.shape[:2] or presence.shape != targets.shape:
        raise ValueError("CORN shapes must be logits [B,P,2], targets/presence [B,P]")
    if sample_weight.shape != (logits.shape[0],) or not torch.isfinite(sample_weight).all() or (sample_weight <= 0).any():
        raise ValueError("Sample weights must be finite positive [B]")
    present = presence > 0.5
    if not bool(present.any()):
        return logits.sum() * 0.0
    weight = sample_weight[:, None].expand_as(targets)
    first = F.binary_cross_entropy_with_logits(logits[..., 0], (targets >= 2).to(logits.dtype), reduction="none")
    second_mask = present & (targets >= 2)
    second = F.binary_cross_entropy_with_logits(logits[..., 1], (targets >= 3).to(logits.dtype), reduction="none")
    numerator = (first * weight * present).sum() + (second * weight * second_mask).sum()
    denominator = (weight * present).sum() + (weight * second_mask).sum()
    return numerator / denominator


def anchor_kl_loss(base: Tensor, corrected: Tensor, presence: Tensor, sample_weight: Tensor) -> Tensor:
    old, new = level_probabilities(base.detach()), level_probabilities(corrected)
    divergence = (old * (old.clamp_min(1e-7).log() - new.clamp_min(1e-7).log())).sum(dim=-1).clamp_min(0.0)
    weights = sample_weight[:, None] * (presence > 0.5)
    return (divergence * weights).sum() / weights.sum().clamp_min(1.0)


GUARD_KEYS = ("class_subset_accuracy", "ordinal_accuracy_present", "ordinal_exact_ternary_water",
              "ordinal_exact_ternary_soil", "ordinal_accuracy_CHL_ternary", "ordinal_accuracy_CHL_ternary_water")


def guard_candidate(candidate: dict, anchor: dict) -> tuple[bool, list[str]]:
    failures = []
    for key in GUARD_KEYS:
        if key not in candidate or key not in anchor or not math.isfinite(float(candidate[key])) or not math.isfinite(float(anchor[key])):
            failures.append(key + ":missing/nonfinite")
        elif float(candidate[key]) < float(anchor[key]) - 1e-12:
            failures.append(key + ":below_anchor")
    key = "ordinal_severe_error_rate"
    if (key not in candidate or key not in anchor or not math.isfinite(float(candidate[key]))
            or not math.isfinite(float(anchor[key]))):
        failures.append(key + ":missing/nonfinite")
    elif float(candidate[key]) > float(anchor[key]) + 1e-12:
        failures.append(key + ":above_anchor")
    return not failures, failures


def selection_score(metrics: dict) -> tuple[float, float, float, float]:
    return (float(metrics["ordinal_exact_ternary"]), float(metrics["ordinal_accuracy_present"]),
            float(metrics["ordinal_exact_profile_accuracy"]), -float(metrics["ordinal_severe_error_rate"]))


def warmup_cosine_factor(step: int, *, total_steps: int, warmup_steps: int,
                         minimum_ratio: float, start_ratio: float = 0.1) -> float:
    """LR for optimizer updates 0..total_steps-1, not validation-loss driven."""
    if total_steps < 2 or not 0 <= warmup_steps < total_steps:
        raise ValueError("total steps >=2 and warmup steps in [0,total)")
    if not 0 < minimum_ratio < 1 or not 0 < start_ratio <= 1:
        raise ValueError("Invalid minimum/start LR ratio")
    step = max(0, int(step))
    if step < warmup_steps:
        if warmup_steps == 1:
            return 1.0
        return start_ratio + (1.0 - start_ratio) * step / (warmup_steps - 1)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps - 1))
    return minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
