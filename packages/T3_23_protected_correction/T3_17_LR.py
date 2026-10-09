"""Per-update linear warmup followed by cosine decay; no torch dependency."""
import math


def validate_schedule(epochs, warmup_epochs, peak_lr, minimum_lr, start_ratio):
    if epochs < 1 or not 0 <= warmup_epochs < epochs:
        raise ValueError("Require 0 <= warmup-epochs < epochs")
    if not all(math.isfinite(x) for x in (peak_lr, minimum_lr, start_ratio)):
        raise ValueError("LR settings must be finite")
    if not 0 < minimum_lr < peak_lr or not 0 < start_ratio <= 1:
        raise ValueError("Require 0 < minimum LR < peak LR and 0 < warmup start ratio <= 1")


def warmup_cosine_factor(step, *, total_steps, warmup_steps, minimum_ratio, start_ratio=0.1):
    if total_steps < 2 or not 0 <= warmup_steps < total_steps:
        raise ValueError("Require total_steps>=2 and 0<=warmup_steps<total_steps")
    if not 0 < minimum_ratio < 1 or not 0 < start_ratio <= 1:
        raise ValueError("Invalid minimum/start ratio")
    step = max(0, int(step))
    if step < warmup_steps:
        if warmup_steps == 1:
            return 1.0
        return start_ratio + (1 - start_ratio) * step / (warmup_steps - 1)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps - 1))
    return minimum_ratio + (1 - minimum_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
