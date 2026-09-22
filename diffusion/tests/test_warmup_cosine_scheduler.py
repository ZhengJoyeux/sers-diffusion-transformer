import math

import torch

from src.ddpm_trainer import (
    WarmupCosineLearningRateScheduler,
)


def _build_scheduler(total_steps=100):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=2.0e-4)

    scheduler = WarmupCosineLearningRateScheduler(
        optimizer=optimizer,
        total_training_steps=total_steps,
        warmup_fraction=0.05,
        warmup_start_learning_rate=1.0e-5,
        peak_learning_rate=2.0e-4,
        minimum_learning_rate=1.0e-6,
    )

    return optimizer, scheduler


def test_warmup_cosine_key_learning_rates():
    optimizer, scheduler = _build_scheduler(total_steps=100)

    assert math.isclose(
        optimizer.param_groups[0]["lr"],
        1.0e-5,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    )

    assert scheduler.warmup_steps == 5

    for _ in range(5):
        scheduler.step()

    assert math.isclose(
        optimizer.param_groups[0]["lr"],
        2.0e-4,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    )

    for _ in range(95):
        scheduler.step()

    assert math.isclose(
        optimizer.param_groups[0]["lr"],
        1.0e-6,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    )


def test_warmup_cosine_is_monotonic_in_each_phase():
    _, scheduler = _build_scheduler(total_steps=100)

    values = [
        scheduler.learning_rate_at_step(step)
        for step in range(101)
    ]

    warmup = values[: scheduler.warmup_steps + 1]
    cosine = values[scheduler.warmup_steps :]

    assert all(
        b >= a
        for a, b in zip(
            warmup,
            warmup[1:],
        )
    )

    assert all(
        b <= a
        for a, b in zip(
            cosine,
            cosine[1:],
        )
    )


def test_warmup_cosine_state_round_trip():
    optimizer_a, scheduler_a = _build_scheduler(total_steps=100)

    for _ in range(37):
        scheduler_a.step()

    state = scheduler_a.state_dict()

    optimizer_b, scheduler_b = _build_scheduler(total_steps=100)
    scheduler_b.load_state_dict(state)

    assert scheduler_b.completed_steps == 37

    assert math.isclose(
        optimizer_a.param_groups[0]["lr"],
        optimizer_b.param_groups[0]["lr"],
        rel_tol=0.0,
        abs_tol=1.0e-15,
    )
