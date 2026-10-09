#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
from pathlib import Path
import shutil
import sys

import yaml


ROOT = Path.cwd()

TRAINER = ROOT / "src/ddpm_trainer.py"
CHECKPOINT = ROOT / "src/checkpoint_manager.py"
SOURCE_CONFIG = (
    ROOT
    / "config/ddpm_training_d4_3_2_14_cross_fitted_pca_medium.yaml"
)
TARGET_CONFIG = (
    ROOT
    / "config/ddpm_training_d4_3_2_15_warmup_cosine.yaml"
)
DIAGNOSTIC_TARGET = (
    ROOT
    / "scripts/analyze_peak_free_background.py"
)
TEST_TARGET = (
    ROOT
    / "tests/test_warmup_cosine_scheduler.py"
)
PATCH_TARGET = (
    ROOT
    / "D4_3_2_15_warmup_cosine_and_background_diagnostic.patch"
)

COMPANION_DIAGNOSTIC = (
    Path(__file__).resolve().parent
    / "analyze_peak_free_background.py"
)


SCHEDULER_CLASS = 'class WarmupCosineLearningRateScheduler:\n    """按optimizer step执行一次linear warmup + cosine decay。"""\n\n    def __init__(\n        self,\n        *,\n        optimizer,\n        total_training_steps: int,\n        warmup_fraction: float,\n        warmup_start_learning_rate: float,\n        peak_learning_rate: float,\n        minimum_learning_rate: float,\n    ) -> None:\n        if total_training_steps <= 0:\n            raise ValueError("total_training_steps必须大于0。")\n\n        if not 0.0 < warmup_fraction < 1.0:\n            raise ValueError("warmup_fraction必须位于(0,1)。")\n\n        if not (\n            0.0\n            < minimum_learning_rate\n            <= warmup_start_learning_rate\n            < peak_learning_rate\n        ):\n            raise ValueError(\n                "学习率必须满足0 < minimum <= warmup_start < peak。"\n            )\n\n        self.optimizer = optimizer\n        self.total_training_steps = int(total_training_steps)\n        self.warmup_fraction = float(warmup_fraction)\n        self.warmup_steps = max(\n            1,\n            min(\n                self.total_training_steps - 1,\n                int(\n                    round(\n                        self.total_training_steps\n                        * self.warmup_fraction\n                    )\n                ),\n            ),\n        )\n\n        self.warmup_start_learning_rate = float(\n            warmup_start_learning_rate\n        )\n        self.peak_learning_rate = float(peak_learning_rate)\n        self.minimum_learning_rate = float(minimum_learning_rate)\n        self.completed_steps = 0\n\n        self._set_learning_rate(\n            self.learning_rate_at_step(0)\n        )\n\n    def learning_rate_at_step(self, completed_steps: int) -> float:\n        import math\n\n        step = int(\n            max(\n                0,\n                min(\n                    completed_steps,\n                    self.total_training_steps,\n                ),\n            )\n        )\n\n        if step <= self.warmup_steps:\n            progress = step / self.warmup_steps\n\n            return (\n                self.warmup_start_learning_rate\n                + (\n                    self.peak_learning_rate\n                    - self.warmup_start_learning_rate\n                )\n                * progress\n            )\n\n        cosine_steps = (\n            self.total_training_steps\n            - self.warmup_steps\n        )\n\n        progress = (\n            step - self.warmup_steps\n        ) / cosine_steps\n\n        cosine_factor = (\n            0.5\n            * (\n                1.0\n                + math.cos(\n                    math.pi * progress\n                )\n            )\n        )\n\n        return (\n            self.minimum_learning_rate\n            + (\n                self.peak_learning_rate\n                - self.minimum_learning_rate\n            )\n            * cosine_factor\n        )\n\n    def _set_learning_rate(self, learning_rate: float) -> None:\n        for parameter_group in self.optimizer.param_groups:\n            parameter_group["lr"] = float(learning_rate)\n\n    def step(self) -> float:\n        self.completed_steps = min(\n            self.completed_steps + 1,\n            self.total_training_steps,\n        )\n\n        learning_rate = self.learning_rate_at_step(\n            self.completed_steps\n        )\n        self._set_learning_rate(learning_rate)\n\n        return learning_rate\n\n    def restore_step(self, completed_steps: int) -> float:\n        completed_steps = int(completed_steps)\n\n        if not 0 <= completed_steps <= self.total_training_steps:\n            raise ValueError("scheduler恢复step超出训练范围。")\n\n        self.completed_steps = completed_steps\n        learning_rate = self.learning_rate_at_step(\n            self.completed_steps\n        )\n        self._set_learning_rate(learning_rate)\n\n        return learning_rate\n\n    def state_dict(self) -> dict:\n        return {\n            "scheduler_type": "warmup_cosine",\n            "completed_steps": self.completed_steps,\n            "total_training_steps": self.total_training_steps,\n            "warmup_fraction": self.warmup_fraction,\n            "warmup_steps": self.warmup_steps,\n            "warmup_start_learning_rate":\n                self.warmup_start_learning_rate,\n            "peak_learning_rate": self.peak_learning_rate,\n            "minimum_learning_rate": self.minimum_learning_rate,\n        }\n\n    def load_state_dict(self, state: dict) -> None:\n        if str(\n            state.get("scheduler_type", "")\n        ) != "warmup_cosine":\n            raise ValueError(\n                "checkpoint中的scheduler类型不兼容。"\n            )\n\n        if int(\n            state.get("total_training_steps", -1)\n        ) != self.total_training_steps:\n            raise ValueError(\n                "checkpoint scheduler的total_training_steps"\n                "与当前配置不一致。"\n            )\n\n        if int(\n            state.get("warmup_steps", -1)\n        ) != self.warmup_steps:\n            raise ValueError(\n                "checkpoint scheduler的warmup_steps"\n                "与当前配置不一致。"\n            )\n\n        for name, expected in {\n            "warmup_start_learning_rate":\n                self.warmup_start_learning_rate,\n            "peak_learning_rate":\n                self.peak_learning_rate,\n            "minimum_learning_rate":\n                self.minimum_learning_rate,\n        }.items():\n            actual = float(state.get(name, float("nan")))\n\n            if abs(actual - expected) > max(\n                1.0e-15,\n                abs(expected) * 1.0e-10,\n            ):\n                raise ValueError(\n                    "checkpoint scheduler与当前配置"\n                    f"{name}不一致。"\n                )\n\n        self.restore_step(\n            int(state.get("completed_steps", 0))\n        )\n'
TEST_CODE = 'import math\n\nimport torch\n\nfrom src.ddpm_trainer import (\n    WarmupCosineLearningRateScheduler,\n)\n\n\ndef _build_scheduler(total_steps=100):\n    parameter = torch.nn.Parameter(torch.tensor([1.0]))\n    optimizer = torch.optim.AdamW([parameter], lr=2.0e-4)\n\n    scheduler = WarmupCosineLearningRateScheduler(\n        optimizer=optimizer,\n        total_training_steps=total_steps,\n        warmup_fraction=0.05,\n        warmup_start_learning_rate=1.0e-5,\n        peak_learning_rate=2.0e-4,\n        minimum_learning_rate=1.0e-6,\n    )\n\n    return optimizer, scheduler\n\n\ndef test_warmup_cosine_key_learning_rates():\n    optimizer, scheduler = _build_scheduler(total_steps=100)\n\n    assert math.isclose(\n        optimizer.param_groups[0]["lr"],\n        1.0e-5,\n        rel_tol=0.0,\n        abs_tol=1.0e-15,\n    )\n\n    assert scheduler.warmup_steps == 5\n\n    for _ in range(5):\n        scheduler.step()\n\n    assert math.isclose(\n        optimizer.param_groups[0]["lr"],\n        2.0e-4,\n        rel_tol=0.0,\n        abs_tol=1.0e-15,\n    )\n\n    for _ in range(95):\n        scheduler.step()\n\n    assert math.isclose(\n        optimizer.param_groups[0]["lr"],\n        1.0e-6,\n        rel_tol=0.0,\n        abs_tol=1.0e-15,\n    )\n\n\ndef test_warmup_cosine_is_monotonic_in_each_phase():\n    _, scheduler = _build_scheduler(total_steps=100)\n\n    values = [\n        scheduler.learning_rate_at_step(step)\n        for step in range(101)\n    ]\n\n    warmup = values[: scheduler.warmup_steps + 1]\n    cosine = values[scheduler.warmup_steps :]\n\n    assert all(\n        b >= a\n        for a, b in zip(warmup, warmup[1:], strict=True)\n    )\n\n    assert all(\n        b <= a\n        for a, b in zip(cosine, cosine[1:], strict=True)\n    )\n\n\ndef test_warmup_cosine_state_round_trip():\n    optimizer_a, scheduler_a = _build_scheduler(total_steps=100)\n\n    for _ in range(37):\n        scheduler_a.step()\n\n    state = scheduler_a.state_dict()\n\n    optimizer_b, scheduler_b = _build_scheduler(total_steps=100)\n    scheduler_b.load_state_dict(state)\n\n    assert scheduler_b.completed_steps == 37\n\n    assert math.isclose(\n        optimizer_a.param_groups[0]["lr"],\n        optimizer_b.param_groups[0]["lr"],\n        rel_tol=0.0,\n        abs_tol=1.0e-15,\n    )\n'


def read(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f"缺少文件：{path}")

    return path.read_text(encoding="utf-8")


def replace_once(text, old, new, label):
    count = text.count(old)
    print(f"{label} 匹配数量 = {count}")

    if count != 1:
        raise RuntimeError(
            f"{label}匹配数量不是1，为避免误改，停止。"
        )

    return text.replace(old, new, 1)


def patch_trainer(text: str) -> str:
    if "class WarmupCosineLearningRateScheduler:" in text:
        raise RuntimeError(
            "WarmupCosineLearningRateScheduler已经存在。"
        )

    marker = "\n\nclass DdpmTrainer:"
    count = text.count(marker)
    print("scheduler class anchor 匹配数量 =", count)

    if count != 1:
        raise RuntimeError(
            "无法唯一定位class DdpmTrainer。"
        )

    text = text.replace(
        marker,
        "\n\n"
        + SCHEDULER_CLASS.strip()
        + "\n\n\nclass DdpmTrainer:",
        1,
    )

    old = """        self.optimizer = AdamW(
            self.diffusion.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training.get("weight_decay", 0.0)),
        )
        self.use_mixed_precision = (
"""

    new = """        self.optimizer = AdamW(
            self.diffusion.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training.get("weight_decay", 0.0)),
        )

        self.learning_rate_scheduler = None

        scheduler_configuration = training.get(
            "learning_rate_scheduler",
            {},
        )

        if bool(
            scheduler_configuration.get(
                "enabled",
                False,
            )
        ):
            scheduler_type = str(
                scheduler_configuration.get(
                    "type",
                    "warmup_cosine",
                )
            ).strip().lower()

            if scheduler_type != "warmup_cosine":
                raise ValueError(
                    "training.learning_rate_scheduler.type"
                    "目前只支持warmup_cosine。"
                )

            peak_learning_rate = float(
                scheduler_configuration.get(
                    "peak_learning_rate",
                    training["learning_rate"],
                )
            )

            configured_optimizer_lr = float(
                training["learning_rate"]
            )

            if abs(
                configured_optimizer_lr
                - peak_learning_rate
            ) > max(
                1.0e-15,
                abs(peak_learning_rate) * 1.0e-10,
            ):
                raise ValueError(
                    "启用warmup_cosine时，"
                    "training.learning_rate必须与"
                    "peak_learning_rate一致。"
                )

            self.learning_rate_scheduler = (
                WarmupCosineLearningRateScheduler(
                    optimizer=self.optimizer,
                    total_training_steps=(
                        self.total_training_steps
                    ),
                    warmup_fraction=float(
                        scheduler_configuration.get(
                            "warmup_fraction",
                            0.05,
                        )
                    ),
                    warmup_start_learning_rate=float(
                        scheduler_configuration.get(
                            "warmup_start_learning_rate",
                            1.0e-5,
                        )
                    ),
                    peak_learning_rate=peak_learning_rate,
                    minimum_learning_rate=float(
                        scheduler_configuration.get(
                            "minimum_learning_rate",
                            1.0e-6,
                        )
                    ),
                )
            )

        self.use_mixed_precision = (
"""

    text = replace_once(
        text,
        old,
        new,
        "trainer scheduler init",
    )

    old = """        self.starting_step = int(checkpoint.get("step", 0))
        self.best_validation_loss = float(
            checkpoint.get(
                "best_validation_loss",
                float("inf"),
            )
        )

        print(
"""

    new = """        self.starting_step = int(checkpoint.get("step", 0))
        self.best_validation_loss = float(
            checkpoint.get(
                "best_validation_loss",
                float("inf"),
            )
        )

        scheduler_state = checkpoint.get(
            "scheduler_state"
        )

        if self.learning_rate_scheduler is not None:
            if scheduler_state:
                self.learning_rate_scheduler.load_state_dict(
                    scheduler_state
                )
            else:
                self.learning_rate_scheduler.restore_step(
                    self.starting_step
                )

                if self.starting_step > 0:
                    print(
                        "警告：旧checkpoint不含scheduler_state；"
                        "已按global step恢复warmup+cosine学习率。"
                        "正式scheduler实验建议从step=0开始。"
                    )

        print(
"""

    text = replace_once(
        text,
        old,
        new,
        "trainer scheduler resume",
    )

    old = """            optimizer_state=self.optimizer.state_dict(),
            scaler_state=self.gradient_scaler.state_dict(),
"""

    new = """            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=(
                None
                if self.learning_rate_scheduler is None
                else self.learning_rate_scheduler.state_dict()
            ),
            scaler_state=self.gradient_scaler.state_dict(),
"""

    text = replace_once(
        text,
        old,
        new,
        "trainer scheduler checkpoint",
    )

    old = """            self.gradient_scaler.step(self.optimizer)
            self.gradient_scaler.update()
            self.ema.update(self.diffusion)
"""

    new = """            scale_before_update = float(
                self.gradient_scaler.get_scale()
            )

            self.gradient_scaler.step(self.optimizer)
            self.gradient_scaler.update()

            scale_after_update = float(
                self.gradient_scaler.get_scale()
            )

            optimizer_step_applied = (
                not self.use_mixed_precision
                or scale_after_update >= scale_before_update
            )

            if (
                optimizer_step_applied
                and self.learning_rate_scheduler is not None
            ):
                self.learning_rate_scheduler.step()

            self.ema.update(self.diffusion)
"""

    text = replace_once(
        text,
        old,
        new,
        "trainer scheduler step",
    )

    return text


def patch_checkpoint(text: str) -> str:
    old = """        best_validation_loss: float,
        file_name: str | None = None,
"""

    new = """        best_validation_loss: float,
        scheduler_state: dict[str, Any] | None = None,
        file_name: str | None = None,
"""

    text = replace_once(
        text,
        old,
        new,
        "checkpoint scheduler signature",
    )

    if """            "optimizer_state": (
                optimizer_state
            ),
            "scaler_state": scaler_state,
""" in text:
        old = """            "optimizer_state": (
                optimizer_state
            ),
            "scaler_state": scaler_state,
"""
        new = """            "optimizer_state": (
                optimizer_state
            ),
            "scheduler_state": scheduler_state,
            "scaler_state": scaler_state,
"""
    else:
        old = """            "optimizer_state": optimizer_state,
            "scaler_state": scaler_state,
"""
        new = """            "optimizer_state": optimizer_state,
            "scheduler_state": scheduler_state,
            "scaler_state": scaler_state,
"""

    text = replace_once(
        text,
        old,
        new,
        "checkpoint scheduler payload",
    )

    return text


def build_config(text: str) -> str:
    configuration = yaml.safe_load(text)
    training = configuration["training"]

    training["learning_rate"] = 2.0e-4
    training["learning_rate_scheduler"] = {
        "enabled": True,
        "type": "warmup_cosine",
        "warmup_fraction": 0.05,
        "warmup_start_learning_rate": 1.0e-5,
        "peak_learning_rate": 2.0e-4,
        "minimum_learning_rate": 1.0e-6,
        "update_unit": "optimizer_step",
    }

    configuration.setdefault(
        "project",
        {},
    )["name"] = (
        "d4_3_2_15_warmup_cosine"
    )

    output = configuration["output"]
    root = (
        "outputs/experiments/"
        "d4_3_2_15_warmup_cosine"
    )

    output["output_directory"] = root
    output["checkpoint_directory"] = (
        f"{root}/checkpoints"
    )
    output["log_directory"] = (
        f"{root}/logs"
    )
    output["generated_spectrum_directory"] = (
        f"{root}/generated"
    )
    output["preview_plot_directory"] = (
        f"{root}/plots"
    )

    header = (
        "# D4.3.2.15 scheduler-only experiment\n"
        "# Derived from D4.3.2.14 cross-fitted PCA medium.\n"
        "# LR: 1e-5 -> 2e-4 (5% warmup) -> 1e-6 cosine.\n"
    )

    return (
        header
        + yaml.safe_dump(
            configuration,
            allow_unicode=True,
            sort_keys=False,
        )
    )


def diff(path, old, new):
    relative = path.relative_to(ROOT).as_posix()

    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{relative}",
            tofile=f"b/{relative}",
        )
    )


def new_file_diff(path, content):
    relative = path.relative_to(ROOT).as_posix()

    return "".join(
        difflib.unified_diff(
            [],
            content.splitlines(keepends=True),
            fromfile="/dev/null",
            tofile=f"b/{relative}",
        )
    )


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("--check", action="store_true")
    group.add_argument("--apply", action="store_true")

    args = parser.parse_args()

    if not COMPANION_DIAGNOSTIC.is_file():
        raise RuntimeError(
            "安装器旁边缺少analyze_peak_free_background.py。"
        )

    trainer_old = read(TRAINER)
    checkpoint_old = read(CHECKPOINT)
    config_old = read(SOURCE_CONFIG)
    diagnostic_code = read(COMPANION_DIAGNOSTIC)

    trainer_new = patch_trainer(trainer_old)
    checkpoint_new = patch_checkpoint(checkpoint_old)
    config_new = build_config(config_old)

    for path, content in (
        (TARGET_CONFIG, config_new),
        (DIAGNOSTIC_TARGET, diagnostic_code),
        (TEST_TARGET, TEST_CODE),
    ):
        if path.exists():
            existing = path.read_text(encoding="utf-8")

            if existing != content:
                raise RuntimeError(
                    f"{path}已存在且内容不同，停止覆盖。"
                )

    patch_text = (
        diff(TRAINER, trainer_old, trainer_new)
        + diff(CHECKPOINT, checkpoint_old, checkpoint_new)
    )

    if not TARGET_CONFIG.exists():
        patch_text += new_file_diff(
            TARGET_CONFIG,
            config_new,
        )

    if not DIAGNOSTIC_TARGET.exists():
        patch_text += new_file_diff(
            DIAGNOSTIC_TARGET,
            diagnostic_code,
        )

    if not TEST_TARGET.exists():
        patch_text += new_file_diff(
            TEST_TARGET,
            TEST_CODE,
        )

    print("===== D4.3.2.15 combined patch =====")
    print("LR: 1e-5 -> 2e-4 -> 1e-6")
    print("warmup: 5% optimizer steps")
    print("cosine: remaining 95%")
    print("scheduler checkpoint/resume: enabled")
    print("background diagnostic: training-only")

    if args.check:
        print("CHECK PASSED；未写入任何文件。")
        return

    TRAINER.write_text(trainer_new, encoding="utf-8")
    CHECKPOINT.write_text(checkpoint_new, encoding="utf-8")
    TARGET_CONFIG.write_text(config_new, encoding="utf-8")

    if not DIAGNOSTIC_TARGET.exists():
        shutil.copy2(
            COMPANION_DIAGNOSTIC,
            DIAGNOSTIC_TARGET,
        )

    if not TEST_TARGET.exists():
        TEST_TARGET.write_text(
            TEST_CODE,
            encoding="utf-8",
        )

    PATCH_TARGET.write_text(
        patch_text,
        encoding="utf-8",
    )

    print("APPLY PASSED。")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"ERROR: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise
