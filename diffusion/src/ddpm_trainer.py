"""D0-D3.3 一维 SERS DDPM 的训练、验证、EMA 和 checkpoint 管理。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import random
from typing import Iterable

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.checkpoint_manager import (
    CheckpointManager,
    load_checkpoint_file,
)
from src.training_logger import TrainingLogger


class ExponentialMovingAverage:
    """维护扩散模型参数和缓冲区的指数移动平均。"""

    def __init__(
        self,
        model: nn.Module,
        decay: float,
        update_every: int,
    ) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay必须在0和1之间。")

        if update_every < 1:
            raise ValueError("EMA update_every必须至少为1。")

        self.decay = float(decay)
        self.update_every = int(update_every)
        self.number_of_updates = 0

        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        self.ema_model.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.number_of_updates += 1

        if self.number_of_updates % self.update_every != 0:
            return

        for ema_parameter, model_parameter in zip(
            self.ema_model.parameters(),
            model.parameters(),
            strict=True,
        ):
            ema_parameter.lerp_(
                model_parameter.detach(),
                1.0 - self.decay,
            )

        for ema_buffer, model_buffer in zip(
            self.ema_model.buffers(),
            model.buffers(),
            strict=True,
        ):
            ema_buffer.copy_(model_buffer)

    def state_dict(self) -> dict:
        return {
            "ema_model": self.ema_model.state_dict(),
            "number_of_updates": self.number_of_updates,
            "decay": self.decay,
            "update_every": self.update_every,
        }

    def load_state_dict(self, state: dict) -> None:
        self.ema_model.load_state_dict(state["ema_model"])
        self.number_of_updates = int(
            state.get("number_of_updates", 0)
        )


def _create_gradient_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=enabled,
        )
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


class WarmupCosineLearningRateScheduler:
    """按optimizer step执行一次linear warmup + cosine decay。"""

    def __init__(
        self,
        *,
        optimizer,
        total_training_steps: int,
        warmup_fraction: float,
        warmup_start_learning_rate: float,
        peak_learning_rate: float,
        minimum_learning_rate: float,
    ) -> None:
        if total_training_steps <= 0:
            raise ValueError("total_training_steps必须大于0。")

        if not 0.0 < warmup_fraction < 1.0:
            raise ValueError("warmup_fraction必须位于(0,1)。")

        if not (
            0.0
            < minimum_learning_rate
            <= warmup_start_learning_rate
            < peak_learning_rate
        ):
            raise ValueError(
                "学习率必须满足0 < minimum <= warmup_start < peak。"
            )

        self.optimizer = optimizer
        self.total_training_steps = int(total_training_steps)
        self.warmup_fraction = float(warmup_fraction)
        self.warmup_steps = max(
            1,
            min(
                self.total_training_steps - 1,
                int(
                    round(
                        self.total_training_steps
                        * self.warmup_fraction
                    )
                ),
            ),
        )

        self.warmup_start_learning_rate = float(
            warmup_start_learning_rate
        )
        self.peak_learning_rate = float(peak_learning_rate)
        self.minimum_learning_rate = float(minimum_learning_rate)
        self.completed_steps = 0

        self._set_learning_rate(
            self.learning_rate_at_step(0)
        )

    def learning_rate_at_step(self, completed_steps: int) -> float:
        import math

        step = int(
            max(
                0,
                min(
                    completed_steps,
                    self.total_training_steps,
                ),
            )
        )

        if step <= self.warmup_steps:
            progress = step / self.warmup_steps

            return (
                self.warmup_start_learning_rate
                + (
                    self.peak_learning_rate
                    - self.warmup_start_learning_rate
                )
                * progress
            )

        cosine_steps = (
            self.total_training_steps
            - self.warmup_steps
        )

        progress = (
            step - self.warmup_steps
        ) / cosine_steps

        cosine_factor = (
            0.5
            * (
                1.0
                + math.cos(
                    math.pi * progress
                )
            )
        )

        return (
            self.minimum_learning_rate
            + (
                self.peak_learning_rate
                - self.minimum_learning_rate
            )
            * cosine_factor
        )

    def _set_learning_rate(self, learning_rate: float) -> None:
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = float(learning_rate)

    def step(self) -> float:
        self.completed_steps = min(
            self.completed_steps + 1,
            self.total_training_steps,
        )

        learning_rate = self.learning_rate_at_step(
            self.completed_steps
        )
        self._set_learning_rate(learning_rate)

        return learning_rate

    def restore_step(self, completed_steps: int) -> float:
        completed_steps = int(completed_steps)

        if not 0 <= completed_steps <= self.total_training_steps:
            raise ValueError("scheduler恢复step超出训练范围。")

        self.completed_steps = completed_steps
        learning_rate = self.learning_rate_at_step(
            self.completed_steps
        )
        self._set_learning_rate(learning_rate)

        return learning_rate

    def state_dict(self) -> dict:
        return {
            "scheduler_type": "warmup_cosine",
            "completed_steps": self.completed_steps,
            "total_training_steps": self.total_training_steps,
            "warmup_fraction": self.warmup_fraction,
            "warmup_steps": self.warmup_steps,
            "warmup_start_learning_rate":
                self.warmup_start_learning_rate,
            "peak_learning_rate": self.peak_learning_rate,
            "minimum_learning_rate": self.minimum_learning_rate,
        }

    def load_state_dict(self, state: dict) -> None:
        if str(
            state.get("scheduler_type", "")
        ) != "warmup_cosine":
            raise ValueError(
                "checkpoint中的scheduler类型不兼容。"
            )

        if int(
            state.get("total_training_steps", -1)
        ) != self.total_training_steps:
            raise ValueError(
                "checkpoint scheduler的total_training_steps"
                "与当前配置不一致。"
            )

        if int(
            state.get("warmup_steps", -1)
        ) != self.warmup_steps:
            raise ValueError(
                "checkpoint scheduler的warmup_steps"
                "与当前配置不一致。"
            )

        for name, expected in {
            "warmup_start_learning_rate":
                self.warmup_start_learning_rate,
            "peak_learning_rate":
                self.peak_learning_rate,
            "minimum_learning_rate":
                self.minimum_learning_rate,
        }.items():
            actual = float(state.get(name, float("nan")))

            if abs(actual - expected) > max(
                1.0e-15,
                abs(expected) * 1.0e-10,
            ):
                raise ValueError(
                    "checkpoint scheduler与当前配置"
                    f"{name}不一致。"
                )

        self.restore_step(
            int(state.get("completed_steps", 0))
        )


class DdpmTrainer:
    """
    执行按 step 计数的一维 DDPM 训练与验证。

    D3.2 性能改动：训练分项损失先以 detached GPU tensor 累计，只在
    真正写日志或完成验证时一次性转换为 Python float，避免每个 step
    对十几个分项逐一调用 ``.item()`` 导致 GPU/CPU 强制同步。
    """

    def __init__(
        self,
        *,
        diffusion: nn.Module,
        training_loader: DataLoader,
        validation_loader: DataLoader,
        device: torch.device,
        configuration: dict,
        metadata: dict,
        checkpoint_manager: CheckpointManager,
        logger: TrainingLogger,
    ) -> None:
        self.diffusion = diffusion.to(device)
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.device = device
        self.configuration = configuration
        self.metadata = metadata
        self.checkpoint_manager = checkpoint_manager
        self.logger = logger

        training = configuration["training"]
        self.total_training_steps = int(
            training["total_training_steps"]
        )
        self.gradient_accumulation_steps = int(
            training["gradient_accumulation_steps"]
        )
        self.maximum_gradient_norm = float(
            training["maximum_gradient_norm"]
        )
        self.log_every_steps = int(training["log_every_steps"])
        self.validate_every_steps = int(
            training["validate_every_steps"]
        )
        self.checkpoint_every_steps = int(
            training["checkpoint_every_steps"]
        )
        self.maximum_validation_batches = int(
            training.get("maximum_validation_batches", 0)
        )
        self.validation_random_seed = int(
            training.get(
                "validation_random_seed",
                configuration.get("project", {}).get("random_seed", 2026),
            )
        )

        self.optimizer = AdamW(
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
            bool(training["use_mixed_precision"])
            and device.type == "cuda"
        )
        self.gradient_scaler = _create_gradient_scaler(
            self.use_mixed_precision
        )
        self.ema = ExponentialMovingAverage(
            model=self.diffusion,
            decay=float(training["ema_decay"]),
            update_every=int(training["ema_update_every"]),
        )

        self.starting_step = 0
        self.best_validation_loss = float("inf")
        self._training_iterator: Iterable | None = None
        self.latest_validation_components: dict[str, float] | None = None

    def resume(self, checkpoint_path: str | Path) -> None:
        checkpoint = load_checkpoint_file(
            checkpoint_path,
            map_location=self.device,
        )
        self.diffusion.load_state_dict(checkpoint["diffusion_state"])
        ema_state = checkpoint.get("ema_state")

        if ema_state:
            self.ema.load_state_dict(ema_state)
        else:
            self.ema.ema_model.load_state_dict(
                checkpoint["diffusion_state"]
            )

        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(
                checkpoint["optimizer_state"]
            )

        scaler_state = checkpoint.get("scaler_state")

        if scaler_state:
            self.gradient_scaler.load_state_dict(scaler_state)

        self.starting_step = int(checkpoint.get("step", 0))
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
            f"从step={self.starting_step}继续训练，"
            f"历史最佳验证损失={self.best_validation_loss:.6f}"
        )

    def _move_batch_to_device(
        self,
        batch,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """兼容历史batch、PCA先验、D4掩码和D4.1条件。"""

        reference_prior = None
        valid_mask = None
        condition = None
        prior_conditioning = None
        full_spectrum_target = None
        local_inverse_slope = None
        if isinstance(batch, dict):
            if "spectrum" not in batch:
                raise KeyError("数据集字典batch缺少spectrum。")
            spectrum = batch["spectrum"]
            reference_prior = batch.get("constraint_reference_prior")
            valid_mask = batch.get("valid_mask")
            condition = batch.get("condition")
            prior_conditioning = batch.get("prior_conditioning")
            full_spectrum_target = batch.get("full_spectrum_target")
            local_inverse_slope = batch.get("local_inverse_slope")
        elif isinstance(batch, (tuple, list)):
            spectrum = batch[0]
            if len(batch) > 1:
                reference_prior = batch[1]
        else:
            spectrum = batch

        if not torch.is_tensor(spectrum):
            raise TypeError("训练batch中的spectrum必须是torch.Tensor。")
        spectrum = spectrum.to(self.device, non_blocking=True)

        if reference_prior is not None:
            if not torch.is_tensor(reference_prior):
                raise TypeError(
                    "constraint_reference_prior必须是torch.Tensor。"
                )
            reference_prior = reference_prior.to(
                self.device,
                non_blocking=True,
            )
            if reference_prior.shape != spectrum.shape:
                raise ValueError(
                    "constraint_reference_prior形状必须与spectrum一致。"
                )

        if valid_mask is not None:
            if not torch.is_tensor(valid_mask):
                raise TypeError("valid_mask必须是torch.Tensor。")
            valid_mask = valid_mask.to(
                self.device,
                non_blocking=True,
            )
            if valid_mask.shape != spectrum.shape:
                raise ValueError("valid_mask形状必须与spectrum一致。")

        if condition is not None:
            if not torch.is_tensor(condition):
                raise TypeError("condition必须是torch.Tensor。")
            condition = condition.to(self.device, non_blocking=True)
            if condition.ndim != 2 or condition.shape[0] != spectrum.shape[0]:
                raise ValueError("condition形状必须为[B,C]。")

        if prior_conditioning is not None:
            if not torch.is_tensor(prior_conditioning):
                raise TypeError("prior_conditioning必须是torch.Tensor。")
            prior_conditioning = prior_conditioning.to(
                self.device, non_blocking=True
            )
            if prior_conditioning.shape != spectrum.shape:
                raise ValueError("prior_conditioning形状必须与spectrum一致。")
            if not torch.isfinite(prior_conditioning).all():
                raise ValueError("prior_conditioning包含NaN或无穷值。")

        for name, value in (
            ("full_spectrum_target", full_spectrum_target),
            ("local_inverse_slope", local_inverse_slope),
        ):
            if value is None:
                continue
            if not torch.is_tensor(value):
                raise TypeError(f"{name}必须是torch.Tensor。")
            value = value.to(self.device, non_blocking=True)
            if value.shape != spectrum.shape:
                raise ValueError(f"{name}形状必须与spectrum一致。")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name}包含NaN或无穷值。")
            if name == "full_spectrum_target":
                full_spectrum_target = value
            else:
                if torch.any(value < 0.0):
                    raise ValueError("local_inverse_slope不能为负。")
                local_inverse_slope = value
        if (full_spectrum_target is None) != (local_inverse_slope is None):
            raise ValueError(
                "full_spectrum_target与local_inverse_slope必须同时提供。"
            )

        return (
            spectrum,
            reference_prior,
            valid_mask,
            condition,
            prior_conditioning,
            full_spectrum_target,
            local_inverse_slope,
        )

    def _calculate_loss(
        self,
        spectrum: torch.Tensor,
        constraint_reference_prior: torch.Tensor | None,
        valid_mask: torch.Tensor | None,
        condition: torch.Tensor | None = None,
        prior_conditioning: torch.Tensor | None = None,
        full_spectrum_target: torch.Tensor | None = None,
        local_inverse_slope: torch.Tensor | None = None,
    ) -> torch.Tensor:
        arguments = {}

        if valid_mask is not None:
            if not bool(
                getattr(
                    self.diffusion,
                    "supports_valid_mask",
                    False,
                )
            ):
                raise RuntimeError(
                    "当前扩散模型不支持valid_mask。"
                )
            arguments["valid_mask"] = valid_mask

        if condition is not None:
            if not bool(getattr(self.diffusion, "supports_condition", False)):
                raise RuntimeError("当前扩散模型不支持condition。")
            arguments["condition"] = condition

        if prior_conditioning is not None:
            if not bool(
                getattr(self.diffusion, "supports_prior_conditioning", False)
            ):
                raise RuntimeError("当前扩散模型不支持prior_conditioning。")
            arguments["prior_conditioning"] = prior_conditioning

        if full_spectrum_target is not None or local_inverse_slope is not None:
            if full_spectrum_target is None or local_inverse_slope is None:
                raise ValueError("完整谱训练上下文必须成对提供。")
            arguments["full_spectrum_target"] = full_spectrum_target
            arguments["local_inverse_slope"] = local_inverse_slope

        if constraint_reference_prior is None:
            return self.diffusion(spectrum, **arguments)
        if not bool(
            getattr(
                self.diffusion,
                "supports_constraint_reference_prior",
                False,
            )
        ):
            return self.diffusion(spectrum, **arguments)

        return self.diffusion(
            spectrum,
            constraint_reference_prior=constraint_reference_prior,
            **arguments,
        )

    def _next_training_batch(self) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        if self._training_iterator is None:
            self._training_iterator = iter(self.training_loader)

        try:
            batch = next(self._training_iterator)
        except StopIteration:
            self._training_iterator = iter(self.training_loader)
            batch = next(self._training_iterator)

        return self._move_batch_to_device(batch)

    def _read_loss_components(
        self,
        total_loss: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        getter = getattr(
            self.diffusion,
            "get_latest_loss_components",
            None,
        )

        if not callable(getter):
            scalar = total_loss.detach()

            return {
                "total_loss": scalar,
                "ddpm_loss": scalar,
                "physics_loss": torch.zeros_like(scalar),
            }

        raw_components = getter()

        if not isinstance(raw_components, dict):
            raise RuntimeError(
                "get_latest_loss_components必须返回字典。"
            )

        converted: dict[str, torch.Tensor] = {}

        for name, value in raw_components.items():
            if torch.is_tensor(value):
                converted[name] = value.detach()
            else:
                converted[name] = torch.as_tensor(
                    value,
                    device=self.device,
                    dtype=total_loss.dtype,
                )

        return converted

    @staticmethod
    def _accumulate_components(
        destination: dict[str, torch.Tensor],
        source: dict[str, torch.Tensor],
        *,
        scale: float = 1.0,
    ) -> None:
        for name, value in source.items():
            scaled = value.detach() * float(scale)

            if name in destination:
                destination[name] = destination[name] + scaled
            else:
                destination[name] = scaled.clone()

    @staticmethod
    def _average_components_to_float(
        accumulated: dict[str, torch.Tensor],
        count: int,
    ) -> dict[str, float]:
        if count <= 0:
            return {}

        names = list(accumulated)

        if not names:
            return {}

        # 把所有分项拼接后只进行一次GPU到CPU同步。
        stacked = torch.stack(
            [accumulated[name] / count for name in names]
        ).detach().cpu()

        return {
            name: float(stacked[index])
            for index, name in enumerate(names)
        }

    @torch.no_grad()
    def validate(self) -> float:
        """在固定随机时间步和噪声下计算可比较的验证损失。

        DDPM 的 ``forward`` 会随机抽取时间步和高斯噪声。若每次验证
        使用不同随机输入，best.pt 可能只是一次随机噪声较容易的结果。
        此处暂存并恢复训练 RNG 状态，因此固定验证不会改变后续训练的
        随机序列或 DataLoader shuffle 行为。
        """

        self.diffusion.eval()
        accumulated_loss: torch.Tensor | None = None
        accumulated_components: dict[str, torch.Tensor] = {}
        number_of_batches = 0

        python_random_state = random.getstate()
        cuda_devices = []

        if self.device.type == "cuda":
            cuda_devices = [
                torch.cuda.current_device()
                if self.device.index is None
                else self.device.index
            ]

        try:
            with torch.random.fork_rng(
                devices=cuda_devices,
                enabled=True,
            ):
                random.seed(self.validation_random_seed)
                torch.manual_seed(self.validation_random_seed)

                for batch_index, batch in enumerate(
                    self.validation_loader
                ):
                    if (
                        self.maximum_validation_batches > 0
                        and batch_index >= self.maximum_validation_batches
                    ):
                        break

                    (
                        spectrum,
                        constraint_reference_prior,
                        valid_mask,
                        condition,
                        prior_conditioning,
                        full_spectrum_target,
                        local_inverse_slope,
                    ) = (
                        self._move_batch_to_device(batch)
                    )

                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.float16,
                        enabled=self.use_mixed_precision,
                    ):
                        loss = self._calculate_loss(
                            spectrum,
                            constraint_reference_prior,
                            valid_mask,
                            condition,
                            prior_conditioning,
                            full_spectrum_target,
                            local_inverse_slope,
                        )

                    detached_loss = loss.detach()
                    accumulated_loss = (
                        detached_loss.clone()
                        if accumulated_loss is None
                        else accumulated_loss + detached_loss
                    )
                    self._accumulate_components(
                        accumulated_components,
                        self._read_loss_components(loss),
                    )
                    number_of_batches += 1
        finally:
            random.setstate(python_random_state)
            self.diffusion.train()

        if number_of_batches == 0 or accumulated_loss is None:
            raise RuntimeError("验证集没有产生任何批次。")

        validation_loss = float(
            (accumulated_loss / number_of_batches).detach().cpu()
        )
        self.latest_validation_components = (
            self._average_components_to_float(
                accumulated_components,
                number_of_batches,
            )
        )

        return validation_loss

    def _save_checkpoint(
        self,
        *,
        step: int,
        file_name: str | None = None,
        update_latest: bool = True,
    ) -> Path:
        return self.checkpoint_manager.save(
            step=step,
            diffusion_state=self.diffusion.state_dict(),
            ema_state=self.ema.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=(
                None
                if self.learning_rate_scheduler is None
                else self.learning_rate_scheduler.state_dict()
            ),
            scaler_state=self.gradient_scaler.state_dict(),
            configuration=self.configuration,
            metadata=self.metadata,
            best_validation_loss=self.best_validation_loss,
            file_name=file_name,
            update_latest=update_latest,
        )

    def train(self) -> None:
        if self.starting_step >= self.total_training_steps:
            print(
                "检查点训练步数已经达到或超过"
                "total_training_steps，无需继续训练。"
            )
            return

        self.diffusion.train()
        accumulated_log_loss: torch.Tensor | None = None
        accumulated_log_components: dict[str, torch.Tensor] = {}
        number_of_logged_steps = 0
        for step in range(
            self.starting_step + 1,
            self.total_training_steps + 1,
        ):
            self.optimizer.zero_grad(set_to_none=True)
            step_loss: torch.Tensor | None = None
            step_components: dict[str, torch.Tensor] = {}

            for _ in range(self.gradient_accumulation_steps):
                (
                    spectrum,
                    constraint_reference_prior,
                    valid_mask,
                    condition,
                    prior_conditioning,
                    full_spectrum_target,
                    local_inverse_slope,
                ) = (
                    self._next_training_batch()
                )

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.use_mixed_precision,
                ):
                    loss = self._calculate_loss(
                        spectrum,
                        constraint_reference_prior,
                        valid_mask,
                        condition,
                        prior_conditioning,
                        full_spectrum_target,
                        local_inverse_slope,
                    )
                    backward_loss = (
                        loss / self.gradient_accumulation_steps
                    )

                self.gradient_scaler.scale(backward_loss).backward()
                detached_loss = (
                    loss.detach() / self.gradient_accumulation_steps
                )
                step_loss = (
                    detached_loss.clone()
                    if step_loss is None
                    else step_loss + detached_loss
                )
                self._accumulate_components(
                    step_components,
                    self._read_loss_components(loss),
                    scale=(
                        1.0 / self.gradient_accumulation_steps
                    ),
                )

            if step_loss is None:
                raise RuntimeError("训练step没有产生损失。")

            self.gradient_scaler.unscale_(self.optimizer)
            clip_grad_norm_(
                self.diffusion.parameters(),
                self.maximum_gradient_norm,
            )
            scale_before_update = float(
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

            accumulated_log_loss = (
                step_loss.clone()
                if accumulated_log_loss is None
                else accumulated_log_loss + step_loss
            )
            self._accumulate_components(
                accumulated_log_components,
                step_components,
            )
            number_of_logged_steps += 1

            should_validate = (
                step % self.validate_every_steps == 0
                or step == self.total_training_steps
            )
            validation_loss_for_log: float | None = None
            validation_components_for_log: dict[str, float] | None = None

            if should_validate:
                validation_loss_for_log = self.validate()
                validation_components_for_log = (
                    self.latest_validation_components
                )

                if (
                    validation_loss_for_log
                    < self.best_validation_loss
                ):
                    self.best_validation_loss = validation_loss_for_log
                    self._save_checkpoint(
                        step=step,
                        file_name="best.pt",
                        update_latest=False,
                    )

            should_log = (
                step % self.log_every_steps == 0
                or step == self.total_training_steps
                or should_validate
            )

            if should_log:
                if accumulated_log_loss is None:
                    raise RuntimeError("日志区间没有累计训练损失。")

                average_training_loss = float(
                    (
                        accumulated_log_loss
                        / number_of_logged_steps
                    )
                    .detach()
                    .cpu()
                )
                average_training_components = (
                    self._average_components_to_float(
                        accumulated_log_components,
                        number_of_logged_steps,
                    )
                )
                learning_rate = self.optimizer.param_groups[0]["lr"]

                self.logger.record(
                    step=step,
                    training_loss=average_training_loss,
                    validation_loss=validation_loss_for_log,
                    learning_rate=learning_rate,
                    training_components=average_training_components,
                    validation_components=validation_components_for_log,
                )

                accumulated_log_loss = None
                accumulated_log_components = {}
                number_of_logged_steps = 0

            should_checkpoint = (
                step % self.checkpoint_every_steps == 0
                or step == self.total_training_steps
            )

            if should_checkpoint:
                checkpoint_path = self._save_checkpoint(
                    step=step,
                    update_latest=True,
                )
                print(f"已保存检查点：{checkpoint_path}")
