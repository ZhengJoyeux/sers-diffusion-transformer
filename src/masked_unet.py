"""将Raman有效区掩码作为显式第二输入通道的一维U-Net。"""

from __future__ import annotations

import torch
from torch import nn

from src.one_dimensional_ddpm import Unet1D


def _condition_film_hook(
    module: nn.Module,
    _inputs: tuple[torch.Tensor, ...],
    output: torch.Tensor,
) -> torch.Tensor:
    """Inject the active chemical condition into one backend ResnetBlock."""

    embedding = getattr(module, "_d4_condition_embedding", None)
    film = getattr(module, "condition_film", None)
    if embedding is None or film is None:
        return output
    parameters = film(embedding).unsqueeze(-1)
    scale, shift = parameters.chunk(2, dim=1)
    return output * (1.0 + 0.1 * torch.tanh(scale)) + 0.1 * shift


class MaskConditionedUnet1D(nn.Module):
    """把有效区掩码和可选化学条件共同送入一维U-Net。"""

    def __init__(
        self,
        *,
        dim: int,
        dim_mults: tuple[int, ...],
        channels: int,
        dropout: float,
        condition_dimension: int = 0,
        condition_embedding_dimension: int = 8,
        condition_injection: str = "input_only",
        prior_conditioning_enabled: bool = False,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels必须大于0。")

        self.channels = int(channels)
        self.out_dim = int(channels)
        self.self_condition = False
        self.prior_conditioning_enabled = bool(prior_conditioning_enabled)
        self.condition_dimension = int(condition_dimension)
        if self.condition_dimension < 0:
            raise ValueError("condition_dimension不能小于0。")
        if condition_embedding_dimension <= 0:
            raise ValueError("condition_embedding_dimension必须大于0。")

        self.condition_embedding_dimension = (
            int(condition_embedding_dimension)
            if self.condition_dimension > 0
            else 0
        )
        self.condition_injection = str(condition_injection).strip().lower()
        if self.condition_injection not in {
            "input_only",
            "input_and_all_resnet_blocks_film",
        }:
            raise ValueError(
                "conditioning.injection必须为input_only或"
                "input_and_all_resnet_blocks_film。"
            )
        self.condition_encoder: nn.Module | None
        if self.condition_dimension > 0:
            self.condition_encoder = nn.Sequential(
                nn.Linear(
                    self.condition_dimension,
                    self.condition_embedding_dimension,
                ),
                nn.SiLU(),
                nn.Linear(
                    self.condition_embedding_dimension,
                    self.condition_embedding_dimension,
                ),
                nn.SiLU(),
            )
        else:
            self.condition_encoder = None

        self.network = Unet1D(
            dim=dim,
            dim_mults=dim_mults,
            channels=(
                self.channels
                + 1
                + int(self.prior_conditioning_enabled)
                + self.condition_embedding_dimension
            ),
            out_dim=self.channels,
            dropout=dropout,
            self_condition=False,
        )
        self.conditioned_resnet_blocks: list[nn.Module] = []
        if (
            self.condition_encoder is not None
            and self.condition_injection
            == "input_and_all_resnet_blocks_film"
        ):
            # The third-party U-Net only accepts x and timestep.  FiLM hooks
            # allow the same encoded condition to modulate every residual
            # block without copying or modifying the pinned backend package.
            candidates = [
                module
                for module in self.network.modules()
                if module.__class__.__name__ == "ResnetBlock"
            ]
            if not candidates:
                raise RuntimeError("没有在一维U-Net中找到可注入条件的ResnetBlock。")
            for block in candidates:
                projection = getattr(getattr(block, "block2", None), "proj", None)
                channels_out = getattr(projection, "out_channels", None)
                if channels_out is None:
                    raise RuntimeError("无法确定ResnetBlock输出通道数。")
                block.condition_film = nn.Linear(
                    self.condition_embedding_dimension,
                    2 * int(channels_out),
                )
                nn.init.zeros_(block.condition_film.weight)
                nn.init.zeros_(block.condition_film.bias)
                block.register_forward_hook(_condition_film_hook)
                self.conditioned_resnet_blocks.append(block)

    def forward(
        self,
        spectrum: torch.Tensor,
        timestep: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
        prior_conditioning: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if kwargs:
            raise TypeError(
                "MaskConditionedUnet1D收到不支持的参数："
                f"{sorted(kwargs)}"
            )
        if valid_mask is None:
            valid_mask = torch.ones_like(spectrum)
        if valid_mask.shape != spectrum.shape:
            raise ValueError("valid_mask形状必须与spectrum一致。")

        mask = valid_mask.to(
            device=spectrum.device,
            dtype=spectrum.dtype,
        )
        input_channels = [spectrum * mask, mask]
        if self.prior_conditioning_enabled:
            if prior_conditioning is None:
                raise ValueError("先验条件化U-Net必须提供prior_conditioning。")
            if not torch.is_tensor(prior_conditioning):
                raise TypeError("prior_conditioning必须是torch.Tensor。")
            prior_conditioning = prior_conditioning.to(
                device=spectrum.device,
                dtype=spectrum.dtype,
            )
            if prior_conditioning.shape != spectrum.shape:
                raise ValueError(
                    "prior_conditioning形状必须与spectrum一致："
                    f"实际{tuple(prior_conditioning.shape)}，"
                    f"期望{tuple(spectrum.shape)}。"
                )
            if not torch.isfinite(prior_conditioning).all():
                raise ValueError("prior_conditioning包含NaN或无穷值。")
            input_channels.append(prior_conditioning * mask)
        elif prior_conditioning is not None:
            raise ValueError("未启用先验谱条件化的U-Net不能接收prior_conditioning。")
        if self.condition_encoder is not None:
            if condition is None:
                raise ValueError("条件生成模型必须提供condition。")
            if not torch.is_tensor(condition):
                raise TypeError("condition必须是torch.Tensor。")
            condition = condition.to(
                device=spectrum.device,
                dtype=spectrum.dtype,
            )
            if condition.ndim == 1:
                condition = condition.unsqueeze(0)
            if condition.shape != (
                spectrum.shape[0],
                self.condition_dimension,
            ):
                raise ValueError(
                    "condition形状必须为[B,C]："
                    f"实际{tuple(condition.shape)}，期望"
                    f"({spectrum.shape[0]}, {self.condition_dimension})。"
                )
            if not torch.isfinite(condition).all():
                raise ValueError("condition包含NaN或无穷值。")
            embedding = self.condition_encoder(condition)
            embedding_channels = embedding.unsqueeze(-1).expand(
                -1,
                -1,
                spectrum.shape[-1],
            )
            input_channels.append(embedding_channels)
        elif condition is not None:
            raise ValueError("无条件模型不能接收condition。")

        conditioned_input = torch.cat(input_channels, dim=1)
        if not self.conditioned_resnet_blocks:
            return self.network(conditioned_input, timestep) * mask

        for block in self.conditioned_resnet_blocks:
            block._d4_condition_embedding = embedding
        try:
            result = self.network(conditioned_input, timestep)
        finally:
            for block in self.conditioned_resnet_blocks:
                block._d4_condition_embedding = None
        return result * mask
