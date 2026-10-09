"""Masked 1D Transformer adapter conditioned on the ACTUAL reconstruction base.

No PCA fitting or sampling here. The existing paired train/generation tensor is
the sole context. An output projection initialized to zero preserves the old
backbone at initialization; internal attention gradients start after it opens.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from src.hybrid_unet_configuration import normalize_hybrid_configuration


class ConditionalTransformerBlock(nn.Module):
    def __init__(self, dim, heads, ff_multiplier, dropout, context_dim, cross_attention):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.cross_attention = (nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
                                if cross_attention else None)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(context_dim, 6 * dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        self.ff = nn.Sequential(nn.Linear(dim, int(dim * ff_multiplier)), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(int(dim * ff_multiplier), dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, conditioning, valid):
        # There must be at least one valid key for each item (checked upstream).
        shifts = self.modulation(conditioning).unsqueeze(1).chunk(6, dim=-1)
        gate = valid.unsqueeze(-1).to(x.dtype)
        def modulate(y, shift, scale):
            return y * (1 + 0.1 * torch.tanh(scale)) + 0.1 * shift
        q = modulate(self.norm1(x), shifts[0], shifts[1])
        update, _ = self.self_attention(q, q, q, key_padding_mask=~valid, need_weights=False)
        x = (x + self.dropout(update)) * gate
        if self.cross_attention is not None:
            q = modulate(self.norm2(x), shifts[2], shifts[3])
            update, _ = self.cross_attention(q, context, context, key_padding_mask=~valid, need_weights=False)
            x = (x + self.dropout(update)) * gate
        x = (x + self.dropout(self.ff(modulate(self.norm3(x), shifts[4], shifts[5])))) * gate
        return x


class PriorBottleneckTransformer(nn.Module):
    def __init__(self, *, dim, spectrum_channels, condition_dim, time_dim, downsample_factor, configuration):
        super().__init__()
        self.configuration = normalize_hybrid_configuration(configuration)
        cfg = self.configuration
        if not cfg["enabled"] or dim % cfg["num_heads"]:
            raise ValueError("Transformer必须启用且dim能被num_heads整除。")
        self.downsample_factor = int(downsample_factor)
        self.position_projection = nn.Linear(2 * cfg["position_frequencies"], dim, bias=False)
        self.context_projection = (nn.Sequential(nn.Linear(spectrum_channels, dim), nn.SiLU(), nn.LayerNorm(dim))
                                   if cfg["cross_attention"] else None)
        self.blocks = nn.ModuleList([
            ConditionalTransformerBlock(dim, cfg["num_heads"], cfg["ff_multiplier"],
                                        cfg["dropout"], condition_dim + time_dim, cfg["cross_attention"])
            for _ in range(cfg["depth"])
        ])
        self.output_projection = nn.Linear(dim, dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def token_geometry(self, valid_mask):
        if valid_mask.ndim != 3 or not torch.isfinite(valid_mask).all():
            raise ValueError("valid_mask必须为有限[B,C,L]张量。")
        if not ((valid_mask == 0) | (valid_mask == 1)).all():
            raise ValueError("valid_mask只能取0或1。")
        if not torch.equal(valid_mask, valid_mask[:, :1].expand_as(valid_mask)):
            raise ValueError("所有光谱通道必须具有相同有效区。")
        mask = valid_mask[:, :1].float()
        factor = self.downsample_factor
        if mask.shape[-1] % factor:
            raise ValueError("输入长度未按U-Net下采样倍数补齐。")
        coverage = F.avg_pool1d(mask, factor, factor)
        valid = coverage[:, 0] > 0
        if not valid.any(dim=1).all():
            raise ValueError("每条光谱至少需要一个有效Raman点。")
        cfg = self.configuration
        axis = (torch.arange(mask.shape[-1], device=mask.device, dtype=torch.float32)
                * cfg["raman_step_cm1"] + cfg["raman_start_cm1"])
        centers = F.avg_pool1d(mask * axis, factor, factor) / coverage.clamp_min(1.0 / factor)
        return coverage, valid, centers[:, 0]

    def forward(self, features, *, valid_mask, condition_embedding, time_embedding, prior_conditioning):
        coverage, valid, centers = self.token_geometry(valid_mask)
        if features.shape[-1] != valid.shape[-1]:
            raise RuntimeError("U-Net瓶颈长度与掩码池化长度不一致。")
        cfg = self.configuration
        frequencies = torch.pow(2.0, torch.arange(cfg["position_frequencies"], device=features.device))
        phase = ((centers - cfg["raman_start_cm1"]) / cfg["raman_span_cm1"])[..., None]
        phase = phase * frequencies * (2 * math.pi)
        position = self.position_projection(torch.cat((phase.sin(), phase.cos()), -1).to(features.dtype))
        gate = valid.unsqueeze(-1).to(features.dtype)
        x = (features.transpose(1, 2) + position) * gate
        context = None
        if self.context_projection is not None:
            if prior_conditioning is None or prior_conditioning.shape != valid_mask.shape:
                raise ValueError("交叉注意力必须接收与valid_mask同形状的实际底谱。")
            # torch.where also prevents a masked NaN from leaking into pooling.
            base = torch.where(valid_mask.bool(), prior_conditioning, 0).float()
            if not torch.isfinite(base).all():
                raise ValueError("有效区底谱包含非有限值。")
            pooled = F.avg_pool1d(base, self.downsample_factor, self.downsample_factor)
            pooled = pooled / coverage.clamp_min(1.0 / self.downsample_factor)
            context = (self.context_projection(pooled.transpose(1, 2).to(features.dtype)) + position) * gate
        conditioning = torch.cat((time_embedding, condition_embedding), -1).to(features.dtype)
        for block in self.blocks:
            x = block(x, context, conditioning, valid)
        # Coverage attenuates partially observed boundary bins, not whole spectra.
        delta = self.output_projection(x) * gate * coverage.transpose(1, 2).to(x.dtype)
        return features + delta.transpose(1, 2)
