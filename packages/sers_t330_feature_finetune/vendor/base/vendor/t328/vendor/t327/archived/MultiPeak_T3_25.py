"""Native-resolution peak-window evidence; no condition/label input at inference."""
from __future__ import annotations
import math
import torch
from torch import nn

DEFAULT_CENTERS = (813., 827., 981., 1000., 1013., 1090., 1125., 1216.,
                   1298., 1369., 1463., 1487., 1534., 1578., 1597., 1600., 2230.)


class MultiPeakEvidence(nn.Module):
    def __init__(self, dimension, heads, centers=DEFAULT_CENTERS, half_width=24,
                 axis_start=600., model_length=1901, dropout=0.1, strength=0.25):
        super().__init__()
        if dimension % heads or half_width < 2 or strength <= 0:
            raise ValueError("Invalid multi-peak dimensions/window/strength")
        center = torch.as_tensor(centers, dtype=torch.float32)
        indices = (center - axis_start).round().long()[:, None] + torch.arange(-half_width, half_width + 1)
        if center.numel() < 2 or (indices < 0).any() or (indices >= model_length).any():
            raise ValueError("At least two complete windows within the physical axis are required")
        self.register_buffer("centers", center)
        self.register_buffer("indices", indices)
        self.register_buffer("relative_axis", torch.linspace(-1., 1., indices.shape[1]))
        self.register_buffer("curve_mean", torch.zeros(len(center), 2, indices.shape[1]))
        self.register_buffer("curve_scale", torch.ones(len(center), 2, 1))
        self.register_buffer("stats_mean", torch.zeros(len(center), 9))
        self.register_buffer("stats_scale", torch.ones(len(center), 9))
        self.register_buffer("normalization_fitted", torch.tensor(False))
        self.strength = float(strength)
        self.curve_encoder = nn.Sequential(nn.Conv1d(2, 32, 3, padding=1), nn.GELU(),
                                           nn.Conv1d(32, 32, 3, padding=1), nn.GELU())
        self.curve_projection = nn.Linear(64, dimension // 2)
        self.stats_encoder = nn.Sequential(nn.Linear(9, dimension // 2), nn.GELU())
        self.token_projection = nn.Linear(dimension, dimension)
        # Position embedding identifies spectral windows, not chemical labels.
        pos = (center - axis_start - 8.) / 9.
        freq = torch.exp(torch.arange(0, dimension, 2) * (-math.log(10000.) / dimension))
        pe = torch.zeros(len(center), dimension)
        pe[:, 0::2] = torch.sin(pos[:, None] * freq)
        pe[:, 1::2] = torch.cos(pos[:, None] * freq)
        self.register_buffer("window_positions", pe)
        layer = nn.TransformerEncoderLayer(dimension, heads, dimension * 2, dropout,
                                            activation="gelu", batch_first=True)
        self.window_encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.cross_attention = nn.MultiheadAttention(dimension, heads, dropout, batch_first=True)
        self.output_norm = nn.LayerNorm(dimension)
        self.output_projection = nn.Linear(dimension, dimension)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.last_attention = None

    def extract(self, raw_intensity, measured_mask):
        if raw_intensity.ndim != 3 or raw_intensity.shape[1] != 1:
            raise ValueError("raw_intensity must be [B,1,L]")
        if measured_mask.shape != (raw_intensity.shape[0], raw_intensity.shape[-1]):
            raise ValueError("measured_mask must be [B,L]")
        y = raw_intensity[:, 0, self.indices]
        available = measured_mask[:, self.indices].bool().all(-1)
        y = torch.where(available[..., None], y, torch.zeros_like(y))
        # Background is an additional feature, not a modification of the input.
        left, right = y[..., :5].mean(-1), y[..., -5:].mean(-1)
        t = (self.relative_axis + 1.) / 2.
        baseline = left[..., None] * (1. - t) + right[..., None] * t
        residual = y - baseline
        positive = residual.clamp_min(0.)
        mass = positive.sum(-1).clamp_min(1e-8)
        centroid = (positive * self.relative_axis).sum(-1) / mass
        width = ((positive * (self.relative_axis - centroid[..., None]).square()).sum(-1) / mass).clamp_min(0.).sqrt()
        height = positive.amax(-1)
        log_height = torch.log1p(height)
        # All-window relative evidence, without dividing by a weak/zero peak.
        relative_height = log_height - (log_height * available).sum(-1, keepdim=True) / available.sum(-1, keepdim=True).clamp_min(1)
        absolute = torch.stack((height, residual.mean(-1), baseline.mean(-1), right-left,
                                residual.amin(-1), y.mean(-1)), -1)
        absolute = absolute.sign() * torch.log1p(absolute.abs())
        stats = torch.cat((absolute, centroid[..., None], width[..., None], relative_height[..., None]), -1)
        curves = torch.stack((y, residual), dim=2)
        curves = curves.masked_fill(~available[..., None, None], 0.)
        stats = stats.masked_fill(~available[..., None], 0.)
        return curves, stats, available

    @torch.no_grad()
    def fit_normalization(self, raw_intensity, measured_mask):
        if bool(self.normalization_fitted):
            raise RuntimeError("Do not refit normalization on validation/inference")
        curves, stats, available = self.extract(raw_intensity, measured_mask)
        counts = available.sum(0)
        if (counts == 0).any():
            raise ValueError("Every configured window needs real training support")
        mask_c = available[..., None, None]
        curve_mean = (curves * mask_c).sum(0) / counts[:, None, None]
        curve_var = ((curves-curve_mean).square()*mask_c).sum((0,3)) / (counts[:,None] * curves.shape[-1])
        mask_s = available[...,None]
        stats_mean = (stats*mask_s).sum(0) / counts[:,None]
        stats_var = ((stats-stats_mean).square()*mask_s).sum(0) / counts[:,None]
        self.curve_mean.copy_(curve_mean)
        self.curve_scale.copy_(curve_var.sqrt().clamp_min(1e-4)[...,None])
        self.stats_mean.copy_(stats_mean)
        self.stats_scale.copy_(stats_var.sqrt().clamp_min(1e-4))
        self.normalization_fitted.fill_(True)
        return {"real_training_spectra": len(raw_intensity), "windows": len(self.centers),
                "measured_training_support": counts.cpu().tolist(), "fit_scope": "real_train_only"}

    def forward(self, raw_intensity, measured_mask, queries):
        if not bool(self.normalization_fitted):
            raise RuntimeError("Fit multi-peak normalization on real training spectra before use")
        curves, stats, available = self.extract(raw_intensity, measured_mask)
        curves = ((curves-self.curve_mean)/self.curve_scale).masked_fill(~available[...,None,None],0.)
        stats = ((stats-self.stats_mean)/self.stats_scale).masked_fill(~available[...,None],0.)
        b, w, c, l = curves.shape
        features = self.curve_encoder(curves.reshape(b*w,c,l))
        features = torch.cat((features.mean(-1), features.amax(-1)), -1)
        curve_tokens = self.curve_projection(features).reshape(b,w,-1)
        tokens = self.token_projection(torch.cat((curve_tokens,self.stats_encoder(stats)), -1))
        tokens = (tokens+self.window_positions).masked_fill(~available[...,None],0.)
        any_available = available.any(-1)
        safe_mask = available.clone()
        safe_mask[~any_available,0] = True
        tokens = self.window_encoder(tokens, src_key_padding_mask=~safe_mask)
        tokens = tokens.masked_fill(~available[...,None],0.)
        context, attention = self.cross_attention(queries,tokens,tokens,key_padding_mask=~safe_mask)
        correction = self.output_projection(self.output_norm(context)) * any_available[:,None,None]
        self.last_attention = (attention*available[:,None,:]).detach()
        return queries + self.strength * correction
