"""Spectrum-space DiT with adaLN-Zero, adapted from the official DiT design.

Reference: https://github.com/facebookresearch/DiT/blob/main/models.py
No convolutional denoising layers, VAE, or U-Net skip/downsample path.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def sinusoidal(values, size):
    frequencies = torch.exp(-math.log(10000) * torch.arange(size // 2, device=values.device).float() / (size // 2))
    angles = values.float().unsqueeze(-1) * frequencies
    return torch.cat((angles.cos(), angles.sin()), dim=-1)


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class DiTBlock(nn.Module):
    def __init__(self, width, heads, ratio, dropout):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, int(width * ratio)), nn.GELU(approximate='tanh'),
                                 nn.Dropout(dropout), nn.Linear(int(width * ratio), width))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
        self.dropout = dropout

    def forward(self, x, c, token_mask):
        s1, z1, g1, s2, z2, g2 = self.adaLN_modulation(c).chunk(6, dim=-1)
        b, n, d = x.shape
        qkv = self.qkv(modulate(self.norm1(x), s1, z1)).reshape(b, n, 3, self.heads, d // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = F.scaled_dot_product_attention(q, k, v, attn_mask=token_mask[:, None, None, :],
                                                   dropout_p=self.dropout if self.training else 0.0)
        attention = attention.transpose(1, 2).reshape(b, n, d)
        x = x + g1[:, None] * self.proj(attention)
        x = x + g2[:, None] * self.mlp(modulate(self.norm2(x), s2, z2))
        return x * token_mask[:, :, None]


class SpectrumDiT1D(nn.Module):
    def __init__(self, *, channels, condition_dimension, prior_conditioning_enabled,
                 hidden_size=512, depth=12, num_heads=8, patch_size=8, mlp_ratio=4.0,
                 gradient_checkpointing=True, dropout=0.05):
        super().__init__()
        self.channels = self.out_dim = channels
        self.self_condition = False
        self.condition_dimension = condition_dimension
        self.prior_conditioning_enabled = prior_conditioning_enabled
        self.patch_size = patch_size
        self.gradient_checkpointing = gradient_checkpointing
        self.width = hidden_size
        self.patch_embed = nn.Linear((channels * (1 + int(prior_conditioning_enabled)) + 1) * patch_size, hidden_size)
        self.time_embed = nn.Sequential(nn.Linear(256, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.condition_encoder = nn.Sequential(nn.Linear(condition_dimension, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.final_linear = nn.Linear(hidden_size, patch_size * channels)
        self.apply(self._init)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        for layer in (self.final_modulation[-1], self.final_linear):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        for layer in (self.time_embed[0], self.time_embed[2], self.condition_encoder[0], self.condition_encoder[2]):
            nn.init.normal_(layer.weight, std=0.02)

    @staticmethod
    def _init(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, spectrum, timestep, *, valid_mask=None, condition=None, prior_conditioning=None, **kwargs):
        if kwargs:
            raise TypeError(f'unsupported DiT arguments: {sorted(kwargs)}')
        if spectrum.ndim != 3 or spectrum.shape[1] != self.channels:
            raise ValueError('spectrum must be [B,C,L]')
        b, _, length = spectrum.shape
        mask = torch.ones_like(spectrum) if valid_mask is None else valid_mask.to(spectrum)
        if mask.shape != spectrum.shape or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError('valid_mask must match spectrum and contain only 0/1')
        if not torch.all(mask.sum(dim=(1, 2)) > 0):
            raise ValueError('every spectrum needs valid points')
        if condition is None:
            raise ValueError('DiT requires condition')
        if condition.ndim == 1:
            condition = condition.unsqueeze(0)
        if condition.shape != (b, self.condition_dimension):
            raise ValueError('invalid condition shape')
        if timestep.shape != (b,):
            raise ValueError('timestep must be [B]')
        inputs = [spectrum * mask, mask[:, :1]]
        if self.prior_conditioning_enabled:
            if prior_conditioning is None or prior_conditioning.shape != spectrum.shape:
                raise ValueError('DiT requires matching reconstruction base')
            inputs.append(prior_conditioning.to(spectrum) * mask)
        elif prior_conditioning is not None:
            raise ValueError('prior conditioning is disabled')
        pad = (-length) % self.patch_size
        values = F.pad(torch.cat(inputs, dim=1), (0, pad))
        padded_mask = F.pad(mask[:, :1], (0, pad))
        token_mask = padded_mask.reshape(b, -1, self.patch_size).any(-1)
        patches = values.unfold(-1, self.patch_size, self.patch_size).permute(0, 2, 1, 3).flatten(2)
        x = self.patch_embed(patches)
        # Fixed physical-grid order; masked padding never participates as a key.
        positions = torch.arange(x.shape[1], device=x.device) * self.patch_size + (self.patch_size - 1) / 2
        x = (x + sinusoidal(positions, self.width).to(x.dtype)) * token_mask[:, :, None]
        c = self.time_embed(sinusoidal(timestep, 256).to(x.dtype)) + self.condition_encoder(condition.to(spectrum))
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, c, token_mask, use_reentrant=False)
            else:
                x = block(x, c, token_mask)
        shift, scale = self.final_modulation(c).chunk(2, dim=-1)
        output = self.final_linear(modulate(self.final_norm(x), shift, scale))
        output = output.reshape(b, -1, self.channels, self.patch_size).permute(0, 2, 1, 3).flatten(2)
        return output[:, :, :length] * mask
