"""Full-height rectangular-strip DiT. Input and output are real 2D images."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from src.dit_1d import SpectrumDiT1D, DiTBlock, sinusoidal, modulate
from src.curve_image_codec import CurveImageCodec


class CurveImageDiT2D(nn.Module):
    def __init__(self, *, channels, condition_dimension, prior_conditioning_enabled,
                 curve_configuration, hidden_size=512, depth=12, num_heads=8,
                 patch_size=8, mlp_ratio=4.0, gradient_checkpointing=True, dropout=0.05):
        super().__init__()
        if channels != 1 or not prior_conditioning_enabled:
            raise ValueError('2D curve DiT requires one channel and prior conditioning')
        self.channels = self.out_dim = channels
        self.self_condition = False
        self.condition_dimension = condition_dimension
        self.prior_conditioning_enabled = prior_conditioning_enabled
        self.codec = CurveImageCodec(curve_configuration)
        self.patch_size = curve_configuration['patch_width']
        self.gradient_checkpointing = gradient_checkpointing
        self.width = hidden_size
        self.patch_pixels = self.codec.height * self.patch_size
        # Linear strip embedding is exactly batched patchify + projection.
        # Channels: noisy curve, clean prior curve, per-pixel valid mask.
        self.patch_embed = nn.Linear(3*self.patch_pixels, hidden_size)
        self.time_embed = nn.Sequential(nn.Linear(256, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.condition_encoder = nn.Sequential(nn.Linear(condition_dimension, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2*hidden_size))
        self.final_linear = nn.Linear(hidden_size, self.patch_pixels)
        self.apply(SpectrumDiT1D._init)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        for layer in (self.final_modulation[-1], self.final_linear):
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)
        for layer in (self.time_embed[0], self.time_embed[2], self.condition_encoder[0], self.condition_encoder[2]):
            nn.init.normal_(layer.weight, std=0.02)

    def forward(self, image, timestep, *, valid_mask=None, condition=None, prior_conditioning=None, **kwargs):
        if kwargs:
            raise TypeError(f'unsupported 2D DiT arguments: {sorted(kwargs)}')
        b, _, h, w = image.shape
        if image.shape != (b, 1, self.codec.height, self.codec.width):
            raise ValueError('2D DiT must receive [B,1,H,W] image, not a 1D spectrum')
        if valid_mask is None or valid_mask.shape != image.shape or prior_conditioning is None or prior_conditioning.shape != image.shape:
            raise ValueError('2D image, mask and prior curve must have identical shapes')
        if condition is None or condition.shape != (b, self.condition_dimension) or timestep.shape != (b,):
            raise ValueError('invalid chemical condition or timestep')
        mask = valid_mask.to(image)
        if not torch.all((mask == 0) | (mask == 1)) or not torch.all(mask.flatten(1).sum(1) > 0):
            raise ValueError('invalid pixel mask')
        p = self.patch_size
        inputs = torch.cat((image*mask, prior_conditioning.to(image)*mask, mask), dim=1)
        patches = inputs.reshape(b, 3, h, w//p, p).permute(0, 3, 1, 2, 4).flatten(2)
        token_mask = mask.reshape(b, 1, h, w//p, p).any(dim=(1, 2, 4))
        x = self.patch_embed(patches)
        positions = torch.arange(w//p, device=x.device)*p+(p-1)/2
        x = (x+sinusoidal(positions, self.width).to(x.dtype))*token_mask[..., None]
        c = self.time_embed(sinusoidal(timestep, 256).to(x.dtype))+self.condition_encoder(condition.to(x))
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, c, token_mask, use_reentrant=False)
            else:
                x = block(x, c, token_mask)
        shift, scale = self.final_modulation(c).chunk(2, dim=-1)
        output = self.final_linear(modulate(self.final_norm(x), shift, scale))
        return output.reshape(b, w//p, h, p).permute(0, 2, 1, 3).reshape(b, 1, h, w)*mask
