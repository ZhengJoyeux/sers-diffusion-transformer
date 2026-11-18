"""Fixed-coordinate floating-point curve raster and exact residual bridge.

One measured Raman location per image column. No smoothing/resampling of the
measured intensities and no per-spectrum scaling. A narrow antialiased ridge
represents the sampled curve; floating point is retained during diffusion.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_CURVE = dict(height=768, patch_width=16, vertical_margin=0.25,
                     sigma_pixels=2.0, decode_temperature=0.10,
                     image_loss_weight=1.0, foreground_weight=64.0,
                     save_images_per_condition=4)


def validate_curve_configuration(configuration):
    raw = configuration.get('curve_image', {}) or {}
    if not isinstance(raw, dict) or set(raw) - set(DEFAULT_CURVE) - {'state'}:
        raise ValueError('unknown curve_image configuration')
    cfg = {**DEFAULT_CURVE, **raw}
    for key in ('height', 'patch_width', 'save_images_per_condition'):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int):
            raise ValueError(f'curve_image.{key} must be an integer')
    if cfg['height'] < 32 or cfg['patch_width'] < 1 or cfg['save_images_per_condition'] < 0:
        raise ValueError('invalid image height, patch width or image count')
    for key in ('vertical_margin', 'sigma_pixels', 'decode_temperature', 'image_loss_weight', 'foreground_weight'):
        if not math.isfinite(float(cfg[key])) or cfg[key] <= 0:
            raise ValueError(f'curve_image.{key} must be finite and positive')
    if cfg['sigma_pixels'] < 1 or cfg['decode_temperature'] > 0.25:
        raise ValueError('use sigma_pixels>=1 and 0<decode_temperature<=0.25')
    if configuration.get('model', {}).get('architecture') == 'dit_curve2d':
        diffusion = configuration.get('diffusion', configuration.get('model', {}))
        if diffusion.get('objective') != 'pred_x0':
            raise ValueError('curve-image bridge currently requires pred_x0')
        if configuration.get('model', {}).get('channels', 1) != 1:
            raise ValueError('curve-image mode requires one spectral channel')
    return cfg


def fit_curve_state(*, bank, full, bases, masks, conditions, condition_ids,
                    training_indices, padded_length, configuration):
    """Only training rows may determine axes or condition inverse transforms."""
    cfg = validate_curve_configuration(configuration)
    indices = np.asarray(training_indices, dtype=np.int64)
    if len(indices) == 0 or len(set(indices.tolist())) != len(indices):
        raise ValueError('empty or duplicate training_indices')
    train_mask = np.asarray(masks)[indices].astype(bool)
    values = np.concatenate([np.asarray(full)[indices][train_mask],
                             np.asarray(bases)[indices][train_mask]])
    if not np.isfinite(values).all():
        raise ValueError('training curves contain nonfinite values')
    lower, upper = float(values.min()), float(values.max())
    margin = max(upper - lower, 1.0) * cfg['vertical_margin']
    alignment = math.lcm(32, cfg['patch_width'])
    width = math.ceil(padded_length / alignment) * alignment
    ids = np.asarray(condition_ids, dtype=str)[indices]
    vectors = np.asarray(conditions)[indices]
    bridge = []
    for cid in sorted(set(ids)):
        rows = vectors[ids == cid]
        if not np.allclose(rows, rows[0], atol=1e-6, rtol=0):
            raise ValueError('one condition ID has inconsistent vectors')
        entry = bank.entries[cid]
        local = entry.broad_local
        multiplier = entry.raman_variance_multiplier
        if multiplier is None:
            multiplier = np.ones(entry.valid_length, dtype=np.float32)
        rate = np.ones(padded_length, dtype=np.float32)
        rate[:entry.valid_length] = float(local.local_asinh_normalizer) / float(local.local_target_abs_max) / multiplier
        bridge.append(dict(condition_id=cid, vector=rows[0].tolist(),
                           inverse_rate=rate.tolist(), inverse_scale=float(local.local_residual_scale)))
    return dict(version='d4_28_curve_image_v1', fit_on='train_only',
                height=cfg['height'], width=width, sequence_length=padded_length,
                measured_length=len(bank.model_axis), y_min=lower-margin, y_max=upper+margin,
                axis=np.asarray(bank.model_axis).tolist(), bridge=bridge,
                training_count=len(indices), representation='antialiased_sampled_curve',
                interpolation='none; original samples retained')


class CurveImageCodec(nn.Module):
    def __init__(self, configuration):
        super().__init__()
        self.configuration = dict(configuration)
        state = configuration.get('state')
        if not state or state.get('version') != 'd4_28_curve_image_v1' or state.get('fit_on') != 'train_only':
            raise ValueError('curve_image.state must be fitted on training data before model construction')
        self.state = state
        self.height, self.width = int(state['height']), int(state['width'])
        self.length = int(state['sequence_length'])
        self.y_min, self.y_max = float(state['y_min']), float(state['y_max'])
        if (not self.y_min < self.y_max or self.width < self.length
                or self.height != configuration['height']
                or self.width % configuration['patch_width']):
            raise ValueError('invalid or inconsistent fixed curve coordinates')
        for name, data in (
            ('condition_vectors', [x['vector'] for x in state['bridge']]),
            ('inverse_rate', [x['inverse_rate'] for x in state['bridge']]),
            ('inverse_scale', [x['inverse_scale'] for x in state['bridge']]),
        ):
            tensor = torch.tensor(data, dtype=torch.float32)
            if not torch.isfinite(tensor).all():
                raise ValueError('nonfinite curve bridge')
            self.register_buffer(name, tensor, persistent=False)
        if torch.any(self.inverse_rate <= 0) or torch.any(self.inverse_scale <= 0):
            raise ValueError('nonpositive curve bridge scale')

    def _fields(self, condition, like):
        matches = torch.isclose(condition.float()[:, None], self.condition_vectors.to(condition.device)[None],
                                atol=1e-6, rtol=0).all(-1)
        if not torch.all(matches.sum(-1) == 1):
            raise ValueError('condition missing or ambiguous in train-only curve bridge')
        index = matches.long().argmax(-1)
        rate = self.inverse_rate.to(like.device)[index, None, :like.shape[-1]]
        scale = self.inverse_scale.to(like.device)[index, None, None]
        return rate, scale

    def full_from_residual(self, residual, base, condition, mask):
        rate, scale = self._fields(condition, residual)
        # No silent clipping of residual values or of the intensity axis.
        full = (base.float() + scale * torch.sinh(residual.float() * rate)) * mask
        if not torch.isfinite(full).all():
            raise ValueError('nonfinite residual-to-full reconstruction')
        return full

    def residual_from_full(self, full, base, condition, mask):
        rate, scale = self._fields(condition, full)
        return torch.asinh((full.float() - base.float()) / scale) / rate * mask

    def pixel_mask(self, mask):
        if mask.ndim != 3 or mask.shape[1:] != (1, self.length):
            raise ValueError('curve mask must be [B,1,sequence_length]')
        return F.pad(mask, (0, self.width-self.length)).unsqueeze(2).expand(-1, -1, self.height, -1)

    def render(self, values, mask):
        if values.shape != mask.shape or values.shape[1:] != (1, self.length):
            raise ValueError('curve values and mask have invalid shape')
        values = values.float()
        valid = mask.bool()
        if not torch.isfinite(values[valid]).all():
            raise ValueError('curve contains nonfinite intensities')
        if torch.any((values[valid] < self.y_min) | (values[valid] > self.y_max)):
            raise ValueError('curve outside fixed train-only y-axis; stopped without clipping or rescaling')
        center = (self.y_max - values) / (self.y_max-self.y_min) * (self.height-1)
        center = F.pad(center * mask, (0, self.width-self.length)).unsqueeze(2)
        rows = torch.arange(self.height, device=values.device, dtype=torch.float32)[None, None, :, None]
        ridge = torch.exp(-0.5 * ((rows-center)/self.configuration['sigma_pixels']).square())
        return (ridge * 2 - 1) * self.pixel_mask(mask)

    def decode(self, image, mask):
        if image.shape != (len(mask), 1, self.height, self.width):
            raise ValueError('curve image must have fixed [B,1,H,W] shape')
        # Soft-argmax is differentiable; no hard threshold or post-hoc smoothing.
        weights = torch.softmax(image.float() / self.configuration['decode_temperature'], dim=2)
        rows = torch.arange(self.height, device=image.device, dtype=torch.float32)[None, None, :, None]
        centers = (weights * rows).sum(dim=2)[..., :self.length]
        return (self.y_max-centers/(self.height-1)*(self.y_max-self.y_min)) * mask

    @torch.no_grad()
    def diagnostics(self, image, mask):
        weights = torch.softmax(image.float()/self.configuration['decode_temperature'], dim=2)
        rows = torch.arange(self.height, device=image.device)[None, None, :, None]
        center = (weights * rows).sum(2, keepdim=True)
        spread = ((weights * (rows-center).square()).sum(2).sqrt())[..., :self.length]
        contrast = (image.amax(2)-image.amin(2))[..., :self.length]
        valid = mask.bool()
        return dict(mean_column_spread_pixels=float(spread[valid].mean()),
                    low_contrast_fraction=float((contrast[valid] < 0.5).float().mean()),
                    diffuse_column_fraction=float((spread[valid] > 5).float().mean()),
                    y_min=self.y_min, y_max=self.y_max,
                    note='diagnostics only; invalid columns are not silently filtered')
