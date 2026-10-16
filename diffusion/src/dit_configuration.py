"""Validate the pure DiT denoiser; legacy architectures remain available."""
import math


def validate_dit_context(configuration):
    model = configuration.get('model', configuration)
    kind = model.get('architecture', 'unet')
    if kind not in ('unet', 'dit', 'dit_curve2d'):
        raise ValueError('model.architecture must be unet, dit or dit_curve2d')
    residual_mode = configuration.get('broad_local_residual', {}).get('residual_mode', 'broad_local')
    if residual_mode not in ('broad_local', 'full_residual'):
        raise ValueError('residual_mode must be broad_local or full_residual')
    if residual_mode == 'full_residual' and kind not in ('dit', 'dit_curve2d'):
        raise ValueError('full_residual experiment requires DiT')
    if residual_mode == 'full_residual' and configuration.get('generation', {}).get('joint_prior_sampling', {}).get('enabled', False):
        raise ValueError('full_residual has no broad prior to jointly sample')
    if kind not in ('dit', 'dit_curve2d'):
        return None
    if kind == 'dit_curve2d':
        from src.curve_image_codec import validate_curve_configuration
        validate_curve_configuration(configuration)
    defaults = dict(hidden_size=512, depth=12, num_heads=8, patch_size=8,
                    mlp_ratio=4.0, gradient_checkpointing=True)
    raw = model.get('dit', {})
    if not isinstance(raw, dict) or set(raw) - set(defaults):
        raise ValueError('unknown model.dit configuration')
    defaults.update(raw)
    for key in ('hidden_size', 'depth', 'num_heads', 'patch_size'):
        value = defaults[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f'dit.{key} must be a positive integer')
    if defaults['hidden_size'] % defaults['num_heads'] or defaults['hidden_size'] % 2:
        raise ValueError('hidden_size must be even and divisible by num_heads')
    if not isinstance(defaults['gradient_checkpointing'], bool):
        raise ValueError('gradient_checkpointing must be boolean')
    if not math.isfinite(float(defaults['mlp_ratio'])) or defaults['mlp_ratio'] <= 0:
        raise ValueError('mlp_ratio must be positive')
    if model.get('self_condition', False):
        raise ValueError('DiT adapter requires self_condition=false')
    if model.get('bottleneck_transformer', {}).get('enabled', False):
        raise ValueError('disable the U-Net bottleneck adapter for pure DiT')
    if configuration.get('data', {}).get('raman_axis_mode') != 'union_with_valid_mask':
        raise ValueError('DiT requires union_with_valid_mask')
    condition = configuration.get('conditioning', {})
    if not condition.get('enabled', False) or not condition.get('prior_spectrum', {}).get('enabled', False):
        raise ValueError('DiT requires chemical condition and reconstruction base')
    return defaults
