"""CPU replay of saved postprocessing; never calls diffusion.sample()."""
from __future__ import annotations
import numpy as np
from src.final_spectrum_delivery import apply_final_delivery_guard
from src.intensity_normalizer import GlobalMinMaxNormalizer


def replay_final(trace, arrays, variant):
    """baseline starts after mean; noQQ starts after PCA; minimal starts at 01."""
    if variant not in ('baseline', 'noQQ', 'minimal'):
        raise ValueError('未知离线回放分支。')
    from src.spectrum_generator import (
        apply_condition_mean_fidelity_calibration,
        apply_condition_intensity_envelope_guard,
    )
    generation = trace['generation_configuration']
    start = {'baseline': '04_mean', 'noQQ': '02_pca_spread',
             'minimal': '01_ddpm_reconstruction'}[variant]
    spectra = np.asarray(arrays[start], dtype=np.float32).copy()
    training = arrays['training_full']
    diagnostics = {'variant': variant, 'starts_from': start, 'diagnostic_only': True}
    if variant == 'noQQ' and generation.get('mean_fidelity_calibration', {}).get('enabled', False):
        spectra, diagnostics['mean'] = apply_condition_mean_fidelity_calibration(
            spectra, training, configuration=generation['mean_fidelity_calibration'])
    delivery = trace['delivery_configuration']
    envelope = generation.get('intensity_envelope_guard', {}) or {}
    if envelope.get('enabled', False) and not delivery.get('enabled', False):
        spectra, diagnostics['legacy_envelope'] = apply_condition_intensity_envelope_guard(
            spectra, training, configuration=envelope, raman_shift=arrays['raman_shift'])
    support = trace.get('support')
    if support is not None and support['state']['configuration']['sampling_soft_guard']:
        import torch
        from src.conditional_spectrum_constraints import ConditionalSpectrumSupport
        module = ConditionalSpectrumSupport(support['state'], support['padded_length']).cpu().eval()
        normalizer = GlobalMinMaxNormalizer.from_state_dict(trace['normalization_state'])
        values = torch.as_tensor(normalizer.transform(spectra), dtype=torch.float32).unsqueeze(1)
        condition = torch.as_tensor(trace['condition_vector'], dtype=torch.float32).reshape(1, -1).expand(len(spectra), -1)
        with torch.no_grad():
            bounded = module.bound_full_spectrum(values, condition, torch.ones_like(values)).squeeze(1).numpy()
        spectra = normalizer.inverse_transform(bounded)
    if delivery.get('enabled', False):
        spectra, diagnostics['terminal'] = apply_final_delivery_guard(
            spectra, training, configuration=delivery, raman_shift=arrays['raman_shift'])
    if not np.isfinite(spectra).all():
        raise RuntimeError('离线回放出现NaN/Inf。')
    return spectra.astype(np.float32), diagnostics


def verify_baseline_replay(expected, replayed):
    expected, replayed = np.asarray(expected, dtype=np.float32), np.asarray(replayed, dtype=np.float32)
    if expected.shape != replayed.shape:
        raise ValueError('回放与终点缓存形状不符。')
    delta = replayed.astype(np.float64) - expected.astype(np.float64)
    # CPU/GPU expm1 float32 round-off is permitted, substantive changes are not.
    if not np.allclose(expected, replayed, rtol=3e-6, atol=8e-5):
        raise RuntimeError(f'同配置CPU回放无法复现终点缓存: max|delta|={np.max(np.abs(delta)):.6g}')
    return {'passed': True, 'maximum_absolute_difference': float(np.max(np.abs(delta))),
            'rmse': float(np.sqrt(np.mean(delta ** 2))), 'rtol': 3e-6, 'atol': 8e-5}
