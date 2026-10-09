"""Read-only generation snapshots and train-only decomposition diagnostics.

No random draws, sampler modifications, fitted held-out data or model weights.
Snapshots are diagnostic artifacts, never approved augmentation spectra.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import numpy as np


def _matrix(values, length=None):
    x = np.asarray(values, dtype=np.float32)
    if (x.ndim != 2 or len(x) < 2 or x.shape[1] < 3
            or (length is not None and x.shape[1] != length)
            or not np.isfinite(x).all()):
        raise ValueError('阶段追踪要求有限[N,L]数组，N>=2，轴长度匹配。')
    return x


def mean_pairwise_mse(values):
    x = np.asarray(values, dtype=np.float64)
    if len(x) < 2:
        return 0.0
    centered = x - x.mean(axis=0)
    return float(2 * np.mean(centered ** 2) * len(x) / (len(x) - 1))


def component_variation(outer, broad):
    a, b = np.asarray(outer, dtype=np.float64), np.asarray(broad, dtype=np.float64)
    if a.shape != b.shape or len(a) < 2:
        raise ValueError('outer/broad必须为同批同轴成对数组。')
    ac, bc = a - a.mean(axis=0), b - b.mean(axis=0)
    cross = float(4 * np.mean(ac * bc) * len(a) / (len(a) - 1))
    av, bv = mean_pairwise_mse(a), mean_pairwise_mse(b)
    total = mean_pairwise_mse(a + b)
    return {'outer_mean_pairwise_mse': av, 'broad_mean_pairwise_mse': bv,
            'outer_broad_cross_term': cross, 'base_mean_pairwise_mse': total,
            'variance_identity_error': total - av - bv - cross}


def joint_component_summary(training_outer, training_broad, generated_outer, generated_broad):
    train_a = _matrix(training_outer).astype(np.float64)
    train_b = _matrix(training_broad, train_a.shape[1]).astype(np.float64)
    gen_a = _matrix(generated_outer, train_a.shape[1]).astype(np.float64)
    gen_b = _matrix(generated_broad, train_a.shape[1]).astype(np.float64)
    if len(train_a) != 12 or train_a.shape != train_b.shape or gen_a.shape != gen_b.shape:
        raise ValueError('联合诊断要求12对训练数据及同批生成outer/broad。')
    def basis(values):
        center = values.mean(axis=0)
        _, s, v = np.linalg.svd(values - center, full_matrices=False)
        rank = int(np.sum(s > max(float(s[0]) * 1e-6, 1e-12))) if len(s) else 0
        return center, v[:min(rank, 6)]
    ca, va = basis(train_a)
    cb, vb = basis(train_b)
    def correlation(a, b):
        if not len(va) or not len(vb):
            return np.zeros((len(va), len(vb)))
        sa, sb = (a - ca) @ va.T, (b - cb) @ vb.T
        sa -= sa.mean(axis=0); sb -= sb.mean(axis=0)
        denominator = np.outer(np.sqrt((sa ** 2).sum(axis=0)), np.sqrt((sb ** 2).sum(axis=0)))
        return np.divide(sa.T @ sb, denominator, out=np.zeros_like(denominator), where=denominator > 1e-12)
    tc, gc = correlation(train_a, train_b), correlation(gen_a, gen_b)
    return {'fit_on': 'train_only', 'training_count': 12,
            'training_reference': 'same LOO/final reference rule as DDPM training',
            'outer_basis_rank': len(va), 'broad_basis_rank': len(vb),
            'training': component_variation(train_a, train_b),
            'generated': component_variation(gen_a, gen_b),
            'training_score_cross_correlation': tc.tolist(),
            'generated_score_cross_correlation': gc.tolist(),
            'cross_correlation_difference_frobenius': float(np.linalg.norm(tc - gc)),
            'interpretation': 'Descriptive only: 12 training spectra; this does not prove that joint sampling improves generation.'}


def training_components(bank, condition_id, normalized_training, axis):
    """Rebuild the EXACT OOF reference rule; reuse checkpoint broad state."""
    from src.conditional_prior_residual import _build_outer
    x = _matrix(normalized_training)
    if len(x) != 12:
        raise ValueError('训练分解参照必须恰好12条。')
    entry = bank._entry(condition_id)
    expected_axis = np.asarray(bank.model_axis[:entry.valid_length])
    if (x.shape[1] != entry.valid_length or len(axis) != entry.valid_length
            or not np.allclose(axis, expected_axis, rtol=0, atol=1e-8)):
        raise ValueError('训练追踪轴必须与条件先验的有效模型轴一致，禁止外推。')
    if bank.training_cross_fit_enabled:
        outer = np.empty_like(x)
        for i in range(12):
            keep = np.arange(12) != i
            temporary = _build_outer(bank.prior_configuration)
            temporary.fit(x[keep], raman_shift=axis)
            outer[i:i+1] = temporary.reference_priors_for_spectra(x[i:i+1])
    else:
        outer = entry.outer.reference_priors_for_spectra(x)
    broad, local = entry.broad_local.split_raw_residuals((x - outer).astype(np.float32))
    if not np.allclose(outer + broad + local, x, rtol=1e-5, atol=1e-6):
        raise RuntimeError('训练outer+broad+local闭合检查失败。')
    return {'outer': outer.astype(np.float32), 'broad': broad.astype(np.float32),
            'base': (outer + broad).astype(np.float32), 'local': local.astype(np.float32)}


def raw_components(components, normalizer):
    # A broad/local residual must NOT receive the normalizer's affine offset.
    outer = normalizer.inverse_transform(components['outer'])
    base = normalizer.inverse_transform(components['base'])
    broad = base - outer
    return {'outer': outer, 'broad': broad, 'base': base}


def support_snapshot(module, vector):
    if module is None:
        return None
    references = module.condition_vectors.detach().cpu().numpy()
    selected = np.flatnonzero(np.isclose(references, np.asarray(vector)[None], rtol=0, atol=1e-6).all(axis=1))
    if len(selected) != 1:
        raise ValueError('追踪support条件无法唯一匹配。')
    i = int(selected[0])
    state = {'version': 'd4.25_full_spectrum_support_v1', 'fit_on': 'train_only',
             'configuration': dict(module.configuration),
             'training_counts': [int(module.training_counts[i])]}
    for key in ('condition_vectors', 'valid_masks', 'inverse_rate', 'inverse_scale',
                'lower', 'upper', 'scale', 'peak_mask'):
        state[key] = getattr(module, key).detach().cpu().numpy()[i:i+1].tolist()
    return {'padded_length': int(module.lower.shape[-1]), 'state': state}


class StageTrace:
    def __init__(self, directory, axis, number, metadata):
        self.directory = Path(directory)
        if self.directory.exists():
            raise FileExistsError(f'阶段追踪目录已存在，禁止覆盖: {self.directory}')
        self.axis = np.asarray(axis, dtype=np.float64)
        if self.axis.ndim != 1 or len(self.axis) < 3 or not np.isfinite(self.axis).all() or not np.all(np.diff(self.axis) > 0):
            raise ValueError('追踪Raman轴无效。')
        self.number = int(number)
        self.metadata = dict(metadata)
        self.arrays = {'raman_shift': self.axis.copy()}
        self.stages = []

    def reference(self, name, values):
        if not re.fullmatch('[a-z0-9_]+', name) or name in self.arrays:
            raise ValueError('追踪引用名无效或重复。')
        self.arrays[name] = _matrix(values, len(self.axis)).copy()

    def capture(self, name, values, *, active=True, kind='generated_full_spectrum'):
        if not re.fullmatch('[a-z0-9_]+', name) or name in self.arrays:
            raise ValueError('阶段名无效或重复。')
        x = _matrix(values, len(self.axis))
        if len(x) != self.number:
            raise ValueError('生成阶段光谱数不匹配。')
        self.arrays[name] = x.copy()
        self.stages.append({'name': name, 'active': bool(active), 'kind': kind,
                            'number': len(x), 'minimum': float(x.min()), 'maximum': float(x.max())})

    def save(self, *, status='complete', error=None):
        self.directory.mkdir(parents=True, exist_ok=False)
        cache = self.directory / 'stage_cache.npz'
        np.savez_compressed(cache, **self.arrays)
        metadata = {**self.metadata, 'schema': 'd4_25_generation_stage_trace_v1',
                    'diagnostic_only': True, 'status': status, 'error': error,
                    'stages': self.stages, 'array_names': list(self.arrays),
                    'cache_sha256': hashlib.sha256(cache.read_bytes()).hexdigest(),
                    'generated_count': self.number, 'point_count': len(self.axis)}
        (self.directory / 'trace_manifest.json').write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        return self.directory / 'trace_manifest.json'


def load_trace(directory):
    directory = Path(directory)
    metadata = json.loads((directory / 'trace_manifest.json').read_text())
    cache = directory / 'stage_cache.npz'
    if metadata.get('schema') != 'd4_25_generation_stage_trace_v1' or hashlib.sha256(cache.read_bytes()).hexdigest() != metadata['cache_sha256']:
        raise ValueError('阶段缓存版本或SHA256检查失败。')
    with np.load(cache, allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    return metadata, arrays
