"""Fixed condition-mean residual image diffusion. No PCA, AE or arctan."""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor'))
from src.dit_curve2d import CurveImageDiT2D
from src.curve_image_codec import CurveImageCodec, DEFAULT_CURVE
from train_joint import joint_class
from boundary_loss import difference_losses, bounded_addition, settings
from geometry import image_centers, continuous_image

VERSION = 'fixed_condition_mean_residual_image_v1'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def fresh(path):
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def manifest(folder, exclude=()):
    p = Path(folder)
    return {x.relative_to(p).as_posix(): sha(x) for x in sorted(p.rglob('*'))
            if x.is_file() and x.name not in exclude and '__pycache__' not in x.parts}


def verify(folder, filename):
    root = Path(folder)
    entries = json.loads((root / filename).read_text())
    for rel, expected in entries.items():
        path = root / rel
        if not path.is_file() or sha(path) != expected:
            raise ValueError('Fingerprint mismatch: ' + str(path))
    return sha(root / filename)


def codec_configuration(height, width, length, low, high, axis, vector):
    # The inherited codec is used ONLY for rasterization and image coordinates.
    # Its legacy PCA/asinh bridge is deliberately never called.
    return {**DEFAULT_CURVE, 'height': height, 'patch_width': 16 if width == 1920 else 8,
            'state': dict(version='d4_28_curve_image_v1', fit_on='train_only',
                          height=height, width=width, sequence_length=length,
                          measured_length=length, y_min=float(low), y_max=float(high),
                          axis=np.asarray(axis).tolist(),
                          bridge=[dict(vector=np.asarray(vector).tolist(), inverse_rate=[1.] * length, inverse_scale=1.)])}


def fit_arrays(raw, masks, names, vectors, axis, output, *, height=768, width=1920,
               margin=0.25, provenance=None, synthetic=False):
    """All fitted statistics depend on columns 0..11 only. Masks are structural."""
    raw = np.asarray(raw, np.float32)
    masks = np.asarray(masks, bool)
    if raw.ndim != 3 or raw.shape[1] != 20 or masks.shape != (len(raw), raw.shape[2]):
        raise ValueError('Expected C x 20 x L spectra and C x L measured masks')
    if len(set(names)) != len(names) or len(names) != len(raw) or not 0.1 <= margin <= 2:
        raise ValueError('Invalid condition inventory or prespecified margin')
    if width < raw.shape[-1] or height < 32 or width % (16 if width == 1920 else 8):
        raise ValueError('Invalid image dimensions')
    if not np.isfinite(raw[np.broadcast_to(masks[:, None], raw.shape)]).all():
        raise ValueError('Nonfinite measured data')
    if not masks.any(1).all() or not (masks[:, :-1] >= masks[:, 1:]).all():
        raise ValueError('Only contiguous measured prefixes are supported; no gap filling')
    raw = np.where(masks[:, None], raw, 0.)
    train = raw[:, :12].astype(np.float64)
    mean = train.mean(1)
    delta = train - mean[:, None]
    scales = []
    for c in range(len(raw)):
        values = np.abs(delta[c, :, masks[c]])
        # Positive condition scalar; not sample-dependent, not a quantile clipping bound.
        floor = max(float(np.max(np.abs(train[c, :, masks[c]]))), 1.) * 1e-6
        scales.append(max(float(np.median(values)), floor))
    scales = np.asarray(scales)
    z = np.arcsinh(delta / scales[:, None, None])
    bound = max(1., max(float(np.abs(z[c, :, masks[c]]).max()) for c in range(len(raw)))) * (1 + margin)
    values = train[np.broadcast_to(masks[:, None], train.shape)]
    lo, hi = float(values.min()), float(values.max())
    pad = max(hi - lo, 1.) * margin
    prior_cfg = codec_configuration(height, width, raw.shape[-1], lo-pad, hi+pad, axis, vectors[0])
    residual_cfg = codec_configuration(height, width, raw.shape[-1], -bound, bound, axis, vectors[0])
    state = dict(version=VERSION, prior='condition_mean_of_12_training_spectra', synthetic=synthetic,
                 fit_on='train_only', conditions=list(names), vectors=np.asarray(vectors).tolist(),
                 transform='z=asinh((I-mu_c)/s_c); I=mu_c+s_c*sinh(z)',
                 scale_rule='max(train median absolute deviation from condition mean, amplitude floor)',
                 residual_bound=bound, residual_zero_row=(height-1)/2, margin=margin,
                 prior_codec=prior_cfg, residual_codec=residual_cfg,
                 full_codec=prior_cfg, axis=np.asarray(axis).tolist(),
                 split=dict(train=list(range(12)), validation=list(range(12,16)), test=list(range(16,20))),
                 gates=dict(contrast_min=0.5, spread_max=5.0, border_pixels=2.5,
                            png_roundtrip_pixels=0.1),
                 source=provenance or {})
    output = fresh(output)
    np.savez_compressed(output/'data.npz', raw=raw, masks=masks, mean=mean.astype(np.float32),
                        scales=scales.astype(np.float32), vectors=np.asarray(vectors, np.float32), axis=axis)
    dump(output/'state.json', state)
    # Validation is audited, never used to change fit or drop a row. Test remains untouched.
    report = []
    b = Bundle(output, unchecked=True)
    prior_dir = fresh(output/'priors')
    for c, name in enumerate(names):
        m = torch.tensor(masks[c:c+1, None], dtype=torch.float32)
        prior = torch.tensor(mean[c:c+1, None], dtype=torch.float32)
        image = b.prior.render(prior, m)
        png(prior_dir/(name+'.png'), image[0], b.prior.pixel_mask(m)[0,0,0].bool())
        counts = {}
        for split, indices in [('train', range(12)), ('validation', range(12,16))]:
            x = raw[c, list(indices)]
            r = np.arcsinh((x-mean[c]) / scales[c])
            valid = np.broadcast_to(masks[c], x.shape)
            counts[split] = dict(outside_residual=int((np.abs(r[valid])>bound).sum()),
                                 outside_full_axis=int(((x[valid]<lo-pad)|(x[valid]>hi+pad)).sum()))
        report.append(dict(condition=name, **counts))
    dump(output/'preflight.json', report)
    dump(output/'manifest.json', manifest(output))
    return output


class Bundle:
    def __init__(self, root, device='cpu', unchecked=False):
        self.root = Path(root).resolve()
        self.digest = None if unchecked else verify(self.root, 'manifest.json')
        self.state = json.loads((self.root/'state.json').read_text())
        if self.state['version'] != VERSION:
            raise ValueError('Incompatible representation')
        with np.load(self.root/'data.npz', allow_pickle=False) as x:
            self.raw = x['raw'].copy()
            self.mask = torch.tensor(x['masks'][:,None], device=device, dtype=torch.float32)
            self.mean = torch.tensor(x['mean'][:,None], device=device)
            self.scale = torch.tensor(x['scales'][:,None,None], device=device)
            self.vectors = torch.tensor(x['vectors'], device=device)
        self.residual = CurveImageCodec(self.state['residual_codec']).to(device)
        self.prior = CurveImageCodec(self.state['prior_codec']).to(device)
        self.full = CurveImageCodec(self.state['full_codec']).to(device)
        self.device = device

    def fields(self, ids):
        return self.mean[ids], self.scale[ids], self.mask[ids], self.vectors[ids]

    def encode(self, values, ids):
        base, scale, mask, _ = self.fields(ids)
        return torch.asinh((values-base)/scale)*mask

    def reconstruct(self, z, ids):
        base, scale, mask, _ = self.fields(ids)
        result = (base + scale * torch.sinh(z.float())) * mask
        if not torch.isfinite(result).all():
            raise ValueError('Nonfinite inverse residual transform')
        return result

    def batch(self, samples):
        c = torch.tensor([x[0] for x in samples], device=self.device, dtype=torch.long)
        values = torch.tensor(np.stack([self.raw[i,j] for i,j in samples])[:,None], device=self.device)
        return values, c

    def images(self, values, ids):
        base, _, mask, condition = self.fields(ids)
        return (self.residual.render(self.encode(values, ids), mask),
                self.prior.render(base, mask), self.residual.pixel_mask(mask), condition)


Joint = joint_class(CurveImageDiT2D)


class ResidualDiT(Joint):
    def forward(self, *args, **kwargs):
        # Smooth fixed pixel parameterization, not spectral/intensity clipping.
        return torch.tanh(super().forward(*args, **kwargs))


def model(bundle, config):
    return ResidualDiT(channels=1, condition_dimension=14, prior_conditioning_enabled=True,
                       curve_configuration=bundle.state['residual_codec'],
                       hidden_size=config['hidden'], depth=config['depth'], num_heads=config['heads'],
                       gradient_checkpointing=True, dropout=0.05).to(bundle.device)


class Diffusion:
    def __init__(self, timesteps=200, sampling_steps=100, device='cpu'):
        if not 2 <= sampling_steps <= timesteps:
            raise ValueError('Require 2 <= sampling steps <= timesteps')
        self.steps, self.sampling_steps = timesteps, sampling_steps
        t = torch.linspace(0, timesteps, timesteps+1, dtype=torch.float64, device=device) / timesteps
        ac = torch.cos((t+0.008)/1.008*math.pi/2).square()
        ac = ac / ac[0]
        beta = (1-ac[1:]/ac[:-1]).clamp(0., 0.999)
        self.alpha = (1-beta).cumprod(0).float()

    def loss(self, net, bundle, values, ids, *, generator=None):
        clean, prior, pixels, condition = bundle.images(values, ids)
        t = torch.randint(self.steps, (len(values),), device=values.device, generator=generator)
        a = self.alpha[t, None, None, None]
        noise = torch.randn(clean.shape, device=values.device, generator=generator)
        noisy = (a.sqrt()*clean+(1-a).sqrt()*noise)*pixels
        predicted = net(noisy, t, valid_mask=pixels, condition=condition, prior_conditioning=prior)
        weight = (1+64*((clean+1)/2).detach())*pixels
        image = ((predicted-clean).square()*weight).flatten(1).sum(1)/weight.flatten(1).sum(1)
        target_columns = torch.softmax(clean.detach()/0.1, dim=2)
        kl = (target_columns*(target_columns.clamp_min(1e-30).log()-F.log_softmax(predicted/0.1, dim=2))).sum(2)
        mask = bundle.mask[ids]
        columns = pixels[:,:,0]
        localization = (kl*columns).sum((1,2))/columns.sum((1,2))
        predicted_columns = torch.softmax(predicted.float()/0.1, dim=2)
        target_log = target_columns.clamp_min(1e-30).log()
        predicted_log = predicted_columns.clamp_min(1e-30).log()
        distribution_loss = ((target_columns*(target_log-predicted_log)).sum(2)*columns).sum((1,2)) / columns.sum((1,2)).clamp_min(1)
        rows = torch.arange(predicted.shape[2], device=predicted.device, dtype=predicted.dtype)[None, None, :, None]
        target_center = (target_columns*rows).sum(2)
        predicted_center = (predicted_columns*rows).sum(2)
        target_variance = (target_columns*(rows-target_center.unsqueeze(2)).square()).sum(2)
        predicted_variance = (predicted_columns*(rows-predicted_center.unsqueeze(2)).square()).sum(2)
        variance_loss = (((predicted_variance-target_variance).square())*columns).sum((1,2)) / columns.sum((1,2)).clamp_min(1)
        predicted_entropy = -(predicted_columns*predicted_log).sum(2)
        target_entropy = -(target_columns*target_log).sum(2)
        entropy_loss = (((predicted_entropy-target_entropy).clamp_min(0).square())*columns).sum((1,2)) / columns.sum((1,2)).clamp_min(1)
        center_delta = predicted_center-target_center
        center_continuity = ((center_delta[:,:,1:]-center_delta[:,:,:-1]).square()*columns[:,:,1:]).sum((1,2)) / columns[:,:,1:].sum((1,2)).clamp_min(1)
        center_curvature = (((center_delta[:,:,2:]-2*center_delta[:,:,1:-1]+center_delta[:,:,:-2]).square())*columns[:,:,2:]).sum((1,2)) / columns[:,:,2:].sum((1,2)).clamp_min(1)
        z = bundle.residual.decode(predicted, mask)
        full = bundle.reconstruct(z, ids)
        # Scale with the newly fitted full-spectrum axis, never old PCA bridge state.
        span = bundle.full.y_max - bundle.full.y_min
        numeric = (((full-values)/span).square()*mask).sum((1,2))/mask.sum((1,2))
        snr = self.alpha[t]/(1-self.alpha[t])
        # Keep useful gradients at the high-noise end. Pure min-SNR weighting
        # makes the final reverse-time states contribute almost no training
        # signal, which is where the decoder audit finds the largest drift.
        high_noise = torch.sigmoid((t.float()-120.)/20.)
        w = snr.clamp(max=5) + 0.5*high_noise
        base = ((image+numeric)*w).mean()
        localization_add, _ = bounded_addition(base, (0.01*localization*w).mean(), 0.02)
        distribution_add, _ = bounded_addition(base, (0.05*distribution_loss*w).mean(), 0.05)
        variance_add, _ = bounded_addition(base, (0.02*variance_loss*w).mean(), 0.05)
        entropy_add, _ = bounded_addition(base, (0.01*entropy_loss*w).mean(), 0.05)
        geometry_add, _ = bounded_addition(base, (0.02*(center_continuity+center_curvature)*w).mean(), 0.02)
        cfg = settings()
        all_d, seam = difference_losses(full/span, values/span, mask, net.patch_size, cfg)
        aux = (cfg['weight']*(all_d+cfg['boundary_extra']*seam)*w).mean()
        boundary, _ = bounded_addition(base, aux, cfg['maximum_ratio'])
        loss = base + localization_add + distribution_add + variance_add + entropy_add + geometry_add + boundary
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        return loss, dict(total=float(loss.detach()), image=float(image.mean().detach()),
                          spectral=float(numeric.mean().detach()), boundary=float(boundary.detach()),
                          localization=float(localization_add.detach()),
                          distribution=float(distribution_add.detach()),
                          variance=float(variance_add.detach()),
                          entropy=float(entropy_add.detach()),
                          geometry=float(geometry_add.detach()))

    @torch.no_grad()
    def sample(self, net, bundle, ids, generator):
        base, _, mask, condition = bundle.fields(ids)
        prior = bundle.prior.render(base, mask)
        pixels = bundle.residual.pixel_mask(mask)
        image = torch.randn(pixels.shape, device=pixels.device, generator=generator)*pixels
        times = torch.linspace(-1, self.steps-1, self.sampling_steps+1).long().tolist()[::-1]
        for now, nxt in zip(times[:-1], times[1:]):
            t = torch.full((len(ids),), now, device=pixels.device, dtype=torch.long)
            clean = net(image, t, valid_mask=pixels, condition=condition, prior_conditioning=prior)
            if nxt < 0:
                image = clean
            else:
                a, an = self.alpha[now], self.alpha[nxt]
                noise = (image-a.sqrt()*clean)/(1-a).sqrt()
                image = an.sqrt()*clean+(1-an).sqrt()*noise
            image *= pixels
        return image


def png(path, image, valid):
    x = image.detach().float().cpu().numpy()[0]
    valid = np.asarray(torch.as_tensor(valid).cpu(), bool)
    if not np.isfinite(x[:,valid]).all() or np.any(np.abs(x[:,valid])>1.000001):
        raise ValueError('PNG pixels outside fixed [-1,1] coordinates')
    # Fixed mapping only; invalid columns white and identified by separate measured mask.
    pixels = np.rint((1-x)/2*255).astype(np.uint8)
    pixels[:,~valid] = 255
    Image.fromarray(pixels).save(path)


def strict_decode(native, bundle, cid):
    m = bundle.mask[cid:cid+1]
    valid = bundle.residual.pixel_mask(m)[0,0,0].bool()
    centers, report = image_centers(native, valid, bundle.state['gates'])
    if report['bad_columns']:
        raise ValueError('Native residual localization failed: '+json.dumps(report))
    border = bundle.state['gates']['border_pixels']
    if ((centers[valid]<border)|(centers[valid]>native.shape[1]-1-border)).any():
        raise ValueError('Residual curve touches fixed image boundary')
    z = bundle.residual.decode(native[None], m)
    ids = torch.tensor([cid], device=m.device)
    full = bundle.reconstruct(z, ids)
    # Bounds are checked before any rendering; no retry, projection or spectral clipping.
    bundle.full.render(full, m)
    return z, full, report
