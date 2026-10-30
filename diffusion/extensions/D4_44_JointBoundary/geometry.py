"""Image-domain projection; native images and measured masks are preserved."""
import hashlib
import numpy as np
import torch

VERSION = 'd4_38_image_geometry_v1'
DEFAULT = dict(stroke_pixels=2.0, top_pixels=16, bottom_pixels=8,
               tail_columns=96, fill='noise', temperature=0.10,
               contrast_min=0.5, spread_max=5.0)


def compact_bounds(full, bases, masks, training_indices, height, options=None):
    cfg = {**DEFAULT, **(options or {})}
    ids = np.asarray(training_indices, dtype=np.int64)
    mask = np.asarray(masks)[ids].astype(bool)
    values = np.concatenate([np.asarray(x)[ids][mask] for x in (full, bases)])
    if not len(ids) or not len(values) or not np.isfinite(values).all():
        raise ValueError('Empty/nonfinite training geometry')
    low, high = float(values.min()), float(values.max())
    if high <= low:
        raise ValueError('Degenerate training amplitude range')
    top, bottom = cfg['top_pixels'], cfg['bottom_pixels']
    if min(top, bottom) <= cfg['stroke_pixels']/2 + .5 or top+bottom >= height-1:
        raise ValueError('Insufficient fixed-axis stroke clearance')
    span = (high-low)*(height-1)/(height-1-top-bottom)
    return low-span*bottom/(height-1), high+span*top/(height-1)


def image_centers(native, valid, options=None):
    cfg = {**DEFAULT, **(options or {})}
    image = torch.as_tensor(native, dtype=torch.float32)
    if image.ndim != 3 or image.shape[0] != 1:
        raise ValueError('Expected one native [1,H,W] image')
    valid = torch.as_tensor(valid, device=image.device, dtype=torch.bool)
    if valid.shape != (image.shape[-1],) or not valid.any():
        raise ValueError('Missing measured-column mask')
    measured = torch.nonzero(valid).flatten()
    if not torch.equal(measured, torch.arange(len(measured), device=image.device)):
        raise ValueError('Only a contiguous measured prefix is supported; no gap filling')
    if not torch.isfinite(image[:, :, valid]).all():
        raise ValueError('Nonfinite measured native image')
    p = torch.softmax(image[0]/cfg['temperature'], dim=0)
    row = torch.arange(image.shape[1], device=image.device)[:, None]
    centers = (p*row).sum(0)
    spread = (p*(row-centers).square()).sum(0).sqrt()
    contrast = image[0].amax(0)-image[0].amin(0)
    bad = valid & ((contrast < cfg['contrast_min']) | (spread > cfg['spread_max']))
    report = dict(
        measured_columns=int(valid.sum()),
        bad_columns=int(bad.sum()),
        bad_column_indices=torch.nonzero(bad).flatten().cpu().tolist(),
        max_spread_pixels=float(spread[valid].max()),
        min_contrast=float(contrast[valid].min()))
    return centers, report


def tail_fill(centers, count, height, seed, options=None):
    cfg = {**DEFAULT, **(options or {})}
    y = np.asarray(centers, dtype=np.float32).copy()
    synthetic = np.zeros(len(y), dtype=bool)
    if cfg['fill'] == 'white' or count == len(y):
        return y, synthetic, dict(mode='white', synthetic_columns=0)
    if cfg['fill'] != 'noise' or count < 16:
        raise ValueError('Noise fill needs >=16 measured columns')
    tail = y[max(0, count-cfg['tail_columns']):count].astype(np.float64)
    delta = np.diff(tail)
    sigma = float(np.median(np.abs(delta-np.median(delta)))/0.67448975/np.sqrt(2))
    level = float(np.median(tail))
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=len(y)-count+32)
    noise = np.convolve(noise, np.array([.25, .5, .25]), mode='valid')[-(len(y)-count):]
    if noise.std() > 0:
        noise = noise/noise.std()*sigma
    clearance = cfg['stroke_pixels']/2+.5
    room = min(level-clearance, height-1-clearance-level)
    if room <= 0:
        raise ValueError('Tail background outside fixed stroke bounds')
    gain = min(1., room/max(float(np.max(np.abs(noise))), 1e-12))
    filled = level+noise*gain
    n = min(8, len(filled))
    filled[:n] = np.linspace(y[count-1], filled[n-1], n+1)[1:]
    y[count:] = filled
    synthetic[count:] = True
    return y, synthetic, dict(
        mode='synthetic_background_noise',
        synthetic_columns=int(synthetic.sum()),
        estimated_sigma_pixels=sigma, noise_gain=gain, seed=int(seed))


def continuous_image(centers, height, drawn, stroke_pixels=2.):
    y = torch.as_tensor(centers, dtype=torch.float32)
    drawn = torch.as_tensor(drawn, device=y.device, dtype=torch.bool)
    if not .5 <= stroke_pixels <= 4 or len(y) < 2:
        raise ValueError('Unsupported stroke width or image width')
    radius = stroke_pixels/2
    if not torch.isfinite(y[drawn]).all() or torch.any(
            (y[drawn] < radius+.5) | (y[drawn] > height-1-radius-.5)):
        raise ValueError('Curve stroke exceeds fixed image bounds; no clipping applied')
    output = torch.full((height, len(y)), -1., device=y.device)
    rows = torch.arange(height, device=y.device, dtype=torch.float32)[:, None]
    reach = int(np.ceil(radius+.5))
    for first in range(0, len(y), 128):
        xs = torch.arange(first, min(first+128, len(y)), device=y.device)
        best = torch.full((height, len(xs)), float('inf'), device=y.device)
        for offset in range(-reach, reach+1):
            idx = xs+offset
            safe = idx.clamp(0, len(y)-2)
            permitted = (idx >= 0) & (idx < len(y)-1) & drawn[safe] & drawn[safe+1]
            dy = y[safe+1]-y[safe]
            dx = xs-safe
            fraction = ((dx+(rows-y[safe])*dy)/(1+dy.square())).clamp(0, 1)
            distance = ((dx-fraction).square()+(rows-y[safe]-fraction*dy).square()).sqrt()
            best = torch.minimum(best, distance.masked_fill(~permitted, float('inf')))
        coverage = (radius+.5-best).clamp(0, 1)
        output[:, first:first+len(xs)] = coverage*2-1
    output[:, ~drawn] = -1
    return output[None]


@torch.no_grad()
def project_image(native, valid, *, seed=2026, options=None, require_quality=True):
    cfg = {**DEFAULT, **(options or {})}
    native = torch.as_tensor(native, dtype=torch.float32)
    centers, report = image_centers(native, valid, cfg)
    if require_quality and report['bad_columns']:
        raise ValueError('Native localization failed: '+str(report))
    valid = np.asarray(valid, dtype=bool)
    y, synthetic, fill = tail_fill(
        centers.cpu().numpy(), int(valid.sum()), native.shape[1], seed, cfg)
    output = continuous_image(
        torch.as_tensor(y, device=native.device),
        native.shape[1], valid | synthetic, cfg['stroke_pixels'])
    if synthetic.any():
        measured = continuous_image(
            centers, native.shape[1], valid, cfg['stroke_pixels'])
        columns = torch.as_tensor(valid, device=output.device)
        output[:, :, columns] = measured[:, :, columns]
    report.update(
        version=VERSION, approved=report['bad_columns']==0,
        projection='native-image soft-argmax to constant-width continuous polyline',
        tail=fill)
    return output.cpu().numpy(), synthetic, report


def image_seed(condition, index):
    return int.from_bytes(
        hashlib.sha256(f'{condition}:{index}:2026'.encode()).digest()[:4], 'big')
