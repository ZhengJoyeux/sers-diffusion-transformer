"""One spectrum -> a grayscale line plot with a shared, train-only intensity scale."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PlotScale:
    width: int = 2048
    height: int = 512
    amplitude_scale: float = 1.0
    lower: float = 0.0
    upper: float = 1.0
    axis_start: float = 600.0
    axis_end: float = 2500.0
    line_width: float = 1.5
    supersampling: int = 2
    version: str = 'T3.32_line_asinh_v1'

    def __post_init__(self):
        if self.width < 32 or self.height < 32 or self.supersampling < 1:
            raise ValueError('Image dimensions must be >=32')
        if not np.isfinite([self.amplitude_scale, self.lower, self.upper, self.line_width]).all():
            raise ValueError('Non-finite plotting configuration')
        if self.amplitude_scale <= 0 or self.lower >= self.upper or self.line_width <= 0:
            raise ValueError('Invalid plotting scale')
        if (self.axis_start, self.axis_end) != (600., 2500.):
            raise ValueError('This patch uses the fixed 600..2500 Raman axis')

    @classmethod
    def fit(cls, values, masks, *, width=2048, height=512):
        values = np.asarray(values, dtype=np.float32)
        masks = np.asarray(masks, dtype=bool)
        if values.ndim == 3:
            values = values[:, 0]
        if values.shape != masks.shape or values.ndim != 2 or not masks.any():
            raise ValueError('Scale fitting needs aligned real-training intensities and measured masks')
        valid = values[masks]
        if not np.isfinite(valid).all():
            raise ValueError('Non-finite measured intensity')
        amplitude = max(float(np.quantile(np.abs(valid), .95)), 1e-4)
        transformed = np.arcsinh(valid / amplitude)
        lo, hi = float(transformed.min()), float(transformed.max())
        margin = max((hi-lo)*.05, .05)
        return cls(width=width, height=height, amplitude_scale=amplitude,
                   lower=lo-margin, upper=hi+margin)

    def row_positions(self, values):
        transformed = np.arcsinh(np.asarray(values) / self.amplitude_scale)
        return (self.height-1)*(self.upper-transformed)/(self.upper-self.lower)

    def inverse_rows(self, rows):
        transformed = self.upper - np.asarray(rows)/(self.height-1)*(self.upper-self.lower)
        return self.amplitude_scale*np.sinh(transformed)

    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def resampled_mask(mask, width):
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if mask.size != 1901:
        raise ValueError('Expected 1901-point source mask')
    coordinates = np.linspace(0, 1900, width)
    left = np.floor(coordinates).astype(int)
    right = np.ceil(coordinates).astype(int)
    return mask[left] & mask[right]


def render_line(values, mask, scale):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if values.size != 1901 or mask.size != 1901 or not mask.any():
        raise ValueError('A line plot needs 1901 values and a nonempty measured mask')
    if not np.isfinite(values[mask]).all():
        raise ValueError('Non-finite measured intensity')
    # Invalid source values never enter interpolation or the image.
    safe = np.where(mask, values, 0.)
    coords = np.linspace(0, 1900, scale.width)
    resampled = np.interp(coords, np.arange(1901), safe)
    valid = resampled_mask(mask, scale.width)
    rows = scale.row_positions(resampled)
    clipped = valid & ((rows < 0) | (rows > scale.height-1))
    rows = np.clip(rows, 0, scale.height-1)
    ss = scale.supersampling
    canvas = Image.new('L', (scale.width*ss, scale.height*ss), 255)
    draw = ImageDraw.Draw(canvas)
    width = max(1, round(scale.line_width*ss))
    # Separate runs prevent a line from bridging an unmeasured interval.
    indices = np.flatnonzero(valid)
    for run in np.split(indices, np.flatnonzero(np.diff(indices) > 1)+1):
        points = [(float(i)*ss, float(rows[i])*ss) for i in run]
        if len(points) >= 2:
            draw.line(points, fill=0, width=width)
        elif points:
            draw.point(points[0], fill=0)
    canvas = canvas.resize((scale.width, scale.height), Image.Resampling.LANCZOS)
    pixels = np.asarray(canvas).copy()
    pixels[:, ~valid] = 255
    return pixels, valid, dict(clipped_columns=int(clipped.sum()), measured_columns=int(valid.sum()))


def acquisition_mask(dataset, source, condition, valid):
    records = dataset.repository.real_data if source == 'real' else dataset.repository.generated_data
    record = records[condition]
    start = float(record.get('original_axis_start_cm-1', record['axis'][0]))
    n = int(record.get('original_axis_points', len(record['axis'])))
    end = float(record.get('original_axis_end_cm-1', record['axis'][n-1]))
    axis = np.arange(600., 2501.)
    return np.asarray(valid, dtype=bool) & (axis >= start-1e-6) & (axis <= end+1e-6)


class LineImageDataset(Dataset):
    def __init__(self, original, scale, cache=None):
        self.original, self.scale = original, scale
        self.samples, self.repository, self.split = original.samples, original.repository, original.split
        self.cache = Path(cache)/scale.fingerprint() if cache is not None else None
        if self.cache is not None:
            self.cache.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.original)

    def __getitem__(self, index):
        sample = dict(self.original[index])
        raw = sample['raw_intensity'].numpy()[0]
        mask = acquisition_mask(self.original, sample['source'], sample['condition'], sample['valid_mask'].numpy())
        key = hashlib.sha256(raw.tobytes()+mask.tobytes()).hexdigest()
        path = None if self.cache is None else self.cache/(key+'.png')
        if path is not None and path.exists():
            with Image.open(path) as img:
                pixels = np.asarray(img.convert('L')).copy()
            if pixels.shape != (self.scale.height, self.scale.width):
                raise RuntimeError('Wrong cached image shape')
            valid = resampled_mask(mask, self.scale.width)
            row = self.scale.row_positions(raw[mask])
            # Cache-independent clipping audit counts source points here.
            clipping = int(((row < 0) | (row > self.scale.height-1)).sum())/int(mask.sum())
        else:
            pixels, valid, _ = render_line(raw, mask, self.scale)
            row = self.scale.row_positions(raw[mask])
            clipping = int(((row < 0) | (row > self.scale.height-1)).sum())/int(mask.sum())
            if path is not None:
                # Unique temporary names allow multiple DataLoader workers.
                import os
                tmp = path.with_suffix('.'+str(os.getpid())+'.tmp')
                Image.fromarray(pixels).save(tmp, format='PNG')
                tmp.replace(path)
        sample.update(image=torch.from_numpy(pixels.astype(np.float32)/255.).unsqueeze(0),
                      image_valid_columns=torch.from_numpy(valid.copy()),
                      measured_mask=torch.from_numpy(mask.copy()),
                      image_clipped_fraction=float(clipping))
        return sample


def fit_training_scale(dataset, width, height):
    if dataset.split != 'train':
        raise ValueError('Never fit plot scales on validation/test')
    values, masks, identities = [], [], []
    for i, ref in enumerate(dataset.samples):
        if ref.source != 'real':
            continue
        if not 0 <= int(ref.spectrum_index) < 12:
            raise ValueError('Scale fitting received a held-out real sample')
        sample = dataset[i]
        values.append(sample['raw_intensity'].numpy())
        masks.append(acquisition_mask(dataset, 'real', ref.condition, sample['valid_mask'].numpy()))
        identities.append((ref.condition, int(ref.spectrum_index)))
    if len(identities) != len(set(identities)) or len(values) != 1512:
        raise ValueError('Require the fixed 1512 real-training spectra')
    values, masks = np.stack(values), np.stack(masks)
    scale = PlotScale.fit(values, masks, width=width, height=height)
    return scale, torch.from_numpy(values), torch.from_numpy(masks), dict(
        fit_scope='real_train_only', rows=len(values), generated_rows=0, validation_rows=0,
        test_rows=0, identity_sha256=hashlib.sha256(json.dumps(identities).encode()).hexdigest())
