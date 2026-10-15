"""Existing T3.34 train/validation cache audit, unchanged."""
import torch

def validate_cache(cache, split, fingerprint, dimensions):
    if cache['split'] != split or cache['fingerprint'] != fingerprint:
        raise ValueError('Cache split/fingerprint mismatch: ' + split)
    n = len(cache['metadata'])
    shapes = {'features': (n, dimensions), 'cp': (n, 3), 'op': (n, 3, 2),
              'logits': (n, 3, 2), 'levels': (n, 3), 'presence': (n, 3)}
    for key, shape in shapes.items():
        if tuple(cache[key].shape) != shape or not torch.isfinite(cache[key]).all():
            raise ValueError('Invalid tensor: ' + key)
    keys = set()
    for row in cache['metadata']:
        if row['split'] != split or row['source'] not in ('real', 'generated'):
            raise ValueError('Unexpected row source/split')
        index = row['spectrum_index']
        if row['source'] == 'real' and not (0 <= index < 12 if split == 'train' else 12 <= index < 16):
            raise ValueError('Real row outside fixed 12/4/4 split')
        if split == 'validation' and row['source'] != 'real':
            raise ValueError('Validation must contain real spectra only')
        identity = (row['condition'], row['source'], index)
        if identity in keys:
            raise ValueError('Duplicate cache identity')
        keys.add(identity)
