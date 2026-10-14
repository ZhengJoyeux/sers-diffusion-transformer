"""Training-only cross-combination triples; labels never enter model.forward."""
from collections import defaultdict
import hashlib
import json
import numpy as np
import torch
from torch.nn import functional as F

VARIANTS = ('low_lr_control', 'coarse_multipeak', 'cross_relation')


class ReferencePlan:
    """Same real-training exposures for every arm, with no condition-name parsing.

    Keys balance pesticide, grade, matrix and complexity transfer. Positive is
    the same pesticide/grade in a lower-order mixture. Negative changes only
    that pesticide's grade in the positive's combination and matrix.
    """
    def __init__(self, dataset):
        if dataset.split != 'train':
            raise ValueError('Reference matching is restricted to the training split')
        self.rows = {}
        groups = defaultdict(list)
        for i, sample in enumerate(dataset.samples):
            if sample.source != 'real':
                continue
            if not 0 <= int(sample.spectrum_index) < 12:
                raise ValueError('Non-training real spectrum in reference plan')
            info = dataset.repository.real_data[sample.condition]['condition_info']
            levels = tuple(int(x) for x in info.concentration_target)
            matrix = info.matrix_name
            if len(levels) != 3 or any(x not in (0, 1, 2, 3) for x in levels):
                raise ValueError('Require absent/S/M/H codes, not physical concentrations')
            self.rows[i] = (matrix, levels)
            groups[(matrix, levels)].append(i)
        self.groups = dict(groups)
        self.options = defaultdict(list)
        for (matrix, levels), anchor_rows in self.groups.items():
            count = sum(x > 0 for x in levels)
            if count < 2:
                continue
            for d, grade in enumerate(levels):
                if not grade:
                    continue
                for (pmatrix, plevels), prows in self.groups.items():
                    pcount = sum(x > 0 for x in plevels)
                    if pmatrix != matrix or plevels[d] != grade or pcount >= count:
                        continue
                    negatives = []
                    for other_grade in (grade - 1, grade + 1):
                        if other_grade not in (1, 2, 3):
                            continue
                        nlevels = list(plevels); nlevels[d] = other_grade
                        if (matrix, tuple(nlevels)) in self.groups:
                            negatives.append(self.groups[(matrix, tuple(nlevels))])
                    if negatives:
                        self.options[(d, grade, matrix, count, pcount)].append((anchor_rows, prows, negatives))
        self.keys = sorted(self.options)
        if not self.keys:
            raise ValueError('No valid cross-combination training triples')

    def audit(self):
        serialized = [(i, matrix, levels) for i, (matrix, levels) in sorted(self.rows.items())]
        return dict(real_training_rows=len(self.rows), balanced_keys=len(self.keys),
                    key_list=[list(k) for k in self.keys], split='train', source='real',
                    indices='0..11', negative_other_pesticides_unchanged=True,
                    identity_sha256=hashlib.sha256(json.dumps(serialized).encode()).hexdigest(),
                    labels_enter_forward=False, validation_references=0, test_references=0)

    def epoch(self, epoch, steps, triples_per_step=4, seed=2026):
        rng = np.random.default_rng(seed + epoch * 1009)
        key_indices = []
        while len(key_indices) < steps * triples_per_step:
            key_indices.extend(rng.permutation(len(self.keys)).tolist())
        result = []
        for index in key_indices[:steps * triples_per_step]:
            key = self.keys[index]
            choices = self.options[key]
            anchors, positives, negatives = choices[int(rng.integers(len(choices)))]
            negative_rows = negatives[int(rng.integers(len(negatives)))]
            triple = [int(rng.choice(pool)) for pool in (anchors, positives, negative_rows)]
            result.append(dict(indices=triple, drug=key[0], key=list(key)))
        return [result[i:i + triples_per_step] for i in range(0, len(result), triples_per_step)]

    def validate(self, triples):
        for triple in triples:
            a, p, n = [self.rows[i] for i in triple['indices']]
            d = triple['drug']
            if not (a[0] == p[0] == n[0] and a[1][d] == p[1][d] > 0 and
                    n[1][d] > 0 and n[1][d] != p[1][d] and
                    sum(x > 0 for x in a[1]) > sum(x > 0 for x in p[1]) and
                    all(p[1][j] == n[1][j] for j in range(3) if j != d)):
                raise RuntimeError('Invalid relation triple')


def relation_loss(embeddings, drugs, margin=.2):
    """Triples are contiguous A,P,N; only selected pesticide projection is used.

    A soft distance margin, not equality of spectra, queries or probabilities.
    Other pesticide representations keep their independent supervised losses.
    """
    if embeddings.shape[0] != 3 * len(drugs):
        raise ValueError('Require three reference rows per selected pesticide')
    grouped = embeddings.reshape(len(drugs), 3, 3, -1)
    rows = torch.arange(len(drugs), device=embeddings.device)
    features = F.normalize(grouped[rows, :, drugs, :], dim=-1, eps=1e-6)
    positive_distance = (features[:, 0] - features[:, 1]).square().sum(-1)
    negative_distance = (features[:, 0] - features[:, 2]).square().sum(-1)
    loss = F.relu(positive_distance - negative_distance + margin).mean()
    return loss, dict(positive_distance=float(positive_distance.detach().mean()),
                      negative_distance=float(negative_distance.detach().mean()),
                      active_fraction=float((positive_distance - negative_distance + margin > 0).float().mean()))
