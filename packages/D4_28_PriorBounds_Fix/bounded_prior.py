"""Explicit rejection sampling of whole input priors within checkpoint axes.

This conditions the original PCA proposal on the fixed image domain. It changes
that proposal distribution; diagnostics must accompany all generated results.
It never clips points, edits image axes, or filters generated output spectra.
"""
from contextlib import contextmanager
import numpy as np


class BoundedPriorSampler:
    def __init__(self, original, lower, upper, *, max_redraws=100, callback=None, names=None):
        if not np.isfinite([lower, upper]).all() or lower >= upper or max_redraws < 1:
            raise ValueError('invalid checkpoint bounds or redraw limit')
        self.original = original
        self.lower, self.upper = float(lower), float(upper)
        self.max_redraws = int(max_redraws)
        self.callback = callback
        self.names = names or {}
        self.conditions = {}

    def report(self):
        records = list(self.conditions.values())
        proposed = sum(r['proposed'] for r in records)
        rejected = sum(r['rejected'] for r in records)
        return dict(policy='whole-prior rejection sampling within original fixed image y-axis',
                    distribution='original PCA prior conditioned on all valid points being within fixed bounds',
                    original_y_min=self.lower, original_y_max=self.upper,
                    edits_to_prior_points=0, max_redraws_per_rejected_row=self.max_redraws,
                    proposed=proposed, rejected=rejected,
                    rejection_fraction=rejected/proposed if proposed else 0.,
                    returned=sum(r['returned'] for r in records),
                    conditions=self.conditions)

    def _notify(self):
        if self.callback is not None:
            self.callback(self.report())

    def sample(self, bank, number, *, condition_id, prior_random_generator,
               broad_random_generator, sampled_components=None):
        cid = str(condition_id)
        length = int(bank.entries[cid].valid_length)
        record = self.conditions.setdefault(cid, dict(name=self.names.get(cid, cid),
            proposed=0, rejected=0, returned=0, batches=0, redraws=0, events=[]))
        record['batches'] += 1
        observing = sampled_components is not None

        def draw(count):
            parts = [] if observing else None
            values = np.asarray(self.original(bank, count, condition_id=cid,
                prior_random_generator=prior_random_generator,
                broad_random_generator=broad_random_generator,
                sampled_components=parts), dtype=np.float32)
            if values.ndim != 2 or len(values) != count or values.shape[1] < length or not np.isfinite(values).all():
                raise ValueError('invalid original prior draw; cannot repair by clipping')
            if observing:
                if len(parts) != 1 or set(parts[0]) != {'outer', 'broad', 'base'}:
                    raise ValueError('unexpected original component trace schema')
                components = {k: np.asarray(v, dtype=np.float32).copy() for k,v in parts[0].items()}
                if any(v.shape != (count, length) for v in components.values()):
                    raise ValueError('component trace shape mismatch')
                if not np.array_equal(components['base'], values[:, :length]):
                    raise ValueError('component base differs from actual sampler input')
            else:
                components = None
            record['proposed'] += count
            return values.copy(), components

        values, parts = draw(int(number))
        outside = ((values[:, :length] < self.lower) | (values[:, :length] > self.upper))
        invalid = np.flatnonzero(outside.any(axis=1))
        for row in invalid:
            event = dict(batch=record['batches'], row_in_batch=int(row),
                         minimum=float(values[row, :length].min()), maximum=float(values[row, :length].max()),
                         bad_point_count=int(outside[row].sum()), redraws=0)
            record['events'].append(event)
            record['rejected'] += 1
            for attempt in range(1, self.max_redraws+1):
                candidate, candidate_parts = draw(1)
                record['redraws'] += 1
                event['redraws'] = attempt
                if np.all((candidate[0, :length] >= self.lower) & (candidate[0, :length] <= self.upper)):
                    values[row] = candidate[0]
                    if observing:
                        for key in parts:
                            parts[key][row] = candidate_parts[key][0]
                    event['replacement_minimum'] = float(candidate[0, :length].min())
                    event['replacement_maximum'] = float(candidate[0, :length].max())
                    break
                record['rejected'] += 1
            else:
                event['status'] = 'redraw_limit_exceeded'
                self._notify()
                raise RuntimeError(f'{record["name"]}: prior redraw limit reached; fixed image axis cannot be met')
        if observing:
            if not np.array_equal(parts['base'], values[:, :length]):
                raise RuntimeError('accepted component trace does not match denoiser base')
            sampled_components.append(parts)
        record['returned'] += len(values)
        self._notify()
        if len(invalid):
            print(f'PRIOR_BOUNDS: {record["name"]}; initial rejected={len(invalid)}/{number}; '
                  f'cumulative proposals={record["proposed"]}; cumulative rejected={record["rejected"]}', flush=True)
        return values


@contextmanager
def bounded_prior_context(bank_class, lower, upper, **kwargs):
    original = bank_class.sample_generation_conditioning
    sampler = BoundedPriorSampler(original, lower, upper, **kwargs)
    def wrapped(bank, number, **arguments):
        return sampler.sample(bank, number, **arguments)
    bank_class.sample_generation_conditioning = wrapped
    try:
        yield sampler
    finally:
        bank_class.sample_generation_conditioning = original
