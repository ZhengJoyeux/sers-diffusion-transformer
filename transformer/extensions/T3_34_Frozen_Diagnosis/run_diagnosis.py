"""Read-only diagnosis of the T3.33 frozen-feature correction; no test access."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import zipfile

import numpy as np
import pandas as pd
import torch
from torch import nn

DRUGS = ('DEL', 'CHL', 'TEB')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def grades(op):
    return 1 + (op >= .5).long().sum(-1)


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


def make_frame(cache, output, bound, ordinal_probabilities):
    truth = cache['levels'].long()
    base = grades(cache['op'])
    new = grades(output['op'])
    routed = output['route']
    logits = cache['logits']
    # The probability offset matches T3.33's exact baseline-preservation rule.
    def endpoint(sign):
        candidate = cache['op'] + ordinal_probabilities(logits + sign * bound) - ordinal_probabilities(logits)
        pm = candidate[..., 0].clamp(0, 1)
        ph = torch.minimum(candidate[..., 1].clamp(0, 1), pm)
        return grades(torch.stack((pm, ph), -1))
    low, high = endpoint(-1), endpoint(1)
    possible = (low <= truth) & (truth <= high)
    correct = base == truth
    # CHL and non-routed outputs are fixed, even for this optimistic oracle.
    possible[:, 1] = correct[:, 1]
    possible[~routed] = correct[~routed]
    rows = []
    for i, meta in enumerate(cache['metadata']):
        row = dict(meta)
        row.update(mixture_count=int((truth[i] > 0).sum()), ternary_route=bool(routed[i]))
        base_profile = new_profile = possible_profile = True
        for j, drug in enumerate(DRUGS):
            present = bool(cache['cp'][i, j] >= .5)
            bp = int(base[i, j]) if present else 0
            npred = int(new[i, j]) if present else 0
            target = int(truth[i, j])
            reachable = bool(possible[i, j]) if present and target else bp == target
            row.update({f'true_{drug}': target, f'baseline_{drug}': bp, f'corrected_{drug}': npred,
                        f'within_bound_possible_{drug}': reachable,
                        f'baseline_P_ge_M_{drug}': float(cache['op'][i, j, 0]),
                        f'baseline_P_ge_H_{drug}': float(cache['op'][i, j, 1]),
                        f'corrected_P_ge_M_{drug}': float(output['op'][i, j, 0]),
                        f'corrected_P_ge_H_{drug}': float(output['op'][i, j, 1])})
            if j in (0, 2):
                delta = output['delta'][i, 0 if j == 0 else 1]
                row[f'delta_first_{drug}'] = float(delta[0])
                row[f'delta_conditional_{drug}'] = float(delta[1])
                row[f'near_bound_{drug}'] = bool((delta.abs() >= .9 * bound).any())
            base_profile &= bp == target
            new_profile &= npred == target
            possible_profile &= reachable
        row.update(baseline_full_correct=base_profile, corrected_full_correct=new_profile,
                   within_bound_profile_possible=possible_profile)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(frame, **labels):
    row = dict(labels, samples=len(frame))
    for prefix in ('baseline', 'corrected'):
        profile = frame[prefix + '_full_correct']
        row[prefix + '_full_correct'] = int(profile.sum())
        row[prefix + '_full_accuracy'] = float(profile.mean())
        total_correct = total = 0
        for drug in DRUGS:
            mask = frame['true_' + drug] > 0
            n = int(mask.sum())
            c = int((frame.loc[mask, 'true_' + drug] == frame.loc[mask, prefix + '_' + drug]).sum())
            row[prefix + '_' + drug + '_n'] = n
            row[prefix + '_' + drug + '_correct'] = c
            row[prefix + '_' + drug + '_accuracy'] = c / n if n else None
            total_correct += c
            total += n
        row[prefix + '_present_n'] = total
        row[prefix + '_present_correct'] = total_correct
        row[prefix + '_present_accuracy'] = total_correct / total if total else None
    row['profiles_fixed'] = int((~frame.baseline_full_correct & frame.corrected_full_correct).sum())
    row['profiles_broken'] = int((frame.baseline_full_correct & ~frame.corrected_full_correct).sum())
    return row


def tables(frame):
    groups = []
    for (split, source), subset in frame.groupby(['split', 'source']):
        groups.append(summarize(subset, split=split, source=source, mixture_count='all', matrix='all'))
        for mixture, m in subset.groupby('mixture_count'):
            groups.append(summarize(m, split=split, source=source, mixture_count=int(mixture), matrix='all'))
            for matrix, g in m.groupby('matrix'):
                groups.append(summarize(g, split=split, source=source, mixture_count=int(mixture), matrix=matrix))
    ternary = frame[frame.mixture_count == 3]
    by_condition = [summarize(g, split=k[0], source=k[1], condition=k[2])
                    for k, g in ternary.groupby(['split', 'source', 'condition'])]
    confusion, bounds = [], []
    for (split, source), g in ternary.groupby(['split', 'source']):
        for drug in DRUGS:
            for prefix in ('baseline', 'corrected'):
                for target in (1, 2, 3):
                    for pred in (0, 1, 2, 3):
                        count = int(((g['true_' + drug] == target) & (g[prefix + '_' + drug] == pred)).sum())
                        confusion.append(dict(split=split, source=source, model=prefix, drug=drug,
                                              true_grade=target, predicted_grade=pred, count=count))
            if drug == 'CHL':
                continue
            wrong = g['baseline_' + drug] != g['true_' + drug]
            fixed = wrong & (g['corrected_' + drug] == g['true_' + drug])
            broken = ~wrong & (g['corrected_' + drug] != g['true_' + drug])
            bounds.append(dict(split=split, source=source, drug=drug, ternary_samples=len(g),
                               baseline_errors=int(wrong.sum()), fixed=int(fixed.sum()), broken=int(broken.sum()),
                               baseline_errors_outside_bound=int((wrong & ~g['within_bound_possible_' + drug]).sum()),
                               baseline_errors_within_bound=int((wrong & g['within_bound_possible_' + drug]).sum()),
                               rows_near_bound=int(g['near_bound_' + drug].sum()),
                               optimistic_profile_upper_count=int(g.within_bound_profile_possible.sum())))
    return pd.DataFrame(groups), pd.DataFrame(by_condition), pd.DataFrame(confusion), pd.DataFrame(bounds)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--correction-package', type=Path, default=Path('/home/wqzheng/T3_33_Ternary_Correction'))
    p.add_argument('--run-directory', type=Path, default=Path('/home/wqzheng/project_transformer/outputs/t3_33_ternary_correction_seed2026'))
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--output-directory', type=Path, default=Path('/home/wqzheng/project_transformer/outputs/t3_34_frozen_diagnosis_seed2026'))
    args = p.parse_args()
    torch.set_num_threads(2)
    checkpoint = args.checkpoint or args.run_directory / 'train/checkpoints/best_accepted.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['checkpoint_version'] != 'T3.33_frozen_image_ternary_DEL_TEB_v1' or state['ordinal_decoding'] != 'median':
        raise ValueError('Expected T3.33 checkpoint with median decoding')
    package = args.correction_package.resolve()
    for name in ('correction_model.py', 'feature_cache.py'):
        if sha256(package / name) != state['source_hashes'][name]:
            raise RuntimeError('T3.33 source differs from checkpoint: ' + name)
    sys.path.insert(0, str(package))
    from correction_model import TernaryCorrector, ordinal_probabilities, protection_check
    # Only load correction heads/buffers: CNN/Transformer inference is unnecessary.
    model = TernaryCorrector(nn.Identity(), **state['correction_config'])
    model.load_state_dict({k: v for k, v in state['model_state_dict'].items() if not k.startswith('baseline.')}, strict=True)
    model.eval()
    fingerprint = state['cache_fingerprint']
    frames, cache_hashes = [], {}
    for split in ('train', 'validation'):
        path = args.run_directory / 'feature_cache' / fingerprint / (split + '.pt')
        if not path.is_file():
            raise FileNotFoundError('Existing cache required; this script does not recompute it: ' + str(path))
        cache = torch.load(path, map_location='cpu', weights_only=False)
        validate_cache(cache, split, fingerprint, state['correction_config']['feature_dim'])
        output = model.forward_cached({k: cache[k] for k in ('features', 'cp', 'op', 'logits')})
        protection_check(output, cache)
        frames.append(make_frame(cache, output, model.config['bound'], ordinal_probabilities))
        cache_hashes[split] = sha256(path)
        print('Diagnosed existing ' + split + ' cache: ' + str(len(cache['metadata'])), flush=True)
    frame = pd.concat(frames, ignore_index=True)
    val = frame[frame.split == 'validation']
    result = summarize(val)
    expected = state['validation_metrics']
    checks = {'samples': len(val), 'present_correct': result['corrected_present_correct'],
              'full_profile_correct': result['corrected_full_correct'],
              'ternary_all_grade_correct': int(val.loc[val.mixture_count == 3, 'corrected_full_correct'].sum())}
    if any(expected[k] != v for k, v in checks.items()):
        raise RuntimeError('Accepted validation predictions do not reproduce checkpoint metrics: ' + str(checks))
    out = args.output_directory.resolve()
    out.mkdir(parents=True, exist_ok=False)
    frame.to_csv(out / 'paired_predictions.csv', index=False)
    grouped, conditions, confusion, bounds = tables(frame)
    for name, value in [('group_metrics', grouped), ('ternary_conditions', conditions),
                        ('ternary_confusions', confusion), ('correction_bound_audit', bounds)]:
        value.to_csv(out / (name + '.csv'), index=False)
    errors = []
    for _, row in frame[frame.mixture_count == 3].iterrows():
        for drug in DRUGS:
            if row['corrected_' + drug] != row['true_' + drug]:
                errors.append(dict(split=row['split'], source=row['source'], condition=row['condition'],
                                   spectrum_index=row['spectrum_index'], matrix=row['matrix'], drug=drug,
                                   true_grade=row['true_' + drug], predicted_grade=row['corrected_' + drug],
                                   true_DEL=row.true_DEL, true_CHL=row.true_CHL, true_TEB=row.true_TEB))
    pd.DataFrame(errors, columns=['split','source','condition','spectrum_index','matrix','drug',
                                 'true_grade','predicted_grade','true_DEL','true_CHL','true_TEB']).to_csv(out / 'ternary_errors.csv', index=False)
    report = dict(status='PASS', accepted_epoch=state['epoch'], checkpoint=str(checkpoint),
                  checkpoint_sha256=sha256(checkpoint), cache_fingerprint=fingerprint,
                  cache_sha256=cache_hashes, bound=model.config['bound'], validation_reproduction=checks,
                  output_protection='PASS', optimizer_steps=0, test_files_opened=0, test_predictions=0,
                  interpretation=['Training performance is retrospective, not independent generalization evidence.',
                  'A train-validation gap suggests generalization issues; low accuracy on both can also reflect the correction bound or objective.',
                  'The per-row bound oracle is optimistic and does not establish that a shared correction head can attain it.',
                  'These reports alone cannot prove that frozen features lost concentration information.',
                  'Generated-vs-real accuracy differences are descriptive and do not prove generated samples cause harm.'])
    (out / 'diagnosis_summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    summary = grouped[(grouped.mixture_count == 3) & (grouped.matrix == 'all')]
    print(summary[['split','source','samples','baseline_full_accuracy','corrected_full_accuracy',
                   'corrected_DEL_accuracy','corrected_CHL_accuracy','corrected_TEB_accuracy']].to_string(index=False))
    print(bounds.to_string(index=False))
    target = out.with_name(out.name + '_reports.zip')
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.iterdir()):
            archive.write(path, path.name)
    print('REPORT_ZIP=' + str(target), flush=True)


if __name__ == '__main__':
    main()
