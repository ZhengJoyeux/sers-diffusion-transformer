"""Two target-aligned continuation arms on the accepted T3.30 baseline."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from aligned_model import (ROOT, VERSION, ARMS, DEFAULT_BASE, BASE_METRICS, INPUT_KEYS, CACHE_KEYS, DRUGS,
    from_baseline, training_loss, target_loss, optimizer, save_checkpoint, load_checkpoint,
    rt, previous, old, parent, UpdateSchedule)
import run_feature as prior_run

COUNTS = (7560, 504, 3240)
SOURCE_COUNTS = {'real': 1512, 'generated': 6048}


def package_hashes():
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text())
    for name, digest in manifest.items():
        if rt.sha256(ROOT / name) != digest: raise RuntimeError('Package changed: ' + name)
    return manifest


def compact(m): return prior_run.compact(m)
def metric(frame): return prior_run.metrics(frame)
def predict_inputs(cache): return {k: cache[k] for k in INPUT_KEYS}


@torch.no_grad()
def extract_cache(model, dataset, args):
    model.eval(); chunks = {k: [] for k in CACHE_KEYS}; metadata = []
    for i, batch in enumerate(rt.make_loader(dataset, args.batch_size, args.num_workers)):
        c = model.extract(rt.model_inputs(dataset, batch, args.device))
        c.update(classes=batch['class_target'], levels=batch['concentration_target'])
        for key in CACHE_KEYS: chunks[key].append(c[key].detach().cpu())
        metadata.extend(rt.metadata_rows(dataset, batch))
        if (i + 1) % 100 == 0: print('CACHE batches=', i + 1, flush=True)
    return {k: torch.cat(v) for k, v in chunks.items()}, pd.DataFrame(metadata)


def put_predictions(meta, output, decoder):
    rows = prior_run.frame_rows(meta.to_dict('records'), output['cp'], output['grade'], output['op'], output['route'], decoder)
    return pd.DataFrame(rows)


@torch.no_grad()
def evaluate(model, cache, metadata, args, baseline_frame=None):
    model.eval(); chunks = {k: [] for k in ('cp', 'grade', 'op', 'route', 'target_logits')}
    for start in range(0, len(cache['grade']), args.batch_size):
        c = {k: cache[k][start:start + args.batch_size].to(args.device) for k in INPUT_KEYS}
        out = model.predict_cached(c); route = out['route']
        for j in range(3):
            if j != model.target:
                if not torch.equal(out['grade'][:, j], c['grade'][:, j]) or not torch.equal(out['op'][:, j], c['op'][:, j]):
                    raise RuntimeError('Inactive drug changed')
        if not torch.equal(out['cp'], c['cp']) or not torch.equal(out['grade'][~route], c['grade'][~route]) or not torch.equal(out['op'][~route], c['op'][~route]):
            raise RuntimeError('Protected baseline predictions changed')
        for k in chunks: chunks[k].append(out[k].detach().cpu())
    outputs = {k: torch.cat(v) for k, v in chunks.items()}
    frame = put_predictions(metadata, outputs, model.arm); m = metric(frame)
    loss, count = target_loss(outputs['target_logits'], cache['levels'], cache['classes'], model.target)
    l2 = float(model.l2sp())
    m['supervised_validation_loss'] = dict(target=float(loss), l2sp=l2, total=float(loss) + args.l2sp_weight * l2,
                                          ternary_samples=count)
    if baseline_frame is not None:
        identity = ['condition', 'source', 'spectrum_index']
        a = baseline_frame.sort_values(identity).reset_index(drop=True); b = frame.sort_values(identity).reset_index(drop=True)
        if not a[identity].equals(b[identity]): raise RuntimeError('Baseline/cache identities differ')
        y = b[['true_' + d for d in DRUGS]].to_numpy()
        old_pred = a[['pred_' + d for d in DRUGS]].to_numpy(); new_pred = b[['pred_' + d for d in DRUGS]].to_numpy()
        old_good = (y == old_pred).all(1); new_good = (y == new_pred).all(1)
        m.update(changed_spectra=int((old_pred != new_pred).any(1).sum()),
                 full_profiles_fixed=int((~old_good & new_good).sum()), full_profiles_broken=int((old_good & ~new_good).sum()))
    return m, frame


def check(args, package, before):
    folder = args.suite / 'check'; folder.mkdir(exist_ok=False)
    model, state = from_baseline(args.checkpoint, 'teb_corn', args.device)
    baseline_digest = previous.frozen_digest(model.baseline)
    datasets, audit, sources = rt.configure_project(args.project)
    with previous.guard_test_dataset(datasets[2]):
        val, metadata = extract_cache(model, datasets[1], args)
        frame = put_predictions(metadata, dict(cp=val['cp'], grade=val['grade'], op=val['op'], route=(val['cp'] >= .5).all(1)), 'T3.30_verified_TEB_only')
        baseline = metric(frame)
        for key, value in BASE_METRICS.items():
            if baseline[key] != value: raise RuntimeError('Parent reproduction failed: ' + key)
        prior_run.reference_match(frame, pd.read_csv(ROOT / 'reference/baseline_validation.csv'))
        parameters = {}
        for arm in ARMS:
            candidate, _ = from_baseline(args.checkpoint, arm, args.device)
            m, f = evaluate(candidate, val, metadata, args, frame)
            prior_run.reference_match(f, frame)
            if compact(m) != compact(baseline): raise RuntimeError('Initial task branch changes current baseline')
            parameters[arm] = dict(total=sum(p.numel() for p in candidate.parameters()),
                trainable=sum(p.numel() for p in candidate.parameters() if p.requires_grad),
                names=[n for n, p in candidate.named_parameters() if p.requires_grad])
            # DEL and TEB pre-fusion inputs must be identical because the parent's frozen prefix is shared in behavior.
            first = next(iter(rt.make_loader(datasets[1], args.batch_size, args.num_workers)))
            alternate = candidate.extract(rt.model_inputs(datasets[1], first, args.device))
            n = len(first['class_target'])
            for key in INPUT_KEYS:
                torch.testing.assert_close(alternate[key].cpu(), val[key][:n], rtol=0, atol=0)
            del candidate
        print('BASELINE PASS: class=503 grades=985 full=380 ternary=125; extracting pre-fusion train queries', flush=True)
        train, train_meta = extract_cache(model, datasets[0], args)
    if len(train['grade']) != COUNTS[0] or len(val['grade']) != COUNTS[1] or int((train['classes'] > .5).all(1).sum()) != COUNTS[2]:
        raise RuntimeError('Training/validation exposure changed')
    if train_meta.source.value_counts().to_dict() != SOURCE_COUNTS: raise RuntimeError('Source selection changed')
    if previous.frozen_digest(model.baseline) != baseline_digest: raise RuntimeError('Check changed protected baseline')
    torch.save(train, folder / 'train_queries.pt'); torch.save(val, folder / 'validation_queries.pt')
    train_meta.to_csv(folder / 'training_identities.csv', index=False); metadata.to_csv(folder / 'validation_identities.csv', index=False)
    frame.to_csv(folder / 'baseline_validation.csv', index=False)
    train_frame = put_predictions(train_meta, dict(cp=train['cp'], grade=train['grade'], op=train['op'], route=(train['cp'] >= .5).all(1)), 'T3.30_verified_TEB_only')
    train_frame.to_csv(folder / 'baseline_training.csv', index=False)
    rt.save_json(folder / 'baseline_training_sources.json', prior_run.source_report(train_frame))
    names = ('train_queries.pt', 'validation_queries.pt', 'training_identities.csv', 'validation_identities.csv',
             'baseline_validation.csv', 'baseline_training.csv', 'baseline_training_sources.json')
    summary = dict(status='PASS', baseline=baseline, parent_sha256=rt.sha256(args.checkpoint),
        protected_baseline_digest=baseline_digest, parent_config=model.baseline.base_config,
        input_hashes=before, source_sha256=sources, dataset_audit=audit, package_sha256=package,
        cache_sha256={name: rt.sha256(folder / name) for name in names}, parameters=parameters,
        cache_stage='before mixture_aware_query_fusion; query [3,128]; not final concentration features',
        network_training_updates=0, test_sample_extractions=0, test_predictions=0)
    rt.save_json(folder / 'check_summary.json', summary)
    print('CHECK PASS: train=7560 (1512 real + 6048 generated); validation=504; test predictions=0', flush=True)
    print('PARAMETERS:', json.dumps({a: {k: v for k, v in p.items() if k != 'names'} for a, p in parameters.items()}))


def load_cache(args, package, before):
    folder = args.suite / 'check'; c = json.loads((folder / 'check_summary.json').read_text())
    if c['status'] != 'PASS' or c['input_hashes'] != before or c['package_sha256'] != package or c['parent_sha256'] != rt.sha256(args.checkpoint):
        raise RuntimeError('Baseline/input/package changed since check')
    for name, digest in c['cache_sha256'].items():
        if rt.sha256(folder / name) != digest: raise RuntimeError('Protected cache changed: ' + name)
    train = torch.load(folder / 'train_queries.pt', map_location='cpu', weights_only=True)
    val = torch.load(folder / 'validation_queries.pt', map_location='cpu', weights_only=True)
    tr_meta = pd.read_csv(folder / 'training_identities.csv'); va_meta = pd.read_csv(folder / 'validation_identities.csv')
    for cache, meta in ((train, tr_meta), (val, va_meta)):
        if set(cache) != set(CACHE_KEYS) or len(meta) != len(cache['queries']): raise RuntimeError('Cache schema differs')
        truth = torch.where(cache['classes'] > .5, cache['levels'], 0).numpy()
        if not np.array_equal(truth, meta[['true_' + d for d in DRUGS]].to_numpy()): raise RuntimeError('Cache/labels misaligned')
    return c, train, val, tr_meta, va_meta, pd.read_csv(folder / 'baseline_validation.csv'), pd.read_csv(folder / 'baseline_training.csv')


def configuration(args, arm, smoke, checked):
    return dict(arm=arm, smoke=smoke, epochs=2 if smoke else args.epochs, max_train_batches=4 if smoke else None,
        batch_size=args.batch_size, seed=args.seed, warmup_epochs=1 if smoke else args.warmup_epochs,
        feature_lr=args.feature_lr, head_lr=args.head_lr, minimum_ratio=args.minimum_ratio,
        clip_norm=args.clip_norm, l2sp_weight=args.l2sp_weight, parent_sha256=checked['parent_sha256'],
        cache_sha256=checked['cache_sha256'], input_hashes=checked['input_hashes'], package_sha256=checked['package_sha256'],
        precision='fp32', optimizer='AdamW; weight_decay=0', generated_per_condition=48, device=args.device,
        loss='TEB conditional CORN + L2-SP' if arm == 'teb_corn' else 'DEL max-coordinate CE + L2-SP',
        label_smoothing=0., trainable_stage='cross-query fusion, projection, target ordinal head')


def train(args, checked, train_cache, val_cache, tr_meta, va_meta, baseline_val, baseline_train, smoke=False):
    for arm in ARMS if args.arm == 'all' else (args.arm,):
        folder = args.suite / ('smoke' if smoke else 'train') / arm
        last = folder / 'checkpoints/last.pt'; summary = folder / 'run_summary.json'
        if summary.exists():
            if args.resume and json.loads(summary.read_text())['status'] == 'COMPLETE': print('SKIP completed:', arm); continue
            raise FileExistsError('Completed run exists; use --resume')
        if folder.exists() and (not args.resume or not last.exists()): raise FileExistsError('Existing run has no resume checkpoint')
        folder.mkdir(parents=True, exist_ok=True)
        rt.seed_everything(args.seed); model, _ = from_baseline(args.checkpoint, arm, args.device)
        frozen = previous.frozen_digest(model)
        if previous.frozen_digest(model.baseline) != checked['protected_baseline_digest']: raise RuntimeError('Baseline differs from cache')
        opt = optimizer(model, args); steps = math.ceil(len(train_cache['grade']) / args.batch_size)
        if smoke: steps = min(steps, 4)
        config = configuration(args, arm, smoke, checked); sched = UpdateSchedule(opt, config['epochs'] * steps, config['warmup_epochs'] * steps)
        selection = dict(candidate=None, accepted=None); start = 1
        if args.resume and last.exists():
            state = torch.load(last, map_location='cpu', weights_only=False)
            if state['experiment_config'] != config: raise RuntimeError('Resume settings differ')
            model.load_state_dict(state['model_state_dict'], strict=True); opt.load_state_dict(state['optimizer_state_dict'])
            sched.load_state_dict(state['schedule_state']); rt.restore_rng(state['rng_state'])
            selection = state['selection']; start = state['epoch'] + 1
        if previous.frozen_digest(model) != frozen: raise RuntimeError('Frozen parent/normalization differs')
        rt.save_json(folder / 'training_config.json', config)
        dataset = TensorDataset(*(train_cache[k] for k in CACHE_KEYS))
        completed_epoch = start - 1
        for epoch in range(start, config['epochs'] + 1):
            model.train(); seen = ternary = 0; sums = dict(loss=0., target_loss=0., l2sp=0.); recorded = False
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                generator=torch.Generator().manual_seed(args.seed + 1009 * epoch))
            for i, values in enumerate(loader):
                if i >= steps: break
                c = {k: v.to(args.device) for k, v in zip(CACHE_KEYS, values)}
                opt.zero_grad(set_to_none=True); out = model.predict_cached(predict_inputs(c))
                loss, parts = training_loss(model, out, c['classes'], c['levels'], args.l2sp_weight)
                if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite loss')
                loss.backward()
                if epoch == start and parts['ternary_samples'] and not recorded:
                    rt.save_json(folder / 'initial_gradients.json', {n: dict(has_gradient=p.grad is not None,
                        nonzero=p.grad is not None and bool((p.grad != 0).any())) for n, p in model.named_parameters() if p.requires_grad})
                    recorded = True
                norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.clip_norm, error_if_nonfinite=True)
                sched.prepare(); opt.step(); sched.finish()
                count = parts['ternary_samples']; seen += len(c['grade']); ternary += count
                sums['loss'] += float(loss.detach()) * count
                for key in ('target_loss', 'l2sp'): sums[key] += parts[key] * count
            if not smoke and (seen != COUNTS[0] or ternary != COUNTS[2]): raise RuntimeError('Training exposure changed')
            if previous.frozen_digest(model) != frozen: raise RuntimeError('Frozen weights or buffers changed')
            m, frame = evaluate(model, val_cache, va_meta, args, baseline_val)
            decision = old.original.adoption(m, checked['baseline'])
            if smoke: decision = dict(accepted=False, rules={}, failed_rules=['smoke_not_eligible'])
            current = dict(epoch=epoch, metrics=m)
            if selection['candidate'] is None or old.original.rank(m) > old.original.rank(selection['candidate']['metrics']):
                selection['candidate'] = current; save_checkpoint(folder / 'checkpoints/best_candidate.pt', model, epoch, m, config)
                frame.to_csv(folder / 'candidate_validation.csv', index=False)
            if decision['accepted'] and (selection['accepted'] is None or old.original.rank(m) > old.original.rank(selection['accepted']['metrics'])):
                selection['accepted'] = current; save_checkpoint(folder / 'checkpoints/best_accepted.pt', model, epoch, m, config)
                frame.to_csv(folder / 'accepted_validation.csv', index=False)
            rt.save_json(folder / f'epoch_{epoch:03d}.json', dict(epoch=epoch, all_training_samples=seen,
                ternary_training_samples=ternary, completed_updates=sched.completed,
                **{k: v / max(1, ternary) for k, v in sums.items()}, gradient_norm=float(norm),
                learning_rates=[g['lr'] for g in opt.param_groups], validation=m, adoption=decision))
            save_checkpoint(last, model, epoch, m, config, optimizer=opt, schedule=sched, selection=selection)
            completed_epoch = epoch
            print(f'{arm} epoch={epoch}/{config["epochs"]} full={m["full_profile_correct"]} grades={m["present_correct"]} ternary={m["ternary_all_grade_correct"]}', flush=True)
        if not smoke:
            m, frame = evaluate(model, train_cache, tr_meta, args, baseline_train)
            frame.to_csv(folder / 'last_training_predictions.csv', index=False)
            rt.save_json(folder / 'last_training_sources.json', prior_run.source_report(frame, baseline_train))
        if previous.frozen_digest(model) != frozen: raise RuntimeError('Final inference changed protected weights')
        rt.save_json(summary, dict(status='COMPLETE', arm=arm, smoke=smoke, selection=selection,
            epochs_completed=completed_epoch, updates=sched.completed,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            total_parameters=sum(p.numel() for p in model.parameters()), frozen_state_unchanged=True,
            parent_sha256=checked['parent_sha256'], baseline_training_updates=0, test_predictions=0))


def compare(args):
    checked = json.loads((args.suite / 'check/check_summary.json').read_text())
    rows = [dict(model='T3.30_TEB_only_epoch17', origin='verified', **compact(checked['baseline']))]
    eligible = []; missing = []
    for arm in ARMS:
        p = args.suite / 'train' / arm / 'run_summary.json'
        if not p.exists(): missing.append(arm); continue
        s = json.loads(p.read_text())
        if s['smoke'] or s['status'] != 'COMPLETE': raise RuntimeError('Formal run incomplete')
        c, a = s['selection']['candidate'], s['selection']['accepted']
        if c: rows.append(dict(model=arm + '_candidate', origin='epoch_' + str(c['epoch']), **compact(c['metrics'])))
        rows.append(dict(model=arm, origin='epoch_' + str(a['epoch']) if a else 'baseline_fallback',
                         **compact(a['metrics'] if a else checked['baseline'])))
        if a: eligible.append((old.original.rank(a['metrics']), arm, str(args.suite / 'train' / arm / 'checkpoints/best_accepted.pt')))
    frame = pd.DataFrame(rows); frame.to_csv(args.suite / 'comparison.csv', index=False)
    best = max(eligible) if eligible else (None, 'T3.30_TEB_only_epoch17', str(args.checkpoint))
    rt.save_json(args.suite / 'comparison_summary.json', dict(status='COMPLETE' if not missing else 'PARTIAL', recommendation=best[1],
        recommended_checkpoint=best[2], missing_arms=missing, test_predictions=0))
    print(frame.to_string(index=False)); print('RECOMMENDED:', best[1], 'missing:', missing)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--project', type=Path, default=Path('/home/wqzheng/project_transformer'))
    p.add_argument('--checkpoint', type=Path, default=Path(DEFAULT_BASE))
    p.add_argument('--phase', choices=('check', 'smoke', 'train', 'compare', 'pack', 'evaluate'), default='check')
    p.add_argument('--arm', choices=('all',) + ARMS, default='all')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--batch-size', type=int, default=16); p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--threads', type=int, default=2); p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--warmup-epochs', type=int, default=3); p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--feature-lr', type=float, default=5e-6); p.add_argument('--head-lr', type=float, default=1e-5)
    p.add_argument('--minimum-ratio', type=float, default=.01); p.add_argument('--clip-norm', type=float, default=1.)
    p.add_argument('--l2sp-weight', type=float, default=.01); p.add_argument('--resume', action='store_true')
    p.add_argument('--optimized-checkpoint', type=Path)
    a = p.parse_args()
    if min(a.batch_size, a.threads) < 1 or a.num_workers < 0 or a.epochs < 2 or not 0 <= a.warmup_epochs < a.epochs:
        p.error('Invalid integer settings')
    if not all(math.isfinite(x) and x > 0 for x in (a.feature_lr, a.head_lr, a.clip_norm)) or not 0 < a.minimum_ratio <= 1:
        p.error('Invalid learning rate/ratio/clip')
    if not math.isfinite(a.l2sp_weight) or a.l2sp_weight < 0: p.error('Invalid regularization')
    for key in ('suite', 'project', 'checkpoint', 'optimized_checkpoint'):
        if getattr(a, key) is not None: setattr(a, key, getattr(a, key).expanduser().resolve())
    return a


def main():
    args = parse_args(); package = package_hashes()
    if any(args.suite.is_relative_to(p) for p in [ROOT, args.project / 'data'] + [(args.project / 'data' / s).resolve() for s in ('real', 'generated')]):
        raise ValueError('Suite must be outside source package and input data')
    marker = args.suite / 'T331_SUITE.json'
    identity = dict(version=VERSION, project=str(args.project), checkpoint=str(args.checkpoint))
    if marker.exists():
        if json.loads(marker.read_text()) != identity: raise RuntimeError('Suite identity differs')
    else:
        if args.phase != 'check' or (args.suite.exists() and any(args.suite.iterdir())): raise ValueError('Start with check in a new empty suite')
        args.suite.mkdir(parents=True); rt.save_json(marker, identity)
    if args.phase in ('compare', 'pack'):
        if args.phase == 'compare': compare(args)
        else:
            path = args.suite / 't3_31_target_aligned_reports.zip'
            with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
                for f in sorted(args.suite.rglob('*')):
                    if f.is_file() and f.suffix in ('.csv', '.json', '.log'): z.write(f, f.relative_to(args.suite))
            print('REPORTS:', path)
        return
    if args.device == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(args.threads); torch.backends.mha.set_fastpath_enabled(False); rt.seed_everything(args.seed)
    os.chdir(args.project)
    baseline, state = parent.load_checkpoint(args.checkpoint, 'cpu')
    from aligned_model import validate_parent
    baseline.requires_grad_(False); validate_parent(state, baseline); del baseline
    before = old.original.snapshot(args.project, args.checkpoint)
    # Also retain the parent's original inputs and original T3.28 checkpoint identity.
    for name, digest in state['experiment_config']['input_hashes'].items():
        if rt.sha256(name) != digest: raise RuntimeError('Input differs from accepted parent training: ' + name)
        before[name] = digest
    rt.save_json(args.suite / ('input_hashes_before_' + args.phase + '.json'), before)
    complete = False
    try:
        if args.phase == 'check': check(args, package, before)
        else:
            checked, tr, va, tm, vm, bv, bt = load_cache(args, package, before)
            if args.phase in ('train', 'smoke'): train(args, checked, tr, va, tm, vm, bv, bt, args.phase == 'smoke')
            else:
                if args.optimized_checkpoint is None: raise ValueError('--optimized-checkpoint required')
                model, saved = load_checkpoint(args.optimized_checkpoint, args.device)
                if saved['parent_sha256'] != checked['parent_sha256'] or previous.frozen_digest(model.baseline) != checked['protected_baseline_digest']:
                    raise RuntimeError('Optimized model has a different protected parent')
                args.l2sp_weight = saved['experiment_config']['l2sp_weight']
                m, f = evaluate(model, va, vm, args, bv)
                for key in BASE_METRICS:
                    if m[key] != saved['validation_metrics'][key]: raise RuntimeError('Saved checkpoint reproduction failed: ' + key)
                folder = args.suite / 'evaluation'; folder.mkdir(exist_ok=False)
                f.to_csv(folder / 'validation.csv', index=False); rt.save_json(folder / 'metrics.json', m)
                print(json.dumps(compact(m), indent=2))
        complete = True
    finally:
        unchanged = all(rt.sha256(name) == digest for name, digest in before.items())
        rt.save_json(args.suite / ('readonly_audit_' + args.phase + '.json'), dict(status='PASS' if unchanged else 'FAIL',
            phase_completed=complete, original_files_unchanged=unchanged, hashed_files=len(before),
            baseline_training_updates=0, test_sample_extractions=0, test_predictions=0))
        if not unchanged: raise RuntimeError('Original data/code/parent changed')


if __name__ == '__main__': main()
