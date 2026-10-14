"""T3.30 concentration-feature and last-Encoder continuation; train/validation only."""
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
from torch.utils.data import DataLoader
from feature_model import (ROOT, VERSION, BASE_SHA, DEFAULT_BASE, ARMS, MODES, DRUGS,
    BASE_METRICS, FeatureModel, from_baseline, supervised_loss, optimizer, save_checkpoint,
    load_checkpoint, previous, rt, old, UpdateSchedule)

TRAIN_COUNT = 7560
TERNARY_COUNT = 3240
SOURCE_COUNTS = {'real': 1512, 'generated': 6048}


def package_hashes():
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text())
    for name, digest in manifest.items():
        if rt.sha256(ROOT / name) != digest: raise RuntimeError('Package changed: ' + name)
    return manifest


def compact(m):
    return dict(classification_exact_correct=m['classification_exact_correct'],
                full_profile_correct=m['full_profile_correct'], present_correct=m['present_correct'],
                ternary_all_grade_correct=m['ternary_all_grade_correct'],
                ternary_present_correct=m['by_mixture']['3']['present_correct'],
                **{'ternary_' + d + '_correct': m['by_mixture']['3']['per_drug'][d]['correct'] for d in DRUGS})


def metrics(frame): return previous.enrich(rt.summarize(frame), frame)


def reference_match(actual, reference):
    identities = ['condition', 'source', 'spectrum_index']
    a = actual.sort_values(identities).reset_index(drop=True)
    b = reference.sort_values(identities).reset_index(drop=True)
    exact = identities + ['matrix', 'mixture_count'] + [p + d for p in ('true_', 'ordinal_', 'pred_') for d in DRUGS]
    if not a[exact].equals(b[exact]): raise RuntimeError('Verified baseline identities/grades differ')
    probability = [p + d for p in ('presence_probability_', 'P_ge_M_', 'P_ge_H_') for d in DRUGS]
    if not np.allclose(a[probability], b[probability], rtol=1e-5, atol=2e-6):
        raise RuntimeError('Verified baseline probabilities differ')


def frame_rows(metadata, cp, grade, op, route, decoder):
    cp, grade, op, route = [x.detach().cpu().numpy() for x in (cp, grade, op, route)]
    rows = []
    for i, raw in enumerate(metadata):
        row = dict(raw, ternary_route=bool(route[i]), decoder=decoder if route[i] else 'verified_baseline')
        for j, drug in enumerate(DRUGS):
            row['ordinal_' + drug] = int(grade[i, j])
            row['pred_' + drug] = int(grade[i, j]) if cp[i, j] >= .5 else 0
            row['presence_probability_' + drug] = float(cp[i, j])
            row['P_ge_M_' + drug] = float(op[i, j, 0]); row['P_ge_H_' + drug] = float(op[i, j, 1])
        rows.append(row)
    return rows


@torch.no_grad()
def evaluate(model, dataset, args, baseline_only=False, baseline_frame=None):
    model.eval(); modes = ('baseline',) if baseline_only else MODES
    rows = {mode: [] for mode in modes}; all_scores = []; all_logits = []; all_classes = []; all_levels = []
    for batch in rt.make_loader(dataset, args.batch_size, args.num_workers):
        data = rt.model_inputs(dataset, batch, args.device)
        meta = rt.metadata_rows(dataset, batch)
        if baseline_only:
            original = model.teacher(data); cp, grade, op = model.teacher.predict(original)
            rows['baseline'].extend(frame_rows(meta, cp, grade, op, original['route'], 'verified_DEL_only'))
            continue
        output = model(data)
        for mode in modes:
            cp, grade, op = model.predict(output, mode)
            route = output['route']
            if not torch.equal(grade[:, 1], output['base_grade'][:, 1]) or not torch.equal(op[:, 1], output['base_op'][:, 1]):
                raise RuntimeError('Protected CHL output changed')
            if not torch.equal(grade[~route], output['base_grade'][~route]) or not torch.equal(op[~route], output['base_op'][~route]):
                raise RuntimeError('Protected baseline route changed')
            inactive = 2 if mode == 'DEL_only' else 0 if mode == 'TEB_only' else None
            if inactive is not None and (not torch.equal(grade[:, inactive], output['base_grade'][:, inactive]) or
                                        not torch.equal(op[:, inactive], output['base_op'][:, inactive])):
                raise RuntimeError('Inactive pesticide output changed')
            rows[mode].extend(frame_rows(meta, cp, grade, op, route, 'feature_' + mode))
        all_scores.append(output['scores'].detach().cpu()); all_logits.append(output['student_logits'].detach().cpu())
        all_classes.append(batch['class_target'].cpu()); all_levels.append(batch['concentration_target'].cpu())
    loss_parts = None
    if not baseline_only:
        # Score all valid supervised tasks together; no averaging of unequal batch denominators.
        scores = torch.cat(all_scores); logits = torch.cat(all_logits)
        classes = torch.cat(all_classes); levels = torch.cat(all_levels)
        # Loss is evaluated on CPU tensors; L2-SP scalar is converted to the same device.
        selected = (classes > .5).all(1)
        if bool(selected.any()):
            from ternary_model import encode_levels
            joint = torch.nn.functional.cross_entropy(scores[selected], encode_levels(levels[selected]), label_smoothing=args.label_smoothing)
            yy = levels[selected, 2]; z = logits[selected, 2]; mh = yy >= 2
            first = torch.nn.functional.binary_cross_entropy_with_logits(z[:, 0], mh.float(), reduction='sum')
            second = torch.nn.functional.binary_cross_entropy_with_logits(z[mh, 1], (yy[mh] >= 3).float(), reduction='sum')
            corn = (first + second) / (len(yy) + int(mh.sum()))
            l2 = float(model.l2sp())
            loss_parts = dict(joint_ce=float(joint), teb_corn=float(corn), l2sp=l2,
                              total=float(joint) + args.teb_weight * float(corn) + args.l2sp_weight * l2)
    results = {}
    for mode in modes:
        frame = pd.DataFrame(rows[mode]); m = metrics(frame)
        if loss_parts is not None: m['supervised_validation_loss'] = loss_parts
        if baseline_frame is not None:
            identities = ['condition', 'source', 'spectrum_index']
            a = baseline_frame.sort_values(identities).reset_index(drop=True)
            b = frame.sort_values(identities).reset_index(drop=True)
            if not a[identities].equals(b[identities]): raise RuntimeError('Evaluation identities differ')
            truth = b[['true_' + d for d in DRUGS]].to_numpy()
            before = a[['pred_' + d for d in DRUGS]].to_numpy(); after = b[['pred_' + d for d in DRUGS]].to_numpy()
            old_correct = (truth == before).all(1); new_correct = (truth == after).all(1)
            m.update(changed_spectra=int((before != after).any(1).sum()),
                     full_profiles_fixed=int((~old_correct & new_correct).sum()),
                     full_profiles_broken=int((old_correct & ~new_correct).sum()))
            cols = ['presence_probability_' + d for d in DRUGS] + ['P_ge_M_CHL', 'P_ge_H_CHL']
            # CSV decimal parsing can differ below float32 precision; tensor protection above is exact.
            if not np.allclose(a[cols], b[cols], rtol=0, atol=2e-7): raise RuntimeError('Baseline classification/CHL protection differs')
            if not np.array_equal(a[['ordinal_CHL', 'pred_CHL']], b[['ordinal_CHL', 'pred_CHL']]):
                raise RuntimeError('Protected CHL labels differ')
            if not np.array_equal(a[['presence_probability_' + d for d in DRUGS]].to_numpy() >= .5,
                                  b[['presence_probability_' + d for d in DRUGS]].to_numpy() >= .5):
                raise RuntimeError('Protected class labels differ')
        results[mode] = (m, frame)
    return results


def source_report(frame, baseline_frame=None):
    result = {}
    for source in ('real', 'generated'):
        part = frame[frame.source == source]
        if not len(part): continue
        for scope, selected in [('all', part), ('ternary', part[part.mixture_count == 3])]:
            m = metrics(selected)
            if baseline_frame is not None:
                old_frame = baseline_frame[(baseline_frame.source == source) &
                    (True if scope == 'all' else baseline_frame.mixture_count == 3)]
                identity = ['condition', 'source', 'spectrum_index']
                old_frame = old_frame.sort_values(identity); new_frame = selected.sort_values(identity)
                if not old_frame[identity].reset_index(drop=True).equals(new_frame[identity].reset_index(drop=True)):
                    raise RuntimeError('Source report identity differs')
                y = new_frame[['true_' + d for d in DRUGS]].to_numpy()
                bp = old_frame[['pred_' + d for d in DRUGS]].to_numpy(); pp = new_frame[['pred_' + d for d in DRUGS]].to_numpy()
                old_good = (y == bp).all(1); new_good = (y == pp).all(1)
                m.update(full_profiles_fixed=int((~old_good & new_good).sum()),
                         full_profiles_broken=int((old_good & ~new_good).sum()))
            result[source + '_' + scope] = m
    return result


def check(args, datasets, audit, sources, package, before):
    folder = args.suite / 'check'; folder.mkdir(exist_ok=False)
    model, _ = from_baseline(args.checkpoint, 'concentration', args.device)
    frozen = previous.frozen_digest(model)
    baseline, frame = evaluate(model, datasets[1], args, baseline_only=True)['baseline']
    for key, value in BASE_METRICS.items():
        if baseline[key] != value: raise RuntimeError('Baseline reproduction failed: ' + key)
    reference_match(frame, pd.read_csv(ROOT / 'reference/combined_validation.csv'))
    frame.to_csv(folder / 'baseline_validation.csv', index=False)
    parameters = {}
    for arm in ARMS:
        candidate, _ = from_baseline(args.checkpoint, arm, args.device)
        evaluated = evaluate(candidate, datasets[1], args, baseline_frame=frame)
        for mode, (m, f) in evaluated.items():
            reference_match(f, frame)
            if compact(m) != compact(baseline): raise RuntimeError('Initial candidate changes baseline')
        parameters[arm] = dict(total=sum(p.numel() for p in candidate.parameters()),
                               trainable=sum(p.numel() for p in candidate.parameters() if p.requires_grad),
                               trainable_names=[n for n, p in candidate.named_parameters() if p.requires_grad])
        del candidate
    train_metrics, train_frame = evaluate(model, datasets[0], args, baseline_only=True)['baseline']
    if len(train_frame) != TRAIN_COUNT or int((train_frame.mixture_count == 3).sum()) != TERNARY_COUNT:
        raise RuntimeError('Training pool exposure differs')
    if train_frame.source.value_counts().to_dict() != SOURCE_COUNTS: raise RuntimeError('Source selection changed')
    if previous.frozen_digest(model) != frozen: raise RuntimeError('Check changed frozen weights/buffers')
    train_frame.to_csv(folder / 'baseline_training.csv', index=False)
    baseline_sources = source_report(train_frame)
    rt.save_json(folder / 'baseline_training_sources.json', baseline_sources)
    files = ('baseline_validation.csv', 'baseline_training.csv', 'baseline_training_sources.json')
    summary = dict(status='PASS', baseline=baseline, baseline_sha256=BASE_SHA, parameters=parameters,
                   teacher_digest=previous.frozen_digest(model.teacher), input_hashes=before,
                   source_sha256=sources, dataset_audit=audit, package_sha256=package,
                   reference_hashes={n: rt.sha256(folder / n) for n in files}, network_training_updates=0,
                   test_sample_extractions=0, test_predictions=0)
    rt.save_json(folder / 'check_summary.json', summary)
    print('CHECK PASS: class=503 grades=984 full=379 ternary=124; train=7560, real=1512, generated=6048', flush=True)
    print('PARAMETERS:', json.dumps({a: {k: v for k, v in p.items() if k != 'trainable_names'} for a, p in parameters.items()}), flush=True)


def settings(args, arm, smoke):
    return dict(arm=arm, smoke=smoke, epochs=2 if smoke else args.epochs,
                max_train_batches=4 if smoke else None, batch_size=args.batch_size,
                seed=args.seed, warmup_epochs=1 if smoke else args.warmup_epochs,
                feature_lr=args.feature_lr, head_lr=args.head_lr, encoder_lr=args.encoder_lr,
                minimum_ratio=args.minimum_ratio, clip_norm=args.clip_norm,
                label_smoothing=args.label_smoothing, teb_weight=args.teb_weight, l2sp_weight=args.l2sp_weight,
                baseline_sha256=BASE_SHA, precision='fp32', optimizer='AdamW; decay=0; L2-SP anchor',
                loss='true ternary joint CE + TEB CORN + L2-SP; no old-grade probability reference',
                generated_per_condition=48, num_workers=args.num_workers, device=args.device,
                student_dropout='eval; stored normalization frozen')


def train(args, datasets, checked, smoke):
    baseline_frame = pd.read_csv(args.suite / 'check/baseline_validation.csv')
    baseline_train = pd.read_csv(args.suite / 'check/baseline_training.csv')
    for arm in ARMS if args.arm == 'all' else (args.arm,):
        folder = args.suite / ('smoke' if smoke else 'train') / arm
        last = folder / 'checkpoints/last.pt'; summary_path = folder / 'run_summary.json'
        if summary_path.exists():
            if args.resume and json.loads(summary_path.read_text())['status'] == 'COMPLETE':
                print('SKIP completed:', arm); continue
            raise FileExistsError('Completed run exists; use --resume or a new suite')
        if folder.exists() and (not args.resume or not last.exists()):
            raise FileExistsError('Existing run lacks a resumable checkpoint')
        folder.mkdir(parents=True, exist_ok=True)
        rt.seed_everything(args.seed); model, _ = from_baseline(args.checkpoint, arm, args.device)
        opt = optimizer(model, args); steps = math.ceil(len(datasets[0]) / args.batch_size)
        if smoke: steps = min(steps, 4)
        config = settings(args, arm, smoke)
        config['input_hashes'] = checked['input_hashes']; config['package_sha256'] = checked['package_sha256']
        sched = UpdateSchedule(opt, config['epochs'] * steps, config['warmup_epochs'] * steps)
        selection = {mode: dict(candidate=None, accepted=None) for mode in MODES}; start = 1
        frozen = previous.frozen_digest(model)
        if args.resume and last.exists():
            state = torch.load(last, map_location='cpu', weights_only=False)
            if state['experiment_config'] != config: raise RuntimeError('Resume configuration differs')
            model.load_state_dict(state['model_state_dict'], strict=True); opt.load_state_dict(state['optimizer_state_dict'])
            sched.load_state_dict(state['schedule_state']); rt.restore_rng(state['rng_state'])
            selection = state['selection']; start = state['epoch'] + 1
        if previous.frozen_digest(model) != frozen or previous.frozen_digest(model.teacher) != checked['teacher_digest']:
            raise RuntimeError('Frozen state differs from baseline')
        rt.save_json(folder / 'training_config.json', config)
        totals = dict(epochs_completed=start - 1, updates=sched.completed)
        for epoch in range(start, config['epochs'] + 1):
            model.train(); seen = ternary_seen = 0; sums = dict(loss=0., joint_ce=0., teb_corn=0., l2sp=0.)
            loader = DataLoader(datasets[0], batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=args.device == 'cuda',
                generator=torch.Generator().manual_seed(args.seed + 1009 * epoch))
            for index, batch in enumerate(loader):
                if index >= steps: break
                opt.zero_grad(set_to_none=True)
                out = model(rt.model_inputs(datasets[0], batch, args.device))
                loss, parts = supervised_loss(model, out, batch['class_target'].to(args.device),
                    batch['concentration_target'].to(args.device), args.label_smoothing, args.teb_weight, args.l2sp_weight)
                if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                if epoch == start and index == 0:
                    rt.save_json(folder / 'initial_gradients.json', {n: dict(has_gradient=p.grad is not None,
                        nonzero=p.grad is not None and bool((p.grad != 0).any())) for n, p in model.named_parameters() if p.requires_grad})
                norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.clip_norm, error_if_nonfinite=True)
                sched.prepare(); opt.step(); sched.finish()
                count = parts['ternary_samples']; seen += len(batch['class_target']); ternary_seen += count
                sums['loss'] += float(loss.detach()) * count
                for key in ('joint_ce', 'teb_corn', 'l2sp'): sums[key] += parts[key] * count
            if not smoke and (seen != TRAIN_COUNT or ternary_seen != TERNARY_COUNT):
                raise RuntimeError('Training exposure changed')
            if previous.frozen_digest(model) != frozen: raise RuntimeError('Frozen parameters or normalization changed')
            evaluated = evaluate(model, datasets[1], args, baseline_frame=baseline_frame); records = {}
            for mode, (m, frame) in evaluated.items():
                decision = old.original.adoption(m, checked['baseline'])
                if smoke: decision = dict(accepted=False, rules={}, failed_rules=['smoke_not_eligible'])
                branch = selection[mode]; current = dict(epoch=epoch, mode=mode, metrics=m)
                if branch['candidate'] is None or old.original.rank(m) > old.original.rank(branch['candidate']['metrics']):
                    branch['candidate'] = current
                    save_checkpoint(folder / ('checkpoints/' + mode + '/best_candidate.pt'), model, epoch, m, config, mode)
                    frame.to_csv(folder / (mode + '_candidate_validation.csv'), index=False)
                if decision['accepted'] and (branch['accepted'] is None or old.original.rank(m) > old.original.rank(branch['accepted']['metrics'])):
                    branch['accepted'] = current
                    save_checkpoint(folder / ('checkpoints/' + mode + '/best_accepted.pt'), model, epoch, m, config, mode)
                    frame.to_csv(folder / (mode + '_accepted_validation.csv'), index=False)
                records[mode] = dict(validation=m, adoption=decision)
            rt.save_json(folder / f'epoch_{epoch:03d}.json', dict(epoch=epoch, all_training_samples=seen,
                ternary_training_samples=ternary_seen, completed_updates=sched.completed,
                **{k: v / max(1, ternary_seen) for k, v in sums.items()}, gradient_norm=float(norm),
                learning_rates=[g['lr'] for g in opt.param_groups], modes=records))
            save_checkpoint(last, model, epoch, evaluated['DEL_TEB'][0], config,
                            optimizer=opt, schedule=sched, selection=selection)
            totals = dict(epochs_completed=epoch, updates=sched.completed)
            print(f'{arm} epoch={epoch}/{config["epochs"]} ternary=' + str({k: m['validation']['ternary_all_grade_correct'] for k, m in records.items()}), flush=True)
        # Source audit of the LAST weights, not used to choose a candidate or accept a model.
        sources = {}
        if not smoke:
            evaluated_train = evaluate(model, datasets[0], args, baseline_frame=baseline_train)
            for mode, (m, frame) in evaluated_train.items():
                sources[mode] = source_report(frame, baseline_train)
                frame.to_csv(folder / (mode + '_last_training.csv'), index=False)
            rt.save_json(folder / 'last_training_sources.json', sources)
        if previous.frozen_digest(model) != frozen: raise RuntimeError('Source inference changed frozen state')
        rt.save_json(summary_path, dict(status='COMPLETE', arm=arm, smoke=smoke, selection=selection,
            **totals, training_samples_per_epoch=TRAIN_COUNT if not smoke else min(TRAIN_COUNT, steps * args.batch_size),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            total_parameters=sum(p.numel() for p in model.parameters()), baseline_training_updates=0,
            frozen_state_unchanged=True, test_predictions=0))


def compare(args):
    checked = json.loads((args.suite / 'check/check_summary.json').read_text())
    rows = [dict(model='T3.28_DEL_only_base', origin='verified', **compact(checked['baseline']))]
    eligible = []; missing = []
    for arm in ARMS:
        path = args.suite / 'train' / arm / 'run_summary.json'
        if not path.exists(): missing.append(arm); continue
        summary = json.loads(path.read_text())
        if summary.get('smoke') or summary['status'] != 'COMPLETE': raise RuntimeError('Formal run incomplete')
        for mode in MODES:
            chosen = summary['selection'][mode]
            c = chosen['candidate']; a = chosen['accepted']; name = arm + '/' + mode
            if c: rows.append(dict(model=name + '_candidate', origin='epoch_' + str(c['epoch']), **compact(c['metrics'])))
            rows.append(dict(model=name, origin='epoch_' + str(a['epoch']) if a else 'baseline_fallback',
                             **compact(a['metrics'] if a else checked['baseline'])))
            if a: eligible.append((old.original.rank(a['metrics']), name,
                                   str(args.suite / 'train' / arm / 'checkpoints' / mode / 'best_accepted.pt')))
    recommendation = max(eligible)[1] if eligible else 'T3.28_DEL_only_base'
    frame = pd.DataFrame(rows); frame.to_csv(args.suite / 'comparison.csv', index=False)
    rt.save_json(args.suite / 'comparison_summary.json', dict(status='COMPLETE' if not missing else 'PARTIAL',
        recommendation=recommendation, recommended_checkpoint=max(eligible)[2] if eligible else str(args.checkpoint),
        missing_arms=missing, test_predictions=0))
    print(frame.to_string(index=False)); print('RECOMMENDED:', recommendation, 'missing:', missing)


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
    p.add_argument('--encoder-lr', type=float, default=1e-6); p.add_argument('--minimum-ratio', type=float, default=.01)
    p.add_argument('--clip-norm', type=float, default=1.); p.add_argument('--label-smoothing', type=float, default=.05)
    p.add_argument('--teb-weight', type=float, default=.5); p.add_argument('--l2sp-weight', type=float, default=.01)
    p.add_argument('--resume', action='store_true'); p.add_argument('--optimized-checkpoint', type=Path)
    a = p.parse_args()
    if min(a.batch_size, a.threads) < 1 or a.num_workers < 0 or a.epochs < 2 or not 0 <= a.warmup_epochs < a.epochs:
        p.error('Invalid integer settings')
    if not all(math.isfinite(x) and x > 0 for x in (a.feature_lr, a.head_lr, a.encoder_lr, a.clip_norm)):
        p.error('Learning rates/clip must be finite and positive')
    if not 0 < a.minimum_ratio <= 1 or not 0 <= a.label_smoothing < 1:
        p.error('Invalid ratio/smoothing')
    if not all(math.isfinite(x) and x >= 0 for x in (a.teb_weight, a.l2sp_weight)):
        p.error('Invalid loss weight')
    for key in ('suite', 'project', 'checkpoint', 'optimized_checkpoint'):
        if getattr(a, key) is not None: setattr(a, key, getattr(a, key).expanduser().resolve())
    return a


def main():
    args = parse_args(); package = package_hashes()
    protected_dirs = [ROOT, args.project / 'data'] + [(args.project / 'data' / s).resolve() for s in ('real', 'generated')]
    if any(args.suite.is_relative_to(p) for p in protected_dirs): raise ValueError('Suite must be outside package/input data')
    marker = args.suite / 'T330_SUITE.json'
    identity = dict(version=VERSION, project=str(args.project), checkpoint=str(args.checkpoint))
    if marker.exists():
        if json.loads(marker.read_text()) != identity: raise RuntimeError('Suite identity differs')
    else:
        if args.phase != 'check' or (args.suite.exists() and any(args.suite.iterdir())):
            raise ValueError('Begin with --phase check in a new empty suite')
        args.suite.mkdir(parents=True); rt.save_json(marker, identity)
    if args.phase in ('compare', 'pack'):
        if args.phase == 'compare': compare(args)
        else:
            archive = args.suite / 't3_30_feature_reports.zip'
            with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
                for path in sorted(args.suite.rglob('*')):
                    if path.is_file() and path.suffix in ('.json', '.csv', '.log'): z.write(path, path.relative_to(args.suite))
            print('REPORTS:', archive)
        return
    if args.device == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(args.threads); torch.backends.mha.set_fastpath_enabled(False); rt.seed_everything(args.seed)
    os.chdir(args.project)
    if rt.sha256(args.checkpoint) != BASE_SHA: raise ValueError('Baseline SHA256 differs')
    before = old.original.snapshot(args.project, args.checkpoint)
    rt.save_json(args.suite / ('input_hashes_before_' + args.phase + '.json'), before)
    complete = False
    try:
        datasets, audit, sources = rt.configure_project(args.project)
        with previous.guard_test_dataset(datasets[2]):
            if args.phase == 'check': check(args, datasets, audit, sources, package, before)
            else:
                checked = json.loads((args.suite / 'check/check_summary.json').read_text())
                if checked['status'] != 'PASS' or checked['input_hashes'] != before or checked['package_sha256'] != package:
                    raise RuntimeError('Source/input/package changed since check')
                if checked['source_sha256'] != sources or checked['dataset_audit'] != audit:
                    raise RuntimeError('Dataset audit changed since check')
                for name, digest in checked['reference_hashes'].items():
                    if rt.sha256(args.suite / 'check' / name) != digest: raise RuntimeError('Checked reference changed')
                if args.phase in ('train', 'smoke'): train(args, datasets, checked, args.phase == 'smoke')
                else:
                    if args.optimized_checkpoint is None: raise ValueError('--optimized-checkpoint required')
                    model, state = load_checkpoint(args.optimized_checkpoint, args.device)
                    if previous.frozen_digest(model.teacher) != checked['teacher_digest']:
                        raise RuntimeError('Optimized checkpoint has a different baseline')
                    # Evaluation uses the checkpoint's loss settings, not new CLI defaults.
                    for name in ('label_smoothing', 'teb_weight', 'l2sp_weight'):
                        setattr(args, name, state['experiment_config'][name])
                    baseline = pd.read_csv(args.suite / 'check/baseline_validation.csv')
                    results = evaluate(model, datasets[1], args, baseline_frame=baseline)
                    mode = model.options['deployment_mode']; m, frame = results[mode]
                    folder = args.suite / 'evaluation'; folder.mkdir(exist_ok=False)
                    frame.to_csv(folder / 'validation.csv', index=False); rt.save_json(folder / 'metrics.json', m)
                    print(json.dumps(compact(m), indent=2))
        complete = True
    finally:
        unchanged = before == old.original.snapshot(args.project, args.checkpoint)
        rt.save_json(args.suite / ('readonly_audit_' + args.phase + '.json'), dict(status='PASS' if unchanged else 'FAIL',
            phase_completed=complete, original_files_unchanged=unchanged, hashed_files=len(before),
            baseline_training_updates=0, test_sample_extractions=0, test_predictions=0))
        if not unchanged: raise RuntimeError('Original data/code/baseline changed')


if __name__ == '__main__': main()
