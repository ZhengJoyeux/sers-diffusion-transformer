"""Isolated T3.27 continuation experiments; train/validation only."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'archived'))
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import runtime_t326 as rt
from Model_T3_26 import TransformerClassifyRegress_sep as Model
from peft import VARIANTS, configure, train_mode, build_optimizer
from schedule import UpdateSchedule

VERSION = 'T3.27_optimization_v1'
ANCHOR_SHA = 'fe091a356ab46836b560f4e16d866c4a4b72f8b0c3fd516278613893e093547e'
DEFAULT_ANCHOR = '/home/wqzheng/project_transformer/outputs/t3_26_cross_combination_seed2026/train/cross_relation/checkpoints/best_accepted.pt'


def verify_package():
    expected = json.loads((ROOT / 'MANIFEST.json').read_text())
    bad = [name for name, digest in expected.items() if not (ROOT / name).is_file() or rt.sha256(ROOT / name) != digest]
    if bad:
        raise RuntimeError('Package integrity failed: ' + ', '.join(bad))
    return expected


def input_snapshot(project, checkpoint):
    result = {}
    # Resolve EACH known source root: Path.rglob does not traverse directory
    # symlinks nested under data. Both original and generated files are protected.
    for name in ('real', 'generated'):
        root = (project / 'data' / name).resolve(strict=True)
        files = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in ('.csv', '.xls', '.xlsx'))
        if not files:
            raise FileNotFoundError(f'No spectrum files in {root}')
        for file in files:
            result[str(file.resolve())] = rt.sha256(file)
    expected = json.loads((ROOT / 'archived/expected_project_sources.json').read_text())
    for name in expected:
        result[str(project / name)] = rt.sha256(project / name)
    result[str(checkpoint)] = rt.sha256(checkpoint)
    if result[str(checkpoint)] != ANCHOR_SHA:
        raise ValueError('Anchor differs from the archived epoch-24 T3.26 checkpoint')
    return result


def tensor_digest(named_values):
    h = hashlib.sha256()
    for name, value in sorted(named_values):
        h.update(name.encode())
        h.update(str((tuple(value.shape), value.dtype)).encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def frozen_digest(model):
    values = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
    # Preserve all buffers (BatchNorm stats, physical axis, fitted peak priors).
    values += [('buffer:' + n, b) for n, b in model.named_buffers()]
    return tensor_digest(values)


@contextlib.contextmanager
def guard_test_dataset(dataset):
    cls = type(dataset)
    original = cls.__getitem__
    def guarded(self, index):
        if self.split == 'test':
            raise RuntimeError('Test spectra may not be read during optimization')
        return original(self, index)
    cls.__getitem__ = guarded
    try:
        yield
    finally:
        cls.__getitem__ = original


def enrich(metrics, frame):
    metrics = dict(metrics)
    metrics['by_mixture'] = {}
    for count in (1, 2, 3):
        part = frame[frame.mixture_count == count]
        if len(part):
            m = rt.summarize(part)
            metrics['by_mixture'][str(count)] = {k: m[k] for k in (
                'samples', 'present_n', 'classification_exact_correct', 'present_correct', 'full_profile_correct')}
            metrics['by_mixture'][str(count)]['per_drug'] = m['per_drug']
    return metrics


def evaluate(model, dataset, args):
    metrics, frame = rt.evaluate(model, dataset, args.device, args.batch_size, args.num_workers)
    return enrich(metrics, frame), frame


def metric_rank(m):
    # The user's comprehensive recognition objective comes first.
    return (m['full_profile_correct'], m['present_correct'], m['ternary_all_grade_correct'], -m['present_mae'])


def acceptance(metrics, anchor):
    rules = {key: metrics[key] >= anchor[key] for key in (
        'classification_exact_correct', 'full_profile_correct', 'present_correct', 'ternary_all_grade_correct')}
    rules['classification_macro_f1'] = metrics['classification_macro_f1'] + 1e-12 >= anchor['classification_macro_f1']
    for drug in ('DEL', 'CHL', 'TEB'):
        rules['grade_' + drug] = metrics['per_drug'][drug]['correct'] >= anchor['per_drug'][drug]['correct']
    for count in ('1', '2', '3'):
        rules['full_mixture_' + count] = (metrics['by_mixture'][count]['full_profile_correct'] >=
                                        anchor['by_mixture'][count]['full_profile_correct'])
    rules['strict_improvement'] = any(metrics[k] > anchor[k] for k in (
        'full_profile_correct', 'present_correct', 'ternary_all_grade_correct'))
    return dict(accepted=all(rules.values()), rules=rules)


def specification(args, variant):
    return dict(variant=variant, rank=args.rank, alpha=args.alpha, bottleneck=args.bottleneck,
        backbone_lr=args.backbone_lr, head_lr=args.head_lr, evidence_lr=args.evidence_lr,
        peft_lr=args.peft_lr, weight_decay=0. if variant == 'adam_control' else args.weight_decay,
        minimum_ratio=args.minimum_ratio, layer_decay=args.layer_decay)


def load_anchor(path, device):
    if rt.sha256(path) != ANCHOR_SHA:
        raise ValueError('Wrong anchor SHA256')
    model, payload = rt.load_model(path, device)
    if payload.get('checkpoint_version') != 'T3.26_cross_combination_v1':
        raise ValueError('Expected T3.26 anchor')
    return model, payload


def save_checkpoint(path, model, model_config, spec, epoch, metrics, config,
                    optimizer=None, schedule=None, selection=None):
    payload = dict(checkpoint_version=VERSION, model_config=model_config, optimization_spec=spec,
        model_state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        epoch=epoch, validation_metrics=metrics, experiment_config=config, selection=selection,
        test_predictions=0)
    if optimizer is not None:
        payload.update(optimizer_state_dict=optimizer.state_dict(), schedule_state=schedule.state_dict(),
                       rng_state=rt.capture_rng())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def load_optimized(path, device='cpu'):
    """Standalone: the original anchor file is NOT needed for inference."""
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('checkpoint_version') != VERSION:
        raise ValueError('Use load_anchor for an original T3.26 checkpoint')
    model = configure(Model(**payload['model_config']), payload['optimization_spec'])
    model.load_state_dict(payload['model_state_dict'], strict=True)
    return model.to(device).eval(), payload


def run_check(args, datasets, audit, sources, package):
    destination = args.suite / 'check'
    destination.mkdir(exist_ok=False)
    anchor, payload = load_anchor(args.checkpoint, args.device)
    baseline, frame = evaluate(anchor, datasets[1], args)
    expected = dict(samples=504, classification_exact_correct=503, present_correct=978,
                    full_profile_correct=372, ternary_all_grade_correct=120)
    for key, value in expected.items():
        if baseline[key] != value:
            raise RuntimeError(f'Anchor reproduction failed: {key}={baseline[key]}, expected {value}')
    frame.to_csv(destination / 'anchor_validation.csv', index=False)
    batch = next(iter(rt.make_loader(datasets[1], args.batch_size, args.num_workers)))
    data = rt.model_inputs(datasets[1], batch, args.device)
    with torch.no_grad():
        original = rt.outputs(anchor, data)
    variants = {}
    for variant in VARIANTS:
        spec = specification(args, variant)
        model, _ = load_anchor(args.checkpoint, 'cpu')
        model = configure(model, spec).to(args.device).eval()
        with torch.no_grad():
            output = rt.outputs(model, data)
        errors = [float((a-b).abs().max()) for a, b in zip(original, output)]
        if max(errors) > 1e-5:
            raise RuntimeError(f'{variant}: zero-initialized model differs from anchor: {errors}')
        optimizer = build_optimizer(model, spec)
        variants[variant] = dict(spec=spec, initial_max_abs_error=errors,
            total_parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            groups=[{k: v for k, v in g.items() if k != 'params'} for g in optimizer.param_groups])
        del optimizer, model
    rt.save_json(destination / 'check_summary.json', dict(status='PASS', baseline=baseline,
        variants=variants, dataset_audit=audit, source_sha256=sources, package_sha256=package,
        anchor_sha256=ANCHOR_SHA, test_predictions=0))
    print('CHECK PASS: anchor 978 / 372 / 120; all variants preserve initial function', flush=True)


def training_config(args, spec, smoke, audit, sources, package):
    return dict(spec=spec, smoke=smoke, epochs=2 if smoke else args.epochs,
        warmup_epochs=1 if smoke else args.warmup_epochs, batch_size=args.batch_size,
        max_batches=4 if smoke else None, seed=args.seed, clip_norm=args.clip_norm,
        precision=args.precision, num_workers=args.num_workers, device=args.device,
        loss_class=.3, loss_corn=.7, relation_weight=0., reference_weight=0.,
        batchnorm_buffers_frozen=True, generated_per_condition=48,
        dataset_audit=audit, source_sha256=sources, package_sha256=package, anchor_sha256=ANCHOR_SHA)


def run_train(args, datasets, audit, sources, package, smoke=False):
    check = json.loads((args.suite / 'check/check_summary.json').read_text())
    if check['status'] != 'PASS' or check['dataset_audit'] != audit or check['package_sha256'] != package:
        raise RuntimeError('Run check with this exact package and dataset first')
    if check['source_sha256'] != sources or check['anchor_sha256'] != ANCHOR_SHA:
        raise RuntimeError('Sources/anchor differ from checked experiment')
    anchor_metrics = check['baseline']
    train, val, _ = datasets
    if smoke:
        val = rt.smoke_validation_view(val, 32)
    chosen = VARIANTS if args.variant == 'all' else (args.variant,)
    for variant in chosen:
        directory = args.suite / ('smoke' if smoke else 'train') / variant
        last = directory / 'checkpoints/last.pt'
        summary_path = directory / 'run_summary.json'
        spec = specification(args, variant)
        if check['variants'][variant]['spec'] != spec:
            raise ValueError('Optimizer/PEFT options differ from check; check them in a new suite')
        config = training_config(args, spec, smoke, audit, sources, package)
        steps = min(math.ceil(len(train) / args.batch_size), 4) if smoke else math.ceil(len(train) / args.batch_size)
        total, warmup = config['epochs'] * steps, config['warmup_epochs'] * steps
        if directory.exists() and not args.resume:
            raise FileExistsError(f'{directory}: use --resume or choose a new suite')
        if args.resume and not last.exists():
            if directory.exists():
                raise FileNotFoundError(f'No epoch checkpoint to resume: {last}')
        directory.mkdir(parents=True, exist_ok=True)
        rt.seed_everything(args.seed)
        model, anchor_payload = load_anchor(args.checkpoint, 'cpu')
        model = configure(model, spec).to(args.device)
        rt.seed_everything(args.seed)
        optimizer = build_optimizer(model, spec)
        schedule = UpdateSchedule(optimizer, total, warmup)
        frozen_before = frozen_digest(model)
        best_candidate, best_accepted, start = None, None, 1
        if args.resume and last.exists():
            payload = torch.load(last, map_location='cpu', weights_only=False)
            if payload['experiment_config'] != config:
                raise ValueError('Resume configuration differs; use the original CLI options')
            model.load_state_dict(payload['model_state_dict'], strict=True)
            optimizer.load_state_dict(payload['optimizer_state_dict'])
            schedule.load_state_dict(payload['schedule_state'])
            rt.restore_rng(payload['rng_state'])
            start = payload['epoch'] + 1
            best_candidate = payload['selection']['best_candidate']
            best_accepted = payload['selection']['best_accepted']
            if frozen_digest(model) != frozen_before:
                raise RuntimeError('Frozen parameters or buffers changed in resumed checkpoint')
        rt.save_json(directory / 'training_config.json', config)
        rt.save_json(directory / 'optimizer_groups.json', [
            {k: v for k, v in g.items() if k != 'params'} for g in optimizer.param_groups])
        for epoch in range(start, config['epochs'] + 1):
            train_mode(model)
            generator = torch.Generator().manual_seed(args.seed + epoch * 1009)
            loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, generator=generator,
                pin_memory=args.device == 'cuda', persistent_workers=False)
            losses, seen = 0., 0
            gradient_summary = None
            for index, batch in enumerate(loader):
                if index >= steps:
                    break
                optimizer.zero_grad(set_to_none=True)
                context = (torch.autocast('cuda', dtype=torch.bfloat16) if args.precision == 'bf16'
                           else contextlib.nullcontext())
                with context:
                    cp, _, logits = rt.outputs(model, rt.model_inputs(train, batch, args.device))
                # BCE probabilities are calculated outside autocast in FP32.
                loss, _, _ = rt.joint_loss(cp.float(), logits.float(),
                    batch['class_target'].to(args.device), batch['concentration_target'].to(args.device))
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                trainable = [p for p in model.parameters() if p.requires_grad]
                norm = torch.nn.utils.clip_grad_norm_(trainable, args.clip_norm, error_if_nonfinite=True)
                if gradient_summary is None:
                    gradient_summary = {name: dict(has_gradient=p.grad is not None,
                        nonzero=p.grad is not None and bool((p.grad != 0).any()))
                        for name, p in model.named_parameters() if p.requires_grad}
                    if not any(v['nonzero'] for v in gradient_summary.values()):
                        raise RuntimeError('No trainable gradients')
                schedule.prepare()
                optimizer.step()
                schedule.finish()
                count = len(batch['class_target'])
                losses += loss.item() * count
                seen += count
            metrics, frame = evaluate(model, val, args)
            if frozen_digest(model) != frozen_before:
                raise RuntimeError('Frozen parameters or buffers changed')
            candidate = dict(epoch=epoch, metrics=metrics)
            if best_candidate is None or metric_rank(metrics) > metric_rank(best_candidate['metrics']):
                best_candidate = candidate
                save_checkpoint(directory / 'checkpoints/best_candidate.pt', model,
                    anchor_payload['model_config'], spec, epoch, metrics, config)
                frame.to_csv(directory / 'candidate_validation.csv', index=False)
            decision = dict(accepted=False, rules={'smoke_cannot_be_adopted': False}) if smoke else acceptance(metrics, anchor_metrics)
            if decision['accepted'] and (best_accepted is None or metric_rank(metrics) > metric_rank(best_accepted['metrics'])):
                best_accepted = candidate
                save_checkpoint(directory / 'checkpoints/best_accepted.pt', model,
                    anchor_payload['model_config'], spec, epoch, metrics, config)
                frame.to_csv(directory / 'accepted_validation.csv', index=False)
            record = dict(epoch=epoch, training_loss=losses / seen, training_samples=seen,
                completed_updates=schedule.completed, last_gradient_norm=float(norm),
                learning_rates={str(i): g['lr'] for i, g in enumerate(optimizer.param_groups)},
                validation=metrics, adoption=decision)
            rt.save_json(directory / f'epoch_{epoch:03d}.json', record)
            if epoch == start:
                rt.save_json(directory / f'gradient_audit_epoch_{epoch:03d}.json', gradient_summary)
            selection = dict(best_candidate=best_candidate, best_accepted=best_accepted)
            save_checkpoint(last, model, anchor_payload['model_config'], spec, epoch, metrics,
                config, optimizer, schedule, selection)
            print(f'{variant} epoch={epoch}/{config["epochs"]} full={metrics["full_profile_correct"]} '
                  f'grade={metrics["present_correct"]} ternary={metrics["ternary_all_grade_correct"]} '
                  f'accepted={decision["accepted"]}', flush=True)
        if schedule.completed != total:
            raise RuntimeError('Unexpected optimizer update count')
        rt.save_json(summary_path, dict(status='COMPLETE', variant=variant, smoke=smoke,
            anchor=anchor_metrics, best_candidate=best_candidate, best_accepted=best_accepted,
            recommendation='best_accepted.pt' if best_accepted else 'original_T3.26_anchor',
            completed_updates=schedule.completed, frozen_state_unchanged=True,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            total_parameters=sum(p.numel() for p in model.parameters()), test_predictions=0))
        del model, optimizer
        if args.device == 'cuda':
            torch.cuda.empty_cache()


def compare(args):
    check = json.loads((args.suite / 'check/check_summary.json').read_text())
    rows = []
    def row(name, origin, metrics):
        result = dict(model=name, origin=origin, **{k: metrics[k] for k in (
            'classification_exact_correct', 'present_correct', 'full_profile_correct', 'ternary_all_grade_correct')})
        result.update({drug + '_correct': metrics['per_drug'][drug]['correct'] for drug in ('DEL', 'CHL', 'TEB')})
        result.update({f'full_mixture_{count}': metrics['by_mixture'][str(count)]['full_profile_correct'] for count in (1, 2, 3)})
        return result
    rows.append(row('T3.26_anchor', 'epoch_24', check['baseline']))
    accepted = []
    missing = []
    for variant in VARIANTS:
        path = args.suite / 'train' / variant / 'run_summary.json'
        if not path.exists():
            missing.append(variant)
            continue
        summary = json.loads(path.read_text())
        if summary['status'] != 'COMPLETE' or summary['smoke']:
            raise RuntimeError('Only completed formal runs can be compared')
        candidate = summary['best_candidate']
        rows.append(row(variant + '_candidate', f'epoch_{candidate["epoch"]}', candidate['metrics']))
        winner = summary['best_accepted']
        rows.append(row(variant, 'accepted' if winner else 'anchor_fallback',
                        winner['metrics'] if winner else check['baseline']))
        if winner:
            accepted.append((metric_rank(winner['metrics']), variant, winner))
    frame = pd.DataFrame(rows)
    frame.to_csv(args.suite / 'comparison.csv', index=False)
    recommended = max(accepted)[1] if accepted else 'T3.26_anchor'
    rt.save_json(args.suite / 'comparison_summary.json', dict(status='COMPLETE' if not missing else 'PARTIAL',
        missing_variants=missing, recommended=recommended,
        recommendation_path=str(args.suite / 'train' / recommended / 'checkpoints/best_accepted.pt') if accepted else str(args.checkpoint),
        selection_scope='exploratory validation; historical validation reused; not an independent test', test_predictions=0))
    print(frame.to_string(index=False), flush=True)
    print('RECOMMENDED:', recommended, 'missing:', missing, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path('/home/wqzheng/project_transformer'))
    parser.add_argument('--checkpoint', type=Path, default=Path(DEFAULT_ANCHOR))
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--phase', choices=('check', 'smoke', 'train', 'compare', 'evaluate'), required=True)
    parser.add_argument('--variant', choices=VARIANTS + ('all',), default='all')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--optimized-checkpoint', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--precision', choices=('fp32', 'bf16'), default='fp32')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--warmup-epochs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--backbone-lr', type=float, default=1e-6)
    parser.add_argument('--head-lr', type=float, default=5e-6)
    parser.add_argument('--evidence-lr', type=float, default=1e-5)
    parser.add_argument('--peft-lr', type=float, default=1e-4)
    parser.add_argument('--minimum-ratio', type=float, default=.01)
    parser.add_argument('--weight-decay', type=float, default=.01)
    parser.add_argument('--layer-decay', type=float, default=.85)
    parser.add_argument('--clip-norm', type=float, default=1.)
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--alpha', type=float, default=8.)
    parser.add_argument('--bottleneck', type=int, default=16)
    args = parser.parse_args()
    if args.epochs < 2 or not 0 <= args.warmup_epochs < args.epochs:
        parser.error('Require epochs >= 2 and 0 <= warmup_epochs < epochs')
    if min(args.batch_size, args.threads, args.rank, args.bottleneck) < 1 or args.num_workers < 0:
        parser.error('Invalid integer settings')
    rates = (args.backbone_lr, args.head_lr, args.evidence_lr, args.peft_lr, args.alpha, args.clip_norm)
    if not all(math.isfinite(v) and v > 0 for v in rates) or not 0 < args.minimum_ratio <= 1 or not 0 < args.layer_decay <= 1 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error('Invalid learning rate/decay/clip setting')
    return args


def main():
    args = parse_args()
    package = verify_package()
    args.project, args.checkpoint, args.suite = args.project.resolve(), args.checkpoint.resolve(), args.suite.resolve()
    protected_roots = [args.project / 'data', ROOT] + [(args.project / 'data' / name).resolve() for name in ('real', 'generated')]
    if any(args.suite.is_relative_to(root) for root in protected_roots):
        raise ValueError('Output suite must be outside input data and this package')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu for check')
    if args.precision == 'bf16' and (args.device != 'cuda' or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('BF16 requires a supported CUDA GPU')
    torch.set_num_threads(args.threads)
    rt.seed_everything(args.seed)
    # Common implementation path for dense Adapter hooks and LoRA weight views.
    torch.backends.mha.set_fastpath_enabled(False)
    args.suite.mkdir(parents=True, exist_ok=True)
    marker = args.suite / 'T327_SUITE.json'
    identity = dict(version=VERSION, project=str(args.project), anchor_sha256=ANCHOR_SHA)
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError('Suite belongs to a different project/version')
    else:
        if args.phase != 'check' or any(p.suffix != '.log' for p in args.suite.iterdir()):
            raise ValueError('First check must use a new empty suite (existing log files are allowed)')
        rt.save_json(marker, identity)
    if args.phase == 'compare':
        compare(args)
        return
    print(f'T3.27 START phase={args.phase} variant={args.variant}; no test inference', flush=True)
    before = input_snapshot(args.project, args.checkpoint)
    rt.save_json(args.suite / f'input_hashes_before_{args.phase}.json', before)
    completed = False
    try:
        os.chdir(args.project)
        datasets, audit, sources = rt.configure_project(args.project)
        with guard_test_dataset(datasets[2]):
            if args.phase == 'check':
                run_check(args, datasets, audit, sources, package)
            elif args.phase in ('smoke', 'train'):
                run_train(args, datasets, audit, sources, package, smoke=args.phase == 'smoke')
            elif args.phase == 'evaluate':
                if args.optimized_checkpoint is None:
                    raise ValueError('--optimized-checkpoint is required')
                model, _ = load_optimized(args.optimized_checkpoint, args.device)
                metrics, frame = evaluate(model, datasets[1], args)
                rt.save_json(args.suite / 'optimized_validation_metrics.json', metrics)
                frame.to_csv(args.suite / 'optimized_validation.csv', index=False)
                print(json.dumps(metrics, ensure_ascii=False, indent=2))
        completed = True
    finally:
        after = input_snapshot(args.project, args.checkpoint)
        unchanged = before == after
        rt.save_json(args.suite / f'input_audit_{args.phase}.json', dict(
            status='PASS' if unchanged else 'FAIL', phase_completed=completed,
            original_files_unchanged=unchanged, hashed_files=len(before), test_predictions=0))
        if not unchanged:
            raise RuntimeError('Source spectra/checkpoint/code changed during execution')


if __name__ == '__main__':
    main()
