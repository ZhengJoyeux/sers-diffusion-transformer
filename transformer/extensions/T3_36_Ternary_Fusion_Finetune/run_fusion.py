"""T3.36: fine-tune copied concentration fusion/projection and DEL/TEB heads."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
import sys
import time
import zipfile

import pandas as pd
import torch
from torch import nn

import metrics_core as runtime
from cached_evaluation import KEYS, evaluate_cached
from cache_audit import validate_cache
from fusion_model import FusionFinetuner, ConcentrationBranch, VERSION, head_loss

DEFAULT_RUN = Path('/home/wqzheng/project_transformer/outputs/t3_33_ternary_correction_seed2026')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def configure(package, state):
    package = Path(package).resolve()
    for name in ('correction_model.py', 'feature_cache.py'):
        if sha256(package / name) != state['source_hashes'][name]:
            raise RuntimeError('T3.33 source differs from teacher checkpoint: ' + name)
    sys.path.insert(0, str(package))
    import correction_model
    if Path(correction_model.__file__).resolve() != package / 'correction_model.py':
        raise RuntimeError('Wrong correction_model imported')
    return correction_model


def configure_image_sources(state, image_package):
    from baseline_bridge import configure as configure_image
    image_runtime, hashes = configure_image(image_package)
    if hashes != state['t332_hashes']:
        raise RuntimeError('T3.32 source differs from teacher dependency')
    return image_runtime


def make_model(state, package, device, image_package):
    core = configure(package, state)
    if state['checkpoint_version'] != core.VERSION or state['ordinal_decoding'] != 'median':
        raise ValueError('Expected accepted T3.33 teacher with median decoding')
    if state['correction_config']['feature_dim'] != 371:
        raise ValueError('Expected original 3x64 concentration features and 371-column cache')
    teacher = core.TernaryCorrector(nn.Identity(), **state['correction_config'])
    weights = state['model_state_dict']
    teacher.load_state_dict({k: v for k, v in weights.items() if not k.startswith('baseline.')}, strict=True)
    configure_image_sources(state, image_package)
    from spectral_core import MixtureAwareQueryFusion
    config = state['baseline_reference']['model_config']
    dropout = config['dropout']
    fusion = MixtureAwareQueryFusion(config['attention_dim'],3,config['heads'],dropout,
                                    use_adaptive_gate=config['adaptive_gate'])
    prefix = 'baseline.mixture_aware_query_fusion.'
    fusion.load_state_dict({k[len(prefix):]: v for k,v in weights.items() if k.startswith(prefix)},strict=True)
    projection = nn.Linear(config['attention_dim'],64)
    prefix = 'baseline.concentration_projection.'
    projection.load_state_dict({k[len(prefix):]: v for k,v in weights.items() if k.startswith(prefix)},strict=True)
    heads = nn.ModuleList()
    for j in (0, 2):
        head = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 2))
        prefix = f'baseline.ordinal_heads.{j}.'
        head.load_state_dict({k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}, strict=True)
        heads.append(head)
    branch = ConcentrationBranch(fusion,projection,heads)
    model = FusionFinetuner(teacher, branch).to(device).eval()
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    if not trainable or any(not n.startswith('branch.') for n in trainable):
        raise RuntimeError('Only copied concentration branch may be trainable')
    return model


def rank(metrics):
    return (metrics['ternary_all_grade_correct'], metrics['full_profile_correct'],
            metrics['present_correct'], -metrics['loss_total'])


def admissible(metrics, reference):
    keys = ('present_correct', 'full_profile_correct', 'classification_exact_correct', 'ternary_all_grade_correct')
    return all(metrics[k] >= reference[k] for k in keys) and all(
        metrics['per_drug'][d]['correct'] >= reference['per_drug'][d]['correct'] for d in ('DEL', 'TEB'))


def assert_reference(metrics, reference):
    keys = ('samples', 'present_correct', 'full_profile_correct', 'classification_exact_correct', 'ternary_all_grade_correct')
    mismatch = {k: (metrics[k], reference[k]) for k in keys if metrics[k] != reference[k]}
    if mismatch:
        raise RuntimeError('Teacher validation reproduction failed: ' + str(mismatch))


def selected_indices(cache):
    if cache['split'] != 'train':
        raise ValueError('Only training cache can enter optimizer')
    select = (cache['presence'] > .5).all(1) & (cache['cp'] >= .5).all(1)
    sources = [m['source'] for m in cache['metadata']]
    real = torch.tensor([i for i in range(len(sources)) if bool(select[i]) and sources[i] == 'real'], dtype=torch.long)
    generated = torch.tensor([i for i in range(len(sources)) if bool(select[i]) and sources[i] == 'generated'], dtype=torch.long)
    if not len(real) or not len(generated):
        raise ValueError('Real and generated ternary training subsets must both exist')
    return real, generated


def paired_batches(real, generated, batch_size, seed):
    generator = torch.Generator().manual_seed(seed)
    real = real[torch.randperm(len(real), generator=generator)]
    # Each real row once per epoch; shuffle generated rows and draw an equal count.
    pieces = []
    available = 0
    while available < len(real):
        part = generated[torch.randperm(len(generated), generator=generator)]
        pieces.append(part)
        available += len(part)
    generated = torch.cat(pieces)[:len(real)]
    half = batch_size // 2
    for start in range(0, len(real), half):
        yield real[start:start+half], generated[start:start+half]


def batch(cache, indices, device):
    return {k: cache[k][indices].to(device) for k in KEYS}


def save_checkpoint(path, model, teacher_state, args, epoch, validation, updates):
    # Embed the unchanged original teacher for standalone future deployment.
    teacher = {k: v for k, v in teacher_state.items() if k != 'optimizer_state_dict'}
    payload = dict(checkpoint_version=VERSION, model_state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                   teacher_checkpoint=teacher, teacher_checkpoint_sha256=args.teacher_sha256,
                   epoch=epoch, validation_metrics=validation, ordinal_decoding='median',
                   completed_updates=updates, frozen_digest=model.frozen_digest(),
                   experiment_config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                   source_hashes={p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')},
                   test_predictions=0, selection_scope='reused validation split; not an independent test result')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, correction_package, device='cpu', image_package=None):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload['checkpoint_version'] != VERSION:
        raise ValueError('Expected T3.36 checkpoint')
    for name, expected in payload['source_hashes'].items():
        if sha256(Path(__file__).parent / name) != expected:
            raise RuntimeError('T3.36 source differs from saved checkpoint: ' + name)
    image_package = image_package or Path(payload['experiment_config']['image_package'])
    model = make_model(payload['teacher_checkpoint'], correction_package, device, image_package)
    model.load_state_dict(payload['model_state_dict'], strict=True)
    model.eval()
    if model.frozen_digest() != payload['frozen_digest']:
        raise RuntimeError('Frozen teacher/head state digest changed')
    return model, payload


def load_for_inference(path, image_package, correction_package, device='cpu'):
    """Return (spectrum-image model, PlotScale, checkpoint) for deployment."""
    model, payload = load_checkpoint(path, correction_package, device, image_package)
    from baseline_bridge import configure as configure_image
    image_runtime, hashes = configure_image(image_package)
    teacher = payload['teacher_checkpoint']
    if hashes != teacher['t332_hashes']:
        raise RuntimeError('T3.32 deployment sources differ')
    from image_model import ImageSERSFormer
    from image_data import PlotScale
    from fusion_model import ImageDeployment
    baseline = ImageSERSFormer(**teacher['baseline_reference']['model_config'])
    prefix = 'baseline.'
    baseline.load_state_dict({k[len(prefix):]: v for k, v in teacher['model_state_dict'].items() if k.startswith(prefix)}, strict=True)
    deployment = ImageDeployment(baseline.to(device), model).to(device).eval()
    return deployment, PlotScale(**teacher['plot_scale']), payload


def load_caches(args, state):
    fingerprint = state['cache_fingerprint']
    caches = []
    for split in ('train', 'validation'):
        path = args.teacher_run / 'feature_cache' / fingerprint / (split + '.pt')
        if not path.is_file():
            raise FileNotFoundError('Existing T3.33 cache required; no automatic re-extraction: ' + str(path))
        cache = torch.load(path, map_location='cpu', weights_only=False)
        validate_cache(cache, split, fingerprint, 371)
        caches.append(cache)
    image_runtime = configure_image_sources(state,args.image_package)
    from run_correction import configure_data
    from baseline_bridge import input_fingerprint, state_digest
    from image_model import ImageSERSFormer
    from image_data import PlotScale
    from query_cache import augment_cache, fingerprint as query_fingerprint
    datasets,audit = configure_data(args,state)
    found,inputs = input_fingerprint(datasets,audit,state['baseline_reference']['checkpoint_sha256'],state['t332_hashes'])
    if found != fingerprint:
        raise RuntimeError('Current data/split/source fingerprint differs from original T3.33 cache')
    baseline = ImageSERSFormer(**state['baseline_reference']['model_config'])
    prefix = 'baseline.'
    baseline.load_state_dict({k[len(prefix):]:v for k,v in state['model_state_dict'].items() if k.startswith(prefix)},strict=True)
    baseline.requires_grad_(False).to(args.device).eval()
    if state_digest(baseline) != state['baseline_digest']:
        raise RuntimeError('Frozen image baseline differs from teacher')
    args.query_fingerprint = query_fingerprint(state)
    scale = PlotScale(**state['plot_scale'])
    augmented = [augment_cache(baseline,datasets[i],scale,cache,args,args.query_fingerprint,image_runtime) for i,cache in enumerate(caches)]
    after_fingerprint,_=input_fingerprint(datasets,audit,state['baseline_reference']['checkpoint_sha256'],state['t332_hashes'])
    if after_fingerprint != fingerprint:
        raise RuntimeError('Source data changed on disk during query extraction')
    if state_digest(baseline) != state['baseline_digest']:
        raise RuntimeError('Image baseline changed while extracting queries')
    runtime.write_json(args.output_directory/'query_extraction_audit.json',dict(status='PASS',
        query_fingerprint=args.query_fingerprint,original_cache_fingerprint=fingerprint,
        train_queries=list(augmented[0]['queries'].shape),validation_queries=list(augmented[1]['queries'].shape),
        baseline_unchanged=True,test_features=0,test_predictions=0))
    return augmented


def check(args):
    state = torch.load(args.teacher_checkpoint, map_location='cpu', weights_only=False)
    args.teacher_sha256 = sha256(args.teacher_checkpoint)
    args.ordinal_decoding = state['ordinal_decoding']
    model = make_model(state, args.correction_package, args.device, args.image_package)
    train, val = load_caches(args, state)
    reference, frame = evaluate_cached(model, val, args, runtime)
    assert_reference(reference, state['validation_metrics'])
    # Compare all probabilities/logits, not only rounded grade counts.
    for start in range(0, len(val['metadata']), args.batch_size):
        inputs = {k: val[k][start:start+args.batch_size].to(args.device) for k in KEYS}
        with torch.no_grad():
            before = model.teacher.forward_cached({k:inputs[k] for k in ('features','cp','op','logits')})
            after = model.forward_cached(inputs)
        for key in ('cp', 'op', 'logits'):
            if not torch.equal(before[key], after[key]):
                raise RuntimeError('Copied-head initialization is not exact identity: ' + key)
    real, generated = selected_indices(train)
    original = model.frozen_digest()
    n = min(args.batch_size // 2, len(real))
    model.train()
    output = model.forward_cached(batch(train, real[:n], args.device))
    loss = head_loss(output, train['levels'][real[:n]].to(args.device), stability_weight=args.stability_weight)
    loss.backward()
    for name,module in [('fusion',model.branch.fusion),('projection',model.branch.projection),('heads',model.branch.heads)]:
        if not any(p.grad is not None and torch.count_nonzero(p.grad) for p in module.parameters()):
            raise RuntimeError('No gradient reached copied '+name)
    if any(p.grad is not None for name, p in model.named_parameters() if not name.startswith('branch.')):
        raise RuntimeError('Gradient reached frozen model')
    model.zero_grad(set_to_none=True)
    model.eval()
    if original != model.frozen_digest():
        raise RuntimeError('Frozen state changed during check')
    report = dict(status='PASS', teacher_epoch=state['epoch'], reference_counts={k: reference[k] for k in
                  ('samples', 'present_correct', 'full_profile_correct', 'ternary_all_grade_correct', 'classification_exact_correct')},
                  train_rows=len(train['metadata']), validation_rows=len(val['metadata']),
                  selected_real_ternary=len(real), selected_generated_ternary=len(generated),
                  trainable_parameters=sum(p.numel() for p in model.branch.parameters()),
                  trainable_modules=['copied fusion','copied concentration projection','copied DEL/TEB ordinal heads'],
                  query_fingerprint=args.query_fingerprint,
                  real_loss_weight=args.real_loss_weight, copied_branch_identity='PASS',
                  frozen_digest=original, teacher_checkpoint_sha256=args.teacher_sha256,
                  cache_fingerprint=state['cache_fingerprint'], optimizer_steps=0, test_features=0, test_predictions=0)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    runtime.write_json(args.output_directory / 'check_report.json', report)
    runtime.save_evaluation(args.output_directory / 'reference_validation', reference, frame)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return model, state, train, val, reference, real, generated


def training(args, prepared, smoke=False):
    model, state, train, val, reference, real, generated = prepared
    directory = args.output_directory / ('smoke' if smoke else 'train')
    directory.mkdir(parents=True, exist_ok=False)
    groups = []
    for name, module, rate in [('fusion',model.branch.fusion,args.fusion_learning_rate),
                               ('projection',model.branch.projection,args.projection_learning_rate),
                               ('heads',model.branch.heads,args.learning_rate)]:
        groups.append(dict(params=list(module.parameters()),family=name,lr=rate,maximum_lr=rate,
                           minimum_lr=rate*args.minimum_learning_rate/args.learning_rate))
    intended={id(p) for p in model.parameters() if p.requires_grad}
    actual=[id(p) for g in groups for p in g['params']]
    if len(actual)!=len(set(actual)) or set(actual)!=intended:
        raise RuntimeError('Optimizer parameters do not match copied branch')
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    steps = math.ceil(len(real) / (args.batch_size // 2))
    epochs = 1 if smoke else args.epochs
    if smoke:
        steps = min(steps, 2)
    total = epochs * steps
    warmup = min(args.warmup_epochs * steps, max(total - 1, 0))
    frozen = model.frozen_digest()
    reference_digest = {k: sha256(p) for k, p in dict(teacher=args.teacher_checkpoint,
        train=args.teacher_run / 'feature_cache' / state['cache_fingerprint'] / 'train.pt',
        validation=args.teacher_run / 'feature_cache' / state['cache_fingerprint'] / 'validation.pt',
        train_queries=Path(train['query_cache_path']),validation_queries=Path(val['query_cache_path'])).items()}
    runtime.write_json(directory / 'training_config.json', dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                       trainable='copied concentration fusion/projection and DEL/TEB ordinal heads',
                       trainable_parameters=sum(p.numel() for p in model.branch.parameters()),
                       selected_real=len(real), selected_generated_pool=len(generated),
                       generated_rows_per_complete_epoch=len(real), source_loss_weights=[args.real_loss_weight, 1-args.real_loss_weight],
                       reference=reference, input_hashes=reference_digest))
    save_checkpoint(directory / 'checkpoints/best_accepted.pt', model, state, args, 0, reference, 0)
    best_rank = rank(reference)
    accepted_epoch = 0
    stale = updates = 0
    history = []
    for epoch in range(1, epochs + 1):
        started = time.time()
        model.train()
        accumulated = 0.
        rows = 0
        for step, (r, g) in enumerate(paired_batches(real, generated, args.batch_size, args.seed + epoch)):
            if step >= steps:
                break
            indices = torch.cat((r, g))
            output = model.forward_cached(batch(train, indices, args.device))
            from correction_model import protection_check
            protection_check(output, batch(train, indices, args.device))
            labels = train['levels'][indices].to(args.device)
            n = len(r)
            def subset(lo, hi):
                return {k: v[lo:hi] for k, v in output.items()}
            real_loss = head_loss(subset(0, n), labels[:n], stability_weight=args.stability_weight)
            gen_loss = head_loss(subset(n, len(indices)), labels[n:], stability_weight=args.stability_weight)
            loss = args.real_loss_weight * real_loss + (1-args.real_loss_weight) * gen_loss + args.anchor_weight * model.anchor_loss()
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite head loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is not None for name, p in model.named_parameters() if not name.startswith('branch.')):
                raise RuntimeError('Gradient reached frozen parameters')
            torch.nn.utils.clip_grad_norm_(model.branch.parameters(), 1.)
            runtime.set_rates(optimizer, updates + 1, total, warmup)
            optimizer.step()
            updates += 1
            accumulated += float(loss.detach()) * n
            rows += n
        metrics, frame = evaluate_cached(model, val, args, runtime)
        if model.frozen_digest() != frozen:
            raise RuntimeError('Frozen teacher/reference branch changed')
        safe = admissible(metrics, reference)
        improved = safe and metrics['ternary_all_grade_correct'] > reference['ternary_all_grade_correct'] and rank(metrics) > best_rank
        if improved:
            best_rank = rank(metrics)
            accepted_epoch = epoch
            stale = 0
            save_checkpoint(directory / 'checkpoints/best_accepted.pt', model, state, args, epoch, metrics, updates)
            runtime.save_evaluation(directory / 'accepted_validation', metrics, frame)
        elif epoch > args.warmup_epochs:
            stale += 1
        save_checkpoint(directory / 'checkpoints/latest.pt', model, state, args, epoch, metrics, updates)
        row = dict(epoch=epoch, train_loss=accumulated/rows, validation_loss=metrics['loss_total'],
                   present_correct=metrics['present_correct'], full_profile_correct=metrics['full_profile_correct'],
                   ternary_correct=metrics['ternary_all_grade_correct'], DEL_correct=metrics['per_drug']['DEL']['correct'],
                   TEB_correct=metrics['per_drug']['TEB']['correct'], classification_exact_correct=metrics['classification_exact_correct'],
                   accepted=improved, passes_no_regression_guard=safe,
                   fusion_lr=optimizer.param_groups[0]['lr'],projection_lr=optimizer.param_groups[1]['lr'],
                   head_lr=optimizer.param_groups[2]['lr'],
                   seconds=time.time()-started, frozen_state_unchanged=True, output_protection='PASS')
        real_cache=subset_cache(train,real)
        real_metrics,_=evaluate_cached(model,real_cache,args,runtime)
        row.update(real_train_ternary_accuracy=real_metrics['ternary_full_profile_accuracy'],
                   validation_ternary_accuracy=metrics['ternary_full_profile_accuracy'])
        history.append(row)
        pd.DataFrame(history).to_csv(directory / 'training_history.csv', index=False)
        runtime.write_json(directory / 'training_history.json', history)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if not smoke and stale >= args.patience:
            break
    runtime.save_evaluation(directory/'latest_validation',metrics,frame)
    _,last_training_frame=evaluate_cached(model,train,args,runtime)
    last_training_frame.to_csv(directory/'latest_training_predictions.csv',index=False)
    runtime.write_json(directory/'latest_training_metrics.json',{source:runtime.metrics(g) for source,g in last_training_frame.groupby('source')})
    accepted, payload = load_checkpoint(directory / 'checkpoints/best_accepted.pt', args.correction_package, args.device,args.image_package)
    metrics, frame = evaluate_cached(accepted, val, args, runtime)
    assert_reference(metrics, payload['validation_metrics'])
    runtime.save_evaluation(directory / 'accepted_validation', metrics, frame)
    # Also compare real and generated training subsets with the same global evaluation definition.
    _, training_frame = evaluate_cached(accepted, train, args, runtime)
    training_frame.to_csv(directory / 'accepted_training_predictions.csv', index=False)
    training_groups = {source: runtime.metrics(g) for source, g in training_frame.groupby('source')}
    runtime.write_json(directory / 'accepted_training_metrics.json', training_groups)
    current = dict(teacher=sha256(args.teacher_checkpoint),
        train=sha256(args.teacher_run / 'feature_cache' / state['cache_fingerprint'] / 'train.pt'),
        validation=sha256(args.teacher_run / 'feature_cache' / state['cache_fingerprint'] / 'validation.pt'),
        train_queries=sha256(Path(train['query_cache_path'])),validation_queries=sha256(Path(val['query_cache_path'])))
    if current != reference_digest:
        raise RuntimeError('An input checkpoint/cache changed on disk during training')
    delta = {k: metrics[k]-reference[k] for k in ('present_correct','full_profile_correct','ternary_all_grade_correct','classification_exact_correct')}
    runtime.write_json(directory / 'comparison.json', dict(reference=reference, accepted=metrics, delta=delta,
                       accepted_epoch=accepted_epoch, status='improved' if accepted_epoch else 'teacher retained',
                       split='validation', test_predictions=0))
    runtime.write_json(directory / 'run_summary.json', dict(status='completed', smoke=smoke, completed_updates=updates,
                       accepted_epoch=accepted_epoch, frozen_state_unchanged=True, input_hashes_unchanged=True,
                       best_accepted=str(directory / 'checkpoints/best_accepted.pt'), test_predictions=0))
    print('Accepted comparison: ' + json.dumps(delta), flush=True)


def subset_cache(cache,indices):
    result=dict(cache)
    for key in KEYS+('levels','presence'):
        result[key]=cache[key][indices]
    result['metadata']=[cache['metadata'][int(i)] for i in indices]
    return result


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('check','smoke','train','evaluate','pack'), default='check')
    p.add_argument('--correction-package', type=Path, default=Path('/home/wqzheng/T3_33_Ternary_Correction'))
    p.add_argument('--image-package',type=Path,default=Path('/home/wqzheng/T3_32_Image_Recognition'))
    p.add_argument('--project',type=Path,default=Path('/home/wqzheng/project_transformer'))
    p.add_argument('--real-root',type=Path)
    p.add_argument('--generated-root',type=Path)
    p.add_argument('--image-cache',type=Path,default=Path('/home/wqzheng/project_transformer/outputs/t3_32_line_image_seed2026/image_cache'))
    p.add_argument('--cache-batch-size',type=int,default=16)
    p.add_argument('--num-workers',type=int,default=2)
    p.add_argument('--teacher-run', type=Path, default=DEFAULT_RUN)
    p.add_argument('--teacher-checkpoint', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--split', choices=('train','validation'), default='validation')
    p.add_argument('--output-directory', type=Path, default=Path('/home/wqzheng/project_transformer/outputs/t3_36_ternary_fusion_seed2026'))
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--warmup-epochs', type=int, default=3)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--learning-rate', type=float, default=3e-5)
    p.add_argument('--minimum-learning-rate', type=float, default=3e-7)
    p.add_argument('--fusion-learning-rate',type=float,default=3e-6)
    p.add_argument('--projection-learning-rate',type=float,default=1e-5)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--real-loss-weight', type=float, default=.75)
    p.add_argument('--stability-weight', type=float, default=.2)
    p.add_argument('--anchor-weight', type=float, default=.01)
    args = p.parse_args()
    args.teacher_checkpoint = args.teacher_checkpoint or args.teacher_run / 'train/checkpoints/best_accepted.pt'
    if min(args.epochs,args.threads,args.patience,args.cache_batch_size) < 1 or args.batch_size < 2 or args.batch_size % 2 or args.num_workers<0:
        p.error('Use positive budgets and an even batch size >=2')
    if not 0 < args.minimum_learning_rate <= args.learning_rate or args.warmup_epochs < 0:
        p.error('Invalid learning-rate schedule')
    if not 0<args.fusion_learning_rate<=args.learning_rate or not 0<args.projection_learning_rate<=args.learning_rate:
        p.error('Fusion/projection learning rates must be positive and no greater than the head learning rate')
    if not .5 < args.real_loss_weight <= 1 or min(args.weight_decay,args.stability_weight,args.anchor_weight) < 0:
        p.error('Use real-majority source loss and nonnegative regularization')
    if args.device.startswith('cuda') and not torch.cuda.is_available() and args.phase != 'pack':
        p.error('CUDA unavailable; CPU is sufficient for cached head training')
    return args


def main():
    args = parse_args()
    if args.phase == 'pack':
        source = args.output_directory
        if not source.is_dir():
            raise FileNotFoundError(source)
        target = source.with_name(source.name + '_reports.zip')
        with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(source.rglob('*')):
                if path.is_file() and path.suffix in ('.csv','.json','.log'):
                    archive.write(path,str(path.relative_to(source)))
        print(target)
        return
    torch.set_num_threads(args.threads)
    runtime.seed_all(args.seed)
    if args.phase == 'evaluate':
        if args.checkpoint is None:
            raise ValueError('--checkpoint required for evaluation')
        model, payload = load_checkpoint(args.checkpoint,args.correction_package,args.device,args.image_package)
        args.ordinal_decoding = payload['ordinal_decoding']
        caches = load_caches(args,payload['teacher_checkpoint'])
        cache = caches[0 if args.split == 'train' else 1]
        metrics, frame = evaluate_cached(model,cache,args,runtime)
        runtime.save_evaluation(args.output_directory/'evaluation'/args.split,metrics,frame)
        print(json.dumps(metrics,ensure_ascii=False))
        return
    if args.phase in ('train','smoke') and (args.output_directory / args.phase).exists():
        raise FileExistsError('Previous run exists; use a different --output-directory')
    prepared = check(args)
    if args.phase in ('train','smoke'):
        training(args,prepared,smoke=args.phase=='smoke')


if __name__ == '__main__':
    main()
