"""D4.38 isolated entry; no network requests or project-source edits."""
import argparse
import copy
import importlib
import json
from datetime import datetime
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
import torch
import yaml
from geometry import DEFAULT, VERSION

HERE = Path(__file__).resolve().parent


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        plain(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def settings():
    path = HERE/'server_config.json'
    if not path.is_file():
        raise FileNotFoundError('Run python run.py configure once first')
    return json.loads(path.read_text())


def configure(args):
    path = HERE/'server_config.json'
    if path.exists():
        print(path.read_text())
        return
    home = Path.home()
    tag = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    write(path, dict(
        project=str(home/'project'),
        real_root=str(home/'project/data/input'),
        checkpoint=str(home/'d4_35_long20_v1/d4_31/boundary/checkpoints/step_00001900.pt'),
        experiment=str(home/f'd4_38_geometry_{tag}'),
        epochs=50, batch_size=16, learning_rate=5e-6,
        geometry=DEFAULT,
        recognition_package=str(home/'T3_38_PureImage1920'),
        recognition_project=str(home/'project_transformer')))
    print(path.read_text())


def load_checkpoint(path):
    checkpoint = torch.load(
        Path(path).expanduser(), map_location='cpu', weights_only=False)
    if checkpoint['configuration']['model']['architecture'] != 'dit_curve2d':
        raise ValueError('Expected existing 2D curve DiT checkpoint')
    if not isinstance(checkpoint.get('ema_state', {}).get('ema_model'), dict):
        raise ValueError('EMA weights are required')
    return checkpoint


def selected_checkpoint(args, root):
    if args.checkpoint and args.epoch:
        raise ValueError('Choose --checkpoint OR --epoch')
    path = Path(args.checkpoint).expanduser() if args.checkpoint else root/'checkpoints/latest.pt'
    saved = load_checkpoint(path)
    if args.epoch:
        train = saved['configuration']['training']
        epochs = int(train['number_of_epochs'])
        total = int(train['total_training_steps'])
        if not 1 <= args.epoch <= epochs or total % epochs:
            raise ValueError('Invalid epoch/step mapping')
        path = root/'checkpoints'/f'step_{args.epoch*(total//epochs):08d}.pt'
        saved = load_checkpoint(path)
    return path, saved


def validate_split(meta):
    files = meta['relative_source_files']
    groups = [[i for i, f in enumerate(files) if f == name]
              for name in sorted(set(files))]
    splits = [set(map(int, meta[k])) for k in
              ('training_indices', 'validation_indices', 'test_indices')]
    if any(splits[i] & splits[j] for i in range(3) for j in range(i)):
        raise ValueError('Split overlap')
    if any([sum(i in s for i in ids) for s in splits] != [12, 4, 4]
           for ids in groups):
        raise ValueError('Each real condition must retain 12/4/4')
    return len(groups)


def prepare(args):
    s = settings()
    root = Path(s['experiment']).expanduser()
    if root.exists():
        raise FileExistsError('Use a new experiment directory: '+str(root))
    saved = load_checkpoint(s['checkpoint'])
    cfg = plain(copy.deepcopy(saved['configuration']))
    validate_split(saved['metadata'])
    fixed = cfg['curve_image']['state']
    if (fixed['height'], fixed['width']) != (768, 1920):
        raise ValueError('Production patch requires existing 1920x768 model')
    if not Path(s['real_root']).expanduser().is_dir():
        raise FileNotFoundError(s['real_root'])
    cfg['curve_image'].pop('state')
    cfg['curve_image']['save_images_per_condition'] = 50
    geometry = {**DEFAULT, **s.get('geometry', {})}
    if geometry['temperature'] != cfg['curve_image']['decode_temperature']:
        raise ValueError('Geometry and native decoder temperatures must match')
    cfg['d4_38'] = dict(
        version=VERSION, geometry=geometry,
        localization=dict(weight=.05, maximum_ratio=.02),
        initializer=str(Path(s['checkpoint']).expanduser().resolve()),
        description='shared train-only y-axis; native retained; continuous image projection')
    cfg['data']['input_directory'] = str(Path(s['real_root']).expanduser().resolve())
    cfg['training'].update(
        device='cuda', number_of_epochs=int(s['epochs']),
        batch_size=int(s['batch_size']), learning_rate=float(s['learning_rate']),
        validate_every_epochs=1, save_every_epochs=1,
        maximum_validation_batches=0,
        learning_rate_scheduler=dict(
            enabled=True, type='warmup_cosine', warmup_fraction=.06,
            peak_learning_rate=float(s['learning_rate']),
            warmup_start_learning_rate=float(s['learning_rate'])/5,
            minimum_learning_rate=float(s['learning_rate'])/10))
    cfg['generation'].update(number_of_spectra=50, model_source='ema')
    cfg['output'].update(
        output_directory=str(root),
        checkpoint_directory=str(root/'checkpoints'),
        log_directory=str(root/'logs'),
        training_log_file=str(root/'logs/training.csv'),
        generated_spectrum_directory=str(root/'generated'),
        preview_plot_directory=str(root/'plots'))
    if 'd4_30' not in cfg:
        raise ValueError('Use the selected D4.31 boundary-budget checkpoint')
    if cfg['d4_30']['loss']['maximum_ratio'] != .02:
        raise ValueError('Expected D4.31 two-percent boundary budget')
    root.mkdir(parents=True)
    (root/'runtime_config.yaml').write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    write(root/'preparation.json', dict(
        version=VERSION, status='prepared', settings=s,
        conditions=validate_split(saved['metadata']),
        pixels=[1920, 768], formal_training='cuda',
        model_selection='compare epoch 20 and 50 on validation; no automatic winner'))
    print('PREPARED:', root, flush=True)


def verify_metadata(actual, expected):
    keys = {
        'relative_source_files', 'training_indices',
        'validation_indices', 'test_indices', 'normalization_state',
        'conditioning_metadata', 'axis_metadata'}
    keys |= {k for source in (actual, expected)
             for k in source if k.endswith('_state')}
    keys.discard('curve_image_state')
    for key in sorted(keys):
        if plain(actual.get(key)) != plain(expected.get(key)):
            raise ValueError('Unexpected preprocessing change: '+key)
    validate_split(actual)


def warm_trainer_class(original, checkpoint, cfg, root):
    class Initialized(original):
        def __init__(self, **kw):
            verify_metadata(kw['metadata'], checkpoint['metadata'])
            kw['diffusion'].load_state_dict(
                checkpoint['ema_state']['ema_model'], strict=True)
            kw['metadata']['d4_38'] = cfg['d4_38']
            super().__init__(**kw)
            write(root/'fitted_geometry.json',
                  kw['configuration']['curve_image']['state'])
            print('D4.38 EMA warm-start; fresh optimizer/EMA; preprocessing verified.',
                  flush=True)
    return Initialized


def train_or_check(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU training fallback')
    s = settings()
    root = Path(s['experiment']).expanduser()
    cfg = yaml.safe_load((root/'runtime_config.yaml').read_text())
    if args.phase == 'train' and (root/'checkpoints/latest.pt').exists():
        raise FileExistsError('Training exists; use resume or a new experiment')
    if args.phase == 'resume' and not (root/'checkpoints/latest.pt').is_file():
        raise FileNotFoundError(root/'checkpoints/latest.pt')
    from runtime import install
    install(Path(s['project']).expanduser().resolve(), cfg)
    checkpoint = load_checkpoint(s['checkpoint'])
    module = importlib.import_module('src.ddpm_trainer')
    module.DdpmTrainer = warm_trainer_class(
        module.DdpmTrainer, checkpoint, cfg, root)
    builder = importlib.import_module('src.model_builder')
    build = builder.build_diffusion_model

    def initialized_build(*args, **kwargs):
        model, diffusion = build(*args, **kwargs)
        diffusion.load_state_dict(
            checkpoint['ema_state']['ema_model'], strict=True)
        return model, diffusion

    builder.build_diffusion_model = initialized_build
    script = importlib.import_module('scripts.start_ddpm_training')
    sys.argv = [str(script.__file__), '--config', str(root/'runtime_config.yaml')]
    if args.phase == 'check':
        sys.argv += ['--pipeline-check-only']
    if args.phase == 'resume':
        sys.argv += ['--resume', str(root/'checkpoints/latest.pt')]
    script.main()
    print(f'{args.phase.upper()} COMPLETE:', root, flush=True)


def generate(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU generation fallback')
    s = settings()
    root = Path(s['experiment']).expanduser()
    cp, saved = selected_checkpoint(args, root)
    cfg = plain(saved['configuration'])
    if cfg.get('d4_38', {}).get('version') != VERSION:
        raise ValueError('Expected a trained D4.38 checkpoint')
    default = f'generated_epoch_{args.epoch:03d}' if args.epoch else 'generated'
    output = Path(args.output).expanduser() if args.output else root/default
    if output.exists():
        raise FileExistsError('Use a fresh generation directory: '+str(output))
    cfg['curve_image']['save_images_per_condition'] = 50
    cfg['data']['input_directory'] = str(Path(s['real_root']).expanduser().resolve())
    from runtime import install
    install(Path(s['project']).expanduser().resolve(), cfg)
    from bounded_prior import bounded_prior_context
    from src.conditional_prior_residual import ConditionalPriorResidualBank
    from src.spectrum_conditioning import source_named_conditions_from_metadata
    from scripts import generate_conditional_spectra

    state = cfg['curve_image']['state']
    names = {cid: name for name, cid, _ in source_named_conditions_from_metadata(
        saved['metadata']['conditioning_metadata'])}
    output.mkdir(parents=True)
    runtime = output/'runtime_config.yaml'
    runtime.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    sys.argv = [
        str(generate_conditional_spectra.__file__),
        '--config', str(runtime), '--checkpoint', str(cp),
        '--condition', args.condition, '--number', '50',
        '--model-source', 'ema', '--device', 'cuda',
        '--output-directory', str(output)]
    with bounded_prior_context(
            ConditionalPriorResidualBank, state['y_min'], state['y_max'],
            max_redraws=100, names=names,
            callback=lambda report: write(output/'prior_bounds_audit.json', report)
    ) as sampler:
        try:
            generate_conditional_spectra.main()
        finally:
            write(output/'prior_bounds_audit.json', sampler.report())
    manifest_path = output/'generation_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['d4_38'] = dict(
        version=VERSION,
        image='continuous geometry projection of native heatmap',
        native='native_heatmap field; numerical decoding remains native',
        synthetic_tail='masked from measured spectra and recognition')
    write(manifest_path, manifest)
    audit(output, expected=50)


def evaluate(args):
    s = settings()
    root = Path(s['experiment']).expanduser()
    cp, saved = selected_checkpoint(args, root)
    cfg = plain(saved['configuration'])
    if cfg.get('d4_38', {}).get('version') != VERSION:
        raise ValueError('Expected D4.38 checkpoint')
    generated = Path(args.generated).expanduser() if args.generated else root/(
        f'generated_epoch_{args.epoch:03d}' if args.epoch else 'generated')
    output = Path(args.output).expanduser() if args.output else root/(
        f'validation_epoch_{args.epoch:03d}' if args.epoch else 'validation')
    if output.exists():
        raise FileExistsError(output)
    if not (generated/'generation_manifest.json').is_file():
        raise FileNotFoundError(generated)
    cfg['data']['input_directory'] = str(Path(s['real_root']).expanduser().resolve())
    from runtime import install
    install(Path(s['project']).expanduser().resolve(), cfg)
    from scripts import evaluate_d4_2_generation
    import tempfile

    with tempfile.TemporaryDirectory(prefix='d438_eval_') as directory:
        path = Path(directory)/'config.yaml'
        path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        sys.argv = [
            str(evaluate_d4_2_generation.__file__),
            '--config', str(path), '--checkpoint', str(cp),
            '--condition', args.condition, '--generated', str(generated),
            '--reference-split', 'validation',
            '--output-directory', str(output), '--plot-mode', 'none']
        evaluate_d4_2_generation.main()
    print('VALIDATION METRICS COMPLETE (FID not requested):', output, flush=True)


def audit(root, expected=None):
    root = Path(root)
    reports = sorted(root.rglob('geometry_quality.jsonl'))
    if not reports:
        raise FileNotFoundError('No geometry reports under '+str(root))
    result = []
    for path in reports:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        accepted = sum(bool(r.get('approved')) for r in rows)
        actual = len(list(path.parent.glob('raw_diffusion_*.npz')))
        result.append(dict(
            condition=path.parent.name, inspected=len(rows),
            approved=accepted, exported=actual, rejected=len(rows)-accepted))
    manifest_path = root/'generation_manifest.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        expected_names = {r['source_condition_name'] for r in manifest['conditions']}
        if {r['condition'] for r in result} != expected_names:
            raise ValueError('Image-condition reports differ from generation manifest')
    complete = all(
        r['rejected']==0 and r['approved']==r['exported']
        and (expected is None or r['exported']==expected)
        for r in result)
    summary = dict(
        status='complete' if complete else 'incomplete',
        conditions=len(result),
        approved_images=sum(r['approved'] for r in result),
        rejected_images=sum(r['rejected'] for r in result),
        per_condition=result,
        scientific_status='geometry export only; performance improvement unverified')
    write(root/'geometry_summary.json', summary)
    print(json.dumps(
        {k: v for k, v in summary.items() if k != 'per_condition'},
        ensure_ascii=False), flush=True)
    if not complete:
        raise RuntimeError(
            'Image export incomplete; see geometry_summary.json and rejected/ native images')


def preview(args):
    s = settings()
    source = Path(args.generated).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    if output == source or source in output.parents:
        raise ValueError('Preview output must be outside input directory')
    saved = load_checkpoint(args.checkpoint or s['checkpoint'])
    coordinate = saved['configuration']['curve_image']['state']
    axes = saved['metadata']['axis_metadata']['source_files']
    axes = {Path(name).stem: row['raman_shift'] for name, row in axes.items()}
    from runtime import save_batch
    from types import SimpleNamespace

    files = sorted(source.rglob('raw_diffusion_*.npz'))
    if not files:
        raise FileNotFoundError('No actual raw_diffusion_*.npz files')
    if args.limit:
        files = files[:args.limit]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for number, path in enumerate(files, 1):
        name = path.parent.name
        with np.load(path, allow_pickle=False) as z:
            native = z['native_heatmap'] if 'native_heatmap' in z else z['image']
            state = json.loads(str(z['coordinate_json'].item()))
            if any(state[k] != coordinate[k] for k in
                   ('height', 'width', 'axis', 'y_min', 'y_max')):
                raise ValueError('Preview checkpoint/image coordinates differ: '+str(path))
            if 'valid_columns' in z:
                valid = z['valid_columns'].astype(bool)
            else:
                if name not in axes:
                    raise ValueError('Missing source axis: '+name)
                valid = np.zeros(state['width'], dtype=bool)
                axis = np.asarray(state['axis'])
                measured = np.asarray(axes[name])
                valid[:len(axis)] = (
                    (axis >= measured.min()) & (axis <= measured.max()))
        state['image_geometry'] = {**DEFAULT, **s.get('geometry', {})}
        proxy = SimpleNamespace(
            codec=SimpleNamespace(state=state),
            betas=torch.zeros(1, device=device),
            last_curve_images=torch.from_numpy(native[None].copy()),
            last_curve_masks=torch.from_numpy(valid[None]),
            curve_configuration={'save_images_per_condition':1000000})
        index = int(path.stem.rsplit('_', 1)[1])-1
        save_batch(proxy, output/name, index)
        if number % 10 == 0 or number == len(files):
            print(f'PREVIEW {number}/{len(files)}', flush=True)
    audit(output)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('phase', choices=[
        'configure', 'prepare', 'check', 'train', 'resume',
        'generate', 'preview', 'audit', 'evaluate'])
    parser.add_argument('--checkpoint')
    parser.add_argument('--output')
    parser.add_argument('--generated')
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--condition', default='all')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.phase == 'configure':
        return configure(args)
    if args.phase == 'prepare':
        return prepare(args)
    if args.phase in ('check', 'train', 'resume'):
        return train_or_check(args)
    if args.phase == 'generate':
        return generate(args)
    if args.phase == 'evaluate':
        return evaluate(args)
    if args.phase == 'preview':
        if not args.generated or not args.output:
            parser.error('preview requires --generated and --output')
        return preview(args)
    return audit(
        args.generated or Path(settings()['experiment'])/'generated',
        expected=50)


if __name__ == '__main__':
    main()
