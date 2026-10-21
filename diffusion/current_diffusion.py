"""Entrypoints for the retained D4.31 epoch-20 model; weights remain external."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
EXTENSION = ROOT / 'extensions' / 'D4_31_Budget2'
SELECTION = REPOSITORY / 'current_configs' / 'D4.31' / 'selection.json'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def selected_model():
    return json.loads(SELECTION.read_text(encoding='utf-8'))


def load_selected_checkpoint(path=None, *, selection=None):
    """Validate a local checkpoint against the selected experiment record.

    Only load trusted checkpoints. No network access, alternate-model search,
    optimizer resume, or CPU training fallback is performed here.
    """
    import torch

    selected = selected_model() if selection is None else selection
    path = Path(path or selected['server_checkpoint']).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f'Selected weights are not in GitHub: {path}. '
            'Provide the existing server checkpoint with --checkpoint.'
        )
    if sha256(path) != selected['checkpoint_sha256']:
        raise ValueError('Checkpoint SHA256 differs from selected D4.31 epoch-20 weights')
    state = torch.load(path, map_location='cpu', weights_only=False)
    cfg = state['configuration']
    if cfg['model']['architecture'] != 'dit_curve2d':
        raise ValueError('Expected the curve-image DiT, not an archived 1D model')
    if state['step'] != selected['step']:
        raise ValueError('Checkpoint step does not match the selected snapshot')
    if cfg['d4_30']['version'] != 'd4_31_boundary_budget2_v1':
        raise ValueError('Expected D4.31 training provenance')
    loss = cfg['d4_30']['loss']
    if loss['maximum_ratio'] != 0.02 or loss['weight'] != 0.1:
        raise ValueError('D4.31 boundary-loss configuration differs')
    if not isinstance(state.get('ema_state', {}).get('ema_model'), dict):
        raise ValueError('Selected checkpoint has no EMA model')
    if selected.get('fixed_coordinates'):
        curve = cfg['curve_image']['state']
        if any(curve[k] != value for k, value in selected['fixed_coordinates'].items()):
            raise ValueError('Saved curve coordinates differ from the selected experiment')
    return path, state


def _configure_project():
    """Reuse the tested extension's path setup before importing legacy scripts."""
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(EXTENSION))
    spec = importlib.util.spec_from_file_location('_current_d431_driver', EXTENSION / 'd4_30.py')
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    driver.configure_project(ROOT)


def _runtime_configuration(checkpoint, output, real_data):
    cfg = copy.deepcopy(checkpoint['configuration'])
    cfg['generation']['model_source'] = 'ema'
    cfg['generation']['number_of_spectra'] = 50
    cfg['output']['generated_spectrum_directory'] = str(output)
    if real_data is not None:
        cfg['data']['input_directory'] = str(Path(real_data).expanduser().resolve())
    else:
        input_path = Path(cfg['data']['input_directory']).expanduser()
        if not input_path.is_absolute():
            raise ValueError('Use --real-data to resolve the checkpoint\'s relative input directory')
    if not Path(cfg['data']['input_directory']).is_dir():
        raise FileNotFoundError('Original training spectra are required for saved calibration; use --real-data')
    # Fitted checkpoint configurations can contain numpy scalar condition IDs.
    # Convert container/scalar types without changing any fitted values.
    from numpy import generic, ndarray

    def plain(value):
        if isinstance(value, dict):
            return {str(k): plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        if isinstance(value, ndarray):
            return plain(value.tolist())
        if isinstance(value, generic):
            return value.item()
        return value

    return plain(cfg)


def generate(*, checkpoint=None, condition, output, real_data=None, device='cuda', save_stage_trace=False):
    import torch
    import yaml

    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    path, state = load_selected_checkpoint(checkpoint)
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'Use a new generation directory: {output}')
    cfg = _runtime_configuration(state, output, real_data)
    fixed = cfg['curve_image']['state']
    conditioning = state['metadata']['conditioning_metadata']
    del state
    _configure_project()
    from bounded_prior import bounded_prior_context
    from src.conditional_prior_residual import ConditionalPriorResidualBank
    from src.spectrum_conditioning import source_named_conditions_from_metadata
    from scripts import generate_conditional_spectra

    names = {cid: name for name, cid, _ in source_named_conditions_from_metadata(conditioning)}
    output.mkdir(parents=True)
    audit = output / 'prior_bounds_audit.json'

    def record(report):
        audit.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))

    with tempfile.TemporaryDirectory(prefix='sers_d431_') as directory:
        config_path = Path(directory) / 'runtime_config.yaml'
        config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        previous = sys.argv
        try:
            sys.argv = [str(generate_conditional_spectra.__file__), '--config', str(config_path),
                        '--checkpoint', str(path), '--condition', condition, '--number', '50',
                        '--model-source', 'ema', '--device', device, '--output-directory', str(output)]
            if save_stage_trace:
                sys.argv.append('--save-stage-trace')
            with bounded_prior_context(ConditionalPriorResidualBank, fixed['y_min'], fixed['y_max'],
                                       max_redraws=100, callback=record, names=names) as sampler:
                try:
                    generate_conditional_spectra.main()
                finally:
                    record(sampler.report())
        finally:
            sys.argv = previous
    if sha256(path) != selected_model()['checkpoint_sha256']:
        raise ValueError('Checkpoint changed during generation')


def evaluate(*, checkpoint=None, generated, output, real_data=None, condition='all', reference_split='validation'):
    """Run ordinary metrics directly; FID is not a dependency."""
    import yaml

    path, state = load_selected_checkpoint(checkpoint)
    output = Path(output).expanduser().resolve()
    generated = Path(generated).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'Use a new evaluation directory: {output}')
    if not generated.is_dir():
        raise FileNotFoundError(generated)
    cfg = _runtime_configuration(state, generated, real_data)
    del state
    _configure_project()
    from scripts import evaluate_d4_2_generation

    with tempfile.TemporaryDirectory(prefix='sers_d431_eval_') as directory:
        config_path = Path(directory) / 'runtime_config.yaml'
        config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        previous = sys.argv
        try:
            sys.argv = [str(evaluate_d4_2_generation.__file__), '--config', str(config_path),
                        '--checkpoint', str(path), '--condition', condition,
                        '--generated', str(generated), '--reference-split', reference_split,
                        '--output-directory', str(output), '--plot-mode', 'none']
            evaluate_d4_2_generation.main()
        finally:
            sys.argv = previous
    report = dict(status='complete', fid='not_requested', checkpoint_sha256=sha256(path),
                  reference_split=reference_split, generated=str(generated),
                  outputs={n: sha256(output / n) for n in
                           ['condition_metrics.csv', 'overall_metric_summary.csv', 'peak_morphology_details.csv']})
    (output / 'current_model_evaluation.json').write_text(json.dumps(report, indent=2))
