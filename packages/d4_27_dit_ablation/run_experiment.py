"""Run checks, train both groups, then generate all conditions; review is separate."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml


def run(command, project, env, log):
    print('Running:', ' '.join(map(str, command)), flush=True)
    with Path(log).open('a') as stream:
        process = subprocess.Popen(list(map(str, command)), cwd=project, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            stream.write(line)
            stream.flush()
        if process.wait():
            raise RuntimeError(f'command failed; see {log}')


def checkpoint_audit(path, c, project, require_complete=False):
    sys.path.insert(0, str(project))
    import torch
    value = torch.load(path, map_location='cpu', weights_only=False)
    metadata = value.get('metadata', {})
    saved = value.get('configuration', {})
    from src.configuration_loader import validate_config
    validate_config(saved)
    validate_config(c)
    for key in ('model', 'data', 'conditioning', 'prior_residual', 'broad_local_residual', 'diffusion', 'normalization'):
        if saved.get(key) != c.get(key):
            raise ValueError(f'checkpoint/config mismatch: {key}')
    if require_complete:
        from src.condition_grouped_batch_sampler import ConditionGroupedBatchSampler
        import math
        # D4.26 grouped sampler puts four spectra of a condition in each group.
        n = len(metadata['training_indices'])
        batches = n // c['training']['batch_size'] if c['training'].get('drop_last') else math.ceil(n / c['training']['batch_size'])
        expected = batches * c['training']['number_of_epochs']
        state = value.get('trainer_state', {})
        step = value.get('step', state.get('step', state.get('global_step', 0)))
        if int(step) < expected:
            raise ValueError(f'training incomplete: {step} < {expected}')
    return metadata


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--suite', type=Path)
    p.add_argument('--phase', required=True, choices=('check', 'train', 'generate', 'train_generate', 'evaluate'))
    p.add_argument('--physical-gpu', default='1')
    p.add_argument('--variants', nargs='+', choices=('broad_local', 'full_residual'),
                   default=['broad_local', 'full_residual'])
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    suite = a.suite or Path((Path(__file__).parent / 'last_suite.txt').read_text().strip())
    index = json.loads((suite / 'run_index.json').read_text())
    project = Path(index['project'])
    for relative, digest in index['source_hashes'].items():
        if hashlib.sha256((project / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f'source changed since suite preparation: {relative}')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.physical_gpu, PYTHONUNBUFFERED='1')
    configs = {}
    for variant, spec in index['configs'].items():
        if variant not in a.variants:
            continue
        path = Path(spec['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != spec['sha256']:
            raise ValueError('suite config changed; create a new suite')
        configs[variant] = yaml.safe_load(path.read_text())
    for phase in (('train', 'generate') if a.phase == 'train_generate' else (a.phase,)):
        for variant, c in configs.items():
            config = index['configs'][variant]['path']
            folder = suite / variant
            if phase == 'check':
                run([sys.executable, '-u', '-m', 'scripts.start_ddpm_training', '--config', config, '--pipeline-check-only'],
                    project, env, folder / 'check_console.log')
            elif phase == 'train':
                command = [sys.executable, '-u', '-m', 'scripts.start_ddpm_training', '--config', config]
                latest = Path(c['output']['checkpoint_directory']) / c['output'].get('latest_checkpoint_name', 'latest.pt')
                if latest.exists():
                    if not a.resume:
                        raise ValueError('checkpoint exists: use --resume or create a new suite')
                    command += ['--resume', str(latest)]
                run(command, project, env, folder / 'training_console.log')
            elif phase == 'generate':
                root = Path(c['output']['checkpoint_directory'])
                checkpoint_audit(root / c['output'].get('latest_checkpoint_name', 'latest.pt'), c, project, True)
                best = root / c['output'].get('best_checkpoint_name', 'best.pt')
                metadata = checkpoint_audit(best, c, project)
                pointer = folder / 'generation_result.json'
                root = Path(json.loads(pointer.read_text())['output_directory']) if pointer.exists() else folder / 'generated'
                manifest_path = root / 'generation_manifest.json'
                complete = False
                if manifest_path.exists():
                    manifest = json.loads(manifest_path.read_text())
                    complete = (manifest.get('checkpoint') == str(best) and manifest.get('model_source') == 'ema'
                                and manifest.get('number_of_spectra_per_condition') == 100
                                and {r['internal_condition_id'] for r in manifest.get('conditions', [])} ==
                                    set(metadata['conditional_prior_residual_state']['conditions']))
                    for record in manifest.get('conditions', []):
                        complete = complete and all(Path(f).is_file() for f in record['output_files'])
                        trace = record.get('diagnostic_stage_trace_manifest')
                        if trace:
                            trace_path = Path(trace)
                            if not trace_path.exists():
                                complete = False
                            else:
                                trace_info = json.loads(trace_path.read_text())
                                cache = trace_path.parent / 'stage_cache.npz'
                                complete = complete and cache.is_file() and hashlib.sha256(cache.read_bytes()).hexdigest() == trace_info['cache_sha256']
                best_hash = hashlib.sha256(best.read_bytes()).hexdigest()
                if pointer.exists():
                    complete = complete and json.loads(pointer.read_text()).get('checkpoint_sha256') == best_hash
                if complete:
                    print('Generation already complete; keeping:', root)
                else:
                    if root.exists() and any(root.iterdir()):
                        root = folder / ('generated_attempt_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
                    run([sys.executable, '-u', '-m', 'scripts.generate_conditional_spectra', '--config', config,
                         '--checkpoint', best, '--condition', 'all', '--number', '100', '--model-source', 'ema',
                         '--output-directory', root, '--save-stage-trace'], project, env, folder / 'generation_console.log')
                pointer.write_text(json.dumps(dict(output_directory=str(root), checkpoint_sha256=best_hash), indent=2))
            else:
                pointer = folder / 'generation_result.json'
                root = Path(json.loads(pointer.read_text())['output_directory']) if pointer.exists() else folder / 'generated'
                best = Path(c['output']['checkpoint_directory']) / c['output'].get('best_checkpoint_name', 'best.pt')
                checkpoint_audit(best, c, project)
                run([sys.executable, '-u', '-m', 'scripts.evaluate_d4_2_generation', '--config', config,
                     '--checkpoint', best, '--condition', 'all', '--generated', root,
                     '--reference-split', 'validation', '--output-directory', folder / 'evaluation'],
                    project, env, folder / 'evaluation_console.log')
                rows = []
                for path in (folder / 'evaluation').rglob('condition_metrics.csv'):
                    with path.open(encoding='utf-8-sig') as f:
                        rows.extend(dict(variant=variant, **row) for row in csv.DictReader(f))
                if rows:
                    with (folder / 'condition_metrics_all.csv').open('w', encoding='utf-8-sig', newline='') as f:
                        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('Completed:', a.phase, '; suite:', suite)


if __name__ == '__main__':
    main()
