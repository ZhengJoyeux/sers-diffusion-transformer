"""Build paired configs from the actual D4.26 runtime config, preserving rules."""
import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import yaml


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(project, source_config, epochs=50, batch_size=16, seed=2026):
    project = Path(project).expanduser().resolve()
    source_config = Path(source_config).expanduser().resolve()
    original = yaml.safe_load(source_config.read_text())
    if original['data']['raman_axis_mode'] != 'union_with_valid_mask':
        raise ValueError('base config must use union_with_valid_mask')
    if not all(original[k].get('enabled', False) for k in ('prior_residual', 'broad_local_residual', 'conditioning')):
        raise ValueError('base config must preserve PCA, residual preprocessing and chemical conditions')
    if not original['conditioning'].get('prior_spectrum', {}).get('enabled', False):
        raise ValueError('base config must enable actual reconstruction-base conditioning')
    if original.get('generation', {}).get('joint_prior_sampling', {}).get('enabled', False):
        raise ValueError('paired experiment requires the original independent prior sampler')
    if epochs <= 0 or batch_size <= 0 or batch_size % 4:
        raise ValueError('epochs must be positive; batch size must be a positive multiple of 4')
    if original['data'].get('fixed_spectrum_split') != {'train_count': 12, 'validation_count': 4, 'test_count': 4}:
        raise ValueError('this matched experiment preserves the D26+T24 12/4/4 split')
    sys.path.insert(0, str(project))
    from src.configuration_loader import validate_config
    suite = project / 'outputs/experiments' / ('d4_27_dit_e%d_b%d_seed%d_' % (epochs, batch_size, seed) + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    suite.mkdir(parents=True)
    index = dict(schema='d4_27_dit_ablation_v1', project=str(project), base_config=str(source_config),
                 base_sha256=digest(source_config), epochs=epochs, batch_size=batch_size, seed=seed,
                 number_per_condition=100, configs={}, source_hashes={})
    for folder in ('src', 'scripts'):
        for path in sorted((project / folder).rglob('*.py')):
            index['source_hashes'][str(path.relative_to(project))] = digest(path)
    for variant in ('broad_local', 'full_residual'):
        c = copy.deepcopy(original)
        c.pop('_paths', None)
        c['project']['name'] = suite.name + '_' + variant
        c['project']['random_seed'] = seed
        c['model']['architecture'] = 'dit'
        c['model']['bottleneck_transformer'] = {'enabled': False}
        c['model']['dit'] = dict(hidden_size=512, depth=12, num_heads=8, patch_size=8,
                                mlp_ratio=4.0, gradient_checkpointing=True)
        c['broad_local_residual']['residual_mode'] = variant
        c['training']['number_of_epochs'] = epochs
        c['training']['batch_size'] = batch_size
        c['generation']['number_of_spectra'] = 100
        input_path = Path(c['data']['input_directory']).expanduser()
        c['data']['input_directory'] = str(input_path if input_path.is_absolute() else project / input_path)
        run = suite / variant
        run.mkdir()
        for key, suffix in {'output_directory': '', 'checkpoint_directory': 'checkpoints',
                            'log_directory': 'logs', 'generated_spectrum_directory': 'generated',
                            'preview_plot_directory': 'plots'}.items():
            c['output'][key] = str(run / suffix)
        validate_config(c)
        path = run / 'runtime_config.yaml'
        path.write_text(yaml.safe_dump(c, allow_unicode=True, sort_keys=False))
        index['configs'][variant] = {'path': str(path), 'sha256': digest(path)}
    (suite / 'run_index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2))
    (Path(__file__).parent / 'last_suite.txt').write_text(str(suite) + '\n')
    return suite


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--project', default='/home/wqzheng/project')
    p.add_argument('--base-config', required=True)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--seed', type=int, default=2026)
    a = p.parse_args()
    print(prepare(a.project, a.base_config, a.epochs, a.batch_size, a.seed))
