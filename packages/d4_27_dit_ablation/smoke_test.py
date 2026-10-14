"""Two synthetic conditions, real CLI training/export. Never uses real data."""
import argparse
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import yaml
from prepare_run import prepare


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project', required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--physical-gpu', default='1')
    a = p.parse_args()
    project = Path(a.project).expanduser().resolve()
    folder = project / 'outputs/experiments' / ('d4_27_synthetic_smoke_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    data = folder / 'data'; data.mkdir(parents=True)
    rng = np.random.default_rng(2026)
    for name, length in [('DEL-H_water', 1901), ('CHL-M_TEB-M_water', 1401)]:
        axis = np.arange(length) + 600
        curves = []
        for i in range(20):
            y = 50 + 10 * np.sin(axis / 160 + rng.normal(0, .04))
            for center in (735, 1000, 1250, 1600):
                y += rng.uniform(50, 100) * np.exp(-.5 * ((axis - center - rng.normal(0, 1)) / rng.uniform(5, 8)) ** 2)
            y += rng.normal(0, 2, length)
            curves.append(y)
        frame = pd.DataFrame(np.column_stack([axis, *curves]), columns=['Raman shift', *[f'spectrum_{i+1}' for i in range(20)]])
        frame.to_excel(data / (name + '.xlsx'), index=False)
    c = yaml.safe_load((Path(__file__).parent / 'reference_D4_26_runtime.yaml').read_text())
    c['data']['input_directory'] = str(data)
    c['training']['device'] = a.device
    c['training']['pin_memory'] = a.device == 'cuda'
    c['training']['validate_every_epochs'] = 1
    c['training']['save_every_epochs'] = 1
    c['training']['log_every_batches'] = 1
    c['diffusion']['diffusion_steps'] = c['model']['diffusion_timesteps'] = 8
    c['diffusion']['sampling_steps'] = c['model']['sampling_timesteps'] = 4
    base = folder / 'base.yaml'; base.write_text(yaml.safe_dump(c, allow_unicode=True, sort_keys=False))
    suite = prepare(project, base, epochs=1, batch_size=4)
    index = json.loads((suite / 'run_index.json').read_text())
    import hashlib
    for spec in index['configs'].values():
        path = Path(spec['path']); config = yaml.safe_load(path.read_text())
        config['model']['dit'].update(hidden_size=32, depth=2, num_heads=4)
        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
        spec['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (suite / 'run_index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2))
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    runner = Path(__file__).parent / 'run_experiment.py'
    for phase in ('check', 'train_generate'):
        subprocess.run([sys.executable, str(runner), '--suite', str(suite), '--phase', phase,
                        '--physical-gpu', a.physical_gpu], env=env, check=True)
    verify(suite)



def verify(suite):
    suite = Path(suite)
    index = json.loads((suite / 'run_index.json').read_text())
    for variant in index['configs']:
        files = list((suite / variant / 'generated').rglob('*_generated.xlsx'))
        assert len(files) == 2, files
        for file in files:
            frame = pd.read_excel(file)
            assert frame.shape[1] == 101, frame.shape
            expected_length = 1401 if file.stem.startswith('CHL-M_TEB-M') else 1901
            assert frame.shape[0] == expected_length, frame.shape
        import torch
        metadata = torch.load(suite / variant / 'checkpoints/best.pt', map_location='cpu', weights_only=False)['metadata']
        for entry in metadata['conditional_prior_residual_state']['conditions'].values():
            assert entry['broad_local_residual_state']['residual_mode'] == variant
    for path in (suite / 'full_residual/generated/_diagnostics').rglob('stage_cache.npz'):
        with np.load(path) as arrays:
            assert not arrays['generated_broad'].any()
            np.testing.assert_array_equal(arrays['00_sampled_base'], arrays['generated_outer'])
    print('SYNTHETIC SMOKE PASSED:', suite)

if __name__ == '__main__':
    main()
