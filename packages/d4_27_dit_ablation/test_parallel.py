import json
from pathlib import Path
import sys
import pytest
import run_parallel


def make_suite(tmp_path):
    for variant in ('broad_local', 'full_residual'):
        (tmp_path / variant).mkdir()
    (tmp_path / 'run_index.json').write_text(json.dumps({'configs': {'broad_local': {}, 'full_residual': {}}}))
    return tmp_path


def test_mapping_and_single_variant_commands():
    commands = run_parallel.worker_commands(Path('/tmp/suite'), 'train', {'broad_local': '0', 'full_residual': '1'}, True)
    for variant, gpu in [('broad_local', '0'), ('full_residual', '1')]:
        command = commands[variant]
        assert command[command.index('--variants') + 1] == variant
        assert command[command.index('--physical-gpu') + 1] == gpu
        assert '--resume' in command


def fake_commands(tmp_path, fail=False):
    script = tmp_path / 'worker.py'
    script.write_text('''import json,sys,time
from pathlib import Path
root,phase,variant,fail=sys.argv[1:]
start=time.time()
time.sleep(.4)
Path(root, phase+'_'+variant+'.json').write_text(json.dumps(dict(start=start,end=time.time())))
sys.exit(7 if fail=='yes' and phase=='train' and variant=='broad_local' else 0)
''')
    def commands(suite, phase, gpu_map, resume=False):
        return {variant: [sys.executable, str(script), str(suite), phase, variant, 'yes' if fail else 'no'] for variant in gpu_map}
    return commands


def test_simultaneous_workers_and_barrier(tmp_path, monkeypatch):
    suite = make_suite(tmp_path)
    monkeypatch.setattr(run_parallel, 'worker_commands', fake_commands(tmp_path))
    monkeypatch.setattr(sys, 'argv', ['run_parallel.py', '--suite', str(suite), '--phase', 'train_generate'])
    run_parallel.main()
    records = {(phase, variant): json.loads((suite / (phase + '_' + variant + '.json')).read_text())
               for phase in ('train', 'generate') for variant in ('broad_local', 'full_residual')}
    for phase in ('train', 'generate'):
        a, b = records[phase, 'broad_local'], records[phase, 'full_residual']
        assert max(a['start'], b['start']) < min(a['end'], b['end'])
    assert min(records['generate', v]['start'] for v in ('broad_local','full_residual')) >= max(records['train', v]['end'] for v in ('broad_local','full_residual'))


def test_failed_training_blocks_generation(tmp_path, monkeypatch):
    suite = make_suite(tmp_path)
    monkeypatch.setattr(run_parallel, 'worker_commands', fake_commands(tmp_path, True))
    monkeypatch.setattr(sys, 'argv', ['run_parallel.py', '--suite', str(suite), '--phase', 'train_generate'])
    with pytest.raises(RuntimeError, match='train failed'):
        run_parallel.main()
    assert not list(suite.glob('generate_*.json'))
    assert (suite / 'train_full_residual.json').exists()
