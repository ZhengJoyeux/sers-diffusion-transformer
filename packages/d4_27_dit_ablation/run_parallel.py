"""Independent one-GPU experiments with a barrier before generation."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def worker_commands(suite, phase, gpu_map, resume=False):
    runner = Path(__file__).parent / 'run_experiment.py'
    commands = {}
    for variant, gpu in gpu_map.items():
        command = [sys.executable, '-u', str(runner), '--suite', str(suite), '--phase', phase,
                   '--variants', variant, '--physical-gpu', str(gpu)]
        if resume and phase == 'train':
            command.append('--resume')
        commands[variant] = command
    return commands


def run_phase(suite, phase, gpu_map, resume=False, cpu_threads=8):
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    workers = {}
    env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS=str(cpu_threads),
               MKL_NUM_THREADS=str(cpu_threads), OPENBLAS_NUM_THREADS=str(cpu_threads))
    status = dict(phase=phase, started=stamp, workers={})
    status_path = suite / ('parallel_' + phase + '_' + stamp + '.json')
    try:
        for variant, command in worker_commands(suite, phase, gpu_map, resume).items():
            folder = suite / variant
            log = folder / ('parallel_' + phase + '_' + stamp + '.log')
            stream = log.open('w')
            try:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                           env=env, start_new_session=True)
            finally:
                stream.close()
            workers[variant] = process
            status['workers'][variant] = dict(physical_gpu=gpu_map[variant], pid=process.pid,
                                              log=str(log), command=command, exit_code=None)
            print(f'{variant}: GPU {gpu_map[variant]}, PID {process.pid}, log={log}', flush=True)
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2))
        announced = set()
        while any(process.poll() is None for process in workers.values()):
            for variant, process in workers.items():
                if process.poll() is not None and variant not in announced:
                    print(f'{phase}: {variant} exited with {process.returncode}', flush=True)
                    announced.add(variant)
            time.sleep(.5)
        for variant, process in workers.items():
            status['workers'][variant]['exit_code'] = process.returncode
        status['finished'] = datetime.now().isoformat()
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2))
        failures = [variant for variant, process in workers.items() if process.returncode]
        if failures:
            raise RuntimeError(f'{phase} failed: {failures}; see {status_path}; subsequent phases did not start')
        print(f'Both experiments completed: {phase}', flush=True)
    finally:
        # Stop only the process groups launched by this invocation if interrupted.
        for process in workers.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in workers.values():
            if process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--suite', type=Path)
    p.add_argument('--phase', required=True, choices=('check', 'train', 'generate', 'train_generate', 'evaluate'))
    p.add_argument('--broad-local-gpu', default='0')
    p.add_argument('--full-residual-gpu', default='1')
    p.add_argument('--cpu-threads', type=int, default=8)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.broad_local_gpu == a.full_residual_gpu or a.cpu_threads <= 0:
        p.error('use two distinct physical GPU IDs and a positive cpu thread count')
    suite = (a.suite or Path((Path(__file__).parent / 'last_suite.txt').read_text().strip())).expanduser().resolve()
    index = json.loads((suite / 'run_index.json').read_text())
    if set(index['configs']) != {'broad_local', 'full_residual'}:
        raise ValueError('suite must contain both residual variants')
    gpu_map = dict(broad_local=a.broad_local_gpu, full_residual=a.full_residual_gpu)
    phases = ('train', 'generate') if a.phase == 'train_generate' else (a.phase,)
    if a.dry_run:
        print(json.dumps(dict(suite=str(suite), phases=[dict(phase=phase, commands=worker_commands(suite, phase, gpu_map, a.resume))
                                                    for phase in phases]), ensure_ascii=False, indent=2))
        return
    with (suite / 'parallel.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('this suite already has an active parallel runner') from None
        for phase in phases:
            run_phase(suite, phase, gpu_map, a.resume, a.cpu_threads)


if __name__ == '__main__':
    main()
