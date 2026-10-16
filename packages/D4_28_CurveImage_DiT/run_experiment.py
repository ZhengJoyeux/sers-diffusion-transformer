"""Explicit phases; never starts formal training as a side effect of check."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
import yaml


def run(command,project,env,log):
    print('Running:',' '.join(map(str,command)),flush=True)
    with Path(log).open('a',encoding='utf-8') as stream:
        process=subprocess.Popen(list(map(str,command)),cwd=project,env=env,stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            print(line,end='',flush=True);stream.write(line);stream.flush()
        if process.wait(): raise RuntimeError(f'command failed; see {log}')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--suite',type=Path)
    p.add_argument('--phase',required=True,choices=('check','train','generate','evaluate','pack'))
    p.add_argument('--physical-gpu',default='0')
    p.add_argument('--number',type=int,default=50)
    p.add_argument('--max-redraws',type=int,default=100)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.number < 2 or a.max_redraws < 1: p.error('number>=2 and max-redraws>=1 required')
    last_suite=Path(__file__).parent/'last_suite.txt'
    if a.suite is None and not last_suite.exists():
        p.error('provide --suite or run prepare_run.py first; last_suite.txt is created during preparation')
    suite=(a.suite or Path(last_suite.read_text().strip())).expanduser().resolve()
    index=json.loads((suite/'run_index.json').read_text());project=Path(index['project'])
    folder=suite/'curve2d';config=Path(index['config']);c=yaml.safe_load(config.read_text())
    if hashlib.sha256(config.read_bytes()).hexdigest()!=index['config_sha256']:
        raise ValueError('runtime config changed after preparation; prepare a separate run')
    if a.phase!='pack':
        for rel,digest in index['source_hashes'].items():
            if hashlib.sha256((project/rel).read_bytes()).hexdigest()!=digest:
                raise ValueError(f'source changed after preparation: {rel}')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.physical_gpu,PYTHONUNBUFFERED='1')
    checkpoint_dir=Path(c['output']['checkpoint_directory'])
    best=checkpoint_dir/c['output'].get('best_checkpoint_name','best.pt')
    latest=checkpoint_dir/c['output'].get('latest_checkpoint_name','latest.pt')
    if a.phase=='check':
        run([sys.executable,'-u','-m','scripts.start_ddpm_training','--config',config,'--pipeline-check-only'],
             project,env,folder/'check_console.log')
        (folder/'check_passed.json').write_text(json.dumps(dict(config_sha256=index['config_sha256'],
            source_hashes=index['source_hashes'],checked_at=datetime.now().isoformat())))
    elif a.phase=='train':
        if not (folder/'check_passed.json').exists(): raise ValueError('run --phase check successfully first')
        command=[sys.executable,'-u','-m','scripts.start_ddpm_training','--config',config]
        if latest.exists():
            if not a.resume: raise ValueError('checkpoint exists; use --resume or prepare another run')
            command+=['--resume',str(latest)]
        elif a.resume: raise FileNotFoundError(latest)
        run(command,project,env,folder/'training_console.log')
        (folder/'training_completed.json').write_text(json.dumps(dict(completed_at=datetime.now().isoformat())))
    elif a.phase in ('generate','evaluate'):
        if a.phase=='generate' and not (folder/'training_completed.json').exists():
            raise ValueError('formal training must finish before generation')
        if not best.exists(): raise FileNotFoundError(best)
        launcher=Path(__file__).resolve().parent.parent/'D4_28_PriorBounds_Fix'/'run_generation.py'
        if not launcher.is_file(): raise FileNotFoundError(launcher)
        run([sys.executable,'-u',launcher,'--suite',suite,'--phase',a.phase,
             '--physical-gpu',a.physical_gpu,'--number',str(a.number),
             '--max-redraws',str(a.max_redraws)],project,env,
             folder/f'{a.phase}_n{a.number}_launcher.log')
    else:
        output=suite/'D4_28_CurveImage_DiT_reports.zip'
        allowed={'.json','.jsonl','.yaml','.csv','.txt','.log','.png','.md'}
        with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
            for path in sorted(suite.rglob('*')):
                if not path.is_file() or path.suffix not in allowed or path.name.endswith('.zip'): continue
                if 'checkpoints' in path.relative_to(suite).parts: continue
                z.write(path,str(path.relative_to(suite)))
        print('REPORT_PACKAGE:',output)
    print('Completed:',a.phase,'; suite:',suite)


if __name__=='__main__': main()
