"""Generation-only compatibility launcher; leaves project source/weights/config intact."""
import argparse
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import zipfile
import yaml
from bounded_prior import bounded_prior_context


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, value):
        for stream in self.streams: stream.write(value); stream.flush()
        return len(value)
    def flush(self):
        for stream in self.streams: stream.flush()
    def isatty(self): return False


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


@contextmanager
def original_cli(project, module, argv, log):
    original_cwd, original_argv = Path.cwd(), sys.argv
    try:
        os.chdir(project)
        sys.argv = [module, *map(str, argv)]
        with Path(log).open('a', encoding='utf-8') as stream:
            with redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
                yield
    finally:
        sys.argv = original_argv
        os.chdir(original_cwd)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--phase', choices=('check','generate','evaluate','pack'), default='generate')
    p.add_argument('--physical-gpu', default='0')
    p.add_argument('--number', type=int, default=50)
    p.add_argument('--max-redraws', type=int, default=100)
    a = p.parse_args()
    if a.number < 2 or a.max_redraws < 1: p.error('number>=2 and max-redraws>=1 required')
    os.environ['CUDA_VISIBLE_DEVICES'] = a.physical_gpu
    os.environ['PYTHONUNBUFFERED'] = '1'
    suite = a.suite.expanduser().resolve()
    index = json.loads((suite/'run_index.json').read_text())
    project = Path(index['project'])
    folder = suite/'curve2d'
    config_path = Path(index['config'])
    if digest(config_path) != index['config_sha256']:
        raise ValueError('runtime config changed; launcher will not bypass its hash check')
    for relative, expected in index['source_hashes'].items():
        if digest(project/relative) != expected:
            raise ValueError(f'project source changed: {relative}; no hash guard bypass')
    runtime = yaml.safe_load(config_path.read_text())
    checkpoint = Path(runtime['output']['checkpoint_directory'])/runtime['output'].get('best_checkpoint_name','best.pt')
    pointer = folder/f'generation_result_n{a.number}_prior_bounds.json'
    sys.path.insert(0, str(project))
    if a.phase == 'pack':
        output = suite/f'D4_28_n{a.number}_prior_bounds_reports.zip'
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(suite.rglob('*')):
                if file.is_file() and file.suffix in {'.json','.jsonl','.yaml','.csv','.log','.png','.txt'} and 'checkpoints' not in file.relative_to(suite).parts:
                    archive.write(file,str(file.relative_to(suite)))
        print('REPORT_PACKAGE:',output); return
    if a.phase == 'evaluate':
        result = json.loads(pointer.read_text())
        if digest(checkpoint) != result['checkpoint_sha256']: raise ValueError('checkpoint changed since generation')
        module = 'scripts.evaluate_d4_2_generation'
        args = ['--config',config_path,'--checkpoint',checkpoint,'--condition','all',
                '--generated',result['output_directory'],'--reference-split','validation',
                '--output-directory',folder/f'evaluation_n{a.number}_prior_bounds']
        with original_cli(project,module,args,folder/f'evaluation_n{a.number}_prior_bounds_console.log'):
            runpy.run_module(module,run_name='__main__')
        print('Completed: evaluate'); return
    import torch
    from src.conditional_prior_residual import ConditionalPriorResidualBank
    from src.spectrum_conditioning import source_named_conditions_from_metadata
    saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
    trained = saved['configuration']
    meta = saved['metadata']
    if trained['model'].get('architecture') != 'dit_curve2d': raise ValueError('expected trained 2D curve checkpoint')
    state = trained['curve_image']['state']
    if state.get('fit_on') != 'train_only': raise ValueError('invalid curve coordinate provenance')
    if runtime.get('generation',{}).get('joint_prior_sampling',{}).get('enabled',False):
        raise ValueError('joint prior sampling is not supported by this targeted fix')
    records = source_named_conditions_from_metadata(meta['conditioning_metadata'])
    names = {cid:name for name,cid,_ in records}
    lower, upper = state['y_min'], state['y_max']
    print(f'CHECKPOINT: {checkpoint}\nFIXED Y-AXIS: [{lower}, {upper}]\n'
          f'POLICY: whole-prior rejection sampling; image coordinates and weights unchanged\n'
          f'NUMBER: {a.number} per condition; physical GPU={a.physical_gpu}',flush=True)
    if a.phase == 'check':
        bank = ConditionalPriorResidualBank.from_state_dict(meta['conditional_prior_residual_state'])
        generation = runtime.get('generation',{})
        bank.apply_score_clip_runtime_override(generation.get('pca_score_clip_override_standard_deviations'))
        seed = int(runtime.get('random',{}).get('seed',runtime.get('project',{}).get('random_seed',2026)))
        batch_size = int(generation.get('batch_size',10))
        import numpy as np
        output = folder/f'prior_bounds_check_n{a.number}.json'
        with bounded_prior_context(ConditionalPriorResidualBank,lower,upper,max_redraws=a.max_redraws,names=names) as sampler:
            try:
                for i,(_,cid,_) in enumerate(records):
                    prng=np.random.default_rng(seed+i); brng=np.random.default_rng(seed+i+1_000_003)
                    for start in range(0,a.number,batch_size):
                        bank.sample_generation_conditioning(min(batch_size,a.number-start),condition_id=cid,
                            prior_random_generator=prng,broad_random_generator=brng)
            finally:
                write_json(output,sampler.report())
            report = sampler.report()
        print('PRIOR_BOUNDS_CHECK_PASSED:',output)
        print('Returned:',report['returned'],'; rejected:',report['rejected'],'; proposals:',report['proposed'])
        return
    del saved, meta, trained
    if not (folder/'training_completed.json').exists(): raise ValueError('training completion marker missing')
    if pointer.exists(): raise FileExistsError(f'completed bounded-prior result already exists: {pointer}')
    root = folder/(f'generated_n{a.number}_prior_bounds_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    root.mkdir()
    audit_path = root/'prior_bounds_audit.json'
    provenance = dict(suite=str(suite),checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
                      config_sha256=digest(config_path),launcher_sha256=digest(__file__),
                      bounded_sampler_sha256=digest(Path(__file__).parent/'bounded_prior.py'),
                      number_per_condition=a.number,physical_gpu=a.physical_gpu,status='running')
    write_json(root/'compatibility_run.json',provenance)
    module='scripts.generate_conditional_spectra'
    args=['--config',config_path,'--checkpoint',checkpoint,'--condition','all','--number',a.number,
          '--model-source','ema','--output-directory',root,'--save-stage-trace']
    print('OUTPUT_DIRECTORY:',root,flush=True)
    with bounded_prior_context(ConditionalPriorResidualBank,lower,upper,max_redraws=a.max_redraws,
                              names=names,callback=lambda report:write_json(audit_path,report)) as sampler:
        try:
            with original_cli(project,module,args,folder/f'generation_n{a.number}_prior_bounds_console.log'):
                runpy.run_module(module,run_name='__main__')
            manifest=json.loads((root/'generation_manifest.json').read_text())
            generated=manifest.get('conditions',[])
            if (manifest.get('number_of_spectra_per_condition') != a.number
                    or manifest.get('model_source') != 'ema'
                    or {row['internal_condition_id'] for row in generated} != set(names)):
                raise RuntimeError('generation manifest does not match requested number/conditions/model source')
            for row in generated:
                if not row.get('output_files') or not all(Path(file).is_file() and Path(file).stat().st_size>0 for file in row['output_files']):
                    raise RuntimeError('generated output files missing or empty')
        except BaseException as error:
            provenance.update(status='failed',error=f'{type(error).__name__}: {error}')
            raise
        else:
            provenance['status']='complete'
        finally:
            write_json(audit_path,sampler.report())
            write_json(root/'compatibility_run.json',provenance)
    # Reaching here means the original script, including final-delivery guard,
    # completed successfully. Retain a separate pointer for explicit evaluation.
    write_json(pointer,dict(output_directory=str(root),checkpoint=str(checkpoint),
               checkpoint_sha256=provenance['checkpoint_sha256'],number_per_condition=a.number,
               prior_bounds_audit=str(audit_path),policy='whole-prior rejection sampling'))
    print('Completed: generate\nRESULT_POINTER:',pointer)


if __name__=='__main__': main()
