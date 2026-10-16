"""Preserve the selected actual DiT runtime config and isolate the 2D variant."""
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


def prepare(project, base_config, *, epochs=None, batch_size=None, height=768, patch_width=16):
    project=Path(project).expanduser().resolve()
    base_config=Path(base_config).expanduser().resolve()
    c=yaml.safe_load(base_config.read_text())
    c.pop('_paths',None)
    if c['model'].get('architecture') != 'dit':
        raise ValueError('base config must be your actual existing 1D DiT run; no guessed baseline')
    if c['data'].get('raman_axis_mode') != 'union_with_valid_mask':
        raise ValueError('preserve union_with_valid_mask')
    for key in ('prior_residual','broad_local_residual','conditioning'):
        if not c.get(key,{}).get('enabled',False):
            raise ValueError(f'base must preserve {key}')
    if not c['conditioning'].get('prior_spectrum',{}).get('enabled',False):
        raise ValueError('actual reconstruction-base conditioning is required')
    if c['diffusion']['objective']!='pred_x0':
        raise ValueError('current numerical constraints and 2D bridge require pred_x0')
    original=copy.deepcopy(c)
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    suite=project/'outputs/experiments'/('d4_28_curve2d_'+stamp)
    run=suite/'curve2d'
    run.mkdir(parents=True)
    c['project']['name']=suite.name
    c['model']['architecture']='dit_curve2d'
    from src.curve_image_codec import DEFAULT_CURVE
    c['curve_image']={**DEFAULT_CURVE,'height':height,'patch_width':patch_width}
    if epochs is not None: c['training']['number_of_epochs']=epochs
    if batch_size is not None: c['training']['batch_size']=batch_size
    if c['training']['number_of_epochs']<=0 or c['training']['batch_size']<=0:
        raise ValueError('epochs and batch size must be positive')
    c['generation']['number_of_spectra']=50
    for key,suffix in dict(output_directory='',checkpoint_directory='checkpoints',log_directory='logs',
                           generated_spectrum_directory='generated',preview_plot_directory='plots').items():
        c['output'][key]=str(run/suffix)
    source=Path(c['data']['input_directory']).expanduser()
    c['data']['input_directory']=str(source if source.is_absolute() else project/source)
    from src.configuration_loader import validate_config
    validate_config(c)
    path=run/'runtime_config.yaml'
    path.write_text(yaml.safe_dump(c,allow_unicode=True,sort_keys=False))
    # Explicit parity evidence: feature modules and original data/split/seed unchanged.
    preserved=('normalization','prior_residual','broad_local_residual','conditioning',
               'diversity_constraints','feature_peak_residual_limiter','diffusion')
    for key in preserved:
        if c.get(key)!=original.get(key): raise AssertionError(f'unexpected module change: {key}')
    if c['data'].get('split_unit')!=original['data'].get('split_unit') or c['data'].get('fixed_spectrum_split')!=original['data'].get('fixed_spectrum_split'):
        raise AssertionError('split changed')
    index=dict(version='d4_28_curve2d_v1',project=str(project),base_config=str(base_config),
               base_sha256=digest(base_config),config=str(path),config_sha256=digest(path),
               preserved_modules=list(preserved),source_hashes={})
    for folder in ('src','scripts'):
        for file in sorted((project/folder).rglob('*.py')):
            index['source_hashes'][str(file.relative_to(project))]=digest(file)
    (suite/'run_index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2))
    (Path(__file__).parent/'last_suite.txt').write_text(str(suite)+'\n')
    print('BASE_CONFIG:',base_config)
    print('Preserved residual mode:',c['broad_local_residual'].get('residual_mode','broad_local'))
    print('CURVE_CONFIG:',path)
    print('SUITE:',suite)
    return suite


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--project',default='/home/wqzheng/project')
    p.add_argument('--base-config',required=True)
    p.add_argument('--epochs',type=int)
    p.add_argument('--batch-size',type=int)
    p.add_argument('--height',type=int,default=768)
    p.add_argument('--patch-width',type=int,default=16)
    a=p.parse_args();sys.path.insert(0,str(Path(a.project).expanduser().resolve()))
    prepare(a.project,a.base_config,epochs=a.epochs,batch_size=a.batch_size,height=a.height,patch_width=a.patch_width)
