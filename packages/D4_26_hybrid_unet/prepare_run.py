"""Create one isolated suite; reuse the existing matched baseline checkpoint."""
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


def prepare(project, baseline, paired, epochs=25, seed=2026, batch_size=None):
    project, baseline, paired = map(lambda p: Path(p).expanduser().resolve(), (project, baseline, paired))
    config_path = baseline / 'runtime_config.yaml'
    checkpoint = baseline / 'checkpoints/best.pt'
    generation_path = paired / 'A_independent.yaml'
    paired_index = json.loads((paired / 'joint_review_index.json').read_text())
    reports = json.loads((paired / 'joint_prior_report.json').read_text())
    for report in reports:
        if not all(report.get(k) is True for k in ('paired_noise_states_match','outer_draws_exactly_match','training_references_exactly_match')):
            raise ValueError('原配对结果未通过随机状态/outer/训练参考检查。')
    if {r['condition'] for r in reports} != {'DEL-H_water','CHL-M_TEB-M_water','DEL-S_TEB-S_CHL-S_soil'}:
        raise ValueError('配对报告必须包含已验证的三个具体条件。')
    if Path(paired_index['checkpoint']).resolve() != checkpoint or digest(checkpoint) != paired_index['checkpoint_sha256']:
        raise ValueError('配对评价使用的checkpoint与指定baseline不同。')
    if digest(generation_path) != paired_index['config_sha256']['A_independent']:
        raise ValueError('独立采样配置SHA256发生变化。')
    original = yaml.safe_load(config_path.read_text())
    g = yaml.safe_load(generation_path.read_text())['generation']
    if g.get('joint_prior_sampling', {}).get('enabled', False):
        raise ValueError('新架构实验必须固定独立先验采样。')
    if not g.get('final_delivery_guard', {}).get('enabled', False):
        raise ValueError('原配对配置未开启已验证的终端保护。')
    if int(g.get('seed',seed))!=seed or int(g.get('batch_size',0))!=10:
        raise ValueError('配对生成必须固定seed2026和batch10。')
    if original['project']['random_seed'] != seed:
        raise ValueError('必须保持原seed2026。')
    batch_size = original['training']['batch_size'] if batch_size is None else batch_size
    if epochs <= 0 or batch_size <= 0:
        raise ValueError('epochs和batch_size必须为正整数。')
    if batch_size % 4:
        raise ValueError('batch_size必须为4的倍数，保持每条件4条光谱成组。')
    budget_matched = (epochs == original['training']['number_of_epochs']
                      and batch_size == original['training']['batch_size'])
    if original['data']['fixed_spectrum_split'] != {'train_count':12,'validation_count':4,'test_count':4}:
        raise ValueError('必须保持每条件12/4/4。')
    if original['model'].get('dimension_multipliers') != [1,2,4] or original['model'].get('model_dimension') != 32:
        raise ValueError('首版要求D4.25的dim32和[1,2,4]。')
    if original['diffusion'].get('diffusion_steps') != 200 or original['diffusion'].get('sampling_steps') != 100:
        raise ValueError('首版必须固定T200/S100。')
    sys.path.insert(0, str(project))
    from src.configuration_loader import validate_config
    package=Path(__file__).resolve().parent
    hashes=json.loads((package/'file_hashes.json').read_text())
    source_hashes={}
    for relative, spec in hashes.items():
        if digest(project/relative) != spec['after']:
            raise ValueError(f'补丁尚未完整应用: {relative}')
        source_hashes[relative]=spec['after']
    for folder in ('src','scripts'):
        for path in sorted((project/folder).rglob('*.py')):
            source_hashes[str(path.relative_to(project))]=digest(path)
    suite = project / 'outputs/experiments' / (f'd4_26_hybrid_e{epochs}_b{batch_size}_seed2026_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    suite.mkdir(parents=True, exist_ok=False)
    index = {'schema':'d4_26_hybrid_suite_v1', 'project':str(project), 'baseline_experiment':str(baseline),
             'baseline_checkpoint':str(checkpoint), 'baseline_checkpoint_sha256':digest(checkpoint),
             'baseline_config_sha256':digest(config_path), 'paired_review':str(paired),
             'paired_report_sha256':digest(paired/'joint_prior_report.json'), 'seed':seed,'epochs':epochs,
             'batch_size':batch_size, 'training_budget_matched':budget_matched,
             'conditions':['DEL-H_water','CHL-M_TEB-M_water','DEL-S_TEB-S_CHL-S_soil'],
             'configs':{},'config_sha256':{},'source_hashes':source_hashes,
             'decision':'Joint sampler not adopted: no consistent shape/QQ/diversity improvement. Independent sampler fixed.',
             'notes':['baseline uses existing seed2026/25epoch checkpoint; no baseline retraining',
                      'self variant is optional ablation; cross is the first candidate to train',
                      'Historical baseline keeps its original budget; candidate budget is recorded separately',
                      'all 126 conditions shared training; pilot generation is only the three conditions']}
    for variant in ('baseline','self','cross'):
        c=copy.deepcopy(original);c.pop('_paths',None);c['generation']=copy.deepcopy(g)
        c['model']['bottleneck_transformer'] = ({'enabled':False} if variant=='baseline' else {
            'enabled':True,'cross_attention':variant=='cross','depth':1,'num_heads':4,
            'ff_multiplier':2.0,'dropout':0.05,'position_frequencies':8,
            'raman_start_cm1':600.0,'raman_step_cm1':1.0,'raman_span_cm1':1900.0,'schema_version':1})
        run=suite/variant;run.mkdir()
        c['project']['name']=suite.name+'_'+variant
        input_path=Path(c['data']['input_directory']).expanduser()
        c['data']['input_directory']=str(input_path if input_path.is_absolute() else project/input_path)
        for key, suffix in {'output_directory':'','checkpoint_directory':'checkpoints','log_directory':'logs',
                            'generated_spectrum_directory':'generated','preview_plot_directory':'plots'}.items():
            c['output'][key]=str(run/suffix)
        c['training']['device']='cuda'
        if variant != 'baseline':
            c['training']['number_of_epochs'] = epochs
            c['training']['batch_size'] = batch_size
        validate_config(c)
        path=run/'runtime_config.yaml';path.write_text(yaml.safe_dump(c,allow_unicode=True,sort_keys=False))
        index['configs'][variant]=str(path);index['config_sha256'][variant]=digest(path)
    (suite/'run_index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2))
    (package/'last_suite.txt').write_text(str(suite)+'\n')
    print('已创建独立候选目录；baseline复用原checkpoint；先训练cross，self仅消融。',file=sys.stderr)
    print(f'候选预算：epochs={epochs}; batch_size={batch_size}; baseline预算匹配={budget_matched}',file=sys.stderr)
    return suite


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',required=True,type=Path)
    p.add_argument('--baseline-experiment',required=True,type=Path)
    p.add_argument('--paired-review',required=True,type=Path)
    p.add_argument('--epochs',type=int,default=25);p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--batch-size',type=int)
    a=p.parse_args();print(prepare(a.project,a.baseline_experiment,a.paired_review,a.epochs,a.seed,a.batch_size))


if __name__=='__main__': main()
