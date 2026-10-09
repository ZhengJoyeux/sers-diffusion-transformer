"""Package controller tests use temporary projects/synthetic traces, not real quality evidence."""
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np
import yaml

PACKAGE=Path(__file__).resolve().parent
SOURCE=Path(os.environ.get('D4_26_TEST_PROJECT',str(PACKAGE.parent/'d425_source')))
if not SOURCE.is_dir():SOURCE=Path('/home/wqzheng/project')
sys.path.insert(0,str(SOURCE))
import src.generation_stage_trace

def module(name):
    spec=importlib.util.spec_from_file_location(name,PACKAGE/(name+'.py'))
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result

prepare=module('prepare_run');runner=module('run_experiment')

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.project=self.root/'project';self.project.mkdir()
        for folder in ('src','scripts','tests'):
            shutil.copytree(SOURCE/folder,self.project/folder,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
        self.specs=json.loads((PACKAGE/'file_hashes.json').read_text())

    def tearDown(self):self.temp.cleanup()

    def call(self,script,*args):
        return subprocess.run([sys.executable,str(PACKAGE/script),*map(str,args)],text=True,capture_output=True)

    def original(self):
        for relative,spec in self.specs.items():
            target=self.project/relative
            if spec['before'] is None:target.unlink()
            else:shutil.copy2(PACKAGE/'test_fixtures/originals'/relative,target)

    def fixture(self):
        exp=self.project/'outputs/experiments/old';(exp/'checkpoints').mkdir(parents=True)
        (exp/'checkpoints/best.pt').write_bytes(b'test fixture checkpoint bytes - never trained')
        cfg=yaml.safe_load((PACKAGE/'test_fixtures/runtime_config.yaml').read_text())
        cfg['data']['input_directory']=str(self.project/'data/input')
        (exp/'runtime_config.yaml').write_text(yaml.safe_dump(cfg))
        paired=exp/'paired';paired.mkdir()
        g=copy.deepcopy(cfg)
        g['generation']['joint_prior_sampling']={'enabled':False,'cross_correlation_strength':.5}
        g['generation']['final_delivery_guard']={'enabled':True}
        gp=paired/'A_independent.yaml';gp.write_text(yaml.safe_dump(g))
        conditions=['DEL-H_water','CHL-M_TEB-M_water','DEL-S_TEB-S_CHL-S_soil']
        (paired/'joint_prior_report.json').write_text(json.dumps([{'condition':c,'paired_noise_states_match':True,
            'outer_draws_exactly_match':True,'training_references_exactly_match':True} for c in conditions]))
        (paired/'joint_review_index.json').write_text(json.dumps({'checkpoint':str(exp/'checkpoints/best.pt'),
            'checkpoint_sha256':digest(exp/'checkpoints/best.pt'),'config_sha256':{'A_independent':digest(gp)}}))
        # Redirect last_suite marker to temporary package without changing production package.
        fake=self.root/'package';fake.mkdir();shutil.copy2(PACKAGE/'file_hashes.json',fake/'file_hashes.json')
        with mock.patch.object(prepare,'__file__',str(fake/'prepare_run.py')):
            suite=prepare.prepare(self.project,exp,paired)
        return suite,exp,paired,cfg,g

    def test_apply_idempotent_and_rollback_exact(self):
        self.original()
        before={r:(self.project/r).read_bytes() if (self.project/r).is_file() else None for r in self.specs}
        result=self.call('apply_fix.py','--project',self.project);self.assertEqual(result.returncode,0,result.stderr)
        backups=list((self.project/'outputs/upgrades').glob('d4_26_hybrid_*'));self.assertEqual(len(backups),1)
        again=self.call('apply_fix.py','--project',self.project);self.assertEqual(again.returncode,0,again.stderr)
        self.assertIn('already applied',again.stdout)
        rolled=self.call('rollback_fix.py','--project',self.project,'--backup',backups[0]);self.assertEqual(rolled.returncode,0,rolled.stderr)
        for r,value in before.items():
            self.assertEqual((self.project/r).read_bytes() if (self.project/r).is_file() else None,value)

    def test_unknown_source_rejected_before_any_mutation(self):
        self.original();q=self.project/'src/masked_unet.py';q.write_text(q.read_text()+'\n# unknown edit\n')
        before={r:(self.project/r).read_bytes() if (self.project/r).is_file() else None for r in self.specs}
        result=self.call('apply_fix.py','--project',self.project);self.assertNotEqual(result.returncode,0)
        for r,value in before.items():self.assertEqual((self.project/r).read_bytes() if (self.project/r).is_file() else None,value)
        self.assertFalse((self.project/'outputs/upgrades').exists())

    def test_rollback_rejects_changed_code(self):
        self.original();result=self.call('apply_fix.py','--project',self.project);self.assertEqual(result.returncode,0)
        backup=next((self.project/'outputs/upgrades').glob('*'));q=self.project/'src/masked_unet.py';q.write_text(q.read_text()+'\n# later edit\n')
        before={r:(self.project/r).read_bytes() for r in self.specs}
        result=self.call('rollback_fix.py','--project',self.project,'--backup',backup);self.assertNotEqual(result.returncode,0)
        for r,value in before.items():self.assertEqual((self.project/r).read_bytes(),value)

    def test_prepare_preserves_training_and_independent_generation(self):
        suite,exp,paired,old,g=self.fixture();index,project=runner.load_suite(suite)
        from src.configuration_loader import validate_config
        validate_config(old)
        self.assertEqual(index['baseline_checkpoint'],str(exp/'checkpoints/best.pt'))
        for variant in ('baseline','self','cross'):
            cfg=yaml.safe_load(Path(index['configs'][variant]).read_text())
            self.assertEqual(cfg['generation'],g['generation']);self.assertEqual(cfg['training'],old['training'])
            for section in ('data','prior_residual','broad_local_residual','diffusion','conditioning','diversity_constraints'):
                self.assertEqual(cfg[section],old[section])
        self.assertFalse(yaml.safe_load(Path(index['configs']['baseline']).read_text())['model']['bottleneck_transformer']['enabled'])
        self.assertTrue(yaml.safe_load(Path(index['configs']['cross']).read_text())['model']['bottleneck_transformer']['cross_attention'])
        self.assertFalse(yaml.safe_load(Path(index['configs']['self']).read_text())['model']['bottleneck_transformer']['cross_attention'])
        config=Path(index['configs']['cross']);config.write_text(config.read_text()+'\n# changed\n')
        with self.assertRaisesRegex(ValueError,'配置'):runner.load_suite(suite)

    def traces(self,suite,index):
        from src.generation_stage_trace import StageTrace
        random=np.random.default_rng(8)
        axis=np.arange(600,641,dtype=float);train=random.normal(size=(12,41)).astype('float32')
        outer=random.normal(size=(200,41)).astype('float32');broad=random.normal(size=(200,41)).astype('float32');full=outer+broad
        for variant in ('baseline','cross'):
            cp=Path(index['baseline_checkpoint']) if variant=='baseline' else suite/variant/'checkpoints/best.pt'
            cp.parent.mkdir(exist_ok=True);cp.write_bytes(b'synthetic checkpoint fixture')
            cfg=Path(index['configs'][variant]);g=yaml.safe_load(cfg.read_text())['generation']
            for condition in index['conditions']:
                root=suite/'pilot_review/generated'/variant/condition;root.mkdir(parents=True)
                trace=StageTrace(root/'_diagnostics'/condition,axis,200,{'source_condition_name':condition,
                    'checkpoint_step':4347,'training_count':12,'model_source':'ema','seed':2026,
                    'diffusion_timesteps':200,'sampling_timesteps':100,
                    'sampling_rng_fingerprints':[{'batch_size':10,'cuda':'a'*64,'cpu':'b'*64} for _ in range(20)],
                    'generation_configuration':g,'delivery_configuration':{'enabled':False},'support':None,
                    'condition_vector':[0.]*14,'normalization_state':{},'training_indices':list(range(12))})
                for key in ('training_full','training_outer','training_broad','training_base'):trace.reference(key,train)
                trace.reference('generated_outer',outer);trace.reference('generated_broad',broad)
                for stage in ('00_sampled_base','01_ddpm_reconstruction','02_pca_spread','03_qq','04_mean',
                              '05_legacy_envelope','06_final_support','07_terminal'):trace.capture(stage,full)
                trace.save()
                (root/'hybrid_provenance.json').write_text(json.dumps({'checkpoint_sha256':digest(cp),
                    'config_sha256':digest(cfg),'checkpoint_step':4347}))
                (root/'generation_manifest.json').write_text(json.dumps({'conditions':[{}]}))
        return train,full

    def evaluator(self,command,project,environment,log):
        self.assertIn('scripts.evaluate_d4_2_generation',command)
        out=Path(command[command.index('--output-directory')+1]);out.mkdir(parents=True,exist_ok=True)
        condition=command[command.index('--condition')+1]
        row={'condition':condition,'reference_split':'validation','generated_count':200,**{k:.8 for k in runner.METRICS}}
        runner.write_csv(out/'condition_metrics.csv',[row])
        runner.write_csv(out/'generated_to_reference_details.csv',[{'first_derivative_pearson':.9}]*200)

    def test_review_stage_exports_replay_pairing_and_cached_evaluation(self):
        suite,exp,paired,old,g=self.fixture();index=json.loads((suite/'run_index.json').read_text());self.traces(suite,index)
        # Fixtures skip envelope replay; this tests orchestration, not guard efficacy.
        for condition in index['conditions']:
            for variant in ('baseline','cross'):
                path=suite/'pilot_review/generated'/variant/condition/'_diagnostics'/condition/'trace_manifest.json'
                trace=json.loads(path.read_text());trace['generation_configuration']['intensity_envelope_guard']['enabled']=False
                path.write_text(json.dumps(trace))
        with mock.patch.object(runner,'validate_checkpoint',return_value=4347),mock.patch.object(runner,'run_logged',side_effect=self.evaluator) as calls:
            runner.review(index,self.project,suite,['baseline','cross'],{},True)
            self.assertEqual(calls.call_count,12)
            runner.review(index,self.project,suite,['baseline','cross'],{},True)
            self.assertEqual(calls.call_count,12)
        with (suite/'pilot_review/hybrid_paired_comparison.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),6);self.assertTrue(all(r['paired_noise_and_base_match']=='True' for r in rows))
        self.assertEqual(len(list((suite/'pilot_review/diagnostic_exports').rglob('*.xlsx'))),12)
        # Changing the paired noise while retaining valid cache data must block a paired report.
        report=suite/'pilot_review/hybrid_paired_comparison.csv';report.unlink()
        path=suite/'pilot_review/generated/cross'/index['conditions'][0]/'_diagnostics'/index['conditions'][0]/'trace_manifest.json'
        trace=json.loads(path.read_text());trace['sampling_rng_fingerprints'][0]['cuda']='c'*64;path.write_text(json.dumps(trace))
        with mock.patch.object(runner,'validate_checkpoint',return_value=4347):
            with self.assertRaisesRegex(ValueError,'随机状态'):
                runner.review(index,self.project,suite,['baseline','cross'],{},True)
        self.assertFalse(report.exists())

    def test_metrics_nonfinite_rejected(self):
        out=self.root/'eval';out.mkdir();row={'condition':'DEL-H_water','reference_split':'validation','generated_count':200,
            **{k:.8 for k in runner.METRICS}};row[runner.METRICS[0]]=float('nan')
        runner.write_csv(out/'condition_metrics.csv',[row]);runner.write_csv(out/'generated_to_reference_details.csv',[{'first_derivative_pearson':.9}]*200)
        with self.assertRaises(ValueError):runner.metrics_from(out,'DEL-H_water')

    def test_checkpoint_budget_axis_counts_and_split_guards(self):
        suite,exp,paired,old,g=self.fixture();index=json.loads((suite/'run_index.json').read_text())
        checkpoint={'configuration':copy.deepcopy(old),'step':4347,'metadata':{
            'conditional_prior_residual_state':{},'training_indices':list(range(1512)),
            'validation_indices':list(range(1512,2016)),'test_indices':list(range(2016,2520))}}
        summary={'number_of_conditions':126,'total_training_spectra':1512,
            'training_spectra_per_condition':[12],'valid_lengths':{'1401':30,'1901':96}}
        latest={'step':4725}
        def load(path):return latest if Path(path).name=='latest.pt' else checkpoint
        with mock.patch('src.checkpoint_manager.load_checkpoint_file',side_effect=load),\
             mock.patch('src.conditional_prior_residual.ConditionalPriorResidualBank.from_state_dict') as restore:
            restore.return_value.summary.return_value=summary
            self.assertEqual(runner.validate_checkpoint(exp/'checkpoints/best.pt','baseline',index,self.project),4347)
            latest['step']=4724
            with self.assertRaisesRegex(ValueError,'4725'):runner.validate_checkpoint(exp/'checkpoints/best.pt','baseline',index,self.project)
            latest['step']=4725;summary['training_spectra_per_condition']=[11,13]
            with self.assertRaisesRegex(ValueError,'126'):runner.validate_checkpoint(exp/'checkpoints/best.pt','baseline',index,self.project)
            summary['training_spectra_per_condition']=[12];checkpoint['metadata']['test_indices'][0]=1512
            with self.assertRaisesRegex(ValueError,'不相交'):runner.validate_checkpoint(exp/'checkpoints/best.pt','baseline',index,self.project)

if __name__=='__main__':unittest.main()
