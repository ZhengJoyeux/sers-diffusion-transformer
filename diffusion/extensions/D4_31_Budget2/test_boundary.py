"""CPU regression tests; the actual 83M-parameter GPU experiment runs on the server."""
import argparse
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import DataLoader
from boundary_loss import settings,difference_losses,bounded_addition,extension_class
from d4_30 import initialized_trainer_class,verify_metadata,canonical

p=argparse.ArgumentParser(add_help=False);p.add_argument('--project',type=Path,required=True)
a,remaining=p.parse_known_args();PROJECT=a.project.resolve()
sys.path.insert(0,str(PROJECT))
from src.curve_image_codec import DEFAULT_CURVE
from src.model_builder import build_diffusion_model
torch.set_num_threads(2)

def tiny_config(height=32):
    state=dict(version='d4_28_curve_image_v1',fit_on='train_only',height=height,width=64,
        sequence_length=64,measured_length=64,y_min=-4.,y_max=4.,axis=(np.arange(64)+600.).tolist(),
        bridge=[dict(vector=[0.]*14,inverse_rate=[1.]*64,inverse_scale=.1,condition_id='test')])
    return dict(model=dict(architecture='dit_curve2d',model_dimension=32,dimension_multipliers=[1,2,4],
        channels=1,dropout=0.,dit=dict(hidden_size=32,depth=2,num_heads=4,patch_size=8,gradient_checkpointing=True)),
        curve_image={**DEFAULT_CURVE,'height':height,'patch_width':8,'state':state},
        diffusion=dict(diffusion_steps=8,sampling_steps=4,objective='pred_x0',beta_schedule='cosine',auto_normalize=False),
        data=dict(raman_axis_mode='union_with_valid_mask'),prior_residual=dict(enabled=True),
        broad_local_residual=dict(enabled=True),conditioning=dict(enabled=True,vector_size=14,prior_spectrum=dict(enabled=True)))

def batch():
    x=torch.randn(2,1,64)*.1;mask=torch.ones_like(x);mask[1,:,48:]=0
    return x*mask,dict(valid_mask=mask,condition=torch.zeros(2,14),
        prior_conditioning=torch.sin(torch.arange(64)/7)[None,None].expand(2,-1,-1)*mask)

class LossTests(unittest.TestCase):
    def test_real_sharp_peak_is_a_zero_error_target(self):
        x=torch.zeros(2,1,64);x[...,15]=1.;x[...,16]=-.4
        full,seam=difference_losses(x,x,torch.ones_like(x),16,settings())
        self.assertTrue(torch.equal(full,torch.zeros(2)))
        self.assertTrue(torch.equal(seam,torch.zeros(2)))

    def test_seam_index_is_the_15_to_16_edge(self):
        target=torch.zeros(1,1,64);mask=torch.ones_like(target)
        seam=target.clone();seam[...,16:]=.1
        interior=target.clone();interior[...,8:]=.1
        f1,s1=difference_losses(seam,target,mask,16,settings())
        f2,s2=difference_losses(interior,target,mask,16,settings())
        torch.testing.assert_close(f1,f2)
        self.assertGreater(s1.item(),0);self.assertEqual(s2.item(),0)

    def test_padding_and_masked_neighbors_never_enter_gradients(self):
        x=torch.randn(2,1,64,requires_grad=True);target=torch.randn_like(x)
        mask=torch.ones_like(x);mask[0,:,48:]=0;mask[1,:,16]=0
        with torch.no_grad(): x[0,:,48:]=float('nan');target[0,:,48:]=float('nan')
        f,s=difference_losses(x,target,mask,16,settings());(f+s).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all());self.assertEqual(x.grad[0,:,48:].abs().sum(),0)
        self.assertEqual(x.grad[1,:,16].abs().sum(),0)

    def test_capped_loss_retains_corrective_gradient(self):
        x=torch.tensor(10.,requires_grad=True);base=torch.tensor(1.)
        extra,gain=bounded_addition(base,x.square(),.05)
        self.assertAlmostEqual(extra.item(),.05,places=6)
        self.assertLess(gain.item(),1);extra.backward();self.assertGreater(x.grad.item(),0)

    def test_invalid_configuration_and_valid_nan_stop(self):
        for options in [{'weight':-1},{'maximum_ratio':.5},{'difference_scale':0},{'weight':float('nan')},{'unknown':1}]:
            with self.assertRaises(ValueError): settings(options)
        x=torch.zeros(1,1,64);x[...,2]=float('nan')
        with self.assertRaises(ValueError): difference_losses(x,torch.zeros_like(x),torch.ones_like(x),16,settings())

class ModelTests(unittest.TestCase):
    def make(self,weight):
        module=importlib.import_module('src.curve_image_diffusion');original=module.CurveImageGaussianDiffusion2D
        with patch.object(module,'CurveImageGaussianDiffusion2D',extension_class(original,{'weight':weight})):
            return build_diffusion_model(tiny_config(height=32),64)

    def test_disabled_extension_is_bitwise_original_and_preserves_rng(self):
        cfg=tiny_config(height=32);_,original=build_diffusion_model(cfg,64)
        model,disabled=self.make(0);disabled.load_state_dict(original.state_dict(),strict=True)
        x,kw=batch();t=torch.tensor([1,5])
        torch.manual_seed(17);first=original.p_losses(x,t,**kw);rng=torch.get_rng_state()
        torch.manual_seed(17);second=disabled.p_losses(x,t,**kw)
        self.assertTrue(torch.equal(first,second));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(set(original.state_dict()),set(disabled.state_dict()))

    def test_active_loss_backpropagates_is_bounded_and_removes_hooks(self):
        model,diff=self.make(.1);x,kw=batch();t=torch.tensor([1,5])
        with torch.no_grad(): model.final_linear.weight.normal_(0,.01)
        loss=diff.p_losses(x,t,**kw);loss.backward();metrics=diff.get_latest_loss_components()
        self.assertGreater(metrics['curve_boundary_aux_loss'],0)
        self.assertAlmostEqual(metrics['curve_boundary_aux_ratio'],.02,places=5)
        self.assertGreater(model.final_linear.weight.grad.abs().sum(),0)
        self.assertFalse(model._forward_hooks)
        vector=diff.p_losses(x,t,loss_reduction='none',**kw)
        self.assertEqual(vector.shape,(2,));self.assertTrue(torch.isfinite(vector).all())

    def test_hook_removed_on_original_error(self):
        model,diff=self.make(.1);x,kw=batch();kw['full_spectrum_target']=torch.ones_like(x)*10
        with self.assertRaises(ValueError): diff.p_losses(x,torch.tensor([1,2]),**kw)
        self.assertFalse(model._forward_hooks)

    def test_checkpoint_loads_in_original_generation_class(self):
        _,extended=self.make(.1);_,original=build_diffusion_model(tiny_config(height=32),64)
        original.load_state_dict(extended.state_dict(),strict=True)
        x,kw=batch();original.eval();extended.eval()
        torch.manual_seed(5);first=original.sample(batch_size=2,**kw)
        torch.manual_seed(5);second=extended.sample(batch_size=2,**kw)
        self.assertTrue(torch.equal(first,second))

    def test_initializer_rejects_changed_split(self):
        with self.assertRaisesRegex(ValueError,'training_indices'):
            verify_metadata({'training_indices':[1,2]},{'training_indices':[1,3]})

    def test_real_trainer_starts_from_selected_weights_with_fresh_optimizer_and_ema(self):
        from src.ddpm_trainer import DdpmTrainer
        from src.checkpoint_manager import CheckpointManager
        from src.training_logger import TrainingLogger
        cfg=tiny_config(height=32);_,old=build_diffusion_model(cfg,64)
        checkpoint=dict(configuration=copy.deepcopy(cfg),metadata={},diffusion_state=old.state_dict(),
                        ema_state={'ema_model':copy.deepcopy(old.state_dict())})
        key=next(k for k,v in checkpoint['ema_state']['ema_model'].items() if k.endswith('final_linear.bias'))
        checkpoint['ema_state']['ema_model'][key].fill_(.125)
        model,diff=self.make(.1)
        cfg['training']=dict(total_training_steps=2,gradient_accumulation_steps=1,maximum_gradient_norm=1.,
            log_every_steps=1,validate_every_steps=1,checkpoint_every_steps=1,learning_rate=1e-4,
            use_mixed_precision=False,ema_decay=.9,ema_update_every=1)
        x,kw=batch();rows=[dict(spectrum=x[i],**{k:v[i] for k,v in kw.items()}) for i in range(2)]
        loader=DataLoader(rows,batch_size=2)
        klass=initialized_trainer_class(DdpmTrainer,checkpoint,{'initial_model_source':'ema','version':'test'})
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            trainer=klass(diffusion=diff,training_loader=loader,validation_loader=loader,device=torch.device('cpu'),
                configuration=cfg,metadata={},checkpoint_manager=CheckpointManager(root/'checkpoints'),
                logger=TrainingLogger(root/'training.csv'))
            self.assertTrue(torch.equal(trainer.diffusion.state_dict()[key],checkpoint['ema_state']['ema_model'][key]))
            self.assertTrue(torch.equal(trainer.ema.ema_model.state_dict()[key],checkpoint['ema_state']['ema_model'][key]))
            self.assertEqual(trainer.starting_step,0);self.assertEqual(len(trainer.optimizer.state),0)
            trainer.train()
            self.assertTrue((root/'checkpoints/latest.pt').is_file())
            saved=torch.load(root/'checkpoints/latest.pt',weights_only=False)
            self.assertEqual(saved['step'],2)
            self.assertTrue(saved['metadata']['d4_30']['fresh_optimizer'])

class PreparationTests(unittest.TestCase):
    def fixture(self,root):
        import importlib.util
        # Reuse the already-tested two-condition real XLSX split fixture when available.
        path=Path(__file__).resolve().parents[1]/'D4_29_CurveDiagnostics/test_diagnostics.py'
        if not path.is_file(): self.skipTest('Optional D4.29 fixture is not next to this package')
        sys.path.insert(0,str(path.parent))
        spec=importlib.util.spec_from_file_location('d429_fixture',path);m=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m);m.PROJECT=PROJECT
        suite,generated,_,_=m.fixture(root)
        checkpoint_path=suite/'curve2d/checkpoints/best.pt'
        checkpoint=torch.load(checkpoint_path,weights_only=False)
        checkpoint['configuration']['curve_image']['state']['bridge'][0]['condition_id']=np.str_(
            checkpoint['configuration']['curve_image']['state']['bridge'][0]['condition_id'])
        torch.save(checkpoint,checkpoint_path)
        pointer_path=suite/'curve2d/generation_result_n50_prior_bounds.json'
        pointer=json.loads(pointer_path.read_text());pointer['checkpoint_sha256']=m.sha256(checkpoint_path)
        m.write_json(pointer_path,pointer)
        (generated/'generation_manifest.json').write_text(json.dumps({'model_source':'ema'}))
        index=json.loads((suite/'run_index.json').read_text());cp=Path(index['config'])
        import yaml
        cfg=yaml.safe_load(cp.read_text());cfg['curve_image'].pop('state')
        cfg['generation']={'number_of_spectra':50,'model_source':'raw'}
        cp.write_text(yaml.safe_dump(cfg));index['config_sha256']=m.sha256(cp)
        m.write_json(suite/'run_index.json',index)
        return suite

    def test_preparation_keeps_fitted_state_uses_actual_generation_source_and_isolates_arms(self):
        import yaml
        from d4_30 import prepare,load_arm
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);suite=self.fixture(root);out=root/'ablation'
            before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
            args=argparse.Namespace(project=PROJECT,suite=suite,output=out,epochs=5,learning_rate=1e-5)
            prepare(args)
            plan=json.loads((out/'plan.json').read_text());self.assertEqual(plan['initial_model_source'],'ema')
            left=yaml.safe_load((out/'control/runtime_config.yaml').read_text())
            right=yaml.safe_load((out/'boundary/runtime_config.yaml').read_text())
            self.assertIn('state',left['curve_image'])
            self.assertEqual(left['training'],right['training'])
            self.assertEqual(left['curve_image'],right['curve_image'])
            self.assertEqual(left['d4_30']['loss']['weight'],0)
            self.assertEqual(right['d4_30']['loss']['weight'],.1)
            self.assertNotEqual(left['output'],right['output'])
            load_arm(out,'control');load_arm(out,'boundary')
            for p,data in before.items(): self.assertEqual(p.read_bytes(),data)
            with self.assertRaises(FileExistsError): prepare(args)
            real=next((root/'real').glob('*.xlsx'));real.write_bytes(real.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'Real input file changed'): load_arm(out,'boundary')

if __name__=='__main__': unittest.main(argv=[sys.argv[0],*remaining])
