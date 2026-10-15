"""Tests for intensity preservation, masking, training and checkpoint contracts."""
import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from image_data import PlotScale, render_line, LineImageDataset, acquisition_mask, fit_training_scale
from image_model import ImageSERSFormer, LineImageCNN, warm_start
from run_image import make_optimizer, set_rates, save_checkpoint, load_checkpoint, supervision, metrics
from bridge import audit_splits

torch.set_num_threads(2)


def batch(short=False, height=64, width=192):
    x=np.arange(1901);raw=5+50*np.exp(-.5*((x-400)/8)**2)+25*np.exp(-.5*((x-1000)/7)**2)
    mask=np.ones(1901,dtype=bool)
    if short: mask[1401:]=False
    scale=PlotScale(width=width,height=height,amplitude_scale=20.,lower=-.1,upper=2.)
    image,columns,_=render_line(raw,mask,scale)
    return dict(image=torch.from_numpy(image.astype(np.float32)/255.)[None,None],
        image_valid_columns=torch.from_numpy(columns)[None],
        raw_intensity=torch.from_numpy(raw.astype(np.float32))[None,None],
        measured_mask=torch.from_numpy(mask)[None])


def args():
    return SimpleNamespace(image_lr=1e-4,head_lr=2e-5,backbone_lr=1e-5,numeric_lr=1e-5,
        from_scratch=True,minimum_ratio=.01,weight_decay=.01,relation_weight=0.,
        classification_weight=.5,concentration_weight=.5,boundary_margin=.1,
        boundary_weight=.1,ternary_multiplier=1.5,ordinal_decoding='median')


class RenderingTests(unittest.TestCase):
    def test_shared_scale_preserves_height_and_negative_values(self):
        scale=PlotScale(width=256,height=128,amplitude_scale=20.,lower=-1.,upper=3.)
        values=np.array([-10.,0.,20.,80.])
        rows=scale.row_positions(values)
        self.assertTrue(np.all(np.diff(rows)<0))
        np.testing.assert_allclose(scale.inverse_rows(rows),values,rtol=1e-6,atol=1e-6)
        low=np.full(1901,20.);high=np.full(1901,80.)
        a,_,_=render_line(low,np.ones(1901,bool),scale)
        b,_,_=render_line(high,np.ones(1901,bool),scale)
        self.assertLess(np.where(b<128)[0].mean(),np.where(a<128)[0].mean())

    def test_invalid_tail_and_gap_never_draw(self):
        scale=PlotScale(width=512,height=128,amplitude_scale=20.,lower=-1.,upper=3.)
        mask=np.ones(1901,bool);mask[1401:]=False;mask[600:650]=False
        original=np.full(1901,20.);altered=original.copy();altered[~mask]=np.nan
        a,columns,_=render_line(original,mask,scale)
        b,_,_=render_line(altered,mask,scale)
        np.testing.assert_array_equal(a,b)
        self.assertTrue((a[:,~columns]==255).all())

    def test_clipping_is_reported(self):
        scale=PlotScale(width=256,height=128,amplitude_scale=1.,lower=0.,upper=1.)
        _,_,audit=render_line(np.full(1901,100.),np.ones(1901,bool),scale)
        self.assertEqual(audit['clipped_columns'],256)

    def test_scale_fit_ignores_invalid_values(self):
        values=np.zeros((2,1901),np.float32);mask=np.ones_like(values,dtype=bool);mask[:,1401:]=False
        changed=values.copy();changed[:,1401:]=1e30
        self.assertEqual(asdict(PlotScale.fit(values,mask)),asdict(PlotScale.fit(changed,mask)))

    def test_acquisition_mask_excludes_imputed_tail(self):
        record=dict(axis=np.arange(600.,2501.),original_axis_points=1401,**{'original_axis_end_cm-1':2000.})
        dataset=SimpleNamespace(repository=SimpleNamespace(real_data={'c':record}))
        mask=acquisition_mask(dataset,'real','c',np.ones(1901,bool))
        self.assertEqual(int(mask.sum()),1401)

    def test_cache_matches_render_without_label_leak(self):
        class Original:
            samples=[SimpleNamespace(source='real',condition='c',spectrum_index=0)]
            split='train'
            repository=SimpleNamespace(real_data={'c':dict(axis=np.arange(600.,2501.),original_axis_points=1901)})
            def __len__(self):return 1
            def __getitem__(self,index):
                return dict(raw_intensity=torch.full((1,1901),20.),valid_mask=torch.ones(1901,dtype=torch.bool),
                            source='real',condition='c',class_target=torch.ones(3))
        with tempfile.TemporaryDirectory() as directory:
            scale=PlotScale(width=128,height=64,amplitude_scale=20.,lower=-1.,upper=3.)
            ds=LineImageDataset(Original(),scale,directory)
            a,b=ds[0],ds[0]
            torch.testing.assert_close(a['image'],b['image'],rtol=0,atol=0)
            self.assertEqual(a['image_clipped_fraction'],b['image_clipped_fraction'])
            self.assertEqual(len(list(Path(directory).rglob('*.png'))),1)

    def test_held_out_scale_fit_rejected(self):
        with self.assertRaises(ValueError):fit_training_scale(SimpleNamespace(split='validation'),2048,512)


class ModelTests(unittest.TestCase):
    def test_vertical_position_survives_frontend(self):
        torch.manual_seed(1);cnn=LineImageCNN(dimension=32,tokens=24).eval()
        a=torch.ones(1,1,64,192);b=a.clone();a[:,:,45,:]=0.;b[:,:,15,:]=0.
        mask=torch.ones(1,192,dtype=torch.bool)
        with torch.no_grad(): x,_=cnn(a,mask);y,_=cnn(b,mask)
        self.assertGreater(float((x-y).abs().max()),1e-3)

    def test_masked_image_values_cannot_change_predictions(self):
        torch.manual_seed(2);model=ImageSERSFormer(numeric_evidence=False,layers=2).eval()
        a=batch(short=True);b={k:v.clone() for k,v in a.items()}
        b['image'].masked_fill_(~b['image_valid_columns'][:,None,None],0.)
        b['raw_intensity'][...,1401:]=1e9
        with torch.no_grad():x=model(a);y=model(b)
        for key in ('cp','op','logits'):torch.testing.assert_close(x[key],y[key],rtol=0,atol=0)

    def test_backward_reaches_image_and_transformer(self):
        torch.manual_seed(3);model=ImageSERSFormer(numeric_evidence=False,layers=2)
        output=model(batch());loss,_,_=supervision(output,torch.ones(1,3),torch.tensor([[1.,2.,3.]]),args())
        loss.backward()
        self.assertGreater(float(model.image_frontend.convolutions[0].weight.grad.abs().sum()),0.)
        self.assertGreater(float(model.transformer_encoder.layers[0].linear1.weight.grad.abs().sum()),0.)
        self.assertTrue((output['op'][...,1]<=output['op'][...,0]).all())

    def test_full_resolution_forward(self):
        model=ImageSERSFormer(numeric_evidence=False,layers=2).eval()
        with torch.no_grad():output=model(batch(height=512,width=2048))
        self.assertEqual(tuple(output['attention'].shape),(1,3,210))
        self.assertTrue(torch.isfinite(output['cp']).all())

    def test_numeric_evidence_is_trainable_and_masked(self):
        torch.manual_seed(4);model=ImageSERSFormer(layers=2)
        intensity=torch.randn(12,1,1901)*2+5
        model.multi_peak_evidence.fit_normalization(intensity,torch.ones(12,1901,dtype=torch.bool))
        output=model(batch(short=True));loss,_,_=supervision(output,torch.ones(1,3),torch.tensor([[1.,2.,3.]]),args())
        loss.backward()
        self.assertGreater(float(model.multi_peak_evidence.output_projection.weight.grad.abs().sum()),0.)
        self.assertTrue(torch.isfinite(output['cp']).all())
        with self.assertRaises(RuntimeError):model.multi_peak_evidence.fit_normalization(intensity,torch.ones(12,1901,dtype=torch.bool))

    def test_no_labels_enter_forward(self):
        data=batch();data['class_target']=torch.ones(1,3)
        with self.assertRaises(ValueError):ImageSERSFormer(numeric_evidence=False,layers=2)(data)

    def test_checkpoint_roundtrip_and_scheduler(self):
        torch.manual_seed(5);model=ImageSERSFormer(numeric_evidence=False,layers=2).eval()
        optimizer=make_optimizer(model,args())
        set_rates(optimizer,1,10,3);first=optimizer.param_groups[0]['lr']
        set_rates(optimizer,3,10,3);self.assertGreater(optimizer.param_groups[0]['lr'],first)
        set_rates(optimizer,10,10,3)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'],optimizer.param_groups[0]['minimum_lr'])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.pt';scale=PlotScale()
            save_checkpoint(path,model,scale,args(),1,{},optimizer,10,{})
            restored,actual,_=load_checkpoint(path,'cpu')
            self.assertEqual(asdict(scale),asdict(actual))
            with torch.no_grad():left=model(batch());right=restored(batch())
            torch.testing.assert_close(left['op'],right['op'],rtol=0,atol=0)

    def test_warmstart_requires_complete_encoder(self):
        source=ImageSERSFormer(numeric_evidence=False,layers=2)
        state={'student.'+k:v for k,v in source.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.pt'
            torch.save(dict(checkpoint_version='T3.30_feature_finetune_v1',model_state_dict=state),path)
            target=ImageSERSFormer(numeric_evidence=False,layers=2)
            report=warm_start(target,path)
            self.assertTrue(report['copied_keys'])
            del state['student.transformer_encoder.layers.0.linear1.weight']
            torch.save(dict(checkpoint_version='T3.30_feature_finetune_v1',model_state_dict=state),path)
            with self.assertRaises(ValueError):warm_start(target,path)


class SplitTests(unittest.TestCase):
    def datasets(self):
        result=[]
        for split,indices in [('train',range(12)),('validation',range(12,16)),('test',range(16,20))]:
            refs=[SimpleNamespace(source='real',condition=f'c{c}',spectrum_index=i) for c in range(126) for i in indices]
            ds=SimpleNamespace(split=split,samples=refs)
            class View:
                def __init__(self,obj):self.__dict__.update(obj.__dict__)
                def __len__(self):return len(self.samples)
            result.append(View(ds))
        return result

    def test_fixed_split_counts(self):
        result=audit_splits(self.datasets())
        self.assertEqual([result[k]['real_rows'] for k in ('train','validation','test')],[1512,504,504])

    def test_generated_heldout_rejected(self):
        datasets=self.datasets();datasets[1].samples.append(SimpleNamespace(source='generated',condition='c0',spectrum_index=0))
        with self.assertRaises(RuntimeError):audit_splits(datasets)

    def test_changed_real_identity_rejected(self):
        datasets=self.datasets();datasets[2].samples[0].spectrum_index=0
        with self.assertRaises(RuntimeError):audit_splits(datasets)


if __name__=='__main__': unittest.main(verbosity=2)
