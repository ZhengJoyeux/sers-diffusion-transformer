"""Actual execution tests; analytic image fixtures are explicitly distinguished."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

import core
from core import (Bundle, Diffusion, dump, fit_arrays, fresh, manifest, model,
                  png, sha, strict_decode)


def synthetic(output, full=False):
    n=1901 if full else 64
    axis=np.arange(600.,600.+n,dtype=np.float32)
    rng=np.random.default_rng(2026)
    masks=np.ones((2,n),bool)
    masks[1,1401 if full else 48:]=False
    raw=[]
    x=np.linspace(0,1,n)
    # Independent rows; held-out data never determines statistics or image bounds.
    for c in range(2):
        base=10+5*np.exp(-.5*((x-.3)/.06)**2)+c
        curves=[]
        for j in range(20):
            amplitude=.12*np.sin((j+1)*1.17)
            curves.append(base+amplitude*np.sin(x*13)+.02*np.cos(x*8+j))
        raw.append(curves)
    raw=np.asarray(raw,np.float32)
    vectors=np.zeros((2,14),np.float32)
    vectors[0,[3,4,8,12]]=1
    vectors[1,[0,7,8,13]]=1
    return fit_arrays(raw,masks,['DEL-H_water','TEB-H_soil'],vectors,axis,output,
                      height=768 if full else 64,width=1920 if full else 64,synthetic=True)


def selftest(a):
    from run import device, seed_all, train, generate, check
    from metrics import evaluate, condition_metrics, compare
    seed_all(2026)
    output=fresh(a.output)
    d=device(a.device)
    root=synthetic(output/'prepared',a.full_resolution)
    b=Bundle(root,d)
    cfg=dict(hidden=512 if a.production_model else 32,depth=12 if a.production_model else 2,
             heads=8 if a.production_model else 4)
    net=model(b,cfg)
    opt=torch.optim.Adam(net.parameters(),lr=1e-3)
    diff=Diffusion(8,4,d)
    checks=[]
    def passed(name):
        checks.append(name);print('PASS',name,flush=True)
    def rejects(name,fn):
        try:fn()
        except (ValueError,RuntimeError,FileExistsError):passed(name)
        else:raise AssertionError('Expected rejection: '+name)
    values,ids=b.batch([(0,0),(1,0)])
    base,scale,mask,condition=b.fields(ids)
    train_mean=torch.tensor(b.raw[:,:12].mean(1)[:,None],device=d)*mask
    torch.testing.assert_close(base,train_mean,atol=2e-6,rtol=2e-6)
    passed('fixed prior equals training mean; invalid tail stays masked')
    z=b.encode(values,ids)
    torch.testing.assert_close(b.reconstruct(z,ids),values*mask,atol=2e-6,rtol=2e-6)
    passed('new asinh inverse roundtrip against same mean')
    check(SimpleNamespace(prepared=str(root),device=d,output=str(output/'target_check')))
    passed('all synthetic train/validation target roundtrips; no test rows audited')
    import pandas as pd
    from src.spectrum_file_reader import read_one_spectrum_file
    frame=pd.DataFrame(b.raw[0].T,columns=[f'spectrum_{j}' for j in range(20)])
    frame.insert(0,'Raman',b.state['axis'])
    file=output/'reader_fixture.xlsx';frame.to_excel(file,index=False)
    axis,loaded,names=read_one_spectrum_file(file,dict(header_row=0,sheet_name=0,raman_shift_column=0,intensity_start_column=1))
    np.testing.assert_allclose(loaded,b.raw[0]);assert names==[f'spectrum_{j}' for j in range(20)]
    passed('original server Excel reader preserves 20 physical spectrum-column identities')
    mu0=b.mean.clone();scale0=b.scale.clone()
    # Refit after changing ONLY held-out values. This must not change fit.
    raw=b.raw.copy();raw[:,12:]+=1234
    second=fit_arrays(raw,b.mask[:,0].cpu().bool().numpy(),b.state['conditions'],b.vectors.cpu().numpy(),
                      np.asarray(b.state['axis']),output/'heldout_changed',height=b.residual.height,
                      width=b.residual.width,synthetic=True)
    b2=Bundle(second,d)
    assert b.state['residual_codec']==b2.state['residual_codec']
    assert b.state['prior_codec']==b2.state['prior_codec']
    torch.testing.assert_close(mu0,b2.mean,rtol=0,atol=0)
    torch.testing.assert_close(scale0,b2.scale,rtol=0,atol=0)
    passed('validation/test values cannot alter prior, residual scale or image coordinates')
    clean,prior,pixels,c=b.images(values,ids)
    decoded=b.residual.decode(clean,mask)
    error=float(((decoded-z)*mask).abs().max()/(2*b.state['residual_bound'])*(b.residual.height-1))
    assert error<.1,error
    passed('signed residual raster numeric roundtrip; fixed zero line')
    rejects('out-of-range residual rejected without clipping',lambda:b.residual.render(z+1000,mask))
    rejects('out-of-range full intensity rejected',lambda:b.full.render(values+1000,mask))
    rejects('existing preparation cannot be overwritten',lambda:synthetic(root,a.full_resolution))
    net.train()
    loss_history=[]
    for _ in range(4):
        opt.zero_grad(set_to_none=True)
        loss,parts=diff.loss(net,b,values,ids)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
        opt.step();loss_history.append(float(loss.detach()))
    for prefix in ('patch_embed.','condition_encoder.','blocks.0.qkv.','final_linear.joint.'):
        grads=[p.grad for name,p in net.named_parameters() if name.startswith(prefix) and p.grad is not None]
        assert grads and any(g.abs().sum()>0 for g in grads),prefix
    passed('actual pixel DiT forward/backward/optimizer with joint head and chemical/prior input')
    net.eval()
    generator=torch.Generator(device=d).manual_seed(2026)
    sampled=diff.sample(net,b,ids,generator)
    assert sampled.shape==clean.shape and torch.isfinite(sampled).all()
    reconstructed=b.reconstruct(b.residual.decode(sampled,mask),ids)
    assert torch.isfinite(reconstructed).all() and not reconstructed[1,:,48 if not a.full_resolution else 1401:].any()
    np.savez_compressed(output/'actual_model_sampling.npz',native=sampled.cpu().numpy(),
                        full=reconstructed.cpu().numpy(),mask=mask.cpu().numpy())
    reports=[]
    for i in range(2):
        try:
            _,_,quality=strict_decode(sampled[i],b,i)
            reports.append(dict(approved=True,quality=quality))
        except ValueError as exc:reports.append(dict(approved=False,reason=str(exc)))
    passed('actual learned-model DDIM sampling and inverse reconstruction, measured mask retained')
    ambiguous=torch.zeros_like(clean[0])
    rejects('diffuse native image fails original contrast/spread gates',lambda:strict_decode(ambiguous,b,0))
    # Two separate peaks far apart cannot be replaced by their midpoint silently.
    multiple=torch.full_like(clean[0],-1.)
    multiple[:,10:13,:]=1.;multiple[:,-13:-10,:]=1.
    rejects('multiple separated ridges rejected',lambda:strict_decode(multiple,b,0))
    # Both conditioning inputs must influence nonzero-weight trained predictions.
    t=torch.ones(2,device=d,dtype=torch.long)
    with torch.no_grad():
        baseline=net(clean,t,valid_mask=pixels,condition=c,prior_conditioning=prior)
        cp=net(clean,t,valid_mask=pixels,condition=c.flip(0),prior_conditioning=prior)
        pp=net(clean,t,valid_mask=pixels,condition=c,prior_conditioning=-prior)
    assert not torch.equal(baseline,cp) and not torch.equal(baseline,pp)
    passed('prior image and chemical condition both affect predictions')
    # Joint correction cannot modify invalid columns, including the short-spectrum cutoff.
    assert not baseline[pixels==0].any()
    assert net.final_linear.pairs is None
    passed('joint boundary head respects measured/padded masks and clears runtime state')
    # Check float -> PNG -> residual -> full independently of learned sample quality.
    z_fixture=torch.zeros_like(mask)
    z_fixture[0,0]=torch.linspace(-.1,.1,mask.shape[-1],device=d)*b.state['residual_bound']
    fixture=b.residual.render(z_fixture,mask)
    for i in range(2):
        strict_decode(fixture[i],b,i)
        file=output/f'fixture_residual_{i}.png'
        png(file,fixture[i],pixels[i,0,0].bool())
        png_image=torch.tensor(1-2*np.asarray(Image.open(file),dtype=np.float32)/255,device=d)[None,None]
        png_z=b.residual.decode(png_image,mask[i:i+1])
        float_z=b.residual.decode(fixture[i:i+1],mask[i:i+1])
        discrepancy=float(((png_z-float_z)*mask[i:i+1]).abs().max()/(2*b.state['residual_bound'])*(b.residual.height-1))
        assert discrepancy<.1,discrepancy
    passed('PNG residual decoding and full reconstruction pass with analytic raster fixture')
    # Exercise real CLI trainer, checkpoint reload/resume. Synthetic small architecture only.
    args=SimpleNamespace(prepared=str(root),output=str(output/'training'),device=d,epochs=1,
                         batch_size=2,accumulation=3,hidden=cfg['hidden'],depth=cfg['depth'],heads=cfg['heads'],
                         lr=1e-3,timesteps=8,sampling_steps=4,seed=2026,resume=None,smoke_batches=2,checkpoint_every=10)
    train(args)
    args.resume=str(output/'training/epoch_0001.pt');args.epochs=2;args.output=str(output/'resumed')
    train(args)
    passed('actual trainer, EMA, train/validation split, checkpoint and optimizer/RNG resume')
    # The complete accepted export/evaluation branch needs a known localized image.
    # Explicit oracle fixture; NOT presented as trained-model generation quality.
    from unittest.mock import patch
    def oracle(self,net,bundle,idx,generator):
        m=bundle.mask[idx]
        values=torch.zeros_like(m)
        return bundle.residual.render(values,m)
    gen=SimpleNamespace(prepared=str(root),checkpoint=str(output/'resumed/epoch_0002.pt'),
                        output=str(output/'oracle_export_fixture'),device=d,condition='all',number=3 if a.full_resolution else 50,batch_size=2,seed=2026)
    with patch.object(Diffusion,'sample',oracle):generate(gen)
    ev=SimpleNamespace(prepared=str(root),generated=gen.output,output=str(output/'oracle_metrics_fixture'),
                       split='validation',locked_test_protocol=False)
    evaluate(ev)
    passed('accepted export/inverse PNG/full PNG/evaluation integration with explicitly analytic oracle fixture')
    if not a.full_resolution:
        baseline=fresh(output/'baseline_xlsx_fixture')
        checkpoint_file=baseline/'baseline_fixture.pt';checkpoint_file.write_bytes(b'explicit synthetic baseline attribution fixture')
        entries=[]
        for cid,name in enumerate(b.state['conditions']):
            table=pd.DataFrame(np.repeat(b.mean[cid,0].cpu().numpy()[:,None],50,axis=1))
            table.insert(0,'Raman',b.state['axis']);table.to_excel(baseline/(name+'_generated.xlsx'),index=False)
            entries.append(dict(source_condition_name=name))
        dump(baseline/'generation_manifest.json',dict(checkpoint=str(checkpoint_file),model_source='ema',
                                                     number_of_spectra_per_condition=50,conditions=entries))
        attribution=fresh(output/'baseline_attribution_fixture')
        dump(attribution/'provenance.json',dict(checkpoint_sha256=sha(checkpoint_file)))
        with patch.object(core,'ROOT',attribution):
            compare(SimpleNamespace(prepared=str(root),generated=gen.output,baseline_root=str(baseline),
                                    baseline_checkpoint=str(checkpoint_file),output=str(output/'comparison_fixture')))
        passed('same-metric paired comparison with synthetic Excel baseline and explicit attribution fixture')
    # Real untrained/short-trained sample failure must retain all samples and block evaluation.
    gen.output=str(output/'actual_generation');gen.number=2
    try:generate(gen)
    except RuntimeError:
        summary=json.loads((Path(gen.output)/'generation_summary.json').read_text())
        assert summary['rejected']>0
        assert len(list(Path(gen.output).glob('*/*_native.npz')))==4
        ev.generated=gen.output;ev.output=str(output/'must_not_evaluate_rejected')
        rejects('rejected learned samples preserved and evaluation cannot silently select',lambda:evaluate(ev))
    else:passed('all actual short-trained sampler outputs passed strict export gates')
    wrong=output/'tamper_copy'
    import shutil
    shutil.copytree(root,wrong)
    (wrong/'state.json').write_text('{}')
    rejects('tampered preparation rejected',lambda:Bundle(wrong))
    dump(output/'validation_report.json',dict(checks=checks,passed=len(checks),device=d,
         torch=torch.__version__,parameters=sum(p.numel() for p in net.parameters()),
         dimensions=list(clean.shape),production_model=a.production_model,full_resolution=a.full_resolution,
         losses=loss_history,residual_raster_roundtrip_pixels=error,actual_sampler_quality=reports,
         analytic_fixture='accepted export/evaluation used an explicit oracle raster; not learned generation quality',
         real_gpu_experiment=False,real_data=False,formal_checkpoint_loaded=False))
    print('SELFTEST COMPLETE',output/'validation_report.json',flush=True)
