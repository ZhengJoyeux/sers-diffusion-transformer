"""Independent fixed-condition-mean residual image experiment CLI."""
from __future__ import annotations
import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from PIL import Image

from core import (ROOT, VERSION, Bundle, Diffusion, dump, fit_arrays, fresh, manifest,
                  model, png, sha, strict_decode, verify)
from src.spectrum_file_reader import read_one_spectrum_file
from src.spectrum_conditioning import parse_condition_text


def audit(project, extension, checkpoint=None):
    p = json.loads((ROOT/'provenance.json').read_text())
    roots = {'project': Path(project), 'D4_38_ImageGeometry': Path(extension)}
    for rel, expected in p['source_sha256'].items():
        prefix, suffix = rel.split('/', 1)
        source = roots[prefix]/suffix
        if not source.is_file() or sha(source) != expected:
            raise ValueError('Server source changed; preserve it and review separately: '+str(source))
    if checkpoint is not None and sha(checkpoint) != p['checkpoint_sha256']:
        raise ValueError('D4.44 checkpoint SHA256 mismatch')
    return dict(source_files_verified=len(p['source_sha256']), checkpoint_verified=checkpoint is not None,
                expected_checkpoint_sha256=p['checkpoint_sha256'])


def prepare(a):
    verification = audit(a.project, a.extension, a.baseline_checkpoint)
    prov = json.loads((ROOT/'provenance.json').read_text())
    root = Path(a.real_data).expanduser().resolve()
    found = {x.relative_to(root).as_posix(): x for x in root.rglob('*')
             if x.is_file() and x.suffix.lower() in prov['data']['allowed_extensions']}
    expected = prov['input_sha256']
    if set(found) != set(expected) or len(found) != 126:
        raise ValueError('Input inventory differs from recorded 126 files; missing='+str(set(expected)-set(found))+
                         '; extra='+str(set(found)-set(expected)))
    names, vectors, arrays, masks, rows = [], [], [], [], []
    axis = np.arange(600., 2501., dtype=np.float32)
    for rel in sorted(found):
        path = found[rel]
        if sha(path) != expected[rel]:
            raise ValueError('Real input SHA256 mismatch: '+rel)
        original_axis, spectra, columns = read_one_spectrum_file(path, prov['data'])
        # The original data reader validates numeric/monotonic data and preserves column order.
        # New experiment deliberately requires the documented measured grids exactly.
        length = len(original_axis)
        if length not in (1401,1901) or not np.allclose(original_axis, axis[:length], atol=1e-6, rtol=0):
            raise ValueError('Expected measured 600..2000 or 600..2500 integer grid: '+rel)
        if spectra.shape != (20, length):
            raise ValueError('Expected exactly 20 source intensity columns: '+rel)
        condition = parse_condition_text(path.stem)
        names.append(condition.condition_id)
        vectors.append(condition.to_vector())
        values = np.zeros((20,1901), np.float32)
        values[:,:length] = spectra
        arrays.append(values)
        masks.append(axis <= original_axis[-1])
        rows.append(dict(condition=condition.condition_id, source=rel, source_sha256=expected[rel],
                         columns=[dict(index=i, name=name, split='train' if i<12 else 'validation' if i<16 else 'test')
                                  for i,name in enumerate(columns)], measured_points=length))
    provenance = dict(verification=verification, real_data_root=str(root), records=rows,
                      source_manifest_sha256=sha(ROOT/'provenance.json'), package_manifest_sha256=sha(ROOT/'PACKAGE_SHA256.json'))
    result = fit_arrays(np.stack(arrays), np.stack(masks), names, vectors, axis, a.output,
                        margin=a.margin, provenance=provenance)
    print('PREPARED:', result, '; test values stored only; no test metrics computed')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device(text):
    d = torch.device(text)
    if d.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; choose --device cpu for synthetic validation only')
    return str(d)


def config_for(a):
    return dict(hidden=a.hidden, depth=a.depth, heads=a.heads, timesteps=a.timesteps,
                sampling_steps=a.sampling_steps, epochs=a.epochs, batch_size=a.batch_size,
                accumulation=a.accumulation, learning_rate=a.lr,
                weight_decay=getattr(a, 'weight_decay', 0.0),
                warmup_epochs=getattr(a, 'warmup_epochs', 0),
                minimum_learning_rate=getattr(a, 'minimum_learning_rate', a.lr),
                seed=a.seed,
                ema_decay=0.995, initialization='from_scratch; no D4.44 weights loaded',
                architecture='direct residual pixels; inherited D4.44 joint image head',
                smoke_batches=a.smoke_batches,checkpoint_every=a.checkpoint_every,
                validation='all 4 validation spectra per condition; fixed diffusion-noise seed')


def preflight(bundle):
    report = json.loads((bundle.root/'preflight.json').read_text())
    failures = [r for r in report if any(v for split in ('train','validation') for v in r[split].values())]
    if failures:
        raise RuntimeError('Fixed train-only image coordinates cannot encode these rows. No rows dropped, bounds unchanged. '
                           'Preserve this preparation and review an explicitly new training-only calibration protocol: '+json.dumps(failures))


def check(a):
    """Real train/validation target roundtrip audit; no test data or model selection."""
    b=Bundle(a.prepared,device(a.device));preflight(b)
    out=fresh(a.output)
    results=[]
    for cid,name in enumerate(b.state['conditions']):
        maximum_pixel=0.;maximum_inverse=0.
        for j in range(16):
            values,ids=b.batch([(cid,j)])
            with torch.no_grad():
                clean,_,_,_=b.images(values,ids)
                z,full,quality=strict_decode(clean[0],b,cid)
                original=b.encode(values,ids)
                error=float(((z-original)*b.mask[cid]).abs().max()/(2*b.state['residual_bound'])*(b.residual.height-1))
                inverse_error=float(((full-values)*b.mask[cid]).abs().max())
            if error>.1:raise RuntimeError(f'Target raster roundtrip failed: {name}, column {j}: {error} pixels')
            maximum_pixel=max(maximum_pixel,error);maximum_inverse=max(maximum_inverse,inverse_error)
        results.append(dict(condition=name,checked_train=12,checked_validation=4,test_checked=0,
                            max_residual_pixel_error=maximum_pixel,max_raw_intensity_error=maximum_inverse))
        print('CHECK',name,results[-1],flush=True)
    dump(out/'target_roundtrip.json',dict(prepared_digest=b.digest,conditions=results,all_passed=True,
                                         note='Checks representation/target inversion only; generated quality remains unproven'))
    dump(out/'manifest.json',manifest(out))


def validation(net, diffusion, b, batch_size, seed):
    net.eval()
    generator = torch.Generator(device=b.device).manual_seed(seed)
    rows = [(c,j) for c in range(len(b.raw)) for j in range(12,16)]
    total = 0.
    with torch.no_grad():
        for start in range(0,len(rows),batch_size):
            values, ids = b.batch(rows[start:start+batch_size])
            loss, _ = diffusion.loss(net,b,values,ids,generator=generator)
            total += float(loss)*len(values)
    return total/len(rows)


def checkpoint(path, net, ema, optimizer, epoch, cfg, bundle, best, updates, replace=False):
    state = dict(version=VERSION, configuration=cfg, prepared_digest=bundle.digest,
                 package_digest=sha(ROOT/'PACKAGE_SHA256.json'), model=net.state_dict(),
                 ema=ema.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                 best_validation=best, optimizer_updates=updates, torch_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                 numpy_rng=np.random.get_state(), python_rng=random.getstate())
    # Only rolling files within this command's freshly created output may be replaced.
    target=Path(path)
    destination=target.with_suffix('.pt.tmp') if replace else target
    with destination.open('xb') as stream:
        torch.save(state, stream)
    if replace:os.replace(destination,target)


def train(a):
    d = device(a.device)
    seed_all(a.seed)
    b = Bundle(a.prepared,d)
    preflight(b)
    cfg = config_for(a)
    if cfg['hidden'] % cfg['heads'] or cfg['hidden'] % 2 or min(cfg['depth'],a.accumulation,a.epochs,a.batch_size,a.lr,a.checkpoint_every)<=0:
        raise ValueError('Invalid architecture/training parameters')
    output = fresh(a.output)
    dump(output/'configuration.json',dict(**cfg, prepared=str(b.root),prepared_digest=b.digest,
                                        torch=torch.__version__,device=d, synthetic=b.state['synthetic']))
    net = model(b,cfg)
    ema = copy.deepcopy(net).requires_grad_(False).eval()
    opt = torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=getattr(a, 'weight_decay', 0.0))
    diff = Diffusion(a.timesteps,a.sampling_steps,d)
    start, best, updates = 0, math.inf, 0
    if a.resume:
        saved = torch.load(a.resume,map_location=d,weights_only=False)
        if saved['version']!=VERSION or saved['prepared_digest']!=b.digest or saved['package_digest']!=sha(ROOT/'PACKAGE_SHA256.json'):
            raise ValueError('Resume fingerprint mismatch')
        old = saved['configuration']
        mutable = {
            'epochs', 'batch_size', 'accumulation', 'checkpoint_every',
            'warmup_epochs', 'minimum_learning_rate', 'weight_decay',
        }
        current_fixed = {k:v for k,v in cfg.items() if k not in mutable}
        old_fixed = {k:v for k,v in old.items() if k not in mutable}
        if current_fixed != old_fixed:
            raise ValueError('Resume requires identical model/data settings')
        net.load_state_dict(saved['model'],strict=True)
        ema.load_state_dict(saved['ema'],strict=True)
        opt.load_state_dict(saved['optimizer'])
        start,best,updates=saved['epoch'],saved['best_validation'],saved['optimizer_updates']
        torch.set_rng_state(saved['torch_rng'].cpu())
        if saved['cuda_rng']:
            if not torch.cuda.is_available():raise ValueError('Cannot restore CUDA RNG on CPU')
            torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
        np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
        if start>=a.epochs:raise ValueError('Resume epoch budget already exhausted')
    torch.save(dict(model=net.state_dict(),configuration=cfg),output/'initial_state.pt')
    parameters=sum(p.numel() for p in net.parameters())
    import shutil
    estimated=parameters*4*(1+4*(2+math.ceil(a.epochs/a.checkpoint_every)))
    free=shutil.disk_usage(output).free
    dump(output/'resources.json',dict(parameters=parameters,estimated_checkpoint_bytes=estimated,
                                      available_disk_bytes=free,cuda_device=torch.cuda.get_device_name() if d.startswith('cuda') else None))
    if free<estimated*1.1:raise RuntimeError('Insufficient free disk for configured checkpoint retention; choose a new larger output filesystem')
    print('PARAMETERS',parameters,'; train samples',len(b.raw)*12,'; device',d,flush=True)
    rows=[(c,j) for c in range(len(b.raw)) for j in range(12)]
    if a.smoke_batches:
        if a.smoke_batches<1:raise ValueError('smoke-batches must be nonnegative')
        rows=rows[:a.smoke_batches*a.batch_size]
    steps=math.ceil(len(rows)/a.batch_size)
    total_updates=steps*a.epochs
    warmup_updates=min(getattr(a, 'warmup_epochs', 0)*steps,max(total_updates-1,0))

    def set_learning_rate(update):
        if warmup_updates and update <= warmup_updates:
            ratio=update/warmup_updates
            rate=a.lr*ratio
        else:
            progress=(update-warmup_updates)/max(total_updates-warmup_updates,1)
            progress=min(max(progress,0.),1.)
            rate=getattr(a, 'minimum_learning_rate', a.lr) + (
                a.lr-getattr(a, 'minimum_learning_rate', a.lr)
            )*(1+math.cos(math.pi*progress))/2
        for group in opt.param_groups:
            group['lr']=rate
        return rate

    for epoch in range(start+1,a.epochs+1):
        begin=time.monotonic();net.train()
        order=np.random.permutation(len(rows))
        loss_sum=0.;opt.zero_grad(set_to_none=True)
        for step,first in enumerate(range(0,len(rows),a.batch_size)):
            group_first=(step//a.accumulation)*a.accumulation
            group_end=min(group_first+a.accumulation,steps)
            group_samples=min(group_end*a.batch_size,len(rows))-group_first*a.batch_size
            values, ids=b.batch([rows[int(i)] for i in order[first:first+a.batch_size]])
            loss, parts=diff.loss(net,b,values,ids)
            (loss*len(values)/group_samples).backward()
            loss_sum+=float(loss.detach())*len(values)
            if step+1==group_end:
                grad=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
                set_learning_rate(updates+1)
                opt.step();opt.zero_grad(set_to_none=True);updates+=1
                with torch.no_grad():
                    for e,p in zip(ema.parameters(),net.parameters()):e.lerp_(p,1-cfg['ema_decay'])
            if step%10==0:print(json.dumps(dict(epoch=epoch,batch=step+1,**parts)),flush=True)
        score=validation(ema,diff,b,a.batch_size,a.seed+100001)
        improved=score<best;best=min(best,score)
        cp=output/'last.pt'
        checkpoint(cp,net,ema,opt,epoch,cfg,b,best,updates,replace=True)
        if epoch%a.checkpoint_every==0 or epoch==a.epochs:
            checkpoint(output/f'epoch_{epoch:04d}.pt',net,ema,opt,epoch,cfg,b,best,updates)
        if improved:
            cp=output/'best.pt'
            checkpoint(cp,net,ema,opt,epoch,cfg,b,best,updates,replace=True)
            dump(output/'best.json',dict(epoch=epoch,path=cp.name,sha256=sha(cp),validation=score))
        row=dict(epoch=epoch,train_loss=loss_sum/len(rows),ema_validation_loss=score,
                 seconds=time.monotonic()-begin,optimizer_updates=updates)
        if d.startswith('cuda'):row['cuda_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        with (output/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    dump(output/'COMPLETE.json',dict(epochs=a.epochs,updates=updates,best=best,parameters=parameters,
                                    prepared_digest=b.digest,synthetic=b.state['synthetic']))


def load(a):
    b=Bundle(a.prepared,device(a.device))
    saved=torch.load(a.checkpoint,map_location=b.device,weights_only=False)
    if saved['version']!=VERSION or saved['prepared_digest']!=b.digest or saved['package_digest']!=sha(ROOT/'PACKAGE_SHA256.json'):
        raise ValueError('Checkpoint/preparation/package mismatch')
    net=model(b,saved['configuration'])
    net.load_state_dict(saved['ema'],strict=True);net.eval()
    cfg=saved['configuration']
    return b,net,Diffusion(cfg['timesteps'],cfg['sampling_steps'],b.device)


def generate(a):
    if a.number<1 or a.batch_size<1:raise ValueError('Positive sample and batch counts required')
    b,net,diff=load(a)
    out=fresh(a.output)
    names=b.state['conditions']
    selected=list(range(len(names))) if a.condition=='all' else [names.index(parse_condition_text(a.condition).condition_id)]
    identity=dict(version=VERSION,checkpoint_sha256=sha(a.checkpoint),prepared_digest=b.digest,
                  package_digest=sha(ROOT/'PACKAGE_SHA256.json'),number=a.number,conditions=[names[c] for c in selected],
                  seed=a.seed,reference='none; sampling never uses validation/test spectra',synthetic=b.state['synthetic'])
    dump(out/'protocol.json',identity)
    records=[]
    for cid in selected:
        folder=fresh(out/names[cid])
        generator=torch.Generator(device=b.device).manual_seed(a.seed+cid*100003)
        for start in range(0,a.number,a.batch_size):
            count=min(a.batch_size,a.number-start)
            ids=torch.full((count,),cid,device=b.device,dtype=torch.long)
            native=diff.sample(net,b,ids,generator)
            for i in range(count):
                index=start+i
                name=f'{index:04d}'
                mask=b.mask[cid:cid+1]
                # ALWAYS preserve the untouched sampler output, even for rejected samples.
                native_path=folder/(name+'_native.npz')
                np.savez_compressed(native_path,native=native[i].cpu().numpy(),
                                    measured_mask=mask[0,0].bool().cpu().numpy(),
                                    axis=np.asarray(b.state['axis']))
                record=dict(condition=names[cid],index=index,approved=False,native_sha256=sha(native_path))
                try:
                    z,full,quality=strict_decode(native[i],b,cid)
                    valid=b.residual.pixel_mask(mask)[0,0,0].bool()
                    png(folder/(name+'_residual.png'),native[i],valid)
                    # PNG extraction is independently audited; generation uses float native data.
                    png_array=np.asarray(Image.open(folder/(name+'_residual.png')),np.float32)
                    png_native=torch.tensor(1-2*png_array/255,device=b.device)[None]
                    z_png=b.residual.decode(png_native[None],mask)
                    pixel_error=float(((z_png-z)*mask).abs().max()/(b.residual.y_max-b.residual.y_min)*(b.residual.height-1))
                    if pixel_error>b.state['gates']['png_roundtrip_pixels']:
                        raise ValueError('PNG residual roundtrip exceeds fixed tolerance')
                    # Render reconstructed FULL spectrum with fixed global intensity calibration.
                    image=b.full.render(full,mask)[0]
                    png(folder/(name+'_full.png'),image,valid)
                    full_png_array=np.asarray(Image.open(folder/(name+'_full.png')),np.float32)
                    full_png_native=torch.tensor(1-2*full_png_array/255,device=b.device)[None,None]
                    decoded_full_png=b.full.decode(full_png_native,mask)
                    full_pixel_error=float(((decoded_full_png-full)*mask).abs().max()/(b.full.y_max-b.full.y_min)*(b.full.height-1))
                    if full_pixel_error>b.state['gates']['png_roundtrip_pixels']:
                        raise ValueError('Full-spectrum PNG roundtrip exceeds fixed tolerance')
                    full_npz=folder/(name+'_reconstruction.npz')
                    np.savez_compressed(full_npz,full=full[0,0].cpu().numpy(),z=z[0,0].cpu().numpy(),
                                        mean=b.mean[cid,0].cpu().numpy(),scale=b.scale[cid].cpu().numpy(),
                                        mask=mask[0,0].bool().cpu().numpy(),axis=np.asarray(b.state['axis']))
                    record.update(approved=True,quality=quality,png_roundtrip_pixels=pixel_error,
                                  full_png_roundtrip_pixels=full_pixel_error,
                                  reconstruction_sha256=sha(full_npz))
                except ValueError as error:
                    record['reason']=str(error)
                records.append(record)
                with (out/'quality.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
    failed=sum(not r['approved'] for r in records)
    dump(out/'generation_summary.json',dict(requested=len(records),approved=len(records)-failed,rejected=failed,
                                           complete=failed==0,protocol=identity))
    dump(out/'manifest.json',manifest(out))
    if failed:raise RuntimeError(f'{failed}/{len(records)} rejected. All native samples retained; no retries or selection. Evaluation will stop.')
    print('GENERATION COMPLETE',out)


def parser():
    p=argparse.ArgumentParser(__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    audit_p=sub.add_parser('audit')
    for x in (audit_p,):
        x.add_argument('--project',required=True);x.add_argument('--extension',required=True)
        x.add_argument('--baseline-checkpoint')
    q=sub.add_parser('prepare')
    q.add_argument('--project',required=True);q.add_argument('--extension',required=True)
    q.add_argument('--baseline-checkpoint',required=True);q.add_argument('--real-data',required=True)
    q.add_argument('--output',required=True);q.add_argument('--margin',type=float,default=.25)
    q=sub.add_parser('train')
    q.add_argument('--prepared',required=True);q.add_argument('--output',required=True)
    q.add_argument('--device',default='cuda');q.add_argument('--epochs',type=int,default=100)
    q.add_argument('--batch-size',type=int,default=2);q.add_argument('--accumulation',type=int,default=8)
    q.add_argument('--hidden',type=int,default=512);q.add_argument('--depth',type=int,default=12)
    q.add_argument('--heads',type=int,default=8);q.add_argument('--lr',type=float,default=1e-4)
    q.add_argument('--timesteps',type=int,default=200);q.add_argument('--sampling-steps',type=int,default=100)
    q.add_argument('--seed',type=int,default=2026);q.add_argument('--resume')
    q.add_argument('--warmup-epochs',type=int,default=3)
    q.add_argument('--minimum-learning-rate',type=float,default=1e-6)
    q.add_argument('--weight-decay',type=float,default=.01)
    q.add_argument('--smoke-batches',type=int,default=0,help='Explicit limited training smoke run, never a formal experiment')
    q.add_argument('--checkpoint-every',type=int,default=10)
    q=sub.add_parser('check')
    q.add_argument('--prepared',required=True);q.add_argument('--output',required=True)
    q.add_argument('--device',default='cuda')
    q=sub.add_parser('generate')
    q.add_argument('--prepared',required=True);q.add_argument('--checkpoint',required=True)
    q.add_argument('--output',required=True);q.add_argument('--device',default='cuda')
    q.add_argument('--condition',default='all');q.add_argument('--number',type=int,default=50)
    q.add_argument('--batch-size',type=int,default=1);q.add_argument('--seed',type=int,default=2026)
    q=sub.add_parser('evaluate')
    q.add_argument('--prepared',required=True);q.add_argument('--generated',required=True)
    q.add_argument('--output',required=True);q.add_argument('--split',choices=['validation','test'],default='validation')
    q.add_argument('--locked-test-protocol',action='store_true')
    q=sub.add_parser('selftest')
    q.add_argument('--output',required=True);q.add_argument('--device',default='cpu')
    q.add_argument('--full-resolution',action='store_true')
    q.add_argument('--production-model',action='store_true')
    q=sub.add_parser('compare')
    q.add_argument('--prepared',required=True);q.add_argument('--generated',required=True)
    q.add_argument('--baseline-root',required=True);q.add_argument('--baseline-checkpoint',required=True)
    q.add_argument('--output',required=True)
    return p


def main():
    a=parser().parse_args()
    verify(ROOT,'PACKAGE_SHA256.json')
    torch.set_num_threads(2)
    if a.command=='audit':print(json.dumps(audit(a.project,a.extension,a.baseline_checkpoint),indent=2))
    elif a.command=='prepare':prepare(a)
    elif a.command=='train':train(a)
    elif a.command=='check':check(a)
    elif a.command=='generate':generate(a)
    elif a.command=='evaluate':
        from metrics import evaluate
        evaluate(a)
    elif a.command=='compare':
        from metrics import compare
        compare(a)
    else:
        from selftest import selftest
        selftest(a)


if __name__=='__main__':
    main()
