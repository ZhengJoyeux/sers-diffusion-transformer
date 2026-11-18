"""Explicit new-experiment metrics. Not a claim of identical D4.44 evaluator output."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

from core import Bundle, dump, fresh, manifest, sha, verify


def corr(a,b):
    a=np.asarray(a).reshape(-1);b=np.asarray(b).reshape(-1)
    a=a-a.mean();b=b-b.mean()
    denominator=np.linalg.norm(a)*np.linalg.norm(b)
    return None if denominator<=1e-15 else float(np.dot(a,b)/denominator)


def ratio(a,b):
    return float(a/b) if np.isfinite(a) and np.isfinite(b) and abs(b)>1e-15 else None


def diversity(x):
    if len(x)<2:return 0.
    dist=cdist(x,x,'sqeuclidean')/x.shape[1]
    return float(dist[np.triu_indices(len(x),1)].mean())


def peaks(x, threshold):
    heights=[];widths=[]
    for y in x:
        loc,_=find_peaks(y,prominence=threshold)
        if len(loc):
            heights.extend(y[loc].tolist());widths.extend(peak_widths(y,loc,rel_height=.5)[0].tolist())
    return np.asarray(heights),np.asarray(widths)


def iqr(x):
    return float(np.quantile(x,.75)-np.quantile(x,.25)) if len(x) else 0.


def condition_metrics(g,r,t,patch):
    # Every normalization, projection bandwidth and peak threshold uses train only.
    scale=max(float(np.ptp(t)),1e-6)
    gn,rn,tn=g/scale,r/scale,t/scale
    gm,rm=gn.mean(0),rn.mean(0)
    dg,dr=np.diff(gn,axis=1),np.diff(rn,axis=1)
    seams=np.arange(1,g.shape[1])%patch==0
    rng=np.random.default_rng(2026)
    directions=rng.normal(size=(g.shape[1],32))
    directions/=np.linalg.norm(directions,axis=0,keepdims=True)
    gp,rp=gn@directions,rn@directions
    swd=float(np.mean([wasserstein_distance(gp[:,i],rp[:,i]) for i in range(32)]))
    bandwidth=float(np.median(cdist(tn,tn,'sqeuclidean')[np.triu_indices(len(tn),1)]))
    bandwidth=max(bandwidth,1e-12)
    kernel=lambda x,y: np.exp(-cdist(x,y,'sqeuclidean')/(2*bandwidth))
    kg,kr,kgr=kernel(gn,gn),kernel(rn,rn),kernel(gn,rn)
    # Biased nonnegative estimator, stated explicitly (4 validation references only).
    mmd=max(0.,float(kg.mean()+kr.mean()-2*kgr.mean()))
    low,high=float(tn.min()),float(tn.max())
    bins=np.r_[-np.inf,np.linspace(low,high,65),np.inf]
    pg=np.histogram(gn,bins=bins)[0].astype(float);pr=np.histogram(rn,bins=bins)[0].astype(float)
    pg/=pg.sum();pr/=pr.sum();mid=(pg+pr)/2
    kl=lambda x: float(np.sum(x[x>0]*np.log(x[x>0]/mid[x>0])))
    jsd=(kl(pg)+kl(pr))/2
    qg,qr=np.quantile(gn,[.01,.05,.25,.5,.75,.95,.99]),np.quantile(rn,[.01,.05,.25,.5,.75,.95,.99])
    hg,wg=peaks(gn,.02);hr,wr=peaks(rn,.02)
    return dict(mean_spectrum_normalized_mse=float(np.mean((gm-rm)**2)),
                mean_spectrum_pearson=corr(gm,rm),mean_derivative_pearson=corr(np.diff(gm),np.diff(rm)),
                absolute_seam_jump=float(np.abs(dg[:,seams]).mean()) if seams.any() else None,
                reference_absolute_seam_jump=float(np.abs(dr[:,seams]).mean()) if seams.any() else None,
                mean_derivative_mse=float(np.mean((dg.mean(0)-dr.mean(0))**2)),
                pairwise_diversity_ratio=ratio(diversity(gn),diversity(rn)),
                qq_lower_tail_ratio=ratio(qg[3]-qg[0],qr[3]-qr[0]),
                qq_upper_tail_ratio=ratio(qg[6]-qg[3],qr[6]-qr[3]),
                peak_height_iqr_ratio=ratio(iqr(hg),iqr(hr)),peak_width_iqr_ratio=ratio(iqr(wg),iqr(wr)),
                generated_peaks=len(hg),reference_peaks=len(hr),mmd2_biased=mmd,jsd_histogram=jsd,
                sliced_wasserstein_32=swd,train_intensity_span=scale,
                nearest_validation_normalized_mse=float(cdist(gn,rn,'sqeuclidean').min(1).mean()/g.shape[1]))


def evaluate(a):
    if a.split=='test' and not a.locked_test_protocol:
        raise ValueError('Test evaluation requires explicit --locked-test-protocol after protocol/model selection is frozen')
    b=Bundle(a.prepared)
    generated=Path(a.generated)
    verify(generated,'manifest.json')
    summary=json.loads((generated/'generation_summary.json').read_text())
    protocol=summary['protocol']
    if protocol['prepared_digest']!=b.digest:raise ValueError('Generation/preparation mismatch')
    records=[json.loads(x) for x in (generated/'quality.jsonl').read_text().splitlines()]
    if not summary['complete'] or summary['rejected'] or any(not x['approved'] for x in records):
        raise ValueError('Generation contains rejected samples: selection/drop/retry forbidden; no favorable subset evaluation')
    expected={(c,i) for c in protocol['conditions'] for i in range(protocol['number'])}
    actual=[(r['condition'],r['index']) for r in records]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:raise ValueError('Incomplete/duplicate generated inventory')
    output=fresh(a.output)
    rows=[]
    for name in protocol['conditions']:
        c=b.state['conditions'].index(name)
        mask=b.mask[c,0].bool().numpy()
        samples=[]
        for i in range(protocol['number']):
            path=generated/name/f'{i:04d}_reconstruction.npz'
            record=next(r for r in records if r['condition']==name and r['index']==i)
            if sha(path)!=record['reconstruction_sha256']:raise ValueError('Reconstruction hash mismatch')
            with np.load(path,allow_pickle=False) as x:
                if not np.array_equal(x['mask'],mask) or not np.array_equal(x['axis'],np.asarray(b.state['axis'])):
                    raise ValueError('Axis or measured mask mismatch')
                if not np.isfinite(x['full'][mask]).all():raise ValueError('Nonfinite generated spectrum')
                if not np.allclose(x['mean'],b.mean[c,0].numpy(),rtol=0,atol=0):raise ValueError('Prior mismatch')
                reconstructed=x['mean']+x['scale'].reshape(1)*np.sinh(x['z'])
                if not np.allclose(reconstructed[mask],x['full'][mask],rtol=2e-5,atol=2e-5):
                    raise ValueError('Saved residual inverse mismatch')
                samples.append(x['full'][mask])
        # NumPy mixed advanced indexing is avoided for row/column axis clarity.
        reference=b.raw[c,12:16][:,mask] if a.split=='validation' else b.raw[c,16:20][:,mask]
        training=b.raw[c,:12][:,mask]
        metrics=condition_metrics(np.stack(samples),reference,training,b.residual.configuration['patch_width'])
        rows.append(dict(condition=name,generated_count=len(samples),reference_count=4,reference_split=a.split,**metrics))
    with (output/'condition_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    mean={}
    for k in rows[0]:
        if k in ('condition','reference_split'):continue
        values=[x[k] for x in rows if isinstance(x[k],(int,float))]
        mean[k]=dict(macro_mean=float(np.mean(values)) if values else None,defined_conditions=len(values),total_conditions=len(rows))
    dump(output/'summary.json',dict(split=a.split,conditions=len(rows),metrics=mean,
                                  prepared_digest=b.digest,checkpoint_sha256=protocol['checkpoint_sha256'],
                                  generation_manifest_sha256=sha(generated/'manifest.json'),
                                  estimator='new independent metric definitions; see metrics.py; no FID',
                                  limitation='within-source 12/4/4; validation-selected results are not independent test evidence'))
    dump(output/'manifest.json',manifest(output))
    print('EVALUATION COMPLETE',output)


def compare(a):
    """Read-only comparison against existing D4.44 numerical exports; no baseline regeneration."""
    import pandas as pd
    from src.spectrum_conditioning import parse_condition_text
    from core import ROOT
    provenance=json.loads((ROOT/'provenance.json').read_text())
    if sha(a.baseline_checkpoint)!=provenance['checkpoint_sha256']:
        raise ValueError('Baseline D4.44 checkpoint mismatch')
    b=Bundle(a.prepared)
    generated=Path(a.generated)
    verify(generated,'manifest.json')
    summary=json.loads((generated/'generation_summary.json').read_text())
    if not summary['complete'] or summary['rejected'] or summary['protocol']['number']!=50:
        raise ValueError('Require complete unfiltered new-model 50-per-condition generation')
    if summary['protocol']['prepared_digest']!=b.digest:raise ValueError('Prepared digest mismatch')
    names=summary['protocol']['conditions']
    base=Path(a.baseline_root).expanduser().resolve()
    inventory={}
    for path in base.rglob('*_generated.xlsx'):
        name=parse_condition_text(path.stem[:-len('_generated')]).condition_id
        if name in inventory:raise ValueError('Duplicate baseline condition; select one experiment root: '+name)
        inventory[name]=path
    if set(inventory)!=set(names):raise ValueError('Baseline/new condition inventories differ; no intersection-only comparison')
    baseline_manifests=list(base.rglob('generation_manifest.json'))
    if not baseline_manifests:raise ValueError('Original generation_manifest.json files required for baseline attribution')
    attributed={}
    for path in baseline_manifests:
        doc=json.loads(path.read_text())
        if Path(doc['checkpoint']).expanduser().resolve()!=Path(a.baseline_checkpoint).expanduser().resolve():
            raise ValueError('Baseline generation checkpoint path differs: '+str(path))
        if doc['model_source']!='ema' or doc['number_of_spectra_per_condition']!=50:
            raise ValueError('Baseline must use EMA and 50 per condition')
        for entry in doc['conditions']:
            name=parse_condition_text(entry['source_condition_name']).condition_id
            if name in attributed:raise ValueError('Duplicate baseline manifest attribution')
            attributed[name]=path
    if set(attributed)!=set(names):raise ValueError('Baseline manifest condition inventory differs')
    out=fresh(a.output)
    rows=[]
    for name in names:
        cid=b.state['conditions'].index(name);mask=b.mask[cid,0].bool().numpy()
        table=pd.read_excel(inventory[name])
        axis=table.iloc[:,0].to_numpy(float)
        spectra=table.iloc[:,1:].to_numpy(float).T
        required=int(mask.sum())
        if len(axis)<required or spectra.shape[0]!=50 or not np.array_equal(axis[:required],np.asarray(b.state['axis'])[:required]):
            raise ValueError('Baseline axis/count mismatch: '+name)
        old=spectra[:,:required]
        if not np.isfinite(old).all():raise ValueError('Nonfinite baseline values')
        new=[]
        for j in range(50):
            with np.load(generated/name/f'{j:04d}_reconstruction.npz',allow_pickle=False) as x:
                if not np.array_equal(x['mask'],mask):raise ValueError('New model mask mismatch')
                new.append(x['full'][mask])
        training=b.raw[cid,:12][:,mask];reference=b.raw[cid,12:16][:,mask]
        metrics_old=condition_metrics(old,reference,training,b.residual.configuration['patch_width'])
        metrics_new=condition_metrics(np.stack(new),reference,training,b.residual.configuration['patch_width'])
        row=dict(condition=name,reference_split='validation',count_per_model=50,baseline_xlsx_sha256=sha(inventory[name]))
        for k in metrics_new:
            row['d444_'+k]=metrics_old[k];row['residual_'+k]=metrics_new[k]
            row['delta_'+k]=None if metrics_new[k] is None or metrics_old[k] is None else metrics_new[k]-metrics_old[k]
        rows.append(row)
    with (out/'paired_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    dump(out/'protocol.json',dict(baseline_checkpoint_sha256=sha(a.baseline_checkpoint),
                                baseline_generation_manifests={str(p):sha(p) for p in baseline_manifests},
                                new_checkpoint_sha256=summary['protocol']['checkpoint_sha256'],
                                prepared_digest=b.digest,conditions=names,
                                note='Same train-normalized metrics, masks, 4 validation references and 50 samples; no original image-quality claim'))
    dump(out/'manifest.json',manifest(out))
    print('PAIRED COMPARISON COMPLETE',out)
