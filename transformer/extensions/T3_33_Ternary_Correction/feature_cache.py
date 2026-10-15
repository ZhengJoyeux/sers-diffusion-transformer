"""Cache frozen image-derived features once, without fitting or predicting on test."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,Dataset
from correction_model import extract_frozen

KEYS=('features','cp','op','logits')


@torch.no_grad()
def build_cache(baseline,original,scale,args,fingerprint,runtime):
    from image_data import LineImageDataset
    path=args.output_directory/'feature_cache'/fingerprint/(original.split+'.pt')
    if path.exists():
        cache=torch.load(path,map_location='cpu',weights_only=False)
        if cache['fingerprint']!=fingerprint or cache['split']!=original.split or len(cache['metadata'])!=len(original):
            raise RuntimeError('Cached features belong to different data/weights')
        print(f'Using {original.split} feature cache: {len(original)} rows',flush=True)
        return cache
    if original.split=='test' and args.phase!='evaluate':
        raise ValueError('Test features are not used during check/smoke/train')
    wrapped=LineImageDataset(original,scale,args.image_cache)
    loader=DataLoader(wrapped,batch_size=args.cache_batch_size,shuffle=False,
        num_workers=args.num_workers,pin_memory=args.device.startswith('cuda'))
    tensors={k:[] for k in KEYS};levels=[];presence=[];metadata=[];max_clipped=0.
    baseline.eval()
    for step,batch in enumerate(loader):
        values=extract_frozen(baseline,runtime.model_inputs(batch,args.device))
        for key in KEYS:tensors[key].append(values[key].cpu())
        levels.append(batch['concentration_level'].cpu());presence.append(batch['class_target'].cpu())
        max_clipped=max(max_clipped,float(batch['image_clipped_fraction'].max()))
        for i in range(len(batch['class_target'])):
            metadata.append(dict(condition=batch['condition'][i],source=batch['source'][i],
                spectrum_index=int(batch['spectrum_index'][i]),matrix=batch['matrix_name'][i],split=original.split))
        if step%25==0 or step+1==len(loader):
            print(f'Extracting frozen {original.split} features: {len(metadata)}/{len(original)}',flush=True)
    cache={k:torch.cat(v) for k,v in tensors.items()}
    cache.update(levels=torch.cat(levels),presence=torch.cat(presence),metadata=metadata,
        split=original.split,fingerprint=fingerprint,max_image_clipped_fraction=max_clipped)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');torch.save(cache,temporary);temporary.replace(path)
    return cache


class CachedTraining(Dataset):
    def __init__(self,cache):
        if cache['split']!='train':raise ValueError('Only training features may enter the optimizer')
        selected=(cache['presence']>.5).all(1) & (cache['cp']>=.5).all(1)
        self.indices=selected.nonzero().flatten()
        if not len(self.indices):raise ValueError('No true and predicted ternary training rows')
        self.cache=cache

    def __len__(self):return len(self.indices)

    def __getitem__(self,index):
        i=int(self.indices[index])
        result={k:self.cache[k][i] for k in KEYS}
        result.update(levels=self.cache['levels'][i],presence=self.cache['presence'][i])
        return result


def tensor_inputs(batch,device):
    return {key:batch[key].to(device) for key in KEYS}


def prediction_frame(cache,cp,op,decoding):
    from spectral_core import decode_ordinal_numpy
    cp=cp.cpu().numpy();op=op.cpu().numpy()
    grades=decode_ordinal_numpy(op,mode=decoding)
    truth=cache['levels'].numpy().astype(int)
    rows=[]
    for i,meta in enumerate(cache['metadata']):
        row=dict(meta,mixture_count=int((truth[i]>0).sum()),ternary_route=bool((cp[i]>=.5).all()))
        for j,drug in enumerate(('DEL','CHL','TEB')):
            row.update({'true_'+drug:int(truth[i,j]),'ordinal_'+drug:int(grades[i,j]),
                'pred_'+drug:int(grades[i,j]) if cp[i,j]>=.5 else 0,
                'presence_probability_'+drug:float(cp[i,j]),'P_ge_M_'+drug:float(op[i,j,0]),
                'P_ge_H_'+drug:float(op[i,j,1])})
        rows.append(row)
    return pd.DataFrame(rows)


@torch.no_grad()
def evaluate_cached(model,cache,args,runtime):
    from correction_model import protection_check
    from spectral_core import present_only_corn_ordinal_loss
    model.eval();output=[];raw_logits=[]
    for start in range(0,len(cache['metadata']),args.batch_size):
        batch={k:cache[k][start:start+args.batch_size].to(args.device) for k in KEYS}
        predicted=model.forward_cached(batch)
        protection_check(predicted,batch)
        output.append(predicted['op'].cpu())
        raw_logits.append(predicted['logits'].cpu())
    op=torch.cat(output)
    frame=prediction_frame(cache,cache['cp'],op,args.ordinal_decoding)
    result=runtime.metrics(frame)
    labels=cache['levels'];presence=cache['presence']
    corn=present_only_corn_ordinal_loss(torch.cat(raw_logits),labels,presence)
    classification=torch.nn.functional.binary_cross_entropy(cache['cp'],presence)
    result.update(loss_total=float(.5*classification+.5*corn),loss_class=float(classification),
        loss_corn=float(corn),evaluation_split=cache['split'],
        ordinal_decoding=args.ordinal_decoding,max_image_clipped_fraction=cache['max_image_clipped_fraction'],
        loss_definition='0.5 presence BCE + 0.5 all-drug conditional CORN; correction training has additional regularization',
        output_protection='PASS')
    return result,frame
