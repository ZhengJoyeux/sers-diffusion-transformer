"""Extract fusion-input queries once; retain the original T3.33 cache unchanged."""
from pathlib import Path
import hashlib
import json
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def extract_queries(baseline, image_inputs):
    from correction_model import extract_frozen
    captured = []
    def hook(module, inputs):
        captured.append(inputs[0].detach())
    handle = baseline.mixture_aware_query_fusion.register_forward_pre_hook(hook)
    try:
        result = extract_frozen(baseline, image_inputs)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError('Expected one concentration-fusion input tensor')
    result['queries'] = captured[0]
    return result


def fingerprint(state):
    payload = dict(original_cache=state['cache_fingerprint'], baseline_digest=state['baseline_digest'],
                   t332_sources=state['t332_hashes'], extractor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


@torch.no_grad()
def augment_cache(baseline, dataset, scale, original, args, query_fingerprint, runtime):
    if original['split'] not in ('train','validation') or dataset.split != original['split']:
        raise ValueError('Query extraction is restricted to train and validation')
    path = args.output_directory/'query_cache'/query_fingerprint/(dataset.split+'.pt')
    expected = original['metadata']
    dimension = baseline.concentration_projection.in_features
    if path.exists():
        saved = torch.load(path,map_location='cpu',weights_only=False)
        if saved['query_fingerprint'] != query_fingerprint or saved['metadata'] != expected or saved['original_fingerprint'] != original['fingerprint']:
            raise RuntimeError('Query cache identity/provenance mismatch')
        queries = saved['queries']
        if tuple(queries.shape) != (len(expected),3,dimension) or not torch.isfinite(queries).all():
            raise RuntimeError('Invalid cached queries')
        print(f'Using {dataset.split} query cache: {len(expected)} rows',flush=True)
        result = dict(original,queries=queries,query_fingerprint=query_fingerprint,query_cache_path=str(path))
        return result
    from image_data import LineImageDataset
    wrapped = LineImageDataset(dataset,scale,args.image_cache)
    loader = DataLoader(wrapped,batch_size=args.cache_batch_size,shuffle=False,
                        num_workers=args.num_workers,pin_memory=args.device.startswith('cuda'))
    queries, metadata = [], []
    worst = 0.
    baseline.requires_grad_(False).eval()
    for step,batch in enumerate(loader):
        start = len(metadata)
        actual = extract_queries(baseline,runtime.model_inputs(batch,args.device))
        n = len(actual['cp'])
        for key in ('cp','op','logits'):
            values = actual[key].cpu()
            target = original[key][start:start+n]
            if values.shape != target.shape or not torch.allclose(values,target,rtol=2e-4,atol=2e-4):
                raise RuntimeError('Re-extracted baseline differs from original cache: '+key)
            worst = max(worst,float((values-target).abs().max()))
        if not torch.equal(batch['concentration_level'].cpu(),original['levels'][start:start+n]) or not torch.equal(batch['class_target'].cpu(),original['presence'][start:start+n]):
            raise RuntimeError('Query extraction labels differ from original fixed split')
        queries.append(actual['queries'].cpu())
        for i in range(n):
            metadata.append(dict(condition=batch['condition'][i],source=batch['source'][i],
                spectrum_index=int(batch['spectrum_index'][i]),matrix=batch['matrix_name'][i],split=dataset.split))
        if metadata[start:] != expected[start:start+n]:
            raise RuntimeError('Query extraction row identities differ from original cache')
        if step%25==0 or step+1==len(loader):
            print(f'Extracting frozen {dataset.split} fusion queries: {len(metadata)}/{len(expected)}',flush=True)
    if metadata != expected:
        raise RuntimeError('Query extraction row count differs')
    queries = torch.cat(queries)
    if tuple(queries.shape)!=(len(expected),3,dimension) or not torch.isfinite(queries).all():
        raise RuntimeError('Invalid extracted queries')
    saved = dict(queries=queries,metadata=metadata,split=dataset.split,query_fingerprint=query_fingerprint,
                 original_fingerprint=original['fingerprint'],baseline_prediction_max_abs_difference=worst)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(saved,temporary)
    temporary.replace(path)
    return dict(original,queries=queries,query_fingerprint=query_fingerprint,query_cache_path=str(path))
