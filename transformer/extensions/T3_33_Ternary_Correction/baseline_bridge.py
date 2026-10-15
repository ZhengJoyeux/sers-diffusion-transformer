"""Resolve the installed T3.32 code and preserve its trained input/decoding rules."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

EXPECTED = dict(samples=504, present_correct=1026, full_profile_correct=406,
                ternary_all_grade_correct=131, classification_exact_correct=504)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def configure(image_package):
    root=Path(image_package).expanduser().resolve()
    expected=json.loads((Path(__file__).parent/'t332_sources.json').read_text())
    actual={name:sha256(root/name) for name in expected}
    mismatch=[name for name in expected if expected[name]!=actual[name]]
    if mismatch:raise RuntimeError('T3.32 source differs from the delivered baseline: '+str(mismatch))
    sys.path.insert(0,str(root))
    import run_image as image_runtime
    if Path(image_runtime.__file__).resolve()!=root/'run_image.py':
        raise RuntimeError('Wrong T3.32 module imported')
    return image_runtime,actual


def load_baseline(runtime,path,device='cpu',strict_report=True):
    baseline,scale,state=runtime.load_checkpoint(path,device)
    if state.get('epoch')!=16:
        raise ValueError('Use T3.32 best_ordinal.pt from epoch 16, not latest epoch 28')
    if strict_report:
        found=state.get('validation_metrics',{})
        mismatch={k:dict(expected=v,actual=found.get(k)) for k,v in EXPECTED.items() if found.get(k)!=v}
        if mismatch:raise ValueError('Checkpoint is not the reported epoch-16 baseline: '+json.dumps(mismatch))
    if not state['model_config'].get('numeric_evidence'):
        raise ValueError('This correction uses the existing numeric-evidence T3.32 baseline')
    baseline.requires_grad_(False).eval()
    return baseline,scale,state


def state_digest(model):
    h=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        tensor=value.detach().cpu().contiguous()
        h.update(key.encode());h.update(str(tensor.dtype).encode());h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def assert_reference(actual,expected,context):
    mismatch={k:dict(expected=expected[k],actual=actual.get(k)) for k in EXPECTED if actual.get(k)!=expected[k]}
    if mismatch:raise RuntimeError(context+': '+json.dumps(mismatch))


def input_fingerprint(datasets,audit,baseline_sha,source_hashes):
    files={}
    repository=datasets[0].repository
    for source,records in [('real',repository.real_data),('generated',repository.generated_data or {})]:
        for condition,record in sorted(records.items()):
            files[source+'/'+condition]=sha256(record['path'])
    folder=Path(__file__).parent
    extraction_sources={name:sha256(folder/name) for name in ('feature_cache.py','correction_model.py')}
    payload=dict(split_audit=audit,data_file_sha256=files,baseline_sha256=baseline_sha,t332_sources=source_hashes,
                 extraction_sources=extraction_sources)
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    return digest,payload
