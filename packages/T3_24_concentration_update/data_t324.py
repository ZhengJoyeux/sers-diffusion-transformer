"""Exact current split audit and acquisition-range masks, no test inference."""
import hashlib
import json
from collections import Counter,defaultdict
import numpy as np
import torch

EXPECTED_IDS={"train":"5ecf918740a882dd33ebc984209264786fc06326e5483123f4d53a92026d40eb",
              "validation":"35fe0e060601c5628ee11f219e509425dbb8c707ff55180b4b43e988a629c9f3",
              "test":"8f74cbc2c26d8cfe5089e093b14cd304dd747b01c02c5b556cb4da939a84585e"}


def audit_selection(datasets):
    from Dataset import _generated_sample_indices
    if [len(d) for d in datasets]!=[7560,504,504]:raise RuntimeError("Dataset counts changed")
    ids={d.split:hashlib.sha256(json.dumps([(s.source,s.condition,int(s.spectrum_index)) for s in d.samples],ensure_ascii=False).encode()).hexdigest() for d in datasets}
    if ids!=EXPECTED_IDS:raise RuntimeError("Sample identities changed")
    train,val,test=datasets
    if Counter(s.source for s in train.samples)!={"real":1512,"generated":6048}:raise RuntimeError("Source counts changed")
    for ds,expected in zip(datasets,(list(range(12)),list(range(12,16)),list(range(16,20)))):
        groups=defaultdict(list)
        for ref in ds.samples:
            if ref.source=="real":groups[ref.condition].append(ref.spectrum_index)
        if len(groups)!=126 or any(x!=expected for x in groups.values()):raise RuntimeError("Real split changed")
    groups=defaultdict(list)
    for ref in train.samples:
        if ref.source=="generated":groups[ref.condition].append(ref.spectrum_index)
    for condition,indices in groups.items():
        pool=train.repository.generated_data[condition]["spectra"].shape[1]
        if pool!=200 or indices!=_generated_sample_indices(condition,pool,48,seed=2026):raise RuntimeError("Generated selection changed")
    print("SOURCE/SUBSET/SPLIT AUDIT: PASS",flush=True)
    return dict(status="PASS",counts=dict(train=7560,validation=504,test=504),
                source_counts=dict(real_train=1512,generated_train=6048,real_validation=504),
                sample_identity_sha256=ids,test_features=0,test_predictions=0)


def measured_masks(dataset,batch):
    from Dataset import MODEL_RAMAN_AXIS
    result=[]
    for source,condition in zip(batch["source"],batch["condition"]):
        record=(dataset.repository.real_data if source=="real" else dataset.repository.generated_data)[condition]
        n=int(record["original_axis_points"]);axis=np.asarray(record["axis"])
        start=float(record.get("original_axis_start_cm-1",axis[0]));end=float(record.get("original_axis_end_cm-1",axis[n-1]))
        result.append((MODEL_RAMAN_AXIS>=start-1e-6)&(MODEL_RAMAN_AXIS<=end+1e-6))
    return torch.from_numpy(np.asarray(result))&batch["valid_mask"].bool()


def metadata_rows(dataset,batch):
    rows=[]
    truth=batch["concentration_target"].round().long()
    for i in range(len(truth)):
        row=dict(condition=batch["condition"][i],source=batch["source"][i],
                 spectrum_index=int(batch["spectrum_index"][i]),matrix=batch["matrix_name"][i],
                 split=dataset.split,scope="real_validation" if dataset.split=="validation" else batch["source"][i]+"_train",
                 mixture_count=int((truth[i]>0).sum()))
        row.update({"true_"+d:int(truth[i,j]) for j,d in enumerate(("DEL","CHL","TEB"))})
        rows.append(row)
    return rows
