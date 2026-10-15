"""Read the existing dataset adapter without modifying project or data files."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from collections import defaultdict


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def load_dataset_module(project):
    root = Path(project).expanduser().resolve()
    path = root/'Dataset.py'
    if not path.exists():
        path = root/'transformer'/'Dataset.py'
    if not path.exists():
        raise FileNotFoundError('Cannot find the current Dataset.py under '+str(root))
    spec = importlib.util.spec_from_file_location('Dataset', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['Dataset'] = module
    spec.loader.exec_module(module)
    if tuple(module.REAL_TRAIN_INDICES) != tuple(range(12)) or tuple(module.REAL_VALIDATION_INDICES) != tuple(range(12,16)) or tuple(module.REAL_TEST_INDICES) != tuple(range(16,20)):
        raise RuntimeError('The existing fixed 12/4/4 split has changed')
    return module, path


def build_data(args):
    module, path = load_dataset_module(args.project)
    real = args.real_root or path.parent/'data'/'real'
    generated = args.generated_root or path.parent/'data'/'generated'
    datasets = module.build_datasets(include_generated_train=args.include_generated,
        maximum_generated_per_condition=args.maximum_generated_per_condition,
        require_complete_generated=True, concentration_map=None,
        real_root=real, generated_root=generated)
    audit = audit_splits(datasets)
    audit.update(dataset_source=str(path), dataset_sha256=sha256(path),
                 real_root=str(Path(real).resolve()), generated_root=str(Path(generated).resolve()))
    return module, datasets, audit


def audit_splits(datasets):
    result, split_sets = {}, []
    for ds, expected in zip(datasets, (range(12), range(12,16), range(16,20))):
        groups = defaultdict(list)
        identities = []
        generated = 0
        for ref in ds.samples:
            identities.append((ref.source, ref.condition, int(ref.spectrum_index)))
            if ref.source == 'real':
                groups[ref.condition].append(int(ref.spectrum_index))
            else:
                generated += 1
                if ds.split != 'train':
                    raise RuntimeError('Generated spectra leaked into a held-out split')
        if len(groups) != 126 or any(indices != list(expected) for indices in groups.values()):
            raise RuntimeError('Real 12/4/4 identities differ')
        real_ids = {(c, i) for c, indices in groups.items() for i in indices}
        split_sets.append(real_ids)
        if len(identities) != len(set(identities)):
            raise RuntimeError('Duplicate spectrum identities')
        result[ds.split] = dict(real_rows=len(real_ids), generated_rows=generated, rows=len(ds),
            identity_sha256=hashlib.sha256(json.dumps(identities, ensure_ascii=False).encode()).hexdigest())
    if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i)):
        raise RuntimeError('Overlapping real identities')
    result['test_features_read'] = 0
    return result
