"""Metrics and report writer extracted unchanged from T3.32."""
import json, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
DRUGS=("DEL","CHL","TEB")

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def metrics(frame):
    truth = frame[['true_'+d for d in DRUGS]].to_numpy(dtype=int)
    predicted = frame[['pred_'+d for d in DRUGS]].to_numpy(dtype=int)
    ordinal = frame[['ordinal_'+d for d in DRUGS]].to_numpy(dtype=int)
    actual = truth > 0; detected = predicted > 0
    present_correct = ((ordinal == truth) & actual).sum()
    profile = (predicted == truth).all(1)
    ternary = actual.all(1)
    def accuracy(num, den): return float(num/den) if den else None
    result = dict(samples=len(frame), present_n=int(actual.sum()), present_correct=int(present_correct),
        present_accuracy=accuracy(present_correct, actual.sum()),
        classification_exact_correct=int((actual == detected).all(1).sum()),
        classification_exact_accuracy=float((actual == detected).all(1).mean()),
        full_profile_correct=int(profile.sum()), full_profile_accuracy=float(profile.mean()),
        ternary_n=int(ternary.sum()), ternary_all_grade_correct=int(profile[ternary].sum()),
        ternary_full_profile_accuracy=accuracy(profile[ternary].sum(), ternary.sum()),
        present_level_code_mae=float(np.abs(ordinal-truth)[actual].mean()),
        per_drug={}, by_mixture={}, by_matrix={})
    for j, d in enumerate(DRUGS):
        tp = int((actual[:, j] & detected[:, j]).sum())
        fp = int((~actual[:, j] & detected[:, j]).sum())
        fn = int((actual[:, j] & ~detected[:, j]).sum())
        result['per_drug'][d] = dict(present_n=int(actual[:, j].sum()),
            correct=int(((ordinal[:, j] == truth[:, j]) & actual[:, j]).sum()),
            accuracy=accuracy(((ordinal[:, j] == truth[:, j]) & actual[:, j]).sum(), actual[:, j].sum()),
            classification_f1=2*tp/max(2*tp+fp+fn, 1),
            confusion_S_M_H=[[int(((truth[:, j] == a) & (ordinal[:, j] == b)).sum()) for b in (1,2,3)] for a in (1,2,3)])
    result['classification_macro_f1'] = float(np.mean([v['classification_f1'] for v in result['per_drug'].values()]))
    for k in (1,2,3):
        select = actual.sum(1) == k
        result['by_mixture'][str(k)] = dict(samples=int(select.sum()), full_profile_correct=int(profile[select].sum()),
            full_profile_accuracy=accuracy(profile[select].sum(), select.sum()))
    for matrix in sorted(frame['matrix'].unique()):
        select = frame['matrix'].to_numpy() == matrix
        result['by_matrix'][matrix] = dict(samples=int(select.sum()), full_profile_correct=int(profile[select].sum()),
            full_profile_accuracy=accuracy(profile[select].sum(), select.sum()))
    return result

def save_evaluation(directory, result, frame):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    write_json(directory/'ordinal_metrics.json', result)
    frame.to_csv(directory/'predictions.csv', index=False)
    pd.DataFrame([dict(pesticide=d, **{k:v for k,v in m.items() if k != 'confusion_S_M_H'})
                  for d,m in result['per_drug'].items()]).to_csv(directory/'ordinal_per_pesticide.csv', index=False)
    for d,m in result['per_drug'].items():
        pd.DataFrame(m['confusion_S_M_H'], index=['S','M','H'], columns=['S','M','H']).to_csv(directory/(d+'_confusion.csv'))
    for key, filename in [('by_mixture','ordinal_by_mixture_complexity.csv'), ('by_matrix','ordinal_by_matrix.csv')]:
        pd.DataFrame([dict(group=k, **v) for k,v in result[key].items()]).to_csv(directory/filename, index=False)

def set_rates(optimizer, update, total, warmup):
    for group in optimizer.param_groups:
        if warmup and update <= warmup:
            rate = group['maximum_lr']*update/warmup
        else:
            progress = (update-warmup)/max(total-warmup,1)
            rate = group['minimum_lr']+(group['maximum_lr']-group['minimum_lr'])*(1+math.cos(math.pi*progress))/2
        group['lr'] = rate
