"""Load the archived, already-trained T3.27 Adapter model without external PEFT."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor/t327'))
import run_optimization as previous
rt = previous.rt

DEFAULT_ANCHOR = '/home/wqzheng/project_transformer/outputs/t3_27_optimization_seed2026/train/adapter/checkpoints/best_accepted.pt'
EXPECTED = dict(samples=504, classification_exact_correct=503, present_correct=981,
                full_profile_correct=376, ternary_all_grade_correct=121)


def load_adapter(path, device='cpu'):
    model, payload = previous.load_optimized(path, device)
    if payload['epoch'] != 6 or payload['optimization_spec']['variant'] != 'adapter':
        raise ValueError('Use the T3.27 Adapter epoch-6 checkpoint, not T3.26 or LoRA')
    for key, value in EXPECTED.items():
        if payload['validation_metrics'][key] != value:
            raise ValueError(f'Unexpected anchor checkpoint metric: {key}')
    config = payload['model_config']
    if any(config.get(k, False) for k in ('use_boundary_specific_mixture_gate',
              'use_local_quantitative_branch', 'use_anchored_local_calibration')):
        raise ValueError('This feature tap expects the archived unmodified concentration path')
    return model, payload
