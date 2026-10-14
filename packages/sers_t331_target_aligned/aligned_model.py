"""Target-only feature optimization from the accepted T3.30 TEB-only parent."""
from pathlib import Path
import copy
import json
import sys
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor/t330'))
import feature_model as parent
rt = parent.rt; previous = parent.previous; old = parent.old
UpdateSchedule = parent.UpdateSchedule
VERSION = 'T3.31_target_aligned_v1'
ARMS = ('teb_corn', 'del_map')
DEFAULT_BASE = '/home/wqzheng/project_transformer/outputs/t3_30_feature_finetune_combined_seed2026/train/concentration/checkpoints/TEB_only/best_accepted.pt'
PARENT_OPTIONS = dict(arm='concentration', deployment_mode='TEB_only')
BASE_METRICS = dict(samples=504, classification_exact_correct=503, full_profile_correct=380,
                    present_correct=985, ternary_all_grade_correct=125)
TEACHER_STATE_DIGEST = '0daaf0f37d742754b09aed5873e7a7b38462a51fadc23027028f53b507314ccc'
INPUT_KEYS = ('queries', 'fusion_cp', 'cp', 'grade', 'op')
CACHE_KEYS = INPUT_KEYS + ('classes', 'levels')
DRUGS = ('DEL', 'CHL', 'TEB')


def del_coordinate_scores(joint_scores):
    # argmax over these three group maxima equals DEL coordinate of the global joint MAP.
    return joint_scores.reshape(-1, 3, 9).amax(-1)


class ConcentrationPath(nn.Module):
    def __init__(self, source, target):
        super().__init__()
        self.mixture_aware_query_fusion = copy.deepcopy(source.mixture_aware_query_fusion).requires_grad_(True)
        self.concentration_projection = copy.deepcopy(source.concentration_projection).requires_grad_(True)
        self.ordinal_heads = nn.ModuleList([copy.deepcopy(h).requires_grad_(True) if j == target else h
                                           for j, h in enumerate(source.ordinal_heads)])
    def forward(self, query, cp):
        fused = self.mixture_aware_query_fusion(query, cp)
        if not isinstance(fused, torch.Tensor): raise ValueError('Unsupported concentration fusion')
        features = self.concentration_projection(fused)
        logits = torch.stack([head(features[:, j]) for j, head in enumerate(self.ordinal_heads)], 1)
        return features, logits


class AlignedModel(nn.Module):
    def __init__(self, baseline, arm):
        super().__init__()
        if arm not in ARMS: raise ValueError('Unknown task arm')
        if baseline.options != PARENT_OPTIONS: raise ValueError('Use accepted concentration/TEB_only parent')
        self.baseline = baseline.requires_grad_(False)
        self.arm = arm; self.target = 2 if arm == 'teb_corn' else 0
        source = self.baseline.student if self.target == 2 else self.baseline.teacher.joint.anchor
        config = baseline.base_config['model_config']
        if any(config.get(k, False) for k in ('use_boundary_specific_mixture_gate', 'use_local_quantitative_branch',
                                             'use_anchored_local_calibration', 'use_adaptive_mixture_gate')):
            raise ValueError('Unsupported feature path')
        self.path = ConcentrationPath(source, self.target)
        self.train(False)

    @property
    def source(self):
        return self.baseline.student if self.target == 2 else self.baseline.teacher.joint.anchor

    def train(self, mode=True):
        super().train(mode); self.baseline.eval(); self.path.eval()
        return self

    @torch.no_grad()
    def extract(self, data):
        captured = []
        handle = self.source.mixture_aware_query_fusion.register_forward_pre_hook(
            lambda module, inputs: captured.append((inputs[0].detach(), inputs[1].detach())))
        try:
            out = self.baseline(data); cp, grade, op = self.baseline.predict(out)
        finally: handle.remove()
        if len(captured) != 1 or captured[0][0].shape[1:] != (3, 128):
            raise RuntimeError('Expected exactly one frozen pre-fusion query tensor')
        query, fusion_cp = captured[0]
        return dict(queries=query, fusion_cp=fusion_cp, cp=cp, grade=grade, op=op)

    def predict_cached(self, cache):
        if set(cache) != set(INPUT_KEYS): raise ValueError('Cached prediction accepts no labels or metadata')
        features, logits = self.path(cache['queries'], cache['fusion_cp'])
        op = self.source._ordered_probabilities_from_logits(logits)
        grade = cache['grade'].clone(); cumulative = cache['op'].clone()
        route = (cache['cp'] >= .5).all(1)
        if self.target == 2:
            target_logits = logits[:, 2]
            updated_grade = 1 + (op[:, 2] >= .5).sum(-1)
            updated_op = op[:, 2]
        else:
            probabilities = parent.level_probabilities(op)
            scores = (probabilities[:, 0, :, None, None] * probabilities[:, 1, None, :, None] *
                      probabilities[:, 2, None, None, :]).reshape(-1, 27).clamp_min(1e-20).log()
            scores = scores + self.baseline.teacher.joint.joint_head(features, cache['cp'], op)
            target_logits = del_coordinate_scores(scores)
            updated_grade = 1 + target_logits.argmax(-1)
            marginal = parent.joint_marginals(scores.softmax(-1))[:, 0]
            updated_op = torch.stack((marginal[:, 1:].sum(-1), marginal[:, 2]), -1)
        grade[route, self.target] = updated_grade[route]
        cumulative[route, self.target] = updated_op[route]
        return dict(cp=cache['cp'], grade=grade, op=cumulative, route=route, target_logits=target_logits)

    def forward(self, data): return self.predict_cached(self.extract(data))
    def predict(self, output): return output['cp'], output['grade'], output['op']

    def l2sp(self):
        reference = dict(self.source.named_parameters())
        return sum((p - reference[name]).square().sum() for name, p in self.path.named_parameters() if p.requires_grad)


def target_loss(logits, levels, classes, target):
    selected = (classes > .5).all(1)
    if not bool(selected.any()): return logits.sum() * 0., 0
    y = levels[selected, target]
    if bool(((y < 1) | (y > 3)).any()): raise ValueError('Invalid target concentration grade')
    z = logits[selected]
    if target == 0: return F.cross_entropy(z, y.long() - 1), int(selected.sum())
    mh = y >= 2
    first = F.binary_cross_entropy_with_logits(z[:, 0], mh.float(), reduction='sum')
    second = F.binary_cross_entropy_with_logits(z[mh, 1], (y[mh] >= 3).float(), reduction='sum')
    return (first + second) / (len(y) + int(mh.sum())), int(selected.sum())


def training_loss(model, output, classes, levels, l2sp_weight=.01):
    supervised, count = target_loss(output['target_logits'], levels, classes, model.target)
    anchor = model.l2sp() if count else supervised * 0.
    total = supervised + l2sp_weight * anchor
    return total, dict(ternary_samples=count, target_loss=float(supervised.detach()), l2sp=float(anchor.detach()))


def validate_parent(state, model):
    if state.get('checkpoint_version') != parent.VERSION or state.get('epoch') != 17 or state.get('options') != PARENT_OPTIONS:
        raise ValueError('Use the accepted T3.30 concentration/TEB_only epoch17 checkpoint')
    for key, value in BASE_METRICS.items():
        if state['validation_metrics'][key] != value: raise ValueError('Parent metric differs: ' + key)
    # Verify archived persistent weights/buffers. Sinusoidal position_encoding is
    # reconstructed (persistent=False); its bitwise values can depend on the torch build.
    # Full local buffers, including that encoding, are protected throughout training.
    if previous.tensor_digest(list(model.teacher.state_dict().items())) != TEACHER_STATE_DIGEST:
        raise ValueError('Original protected teacher weights differ')
    experiment = state['experiment_config']
    if experiment.get('smoke') or experiment.get('epochs') != 50 or experiment.get('generated_per_condition') != 48:
        raise ValueError('Wrong parent experiment provenance')


def from_baseline(path, arm, device='cpu'):
    baseline, state = parent.load_checkpoint(path, 'cpu')
    baseline.requires_grad_(False); validate_parent(state, baseline)
    return AlignedModel(baseline, arm).to(device), state


def save_checkpoint(path, model, epoch, metrics, config, optimizer=None, schedule=None, selection=None):
    state = dict(checkpoint_version=VERSION, arm=model.arm, parent_base_config=model.baseline.base_config,
                 parent_options=model.baseline.options, parent_sha256=config['parent_sha256'],
                 model_state_dict={n: v.detach().cpu().clone() for n, v in model.state_dict().items()},
                 epoch=epoch, validation_metrics=metrics, experiment_config=config, selection=selection,
                 test_predictions=0, selection_scope='exploratory reused validation')
    if optimizer is not None:
        state.update(optimizer_state_dict=optimizer.state_dict(), schedule_state=schedule.state_dict(), rng_state=rt.capture_rng())
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + '.tmp'); torch.save(state, tmp); tmp.replace(path)


def load_checkpoint(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('checkpoint_version') != VERSION: raise ValueError('Expected a T3.31 checkpoint')
    baseline = parent.FeatureModel(state['parent_base_config'], state['parent_options']).requires_grad_(False)
    model = AlignedModel(baseline, state['arm'])
    model.load_state_dict(state['model_state_dict'], strict=True)
    return model.to(device).eval(), state


def optimizer(model, args):
    groups = {}
    for name, p in model.path.named_parameters():
        if not p.requires_grad: continue
        head = name.startswith('ordinal_heads.')
        maximum = args.head_lr if head else args.feature_lr
        g = groups.setdefault(head, dict(params=[], names=[], lr=maximum, maximum_lr=maximum,
            minimum_lr=maximum * args.minimum_ratio, weight_decay=0., family='target_head' if head else 'concentration_features'))
        g['params'].append(p); g['names'].append(name)
    identities = [id(p) for g in groups.values() for p in g['params']]
    if len(identities) != len(set(identities)) or set(identities) != {id(p) for p in model.parameters() if p.requires_grad}:
        raise RuntimeError('Optimizer parameter coverage differs')
    return torch.optim.AdamW(list(groups.values()), betas=(.9, .999), eps=1e-8)
