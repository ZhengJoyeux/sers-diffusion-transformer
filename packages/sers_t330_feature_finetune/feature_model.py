"""Fine-tune existing concentration features from the verified T3.28 baseline."""
from pathlib import Path
import copy
import sys
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor/base'))
import combined_model as old
from combined_model import DELOnlyModel, TernaryModel, previous, rt
from ternary_model import extract, level_probabilities, encode_levels, decode_states, joint_marginals
from schedule import UpdateSchedule

VERSION = 'T3.30_feature_finetune_v1'
BASE_SHA = '7819ca86c16ee302cf00ca89021da3489fd93b62c3ffda5e192e648a0a5201fb'
DEFAULT_BASE = '/home/wqzheng/project_transformer/outputs/t3_28_del_only_verified_seed2026/checkpoints/combined_verified.pt'
ARMS = ('concentration', 'last_encoder')
MODES = ('DEL_only', 'TEB_only', 'DEL_TEB')
DRUGS = ('DEL', 'CHL', 'TEB')
BASE_METRICS = dict(samples=504, classification_exact_correct=503, present_correct=984,
                    full_profile_correct=379, ternary_all_grade_correct=124)


class FeatureModel(nn.Module):
    def __init__(self, base_config, options):
        super().__init__()
        self.base_config = base_config; self.options = dict(options)
        self.arm = options['arm']
        if self.arm not in ARMS or options['deployment_mode'] not in MODES:
            raise ValueError('Invalid feature-finetune options')
        self.teacher = DELOnlyModel(TernaryModel(base_config['model_config'],
            base_config['adapter_spec'], base_config['ternary_options'])).requires_grad_(False)
        self.student = copy.deepcopy(self.teacher.joint.anchor).requires_grad_(False)
        if len(self.student.transformer_encoder.layers) != 6:
            raise ValueError('Expected the actual six-layer archived model')
        for name, parameter in self.student.named_parameters():
            concentration = name.startswith(('concentration_projection.', 'mixture_aware_query_fusion.',
                                             'ordinal_heads.0.', 'ordinal_heads.2.'))
            encoder = self.arm == 'last_encoder' and name.startswith('transformer_encoder.layers.5.')
            parameter.requires_grad_(concentration or encoder)
        self.train(False)

    def initialize(self, payload):
        if payload.get('checkpoint_version') != old.VERSION or payload.get('decoding_policy') != old.POLICY:
            raise ValueError('Start from verified T3.28 DEL-only, not a rejected T3.29 head')
        self.teacher.load_state_dict(payload['model_state_dict'], strict=True)
        self.student.load_state_dict(self.teacher.joint.anchor.state_dict(), strict=True)
        return self

    def train(self, mode=True):
        super().train(mode)
        # Keep dropout and all stored normalization fixed; eval permits parameter gradients.
        self.teacher.eval(); self.student.eval()
        return self

    def forward(self, data):
        allowed = {'raw', 'smoothed', 'percentile', 'valid_mask', 'measured_mask', 'raw_intensity'}
        if set(data) != allowed: raise ValueError('Spectrum tensors/masks only; no labels or condition IDs')
        with torch.no_grad():
            reference = self.teacher(data)
            cp, base_grade, base_op = self.teacher.predict(reference)
        _, op, logits, features = extract(self.student, data)
        probabilities = level_probabilities(op)
        prior = (probabilities[:, 0, :, None, None] * probabilities[:, 1, None, :, None] *
                 probabilities[:, 2, None, None, :]).reshape(-1, 27).clamp_min(1e-20).log()
        # Existing epoch-6 joint head is frozen, but differentiable with respect to student features.
        scores = prior + self.teacher.joint.joint_head(features, cp, op)
        return dict(cp=cp, base_grade=base_grade, base_op=base_op, scores=scores,
                    student_op=op, student_logits=logits, route=(cp >= .5).all(1))

    def predict(self, output, mode=None):
        mode = mode or self.options['deployment_mode']
        if mode not in MODES: raise ValueError('Invalid deployment mode')
        grade = output['base_grade'].clone(); op = output['base_op'].clone()
        route = output['route']
        joint_grade = decode_states(output['scores'].argmax(-1))
        marginals = joint_marginals(output['scores'].softmax(-1))
        cumulative = torch.stack((marginals[..., 1:].sum(-1), marginals[..., 2]), -1)
        if mode != 'TEB_only':
            grade[route, 0] = joint_grade[route, 0]; op[route, 0] = cumulative[route, 0]
        if mode != 'DEL_only':
            grade[route, 2] = 1 + (output['student_op'][route, 2] >= .5).sum(-1)
            op[route, 2] = output['student_op'][route, 2]
        return output['cp'], grade, op

    def l2sp(self):
        reference = dict(self.teacher.joint.anchor.named_parameters())
        return sum((p - reference[n]).square().sum() for n, p in self.student.named_parameters() if p.requires_grad)


def supervised_loss(model, output, classes, levels, label_smoothing=.05, teb_weight=.5, l2sp_weight=.01):
    selected = (classes > .5).all(1)
    if not bool(selected.any()):
        zero = output['scores'].sum() * 0.
        return zero, dict(ternary_samples=0, joint_ce=0., teb_corn=0., l2sp=0.)
    y = levels[selected]
    joint = F.cross_entropy(output['scores'][selected], encode_levels(y), label_smoothing=label_smoothing)
    z = output['student_logits'][selected, 2]
    yy = y[:, 2]; mh = yy >= 2
    first = F.binary_cross_entropy_with_logits(z[:, 0], mh.float(), reduction='sum')
    second = F.binary_cross_entropy_with_logits(z[mh, 1], (yy[mh] >= 3).float(), reduction='sum')
    corn = (first + second) / (len(yy) + int(mh.sum()))
    anchor = model.l2sp()
    loss = joint + teb_weight * corn + l2sp_weight * anchor
    return loss, dict(ternary_samples=int(selected.sum()), joint_ce=float(joint.detach()),
                      teb_corn=float(corn.detach()), l2sp=float(anchor.detach()))


def base_config(payload):
    return {k: payload[k] for k in ('model_config', 'adapter_spec', 'ternary_options')}


def from_baseline(path, arm, device='cpu'):
    if rt.sha256(path) != BASE_SHA: raise ValueError('Verified baseline checkpoint SHA256 differs')
    state = torch.load(path, map_location='cpu', weights_only=False)
    model = FeatureModel(base_config(state), dict(arm=arm, deployment_mode='DEL_TEB')).initialize(state)
    return model.to(device), state


def save_checkpoint(path, model, epoch, metrics, config, mode=None, optimizer=None, schedule=None, selection=None):
    options = dict(model.options)
    if mode is not None: options['deployment_mode'] = mode
    state = dict(checkpoint_version=VERSION, base_config=model.base_config, options=options,
                 model_state_dict={n: p.detach().cpu().clone() for n, p in model.state_dict().items()},
                 epoch=epoch, validation_metrics=metrics, experiment_config=config, selection=selection,
                 baseline_sha256=BASE_SHA, test_predictions=0, selection_scope='exploratory reused validation')
    if optimizer is not None:
        state.update(optimizer_state_dict=optimizer.state_dict(), schedule_state=schedule.state_dict(), rng_state=rt.capture_rng())
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + '.tmp'); torch.save(state, tmp); tmp.replace(path)


def load_checkpoint(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('checkpoint_version') != VERSION or state.get('baseline_sha256') != BASE_SHA:
        raise ValueError('Expected a T3.30 feature-finetune checkpoint')
    model = FeatureModel(state['base_config'], state['options'])
    model.load_state_dict(state['model_state_dict'], strict=True)
    return model.to(device).eval(), state


def optimizer(model, args):
    groups = {}
    for name, parameter in model.student.named_parameters():
        if not parameter.requires_grad: continue
        if name.startswith('transformer_encoder.layers.5.'):
            family = 'encoder_6'; maximum = args.encoder_lr
        elif name.startswith('ordinal_heads.'):
            family = 'ordinal'; maximum = args.head_lr
        else:
            family = 'concentration_features'; maximum = args.feature_lr
        # L2-SP anchors trained weights to the accepted model; no simultaneous decay towards zero.
        key = (family, maximum)
        g = groups.setdefault(key, dict(params=[], names=[], family=family, lr=maximum,
                                       maximum_lr=maximum, minimum_lr=maximum * args.minimum_ratio, weight_decay=0.))
        g['params'].append(parameter); g['names'].append(name)
    groups = list(groups.values())
    identities = [id(p) for g in groups for p in g['params']]
    intended = {id(p) for p in model.parameters() if p.requires_grad}
    if len(identities) != len(set(identities)) or set(identities) != intended: raise RuntimeError('Optimizer coverage differs')
    return torch.optim.AdamW(groups, betas=(.9, .999), eps=1e-8)
