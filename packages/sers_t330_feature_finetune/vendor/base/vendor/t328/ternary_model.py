"""A 27-state residual joint head, routed only by frozen predicted presence."""
import torch
from torch import nn
from base_bridge import previous

ARMS = ('joint_head', 'joint_rdrop', 'joint_sam', 'joint_adapter', 'joint_adapter_l2sp')
CONTROLS = ('anchor', 'independent_map')


def level_probabilities(ordinal):
    probabilities = torch.stack((1-ordinal[..., 0], ordinal[..., 0]-ordinal[..., 1],
                                 ordinal[..., 1]), -1).clamp_min(1e-7)
    return probabilities / probabilities.sum(-1, keepdim=True)


def encode_levels(levels):
    values = levels.long()-1
    if values.ndim != 2 or values.shape[1] != 3 or bool(((values < 0) | (values > 2)).any()):
        raise ValueError('Joint labels must be three S/M/H codes in DEL,CHL,TEB order')
    return values[:, 0]*9 + values[:, 1]*3 + values[:, 2]


def decode_states(indices):
    return torch.stack((indices//9+1, indices//3%3+1, indices%3+1), -1)


def joint_marginals(probabilities):
    cube = probabilities.reshape(-1, 3, 3, 3)
    return torch.stack((cube.sum((2,3)), cube.sum((1,3)), cube.sum((1,2))), 1)


def marginal_loss(scores, levels):
    marginal = joint_marginals(scores.softmax(-1))
    values = marginal.gather(-1, (levels.long()-1).unsqueeze(-1)).squeeze(-1)
    return -values.clamp_min(1e-7).log().mean()


def symmetric_kl(left, right):
    lp, lq = left.log_softmax(-1), right.log_softmax(-1)
    return .5 * ((lp.exp()*(lp-lq)).sum(-1)+(lq.exp()*(lq-lp)).sum(-1)).mean()


class JointHead(nn.Module):
    def __init__(self, dimension, hidden=64, dropout=.1, bound=2.):
        super().__init__()
        self.bound = float(bound)
        self.network = nn.Sequential(nn.LayerNorm(3*dimension+9),
            nn.Linear(3*dimension+9, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 27))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features, cp, op):
        value = torch.cat((features.flatten(1), cp, op.flatten(1)), -1)
        return self.bound * torch.tanh(self.network(value))


def extract(model, data):
    capture = []
    handle = model.concentration_projection.register_forward_hook(
        lambda module, args, output: capture.append(output))
    try:
        values = model(data, return_ordinal=True, return_ordinal_logits=True)
    finally:
        handle.remove()
    if len(capture) != 1 or capture[0].ndim != 3 or capture[0].shape[1] != 3:
        raise RuntimeError('Expected one three-query concentration feature tensor')
    return values[0], values[2], values[3], capture[0]


class TernaryModel(nn.Module):
    def __init__(self, model_config, adapter_spec, options):
        super().__init__()
        self.model_config, self.adapter_spec, self.options = model_config, adapter_spec, options
        self.arm = options['arm']
        if self.arm not in ARMS+CONTROLS:
            raise ValueError('Unknown ternary experiment')
        self.anchor = previous.configure(previous.Model(**model_config), adapter_spec).requires_grad_(False)
        self.student = None
        if self.arm in ('joint_adapter', 'joint_adapter_l2sp'):
            self.student = previous.configure(previous.Model(**model_config), adapter_spec).requires_grad_(False)
            for name, parameter in self.student.named_parameters():
                if '.t327_adapter.' in name:
                    parameter.requires_grad_(True)
        self.joint_head = None if self.arm in CONTROLS else JointHead(
            2*model_config['dim_model'], options['hidden'], options['dropout'], options['bound'])
        self.train(False)

    def initialize(self, payload):
        self.anchor.load_state_dict(payload['model_state_dict'], strict=True)
        if self.student is not None:
            self.student.load_state_dict(payload['model_state_dict'], strict=True)
        return self

    def train(self, mode=True):
        super().train(mode)
        # Keep the original network's dropout and all normalization statistics fixed.
        # eval() does not prevent gradients through the student's trainable Adapters.
        self.anchor.eval()
        if self.student is not None:
            self.student.eval()
        if self.joint_head is not None:
            self.joint_head.train(mode)
        return self

    def forward(self, data):
        allowed = {'raw','smoothed','percentile','valid_mask','measured_mask','raw_intensity'}
        if set(data) != allowed:
            raise ValueError('forward accepts spectrum tensors/masks only; no labels or metadata')
        with torch.no_grad():
            cp, op, logits, anchor_features = extract(self.anchor, data)
        features = anchor_features
        if self.student is not None:
            _, _, _, features = extract(self.student, data)
        level_probs = level_probabilities(op)
        independent = (level_probs[:,0,:,None,None] * level_probs[:,1,None,:,None] *
                       level_probs[:,2,None,None,:]).reshape(-1,27)
        scores = independent.clamp_min(1e-20).log()
        if self.joint_head is not None:
            scores = scores + self.joint_head(features, cp, op)
        return dict(cp=cp, op=op, anchor_logits=logits, features=features,
                    scores=scores, route=(cp >= .5).all(1))

    def second_scores(self, output):
        if self.joint_head is None:
            raise ValueError('Control has no trainable head')
        probabilities = level_probabilities(output['op'])
        scores = (probabilities[:,0,:,None,None] * probabilities[:,1,None,:,None] *
                  probabilities[:,2,None,None,:]).reshape(-1,27).clamp_min(1e-20).log()
        return scores + self.joint_head(output['features'], output['cp'], output['op'])

    def predict(self, output):
        grade = 1 + (output['op'] >= .5).sum(-1)
        ordinal = output['op'].clone()
        if self.arm != 'anchor':
            selected = output['route']
            grade = torch.where(selected[:,None], decode_states(output['scores'].argmax(-1)), grade)
            marginals = joint_marginals(output['scores'].softmax(-1))
            cumulative = torch.stack((marginals[...,1:].sum(-1), marginals[...,2]), -1)
            ordinal = torch.where(selected[:,None,None], cumulative, ordinal)
        return output['cp'], grade, ordinal

    def l2sp(self):
        if self.student is None:
            return next(self.anchor.parameters()).sum()*0.
        reference = dict(self.anchor.named_parameters())
        return .5*sum((parameter-reference[name]).square().sum()
            for name,parameter in self.student.named_parameters() if parameter.requires_grad)


def training_loss(model, output, classes, levels, marginal_weight=.3, rdrop_weight=.1, l2sp_weight=.01):
    ternary = (classes > .5).all(1)
    if not bool(ternary.any()):
        return output['scores'].sum()*0., dict(ternary_samples=0, joint_ce=0., marginal_ce=0., rdrop_kl=0., l2sp=0.)
    selected_levels = levels[ternary]
    target = encode_levels(selected_levels)
    scores = output['scores'][ternary]
    ce = nn.functional.cross_entropy(scores, target)
    marginal = marginal_loss(scores, selected_levels)
    kl = scores.sum()*0.
    if model.arm == 'joint_rdrop':
        second = model.second_scores(output)[ternary]
        ce = .5*(ce+nn.functional.cross_entropy(second,target))
        marginal = .5*(marginal+marginal_loss(second,selected_levels))
        kl = symmetric_kl(scores,second)
    prior = model.l2sp() if model.arm == 'joint_adapter_l2sp' else scores.sum()*0.
    loss = ce + marginal_weight*marginal + rdrop_weight*kl + l2sp_weight*prior
    return loss, dict(ternary_samples=int(ternary.sum()), joint_ce=float(ce.detach()),
        marginal_ce=float(marginal.detach()), rdrop_kl=float(kl.detach()), l2sp=float(prior.detach()))
