"""Train a copied concentration-fusion branch on frozen image-derived queries."""
import copy
import hashlib
import torch
from torch import nn
from torch.nn import functional as F

VERSION = 'T3.36_ternary_concentration_fusion_finetune_v1'
DRUGS = (0, 2)


def probabilities(logits):
    first = logits[..., 0].sigmoid()
    return torch.stack((first, first * logits[..., 1].sigmoid()), -1)


class ConcentrationBranch(nn.Module):
    def __init__(self, fusion, projection, heads):
        super().__init__()
        self.fusion = copy.deepcopy(fusion)
        self.projection = copy.deepcopy(projection)
        self.heads = copy.deepcopy(heads)

    def forward(self, queries, cp):
        features = self.projection(self.fusion(queries, cp.detach()))
        return torch.stack([head(features[:, j]) for head, j in zip(self.heads, DRUGS)], 1)


class FusionFinetuner(nn.Module):
    def __init__(self, teacher, original_branch):
        super().__init__()
        self.teacher = teacher.requires_grad_(False).eval()
        self.frozen_branch = copy.deepcopy(original_branch).requires_grad_(False).eval()
        self.branch = copy.deepcopy(original_branch).requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        self.frozen_branch.eval()
        return self

    def forward_cached(self, data):
        if set(data) != {'features', 'cp', 'op', 'logits', 'queries'}:
            raise ValueError('Forward accepts frozen spectrum features/queries only; no labels or condition IDs')
        if data['queries'].shape != (len(data['cp']), 3, self.branch.projection.in_features):
            raise ValueError('Invalid fusion-input query shape')
        with torch.no_grad():
            teacher = self.teacher.forward_cached({k: data[k] for k in ('features','cp','op','logits')})
            original = self.frozen_branch(data['queries'], data['cp'])
        replacement = self.branch(data['queries'], data['cp'])
        difference = replacement - original
        # Identical weights define an exact zero update in evaluation. Different
        # CUDA attention/GEMM paths can otherwise introduce a tiny logit delta.
        # Use exact parameter equality, never a tolerance on learned updates.
        # Training keeps the ordinary differentiable/dropout path unchanged.
        if not self.training:
            with torch.no_grad():
                identical = all(torch.equal(a, b) for a, b in
                                zip(self.branch.parameters(), self.frozen_branch.parameters()))
            if identical:
                difference = torch.zeros_like(difference)
        change = torch.zeros_like(teacher['logits'])
        change[:, 0] = difference[:, 0]
        change[:, 2] = difference[:, 1]
        allowed = teacher['route'][:, None, None] & torch.tensor([True, False, True], device=change.device)[None, :, None]
        logits = torch.where(allowed, teacher['logits'] + change, teacher['logits'])
        updated = teacher['op'] + probabilities(logits) - probabilities(teacher['logits'])
        pm = updated[..., 0].clamp(0, 1)
        ph = torch.minimum(updated[..., 1].clamp(0, 1), pm)
        op = torch.where(allowed, torch.stack((pm, ph), -1), teacher['op'])
        op = torch.where(allowed & (change != 0).any(-1, keepdim=True), op, teacher['op'])
        return dict(cp=teacher['cp'], op=op, logits=logits, route=teacher['route'],
                    teacher_op=teacher['op'], teacher_logits=teacher['logits'], logit_change=change)

    def frozen_digest(self):
        digest = hashlib.sha256()
        for key, value in sorted(self.state_dict().items()):
            if key.startswith('branch.'):
                continue
            x = value.detach().cpu().contiguous()
            digest.update(key.encode())
            digest.update(x.numpy().tobytes())
        return digest.hexdigest()

    def anchor_loss(self):
        numerator = sum((p - old.detach()).square().sum() for p, old in zip(self.branch.parameters(), self.frozen_branch.parameters()))
        denominator = sum(old.detach().square().sum() for old in self.frozen_branch.parameters()).clamp_min(1e-8)
        return numerator / denominator


def head_loss(output, levels, *, stability_weight=.2):
    if not ((levels > 0).all(1) & output['route']).all():
        raise ValueError('Only selected true and predicted ternary training rows may enter loss')
    target = levels[:, DRUGS]
    logits = output['logits'][:, DRUGS]
    op = output['op'][:, DRUGS]
    reference = output['teacher_op'][:, DRUGS]
    y = torch.stack((target >= 2, target >= 3), -1).float()
    active = torch.stack((torch.ones_like(target, dtype=torch.bool), target >= 2), -1)
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
    corn = ((bce * active).sum((0, 2)) / active.sum((0, 2)).clamp_min(1)).mean()
    correct = (1 + (reference >= .5).long().sum(-1)) == target.long()
    stability = ((op - reference).square() * correct[..., None]).sum() / (2 * correct.sum()).clamp_min(1)
    return corn + stability_weight * stability


class ImageDeployment(nn.Module):
    def __init__(self, baseline, branch):
        super().__init__()
        self.baseline = baseline.requires_grad_(False).eval()
        self.fusion_model = branch.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, image_inputs):
        from query_cache import extract_queries
        return self.fusion_model.forward_cached(extract_queries(self.baseline, image_inputs))
