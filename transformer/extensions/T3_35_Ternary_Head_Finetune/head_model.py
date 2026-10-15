"""Copied DEL/TEB ordinal heads; frozen T3.33 teacher and shared features."""
import copy
import hashlib
import torch
from torch import nn
from torch.nn import functional as F

VERSION = 'T3.35_ternary_ordinal_head_finetune_v1'
DRUGS = (0, 2)


def probabilities(logits):
    first = logits[..., 0].sigmoid()
    return torch.stack((first, first * logits[..., 1].sigmoid()), -1)


class HeadFinetuner(nn.Module):
    def __init__(self, teacher, original_heads):
        super().__init__()
        self.teacher = teacher.requires_grad_(False).eval()
        self.frozen_heads = copy.deepcopy(original_heads).requires_grad_(False).eval()
        self.heads = copy.deepcopy(original_heads).requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        self.frozen_heads.eval()
        return self

    def forward_cached(self, data):
        if set(data) != {'features', 'cp', 'op', 'logits'}:
            raise ValueError('Forward accepts frozen spectrum features only, without labels or condition IDs')
        with torch.no_grad():
            teacher = self.teacher.forward_cached(data)
        features = data['features'][:, :192].reshape(-1, 3, 64)
        replacement = torch.stack([h(features[:, j]) for h, j in zip(self.heads, DRUGS)], 1)
        with torch.no_grad():
            original = torch.stack([h(features[:, j]) for h, j in zip(self.frozen_heads, DRUGS)], 1)
        change = torch.zeros_like(teacher['logits'])
        change[:, 0] = replacement[:, 0] - original[:, 0]
        change[:, 2] = replacement[:, 1] - original[:, 1]
        allowed = teacher['route'][:, None, None] & torch.tensor([True, False, True], device=change.device)[None, :, None]
        logits = torch.where(allowed, teacher['logits'] + change, teacher['logits'])
        # Preserve exact epoch-4 output at initialization, including cache roundoff.
        updated = teacher['op'] + probabilities(logits) - probabilities(teacher['logits'])
        pm = updated[..., 0].clamp(0, 1)
        ph = torch.minimum(updated[..., 1].clamp(0, 1), pm)
        op = torch.where(allowed, torch.stack((pm, ph), -1), teacher['op'])
        # The explicit zero-change branch avoids floating-point cancellation at identity.
        op = torch.where(allowed & (change != 0).any(-1, keepdim=True), op, teacher['op'])
        return dict(cp=teacher['cp'], op=op, logits=logits, route=teacher['route'],
                    teacher_op=teacher['op'], teacher_logits=teacher['logits'], logit_change=change)

    def frozen_digest(self):
        digest = hashlib.sha256()
        for key, value in sorted(self.state_dict().items()):
            if key.startswith('heads.'):
                continue
            x = value.detach().cpu().contiguous()
            digest.update(key.encode())
            digest.update(x.numpy().tobytes())
        return digest.hexdigest()

    def anchor_loss(self):
        numerator = sum((p - old.detach()).square().sum() for p, old in zip(self.heads.parameters(), self.frozen_heads.parameters()))
        denominator = sum(old.detach().square().sum() for old in self.frozen_heads.parameters()).clamp_min(1e-8)
        return numerator / denominator


def head_loss(output, levels, *, stability_weight=.2):
    if not ((levels > 0).all(1) & output['route']).all():
        raise ValueError('Only selected true and predicted ternary training rows may enter head loss')
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
    def __init__(self, baseline, heads):
        super().__init__()
        self.baseline = baseline.requires_grad_(False).eval()
        self.head_model = heads.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    def forward(self, image_inputs):
        from correction_model import extract_frozen
        return self.head_model.forward_cached(extract_frozen(self.baseline, image_inputs))
