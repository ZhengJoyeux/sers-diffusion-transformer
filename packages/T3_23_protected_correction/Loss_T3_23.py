"""Training-only middle-grade separation and anchor boundary preservation.

Labels decide training masks only; no label-dependent gate enters inference.
The H decision uses cumulative P(H), not conditional P(H | >=M).
"""
import torch
from torch.nn import functional as F
from Model_T3_17 import present_only_corn_ordinal_loss
from Model_T3_23 import corn_probabilities


def cumulative_logits(conditional_logits):
    """Exact log odds for sigmoid(a) and sigmoid(a)*sigmoid(b), stably."""
    a,b=conditional_logits.unbind(-1)
    high=a+b-torch.logsumexp(torch.stack((torch.zeros_like(a),a,b),-1),-1)
    return torch.stack((a,high),-1)


def masked_mean(values,mask):
    weight=mask.to(values.dtype)
    return (values*weight).sum()/weight.sum().clamp_min(1.)


def correction_objective(logits,base_logits,delta,levels,presence,*,middle_weight=.1,
                         preservation_weight=.2,middle_margin=.35,preservation_margin_cap=.5,
                         correction_penalty=.02):
    if min(middle_weight,preservation_weight,correction_penalty)<0:
        raise ValueError("Loss weights must be non-negative")
    if middle_margin<=0 or preservation_margin_cap<=0:raise ValueError("Margins must be positive")
    active=presence>.5
    cumulative=cumulative_logits(logits)
    base=base_logits.detach()
    anchor_cumulative=cumulative_logits(base)
    direction=torch.stack((levels>=2,levels>=3),-1).to(logits.dtype)*2.-1.
    signed=cumulative*direction
    anchor_signed=anchor_cumulative*direction
    anchor_grade=1+(corn_probabilities(base)>=.5).sum(-1)
    preserve=(active&(anchor_grade==levels))[...,None].expand_as(logits)
    # Preserve at most 0.5 log-odds of the original correct decision; no
    # requirement to copy arbitrary high-confidence anchor probabilities.
    required=anchor_signed.clamp(min=0.,max=preservation_margin_cap)
    preservation=masked_mean(F.relu(required-signed).square(),preserve)
    middle=(active&(levels==2)&(active.sum(-1,keepdim=True)>=2))[...,None].expand_as(logits)
    separation=masked_mean(F.relu(middle_margin-signed).square(),middle)
    penalty=masked_mean(delta.square(),active[...,None].expand_as(delta))
    corn=present_only_corn_ordinal_loss(logits,levels,presence)
    total=.7*corn+correction_penalty*penalty
    # Plain arm must use exactly the old objective, without extra zero-weight
    # branches in its backward graph.
    if middle_weight:total=total+middle_weight*separation
    if preservation_weight:total=total+preservation_weight*preservation
    return dict(total=total,corn=corn,middle=separation,preservation=preservation,penalty=penalty)
