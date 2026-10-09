"""Identical objective in both arms: CORN and anchored total-logit drift.

No T3.23 middle-grade or preservation loss is applied. Training labels only
select supervised CORN events and present-drug drift regularization.
Classification is frozen and contributes no optimization term.
"""
import torch
from Model_T3_17 import present_only_corn_ordinal_loss


def correction_objective(logits,base_logits,delta,levels,presence,*,middle_weight=0.,
                         preservation_weight=0.,middle_margin=.35,preservation_margin_cap=.5,
                         correction_penalty=.02):
    if middle_weight!=0 or preservation_weight!=0:
        raise ValueError("T3.24 comparison excludes middle and preservation extras")
    if correction_penalty<0:raise ValueError("Drift penalty must be non-negative")
    if logits.shape!=base_logits.shape or delta.shape!=logits.shape:
        raise ValueError("Logit/change shapes differ")
    active=(presence>.5)[...,None].expand_as(delta)
    weight=active.to(delta.dtype)
    penalty=(delta.square()*weight).sum()/weight.sum().clamp_min(1.)
    corn=present_only_corn_ordinal_loss(logits,levels,presence)
    return dict(total=.7*corn+correction_penalty*penalty,corn=corn,penalty=penalty)
