"""Frozen image baseline plus bounded DEL/TEB logit corrections on predicted ternaries."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

VERSION='T3.33_frozen_image_ternary_DEL_TEB_v1'
DRUG_INDICES=(0,2)


def ordinal_probabilities(logits):
    ge_m=logits[...,0].sigmoid()
    return torch.stack((ge_m,ge_m*logits[...,1].sigmoid()),-1)


@torch.no_grad()
def extract_frozen(baseline,data):
    if baseline.training or any(p.requires_grad for p in baseline.parameters()):
        raise RuntimeError('Baseline must be fully frozen and in eval mode')
    captured=[]
    handle=baseline.concentration_projection.register_forward_hook(lambda m,i,o:captured.append(o))
    try:output=baseline(data)
    finally:handle.remove()
    if len(captured)!=1:raise RuntimeError('Expected one shared concentration feature tensor')
    evidence=baseline.multi_peak_evidence
    if evidence is None:raise ValueError('Numeric evidence is required by this correction')
    _,stats,available=evidence.extract(data['raw_intensity'],data['measured_mask'])
    numerical=((stats-evidence.stats_mean)/evidence.stats_scale).clamp(-8.,8.)
    numerical=numerical.masked_fill(~available[...,None],0.)
    features=torch.cat((captured[0].flatten(1),output['logits'].flatten(1),output['cp'],
                        numerical.flatten(1),available.to(numerical.dtype)),1)
    if not torch.isfinite(features).all():raise RuntimeError('Non-finite frozen features')
    return dict(features=features.detach(),cp=output['cp'].detach(),op=output['op'].detach(),
                logits=output['logits'].detach())


class TernaryCorrector(nn.Module):
    def __init__(self,baseline,feature_dim=None,hidden=32,bound=.5,dropout=.1):
        super().__init__()
        if hidden<4 or not 0<bound<=1 or not 0<=dropout<1:
            raise ValueError('Invalid correction capacity/bound/dropout')
        if feature_dim is None:
            evidence=baseline.multi_peak_evidence
            if evidence is None:raise ValueError('Baseline requires numeric evidence')
            feature_dim=3*baseline.concentration_projection.out_features+6+3+10*len(evidence.centers)
        self.baseline=baseline.requires_grad_(False).eval()
        self.config=dict(feature_dim=int(feature_dim),hidden=int(hidden),bound=float(bound),dropout=float(dropout))
        self.register_buffer('feature_mean',torch.zeros(feature_dim))
        self.register_buffer('feature_std',torch.ones(feature_dim))
        self.register_buffer('normalization_fitted',torch.tensor(False))
        self.correction_heads=nn.ModuleList([
            nn.Sequential(nn.Linear(feature_dim,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,2))
            for _ in DRUG_INDICES])
        for head in self.correction_heads:
            nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)

    def train(self,mode=True):
        super().train(mode);self.baseline.eval()
        return self

    @torch.no_grad()
    def fit_normalization(self,features,real_training,route,*,split):
        if bool(self.normalization_fitted):raise RuntimeError('Normalization already fitted')
        if split!='train':raise ValueError('Correction normalization uses real training only')
        if features.ndim!=2 or features.shape[1]!=self.config['feature_dim']:
            raise ValueError('Correction feature shape differs')
        if real_training.shape!=route.shape or route.shape!=(len(features),):
            raise ValueError('Normalization row masks differ')
        selected=real_training.bool() & route.bool()
        if int(selected.sum())<8:raise ValueError('At least eight routed real training rows are required')
        values=features[selected].float()
        if not torch.isfinite(values).all():raise ValueError('Non-finite fit features')
        self.feature_mean.copy_(values.mean(0))
        self.feature_std.copy_(values.std(0,unbiased=False).clamp_min(.05))
        self.normalization_fitted.fill_(True)
        return dict(fit_scope='routed_real_train_only',rows=int(selected.sum()),
                    generated_fit_rows=0,validation_fit_rows=0,test_fit_rows=0)

    def forward_cached(self,data):
        if set(data)!={'features','cp','op','logits'}:
            raise ValueError('Cached forward accepts frozen features/predictions only; no labels or metadata')
        if not bool(self.normalization_fitted):raise RuntimeError('Fit correction normalization first')
        features,cp,base_op,base_logits=[data[k] for k in ('features','cp','op','logits')]
        b=len(features)
        if features.shape!=(b,self.config['feature_dim']) or cp.shape!=(b,3) or base_op.shape!=(b,3,2) or base_logits.shape!=(b,3,2):
            raise ValueError('Invalid cached tensor shapes')
        z=((features-self.feature_mean)/self.feature_std).clamp(-8.,8.)
        delta=self.config['bound']*torch.tanh(torch.stack([head(z) for head in self.correction_heads],1))
        route=(cp>=.5).all(1)  # Predicted presence only, never ground-truth classes.
        delta=delta*route[:,None,None]
        changes=torch.zeros_like(base_logits)
        changes[:,0]=delta[:,0];changes[:,2]=delta[:,1]
        logits=base_logits+changes
        # Preserve exact baseline probabilities even when cached/inference batch sizes differ.
        updated=base_op+(ordinal_probabilities(logits)-ordinal_probabilities(base_logits))
        ge_m=updated[...,0].clamp(0.,1.)
        ge_h=torch.minimum(updated[...,1].clamp(0.,1.),ge_m)
        updated=torch.stack((ge_m,ge_h),-1)
        allowed=route[:,None,None] & torch.tensor([True,False,True],device=route.device)[None,:,None]
        op=torch.where(allowed,updated,base_op)
        # Also retain the original CHL/non-routed raw logits exactly.
        logits=torch.where(allowed,logits,base_logits)
        return dict(cp=cp,op=op,logits=logits,delta=delta,route=route,base_op=base_op,base_logits=base_logits)

    def forward(self,data):
        return self.forward_cached(extract_frozen(self.baseline,data))


def protection_check(output,reference):
    route=output['route']
    checks=dict(classification_exact=torch.equal(output['cp'],reference['cp']),
        CHL_probabilities_exact=torch.equal(output['op'][:,1],reference['op'][:,1]),
        CHL_logits_exact=torch.equal(output['logits'][:,1],reference['logits'][:,1]),
        non_routed_probabilities_exact=torch.equal(output['op'][~route],reference['op'][~route]),
        non_routed_logits_exact=torch.equal(output['logits'][~route],reference['logits'][~route]))
    if not all(checks.values()):raise RuntimeError('Frozen output protection failed: '+str(checks))
    if not (output['op'][...,1]<=output['op'][...,0]+1e-7).all():raise RuntimeError('CORN ordering violated')
    return checks


def correction_loss(output,levels,presence,*,error_multiplier=2.,delta_weight=.01,stability_weight=.05):
    # Labels select supervised training rows only; they never determine inference routing.
    selected=(presence>.5).all(1) & output['route']
    if not selected.any():return output['delta'].sum()*0.,dict(supervised_rows=0)
    targets=levels[selected][:,DRUG_INDICES]
    logits=output['logits'][selected][:,DRUG_INDICES]
    base=output['base_op'][selected][:,DRUG_INDICES]
    ordinal=output['op'][selected][:,DRUG_INDICES]
    correct=(1+(base>=.5).long().sum(-1))==targets.long()
    weight=torch.where(correct,torch.ones_like(targets),torch.full_like(targets,error_multiplier))
    y=torch.stack((targets>=2,targets>=3),-1).float()
    active=torch.stack((torch.ones_like(targets,dtype=torch.bool),targets>=2),-1)
    losses=F.binary_cross_entropy_with_logits(logits,y,reduction='none')
    per_drug=(losses*active*weight[...,None]).sum((0,2))/(active*weight[...,None]).sum((0,2)).clamp_min(1.)
    supervised=per_drug.mean()
    penalty=output['delta'][selected].square().mean()
    stable=((ordinal-base).square()*correct[...,None]).sum()/(correct.sum()*2).clamp_min(1.)
    loss=supervised+delta_weight*penalty+stability_weight*stable
    return loss,dict(supervised_rows=int(selected.sum()),corn=float(supervised.detach()),
                     delta_penalty=float(penalty.detach()),stability=float(stable.detach()))
