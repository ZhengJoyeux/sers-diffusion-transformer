"""Frozen T3.18 and joint correction with the T3.22-compatible feature format.

Both T3.24 experiment arms select no_reference: matching dimensions are
masked after normalization. Matched CORN/drift loss is in Loss_T3_24.py.

No labels, matrix name, source type, condition name or sample index enter the
forward path. Matching coefficients are auxiliary evidence, not concentrations.
"""
from __future__ import annotations
import hashlib
import math
import numpy as np
import torch
from torch import nn
from Model_T3_17 import TransformerClassifyRegress_sep

DRUGS=("DEL","CHL","TEB")
MATRICES=("water","soil")
CORE=(1000.,1090.,1597.,1600.,2230.)
CONTRAST_PAIRS=((1000,1600),(1090,1597),(1600,1597),(1000,1090),
                (1000,2230),(1090,2230),(1600,2230))


def digest(model):
    h=hashlib.sha256()
    for key,value in model.state_dict().items():
        h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def corn_probabilities(logits):
    # Preserve the anchor's exact per-boundary computation, including layout.
    return TransformerClassifyRegress_sep._ordered_probabilities_from_logits(logits)


class ReferenceBank(nn.Module):
    def __init__(self,centers,half_width=24):
        super().__init__()
        centers=tuple(float(c) for c in centers)
        if tuple(sorted(set(centers)))!=centers or not set(CORE).issubset(centers):
            raise ValueError("Require sorted unique centers including all five core centers")
        if not 8<=int(half_width)<=60:raise ValueError("Invalid spectral window half-width")
        self.centers=centers;self.half_width=int(half_width)
        offsets=torch.arange(-self.half_width,self.half_width+1)
        indices=(torch.tensor(centers)-600).round().long()[:,None]+offsets[None,:]
        if indices.min()<0 or indices.max()>=1901:raise ValueError("Window exceeds axis")
        self.window_length=len(offsets);self.window_count=len(centers)
        self.signal_dimensions=indices.numel()
        self.register_buffer("indices",indices)
        self.register_buffer("signal_scale",torch.ones(len(centers)))
        self.register_buffer("area_floor",torch.ones(len(centers)))
        self.register_buffer("templates",torch.zeros(18,self.signal_dimensions))
        self.register_buffer("template_mask",torch.zeros(18,self.signal_dimensions,dtype=torch.bool))
        self.register_buffer("fitted",torch.tensor(False))
        self.reference_keys=[(matrix,drug,level) for matrix in MATRICES for drug in DRUGS for level in (1,2,3)]
        self.pair_indices=[(centers.index(float(a)),centers.index(float(b))) for a,b in CONTRAST_PAIRS]
        self.raw_dimensions=self.signal_dimensions+8*self.window_count+self.window_count+len(self.pair_indices)
        self.reference_dimensions=18*4

    def _windows(self,intensity,measured):
        if intensity.ndim!=3 or intensity.shape[1:]!=(1,1901):raise ValueError("Intensity must be [B,1,1901]")
        if measured.shape!=(len(intensity),1901):raise ValueError("Measured mask shape differs")
        if not torch.isfinite(intensity).all():raise ValueError("Non-finite intensity")
        available=measured[:,self.indices].all(-1)
        windows=intensity[:,0,self.indices]
        return windows.masked_fill(~available[...,None],0.),available

    def _measure(self,windows,available):
        flank=max(3,self.window_length//6)
        left=windows[...,:flank].mean(-1,keepdim=True)
        right=windows[...,-flank:].mean(-1,keepdim=True)
        coordinate=torch.arange(self.window_length,device=windows.device,dtype=windows.dtype)
        fraction=(coordinate-(flank-1)/2)/(self.window_length-flank)
        background=left+(right-left)*fraction
        residual=windows-background
        positive=residual.clamp_min(0.)
        area=positive.sum(-1);height=positive.amax(-1)
        offsets=coordinate-self.half_width
        mass=area.clamp_min(1e-8)
        centroid=(positive*offsets).sum(-1)/mass
        width=((positive*(offsets-centroid[...,None]).square()).sum(-1)/mass).clamp_min(0.).sqrt()
        asymmetry=(positive[...,self.half_width+1:].sum(-1)-positive[...,:self.half_width].sum(-1))/mass
        rms=residual.square().mean(-1).sqrt()
        noise=torch.cat((residual[...,:flank],residual[...,-flank:]),-1).std(-1,unbiased=False)
        scale=self.signal_scale[None,:]
        stats=torch.stack((torch.asinh(height/scale),torch.asinh(area/(self.window_length*scale)),
                           centroid/self.half_width,width/self.half_width,asymmetry,
                           torch.asinh(rms/scale),torch.asinh(background.mean(-1)/scale),
                           torch.asinh(noise/scale)),-1)
        return stats.masked_fill(~available[...,None],0.),area.masked_fill(~available,0.)

    @torch.no_grad()
    def fit(self,intensity,measured,levels,metadata):
        """Only real_train rows may fit scaling or any of the 18 templates."""
        if bool(self.fitted):raise RuntimeError("Reference bank already fitted")
        if len(metadata)!=len(intensity) or levels.shape!=(len(intensity),3):raise ValueError("Fit alignment differs")
        if any(row["source"]!="real" or row["split"]!="train" for row in metadata):
            raise ValueError("References and scaling require real training spectra only")
        if not torch.isfinite(levels).all() or not torch.equal(levels,levels.round()) or not ((levels>=0)&(levels<=3)).all():
            raise ValueError("References require 0/S/M/H level codes")
        windows,available=self._windows(intensity,measured)
        for j in range(self.window_count):
            values=windows[available[:,j],j].abs().flatten()
            if not len(values):raise ValueError(f"No measured training data for {self.centers[j]}")
            self.signal_scale[j]=values.median().clamp_min(1.)
        _,areas=self._measure(windows,available)
        for j in range(self.window_count):
            values=areas[available[:,j],j]
            positive=values[values>0]
            self.area_floor[j]=(positive.median()*.05).clamp_min(1.) if len(positive) else 1.
        signals=torch.asinh(windows/self.signal_scale[None,:,None])
        audit=[]
        present=(levels>0).sum(-1)
        for k,(matrix,drug,level) in enumerate(self.reference_keys):
            d=DRUGS.index(drug)
            take=(present==1)&(levels[:,d]==level)&torch.tensor([r["matrix"]==matrix for r in metadata],device=levels.device)
            refs=[row for row,keep in zip(metadata,take.cpu().tolist()) if keep]
            if len(refs)!=12 or len({r["condition"] for r in refs})!=1 or sorted(r["spectrum_index"] for r in refs)!=list(range(12)):
                raise ValueError(f"Expected exactly indices 0..11 of one real single-compound condition: {matrix}/{drug}/{level}")
            mask=available[take].all(0)
            template=signals[take].median(0).values.masked_fill(~mask[:,None],0.)
            self.templates[k].copy_(template.flatten())
            self.template_mask[k].copy_(mask[:,None].expand(-1,self.window_length).flatten())
            audit.append(dict(matrix=matrix,pesticide=drug,level=level,condition=refs[0]["condition"],
                              rows=12,indices=list(range(12)),source="real",split="train",measured_windows=int(mask.sum())))
        self.fitted.fill_(True)
        return dict(template_rows=216,scaling_rows=len(metadata),templates=audit,
                    validation_fit_rows=0,generated_fit_rows=0,test_fit_rows=0,
                    policy="Both water/soil reference banks are provided at inference; true matrix/levels are not input features")

    def forward(self,intensity,measured):
        if not bool(self.fitted):raise RuntimeError("Unfitted reference bank")
        windows,available=self._windows(intensity,measured)
        signals=torch.asinh(windows/self.signal_scale[None,:,None])
        stats,area=self._measure(windows,available)
        soft_log_area=torch.log1p(area/self.area_floor[None,:])
        contrasts=[]
        for a,b in self.pair_indices:
            contrasts.append((soft_log_area[:,a]-soft_log_area[:,b]).masked_fill(~(available[:,a]&available[:,b]),0.))
        raw=torch.cat((signals.flatten(1),stats.flatten(1),available.float(),torch.stack(contrasts,-1)),dim=-1)
        x=signals.flatten(1)[:,None,:]
        t=self.templates[None,:,:]
        mask=available[:,:,None].expand(-1,-1,self.window_length).flatten(1)[:,None,:]&self.template_mask[None,:,:]
        count=mask.sum(-1);weight=mask.to(x.dtype)
        dot=(x*t*weight).sum(-1)
        x2=(x.square()*weight).sum(-1);t2=(t.square()*weight).sum(-1)
        cosine=dot/(x2*t2).clamp_min(1e-12).sqrt()
        gain=dot/(t2+.01)
        error=((x-gain[...,None]*t).square()*weight).sum(-1).div(count.clamp_min(1)).sqrt()
        xrms=(x2/count.clamp_min(1)).sqrt()
        reference=torch.stack((cosine,torch.asinh(gain),error/(xrms+1e-6),count/self.signal_dimensions),dim=-1)
        reference=reference.masked_fill(count[...,None]==0,0.).flatten(1)
        return raw,reference


class JointCorrection(nn.Module):
    def __init__(self,dimensions,reference_slice,variant="reference",hidden=32,logit_bound=2.):
        super().__init__()
        if variant not in {"reference","no_reference"}:raise ValueError("Invalid variant")
        if hidden<4 or not math.isfinite(logit_bound) or not 0<logit_bound<=4:raise ValueError("Invalid correction settings")
        self.variant=variant;self.reference_slice=tuple(reference_slice);self.logit_bound=float(logit_bound)
        self.register_buffer("feature_mean",torch.zeros(dimensions))
        self.register_buffer("feature_std",torch.ones(dimensions))
        self.register_buffer("normalization_fitted",torch.tensor(False))
        self.network=nn.Sequential(nn.Linear(dimensions,hidden),nn.GELU(),nn.Dropout(.1),nn.Linear(hidden,6))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)

    @torch.no_grad()
    def fit_normalization(self,features,sources,splits):
        if bool(self.normalization_fitted):raise RuntimeError("Normalization already fitted")
        if len(features)!=len(sources) or len(features)!=len(splits):raise ValueError("Normalization alignment differs")
        if any(s!="real" for s in sources) or any(s!="train" for s in splits):raise ValueError("Normalization requires real_train only")
        if not torch.isfinite(features).all():raise ValueError("Non-finite features")
        self.feature_mean.copy_(features.mean(0))
        self.feature_std.copy_(features.std(0,unbiased=False).clamp_min(.05))
        self.normalization_fitted.fill_(True)

    def forward(self,features):
        if not bool(self.normalization_fitted):raise RuntimeError("Unfitted feature normalization")
        z=((features-self.feature_mean)/self.feature_std).clamp(-8.,8.)
        if self.variant=="no_reference":
            z=z.clone();z[:,self.reference_slice[0]:self.reference_slice[1]]=0.
        return self.logit_bound*torch.tanh(self.network(z).reshape(-1,3,2))


class ReferenceJointModel(nn.Module):
    def __init__(self,base_model_config,centers,variant="reference",hidden=32,logit_bound=2.,half_width=24):
        super().__init__()
        self.config=dict(base_model_config=base_model_config,centers=list(centers),variant=variant,
                         hidden=hidden,logit_bound=logit_bound,half_width=half_width)
        self.base=TransformerClassifyRegress_sep(**base_model_config)
        self.base.eval().requires_grad_(False)
        self.bank=ReferenceBank(centers,half_width)
        feature_dimension=self.base.concentration_projection(torch.zeros(1,3,base_model_config["attention_dim"])).shape[-1]
        context_dimensions=3*feature_dimension+6+3
        start=self.bank.raw_dimensions;stop=start+self.bank.reference_dimensions
        self.correction=JointCorrection(stop+context_dimensions,(start,stop),variant,hidden,logit_bound)

    def train(self,mode=True):
        super().train(mode)
        self.base.eval();self.bank.eval()
        self.correction.train(mode)
        return self

    @torch.no_grad()
    def frozen_features(self,data):
        if self.base.training:raise RuntimeError("Anchor must remain in eval mode")
        captured=[]
        handle=self.base.concentration_projection.register_forward_hook(lambda m,i,o:captured.append(o.detach()))
        try:
            inputs={k:data[k] for k in ("raw","smoothed","percentile","valid_mask")}
            classify,score,ordinal,logits=self.base(inputs,return_ordinal=True,return_ordinal_logits=True)
        finally:handle.remove()
        if len(captured)!=1:raise RuntimeError("Expected the unchanged shared concentration path")
        raw,reference=self.bank(data["raw_intensity"],data["measured_mask"])
        features=torch.cat((raw,reference,captured[0].flatten(1),logits.flatten(1),classify),-1)
        return features,classify,logits,ordinal

    def forward_cached(self,features,classify,base_logits):
        delta=self.correction(features)
        logits=base_logits+delta
        ordinal=corn_probabilities(logits)
        score=1.+ordinal.sum(-1)
        return classify,score,ordinal,logits,delta

    def forward(self,data):
        features,classify,logits,_=self.frozen_features(data)
        return self.forward_cached(features,classify,logits)


class ConcentrationUpdateModel(ReferenceJointModel):
    """Frozen shared features; an optional copied concentration path can learn.

    Both arms retain exactly the same raw-window residual correction. The
    experimental arm additionally updates copied fusion and ordinal heads.
    Original modules, including concentration projection, remain read-only.
    The cached input ends with the PRE-fusion query [B,3,attention_dim].
    """
    def __init__(self,base_model_config,centers,experiment_variant="concentration_update",
                 hidden=32,logit_bound=2.,half_width=24,variant="no_reference"):
        import copy
        if experiment_variant not in ("concentration_update","residual_control"):
            raise ValueError("Unknown T3.24 experiment")
        if variant!="no_reference":raise ValueError("Reference matching stays masked in both arms")
        if base_model_config.get("concentration_head_mode")!="ordinal" or not base_model_config.get("use_mixture_aware_query_fusion"):
            raise ValueError("T3.24 requires the existing mixture-aware ordinal model")
        for key in ("use_adaptive_mixture_gate","use_boundary_specific_mixture_gate",
                    "use_local_quantitative_branch","use_anchored_local_calibration"):
            if base_model_config.get(key,False):raise ValueError("Unsupported anchor option: "+key)
        super().__init__(base_model_config,centers,variant,hidden,logit_bound,half_width)
        self.config["experiment_variant"]=experiment_variant
        self.experiment_variant=experiment_variant
        self.residual_feature_dimension=int(self.correction.feature_mean.numel())
        self.query_dimension=3*int(base_model_config["attention_dim"])
        if experiment_variant=="concentration_update":
            self.updated_fusion=copy.deepcopy(self.base.mixture_aware_query_fusion).requires_grad_(True)
            self.updated_ordinal_heads=copy.deepcopy(self.base.ordinal_heads).requires_grad_(True)
        self.train(False)

    @torch.no_grad()
    def initialize_anchor(self,state):
        """Only call before training, never while loading a trained checkpoint."""
        self.base.load_state_dict(state,strict=True)
        if self.experiment_variant=="concentration_update":
            self.updated_fusion.load_state_dict(self.base.mixture_aware_query_fusion.state_dict(),strict=True)
            self.updated_ordinal_heads.load_state_dict(self.base.ordinal_heads.state_dict(),strict=True)

    def train(self,mode=True):
        super().train(mode)
        # eval() disables pretrained-path dropout, without disabling gradients.
        # This gives exact zero change at initialization and avoids adding
        # dropout-driven drift to the small parameter update.
        if hasattr(self,"updated_fusion"):
            self.updated_fusion.eval();self.updated_ordinal_heads.eval()
        return self

    @torch.no_grad()
    def frozen_features(self,data):
        queries=[]
        handle=self.base.peak_guided_attention.register_forward_hook(lambda m,i,o:queries.append(o[0].detach()))
        try:features,classify,logits,ordinal=super().frozen_features(data)
        finally:handle.remove()
        if len(queries)!=1:raise RuntimeError("Expected one pre-fusion query")
        q=queries[0]
        if q.shape[1:]!=(3,self.query_dimension//3):raise RuntimeError("Unexpected query dimensions")
        return torch.cat((features,q.flatten(1)),-1),classify,logits,ordinal

    def concentration_logits(self,query,classify,updated):
        fusion=self.updated_fusion if updated else self.base.mixture_aware_query_fusion
        heads=self.updated_ordinal_heads if updated else self.base.ordinal_heads
        fused=fusion(query,classify.detach())
        if isinstance(fused,tuple):raise RuntimeError("Expected the original shared concentration path")
        features=self.base.concentration_projection(fused)
        return torch.stack([head(features[:,i,:]) for i,head in enumerate(heads)],1)

    def delta_components(self,features,classify):
        expected=self.residual_feature_dimension+self.query_dimension
        if features.ndim!=2 or features.shape[1]!=expected:
            raise ValueError(f"Expected a pre-fusion feature cache with {expected} columns")
        residual=self.correction(features[:,:self.residual_feature_dimension])
        branch=torch.zeros_like(residual)
        if self.experiment_variant=="concentration_update":
            q=features[:,self.residual_feature_dimension:].reshape(-1,3,self.query_dimension//3).detach()
            with torch.no_grad():anchor=self.concentration_logits(q,classify,False)
            branch=self.concentration_logits(q,classify,True)-anchor
            # Equal copied weights can still choose a different floating-point
            # kernel (e.g. frozen versus gradient-enabled Linear). Canonicalize
            # only paths whose ENTIRE state is identical, not small learned
            # changes. Subtracting detach keeps the training derivative intact.
            original=self.base.mixture_aware_query_fusion.state_dict()
            updated=self.updated_fusion.state_dict()
            fusion_same=all(torch.equal(original[k],updated[k]) for k in original)
            if fusion_same:
                same=[]
                for old,new in zip(self.base.ordinal_heads,self.updated_ordinal_heads):
                    a,b=old.state_dict(),new.state_dict()
                    same.append(all(torch.equal(a[k],b[k]) for k in a))
                mask=branch.new_tensor(same).reshape(1,3,1)
                branch=branch-branch.detach()*mask
        return residual,branch

    def predict_with_components(self,features,classify,base_logits,base_ordinal=None):
        residual,branch=self.delta_components(features,classify)
        delta=residual+branch
        logits=base_logits.detach()+delta
        ordinal=corn_probabilities(logits)
        if base_ordinal is not None:
            if base_ordinal.shape!=ordinal.shape:raise ValueError("Anchor ordinal shape differs")
            # A mathematically zero change must preserve the original
            # probability, including values next to the median threshold.
            # Recomputing sigmoid in a different CUDA batch need not preserve
            # its final bits. This mask depends only on predicted changes.
            unchanged=(delta==0).all(-1,keepdim=True)
            ordinal=torch.where(unchanged,base_ordinal.detach(),ordinal)
        outputs=(classify.detach(),1.+ordinal.sum(-1),ordinal,logits,delta)
        return outputs,residual,branch

    def forward_cached(self,features,classify,base_logits,base_ordinal=None):
        return self.predict_with_components(features,classify,base_logits,base_ordinal)[0]

    def forward(self,data):
        features,classify,logits,ordinal=self.frozen_features(data)
        return self.forward_cached(features,classify,logits,ordinal)

    def trainable_named_parameters(self):
        allowed=("correction.network.",)
        if self.experiment_variant=="concentration_update":allowed+=("updated_fusion.","updated_ordinal_heads.")
        parameters=[(n,p) for n,p in self.named_parameters() if p.requires_grad]
        if not parameters or any(not n.startswith(allowed) for n,p in parameters):
            raise RuntimeError("Unexpected trainable parameter outside concentration update scope")
        return parameters
