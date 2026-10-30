"""D4.30 training-only target-matched full-spectrum differences; no new weights."""
from __future__ import annotations
import math
import torch
from torch.nn import functional as F

DEFAULT = dict(weight=0.1, difference_scale=0.01, boundary_extra=4.0,
               smooth_l1_beta=1.0, maximum_ratio=0.02)
FIELDS = ('curve_full_difference_loss', 'curve_boundary_difference_loss',
          'curve_boundary_aux_uncapped', 'curve_boundary_aux_loss',
          'curve_boundary_aux_scale', 'curve_boundary_aux_ratio', 'curve_baseline_total_loss')

def settings(raw=None):
    raw = raw or {}
    if set(raw)-set(DEFAULT):
        raise ValueError('Unknown boundary loss option')
    cfg = {**DEFAULT, **raw}
    if any(isinstance(v,bool) or not math.isfinite(float(v)) for v in cfg.values()):
        raise ValueError('Boundary loss options must be finite numbers')
    if (cfg['weight']<0 or cfg['boundary_extra']<0 or cfg['difference_scale']<=0
            or cfg['smooth_l1_beta']<=0 or not 0<cfg['maximum_ratio']<=0.1):
        raise ValueError('Invalid boundary loss scale, weight or budget')
    return cfg

def difference_losses(prediction, target, mask, patch_width, cfg):
    if (prediction.shape!=target.shape or mask.shape!=target.shape
            or prediction.ndim!=3 or prediction.shape[1]!=1 or prediction.shape[-1]<2
            or isinstance(patch_width,bool) or not isinstance(patch_width,int) or patch_width<2):
        raise ValueError('Expected matching [B,1,L] spectra/mask and patch_width>=2')
    if not torch.all((mask==0)|(mask==1)):
        raise ValueError('Mask must be binary')
    valid = mask.bool()
    if not torch.isfinite(prediction[valid]).all() or not torch.isfinite(target[valid]).all():
        raise ValueError('Nonfinite valid spectral value')
    # Remove padding before arithmetic, so invalid NaNs cannot enter gradients.
    p = torch.where(valid, prediction.float(), torch.zeros_like(prediction, dtype=torch.float32))
    y = torch.where(valid, target.detach().float(), torch.zeros_like(target, dtype=torch.float32))
    pairs = valid[...,1:] & valid[...,:-1]
    error = ((p[...,1:]-p[...,:-1])-(y[...,1:]-y[...,:-1]))/cfg['difference_scale']
    loss = F.smooth_l1_loss(error, torch.zeros_like(error), reduction='none', beta=cfg['smooth_l1_beta'])
    seam = (torch.arange(1,p.shape[-1],device=p.device)%patch_width==0)[None,None] & pairs
    def mean(where):
        return (loss*where).sum((1,2))/where.sum((1,2)).clamp_min(1)
    return mean(pairs), mean(seam)

def bounded_addition(base, auxiliary, maximum_ratio):
    if base.shape!=auxiliary.shape or not torch.isfinite(base).all() or not torch.isfinite(auxiliary).all():
        raise ValueError('Nonfinite or mismatched base/auxiliary losses')
    if (base.detach()<0).any() or (auxiliary.detach()<0).any():
        raise ValueError('Loss budgets must be nonnegative')
    cap = base.detach()*maximum_ratio
    # Detach the rescaling factor: a saturated budget must retain a corrective gradient.
    gain = torch.minimum(torch.ones_like(auxiliary),cap/(auxiliary.detach()+1e-12)).detach()
    return auxiliary*gain, gain

def extension_class(original_class, configuration):
    cfg = settings(configuration)
    class BoundaryDiffusion(original_class):
        def p_losses(self, x_start, t, *args, **kwargs):
            if cfg['weight']==0:
                base = super().p_losses(x_start,t,*args,**kwargs)
                if kwargs.get('loss_reduction','mean')!='none':
                    zero=base.detach()*0
                    self._latest_loss_components.update({name:zero for name in FIELDS})
                    self._latest_loss_components['curve_baseline_total_loss']=base.detach()
                return base
            captured=[]
            # Capture the single model forward already required by the original loss.
            # Decode its output again to add a full-spectrum gradient; never resample noise.
            hook=self.model.register_forward_hook(lambda module,inputs,output:captured.append(output))
            try:
                base=super().p_losses(x_start,t,*args,**kwargs)
            finally:
                hook.remove()
            if len(captured)!=1:
                raise RuntimeError('Expected exactly one original DiT forward')
            mask=self._prepare_mask(kwargs.get('valid_mask'),x_start)
            c=self._prepare_condition(kwargs.get('condition'),len(x_start),x_start.device,x_start.dtype)
            prior=self._prepare_prior_conditioning(kwargs.get('prior_conditioning'),x_start)*mask
            target=kwargs.get('full_spectrum_target')
            if target is None:
                target=self.codec.full_from_residual(x_start*mask,prior,c,mask)
            prediction=self.codec.decode(captured[0],mask)
            full,seam=difference_losses(prediction,target,mask,int(self.curve_configuration['patch_width']),cfg)
            from src.masked_diffusion import _extract
            per=cfg['weight']*(full+cfg['boundary_extra']*seam)
            per=per*_extract(self.loss_weight,t,per.shape)
            if kwargs.get('loss_reduction','mean')=='none':
                addition,_=bounded_addition(base,per,cfg['maximum_ratio'])
                return base+addition
            high=self._high_noise_weights(t).to(per)
            uncapped=(per*high).sum()/high.sum().clamp_min(1e-12)
            addition,gain=bounded_addition(base,uncapped,cfg['maximum_ratio'])
            total=base+addition
            self._latest_loss_components.update(
                curve_full_difference_loss=full.mean().detach(),
                curve_boundary_difference_loss=seam.mean().detach(),
                curve_boundary_aux_uncapped=uncapped.detach(), curve_boundary_aux_loss=addition.detach(),
                curve_boundary_aux_scale=gain.detach(),
                curve_boundary_aux_ratio=(addition/(base.detach()+1e-12)).detach(),
                curve_baseline_total_loss=base.detach(),total_loss=total.detach())
            return total
    return BoundaryDiffusion
