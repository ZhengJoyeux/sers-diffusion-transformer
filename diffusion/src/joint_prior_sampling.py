"""Train-only cross-block Gaussian copula, with checkpoint PCA marginals."""
from __future__ import annotations
import numpy as np
from scipy.special import ndtr, ndtri

def _finite_matrix(x, name):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or not np.isfinite(x).all():
        raise ValueError(f'{name}必须为有限二维矩阵。')
    return x

def contract_cross_correlation(cross, strength):
    if not np.isfinite(strength) or not 0 <= strength <= .5:
        raise ValueError('joint cross_correlation_strength必须位于[0,0.5]。')
    cross=_finite_matrix(cross,'cross')
    u,s,v=np.linalg.svd(cross,full_matrices=False)
    # Closest spectral contraction gives [[I,C],[C.T,I]] positive definite.
    # It preserves independent checkpoint scores WITHIN each block.
    effective=(u*np.minimum(s,1.0)*strength)@v
    residual=np.eye(cross.shape[1])-effective.T@effective
    w,q=np.linalg.eigh(residual)
    if np.min(w)<-1e-10:raise RuntimeError('联合协方差不是半正定。')
    root=(q*np.sqrt(np.maximum(w,0)))@q.T
    return effective,root,s

class JointPriorSampler:
    """Only cross-block dependence changes; no refit of marginal PCA states."""
    def __init__(self, entry, training_outer, training_broad, configuration):
        a,b=entry.outer,entry.broad_local
        if (a.prior_method!='pca_reconstruction' or
                a.pca_sampling_strategy!='independent_truncated_gaussian_scores'):
            raise ValueError('联合采样仅支持现有独立截断高斯PCA outer。')
        self.outer=a;self.broad=b
        self.oa=_finite_matrix(a.pca_components,'outer components').copy()
        self.ba=_finite_matrix(b.broad_pca_components,'broad components').copy()
        x,y=_finite_matrix(training_outer,'training_outer'),_finite_matrix(training_broad,'training_broad')
        if x.shape!=y.shape or x.shape!=(12,entry.valid_length):
            raise ValueError('联合先验只能由恰好12条同条件、同轴训练分解拟合。')
        if self.oa.shape[1]!=entry.valid_length or self.ba.shape[1]!=entry.valid_length:
            raise ValueError('PCA与条件有效轴不一致。')
        self.strength=float(configuration.get('cross_correlation_strength',.5))
        self.oc=float(a.pca_score_clip_standard_deviations)
        self.bc=float(b.broad_score_clip_standard_deviations)
        if not (np.isfinite(self.oc) and np.isfinite(self.bc) and self.oc>0 and self.bc>0):
            raise ValueError('PCA截断范围无效。')
        self.om=np.asarray(a.pca_mean,dtype=np.float64).reshape(-1).copy()
        self.bm=np.asarray(b.broad_pca_mean,dtype=np.float64).reshape(-1).copy()
        self.osm=np.asarray(a.pca_training_score_mean,dtype=np.float64).reshape(-1).copy()
        self.bsm=np.asarray(b.broad_training_score_mean,dtype=np.float64).reshape(-1).copy()
        self.oss=np.asarray(a.pca_score_standard_deviation,dtype=np.float64).reshape(-1).copy()
        self.bss=np.asarray(b.broad_score_standard_deviation,dtype=np.float64).reshape(-1).copy()
        for mean,score_mean,std,components in [(self.om,self.osm,self.oss,self.oa),(self.bm,self.bsm,self.bss,self.ba)]:
            if (mean.size!=entry.valid_length or score_mean.size!=len(components) or std.size!=len(components)
                    or not np.isfinite(np.concatenate([mean,score_mean,std])).all() or np.any(std<0)):
                raise ValueError('checkpoint PCA统计状态无效。')
        # Project PAIRED training components onto existing checkpoint axes.
        ps=(x-self.om)@self.oa.T;qs=(y-self.bm)@self.ba.T
        ps-=ps.mean(axis=0);qs-=qs.mean(axis=0)
        pd=np.std(ps,axis=0,ddof=1);qd=np.std(qs,axis=0,ddof=1)
        active_p=(pd>1e-10)&(self.oss>1e-10)
        active_q=(qd>1e-10)&(self.bss>1e-10)
        ps/=np.maximum(pd,1e-10);qs/=np.maximum(qd,1e-10)
        raw=ps.T@qs/11
        raw[~active_p,:]=0;raw[:,~active_q]=0
        self.cross,self.root,s=contract_cross_correlation(raw,self.strength)
        self.diagnostics={'strategy':'checkpoint_marginal_cross_copula_v1','fit_on':'train_only',
          'training_count':12,'cross_correlation_strength':self.strength,
          'outer_score_count':len(self.oa),'broad_score_count':len(self.ba),
          'outer_score_clip':self.oc,'broad_score_clip':self.bc,
          'training_score_cross_correlation':raw.tolist(),'effective_latent_cross_correlation':self.cross.tolist(),
          'raw_cross_singular_values':s.tolist(),'effective_maximum_cross_singular_value':float(np.linalg.norm(self.cross,2)),
          'unchanged_checkpoint_marginal_statistics':True,
          'warning':'12 training spectra; correlation contraction/half-strength is a prespecified regularization, not a validation-fitted optimum'}

    def sample(self, number, prior_random_generator, broad_random_generator):
        if isinstance(number,bool) or int(number)!=number or number<1:raise ValueError('number必须是正整数。')
        for rng in (prior_random_generator,broad_random_generator):
            if not isinstance(rng,np.random.Generator):raise TypeError('必须使用numpy Generator。')
        # Zero-strength exact legacy path, including RNG consumption and float32 rounding.
        if not np.any(self.cross):
            return (self.outer.sample_reference_priors(int(number),random_generator=prior_random_generator),
                    self.broad.sample_broad_residuals(int(number),random_generator=broad_random_generator))
        ol,oh=ndtr(-self.oc),ndtr(self.oc)
        bl,bh=ndtr(-self.bc),ndtr(self.bc)
        op=prior_random_generator.uniform(ol,oh,size=(int(number),len(self.oa)))
        bp=broad_random_generator.uniform(bl,bh,size=(int(number),len(self.ba)))
        outer_scores=self.osm+ndtri(op)*self.oss
        # Same outer draws as old sampler; correlate BROAD Gaussian latent draws.
        eps=np.finfo(np.float64).eps
        ou=np.clip((op-ol)/(oh-ol),eps,1-eps)
        bu=np.clip((bp-bl)/(bh-bl),eps,1-eps)
        og,bg=ndtri(ou),ndtri(bu)
        conditional_latent=og@self.cross+bg@self.root.T
        bprob=np.clip(bl+(bh-bl)*ndtr(conditional_latent),np.nextafter(bl,bh),np.nextafter(bh,bl))
        broad_scores=self.bsm+ndtri(bprob)*self.bss
        prior=self.om+outer_scores@self.oa
        broad=self.bm+broad_scores@self.ba
        if not np.isfinite(prior).all() or not np.isfinite(broad).all():raise RuntimeError('联合先验包含NaN/Inf。')
        return prior.astype(np.float32),broad.astype(np.float32)
