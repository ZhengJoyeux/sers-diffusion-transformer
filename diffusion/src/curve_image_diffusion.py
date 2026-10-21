"""Pixel-space 2D Gaussian diffusion with legacy numerical SERS losses.

The existing dataset/trainer/generator interfaces stay [B,1,L]. Internally the
clean FULL spectrum is rendered BEFORE q_sample; every reverse state is HxW.
PCA, broad/local, variance equalization, spectral constraints and export remain
in their original numerical domain, connected by the exact asinh bridge.
"""
import torch
from torch.nn import functional as F
from src.masked_diffusion import MaskedGaussianDiffusion1D, MaskedModelPrediction, _extract


class CurveImageGaussianDiffusion2D(MaskedGaussianDiffusion1D):
    def __init__(self, *args, curve_configuration, **kwargs):
        super().__init__(*args, **kwargs)
        if self.objective != 'pred_x0':
            raise ValueError('2D curve diffusion currently requires pred_x0')
        self.curve_configuration = dict(curve_configuration)
        self.last_curve_images = None
        self.last_curve_diagnostics = None

    @property
    def codec(self):
        return self.model.codec

    def p_losses(self, x_start, t, noise=None, model_forward_kwargs=None,
                 return_reduced_loss=True, loss_reduction='mean', valid_mask=None,
                 condition=None, prior_conditioning=None, full_spectrum_target=None,
                 local_inverse_slope=None):
        if model_forward_kwargs:
            raise ValueError('unsupported model_forward_kwargs for 2D curve diffusion')
        mask = self._prepare_mask(valid_mask, x_start)
        c = self._prepare_condition(condition, len(x_start), x_start.device, x_start.dtype)
        base = self._prepare_prior_conditioning(prior_conditioning, x_start)*mask
        full = self.codec.full_from_residual(x_start*mask, base, c, mask)
        if full_spectrum_target is not None:
            if not torch.allclose(full*mask, full_spectrum_target.float()*mask, atol=2e-5, rtol=2e-4):
                raise ValueError('curve bridge disagrees with exact full spectrum target')
        clean = self.codec.render(full, mask)
        prior_image = self.codec.render(base, mask)
        pixels = self.codec.pixel_mask(mask)
        if noise is None:
            noise = torch.randn_like(clean)
        elif noise.ndim == 3 and torch.count_nonzero(noise) == 0:
            # Existing deterministic pipeline probe uses zero numeric noise.
            noise = torch.zeros_like(clean)
        elif noise.shape != clean.shape:
            raise ValueError('2D diffusion noise must be [B,1,H,W]; 1D random noise is invalid')
        noisy = self.q_sample(clean, t, noise=noise)*pixels
        predicted_image = self.model(noisy, t, valid_mask=pixels, condition=c, prior_conditioning=prior_image)
        decoded_full = self.codec.decode(predicted_image, mask)
        predicted_residual = self.codec.residual_from_full(decoded_full, base, c, mask)
        # Keep all numerical losses, caps, group gates and exact reconstruction.
        spectral_loss = super().p_losses(
            x_start, t, noise=torch.zeros_like(x_start), valid_mask=mask, condition=c,
            prior_conditioning=base, full_spectrum_target=full_spectrum_target,
            local_inverse_slope=local_inverse_slope,
            prediction_override=predicted_residual,
            return_reduced_loss=return_reduced_loss, loss_reduction=loss_reduction)
        foreground = ((clean+1)*0.5).detach()
        pixel_weight = (1+self.curve_configuration['foreground_weight']*foreground)*pixels
        image_pointwise = (predicted_image.float()-clean).square()*pixel_weight
        per_image = image_pointwise.flatten(1).sum(1)/pixel_weight.flatten(1).sum(1).clamp_min(1)
        per_image = per_image*_extract(self.loss_weight, t, per_image.shape)
        if not return_reduced_loss:
            raise ValueError('pointwise 1D and 2D losses have different shapes; request reduced losses')
        factor = self.curve_configuration['image_loss_weight']
        if loss_reduction == 'none':
            return spectral_loss+factor*per_image
        high = self._high_noise_weights(t).to(per_image)
        image_loss = (per_image*high).sum()/high.sum().clamp_min(1e-12)
        total = spectral_loss+factor*image_loss
        self._latest_loss_components.update(
            curve_image_loss=image_loss.detach(), curve_spectral_total_loss=spectral_loss.detach(),
            total_loss=total.detach())
        return total

    def model_predictions(self, x, t, x_self_cond=None, clip_x_start=False,
                          rederive_pred_noise=False, model_forward_kwargs=None,
                          valid_mask=None, condition=None, prior_conditioning=None, **kwargs):
        if x.ndim != 4:
            raise ValueError('2D curve model_predictions requires an image state')
        if x_self_cond is not None or model_forward_kwargs or kwargs:
            raise ValueError('unsupported 2D curve prediction arguments')
        mask = valid_mask
        if mask is None:
            mask = self._sampling_valid_mask
        c = condition if condition is not None else self._sampling_condition
        base = prior_conditioning if prior_conditioning is not None else self._sampling_prior_conditioning
        if mask is None or c is None or base is None:
            raise ValueError('2D reverse diffusion requires numerical mask, condition and base')
        pixels = self.codec.pixel_mask(mask)
        prior = self.codec.render(base, mask)
        clean = self.model(x*pixels, t, valid_mask=pixels, condition=c, prior_conditioning=prior)
        if clip_x_start:
            clean = clean.clamp(-1, 1)*pixels
            support = self.spectrum_support_module
            if support is not None and support.configuration['sampling_soft_guard']:
                full = self.codec.decode(clean, mask)
                residual = self.codec.residual_from_full(full, base, c, mask)
                guarded = self._clip_prediction(residual, base, c, mask)
                # Apply the original numerical support at every reverse step.
                # Re-rasterize only if that original guard actually changes it.
                if not torch.allclose(guarded, residual, atol=1e-7, rtol=1e-6):
                    full = self.codec.full_from_residual(guarded, base, c, mask)
                    clean = self.codec.render(full, mask)
        predicted_noise = self.predict_noise_from_start(x*pixels, t, clean)*pixels
        return MaskedModelPrediction(predicted_noise, clean*pixels)

    def p_mean_variance(self, x, t, x_self_cond=None, clip_denoised=True, model_forward_kwargs=None):
        prediction = self.model_predictions(x, t, clip_x_start=clip_denoised,
                                            model_forward_kwargs=model_forward_kwargs)
        return (*self.q_posterior(x_start=prediction.pred_x_start, x_t=x, t=t), prediction.pred_x_start)

    @torch.no_grad()
    def sample(self, batch_size=16, *args, valid_mask=None, condition=None,
               prior_conditioning=None, return_all_timesteps=False, **kwargs):
        if args or kwargs or return_all_timesteps:
            raise ValueError('2D sample currently returns final numerical spectra only')
        reference = torch.zeros((batch_size, 1, self.seq_length), device=self.betas.device)
        mask = self._prepare_mask(valid_mask, reference)
        c = self._prepare_condition(condition, batch_size, reference.device, reference.dtype)
        base = self._prepare_prior_conditioning(prior_conditioning, reference)*mask
        self.codec.render(base, mask)  # Validate fixed bounds before allocating noise.
        if self._sampling_valid_mask is not None:
            raise RuntimeError('concurrent sampling on one diffusion instance is unsupported')
        self._sampling_valid_mask, self._sampling_condition, self._sampling_prior_conditioning = mask, c, base
        pixels = self.codec.pixel_mask(mask)
        image = torch.randn((batch_size, 1, self.codec.height, self.codec.width), device=reference.device)*pixels
        try:
            if self.is_ddim_sampling:
                times = torch.linspace(-1, self.num_timesteps-1, steps=self.sampling_timesteps+1).int().tolist()[::-1]
                for now, following in zip(times[:-1], times[1:]):
                    t = torch.full((batch_size,), now, device=image.device, dtype=torch.long)
                    prediction = self.model_predictions(image, t, clip_x_start=True)
                    if following < 0:
                        image = prediction.pred_x_start
                        continue
                    alpha, alpha_next = self.alphas_cumprod[now], self.alphas_cumprod[following]
                    sigma = self.ddim_sampling_eta*((1-alpha/alpha_next)*(1-alpha_next)/(1-alpha)).sqrt()
                    coefficient = (1-alpha_next-sigma.square()).clamp_min(0).sqrt()
                    image = (alpha_next.sqrt()*prediction.pred_x_start+coefficient*prediction.pred_noise+
                             sigma*torch.randn_like(image))*pixels
            else:
                for now in reversed(range(self.num_timesteps)):
                    t = torch.full((batch_size,), now, device=image.device, dtype=torch.long)
                    mean, _, log_variance, _ = self.p_mean_variance(image, t)
                    image = (mean+(0.5*log_variance).exp()*torch.randn_like(image) if now > 0 else mean)*pixels
            full = self.codec.decode(image, mask)
            residual = self.codec.residual_from_full(full, base, c, mask)
            count = min(batch_size, self.curve_configuration['save_images_per_condition'])
            self.last_curve_images = image[:count].detach().cpu()
            self.last_curve_diagnostics = self.codec.diagnostics(image, mask)
            return residual*mask
        finally:
            self._sampling_valid_mask = self._sampling_condition = self._sampling_prior_conditioning = None


def check_curve_gradient_flow(diffusion, spectrum, mask, condition, base):
    """Disposable 2D model probe; never changes the formal model weights."""
    from copy import deepcopy
    probe = deepcopy(diffusion.model).train()
    codec = probe.codec
    full = codec.full_from_residual(spectrum, base, condition, mask)
    clean = codec.render(full, mask)
    prior = codec.render(base, mask)
    pixels = codec.pixel_mask(mask)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    t = torch.ones(len(spectrum), device=spectrum.device, dtype=torch.long)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        predicted = probe(clean, t, valid_mask=pixels, condition=condition, prior_conditioning=prior)
        loss = ((predicted-clean).square()*pixels).sum()/pixels.sum()
        loss.backward(); optimizer.step()
    for prefix in ('patch_embed.', 'time_embed.', 'condition_encoder.', 'blocks.0.qkv.', 'final_linear.'):
        gradients = [p.grad for n,p in probe.named_parameters() if n.startswith(prefix) and p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients) or not any(g.abs().sum()>0 for g in gradients):
            raise RuntimeError(f'2D DiT gradient flow failed: {prefix}')
    print('2D curve DiT: disposable-copy 3-step gradient check passed; formal weights unchanged.')
