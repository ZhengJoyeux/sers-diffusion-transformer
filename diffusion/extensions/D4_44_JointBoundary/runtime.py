"""Process-local extensions; installed project sources remain untouched."""
import importlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from boundary_loss import bounded_addition, extension_class, FIELDS as BOUNDARY_FIELDS
from geometry import DEFAULT, VERSION, compact_bounds, image_seed, project_image

FIELDS = ('curve_localization_kl', 'curve_localization_addition', 'curve_localization_ratio')


def localization_loss(predicted, clean, mask, temperature):
    target = torch.softmax(clean.detach().float()/temperature, dim=2)
    logp = torch.log_softmax(predicted.float()/temperature, dim=2)
    kl = (target*(target.clamp_min(1e-30).log()-logp)).sum(2).clamp_min(0)
    columns = torch.nn.functional.pad(mask, (0, kl.shape[-1]-mask.shape[-1]))
    return (kl*columns).sum((1, 2))/columns.sum((1, 2)).clamp_min(1)


def diffusion_class(original, boundary, geometry, loss):
    base_class = extension_class(original, boundary)

    class GeometryDiffusion(base_class):
        def p_losses(self, x_start, t, *args, **kwargs):
            captured = []
            hook = self.model.register_forward_hook(
                lambda m, a, out: captured.append(out))
            try:
                base = super().p_losses(x_start, t, *args, **kwargs)
            finally:
                hook.remove()
            if len(captured) != 1:
                raise RuntimeError('Localization must reuse the original single DiT forward')
            mask = self._prepare_mask(kwargs.get('valid_mask'), x_start)
            condition = self._prepare_condition(
                kwargs.get('condition'), len(x_start), x_start.device, x_start.dtype)
            prior = self._prepare_prior_conditioning(
                kwargs.get('prior_conditioning'), x_start)*mask
            full = self.codec.full_from_residual(x_start*mask, prior, condition, mask)
            clean = self.codec.render(full, mask)
            kl = localization_loss(captured[0], clean, mask, geometry['temperature'])
            from src.masked_diffusion import _extract
            per = kl*float(loss['weight'])*_extract(self.loss_weight, t, kl.shape)
            if kwargs.get('loss_reduction', 'mean') != 'none':
                high = self._high_noise_weights(t).to(per)
                per = (per*high).sum()/high.sum().clamp_min(1e-12)
            addition, _ = bounded_addition(base, per, float(loss['maximum_ratio']))
            total = base+addition
            if total.ndim == 0:
                self._latest_loss_components.update(
                    curve_localization_kl=kl.mean().detach(),
                    curve_localization_addition=addition.detach(),
                    curve_localization_ratio=(addition/(base.detach()+1e-12)).detach(),
                    total_loss=total.detach())
            return total

        @torch.no_grad()
        def sample(self, batch_size=16, *args, valid_mask=None, **kwargs):
            reference = torch.zeros(
                (batch_size, 1, self.seq_length), device=self.betas.device)
            mask = self._prepare_mask(valid_mask, reference)
            result = super().sample(batch_size, *args, valid_mask=mask, **kwargs)
            self.last_curve_masks = self.codec.pixel_mask(mask)[:, 0, 0].bool().cpu()
            return result

    return GeometryDiffusion


def save_batch(diffusion, directory, start_index):
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    state = {k: v for k, v in diffusion.codec.state.items() if k != 'bridge'}
    options = state.get('image_geometry', DEFAULT)
    (folder/'curve_coordinate_state.json').write_text(
        json.dumps(state, ensure_ascii=False, indent=2))
    images = diffusion.last_curve_images
    remaining = diffusion.curve_configuration['save_images_per_condition']-start_index
    if images is None:
        return
    for i, native in enumerate(images[:remaining]):
        index = start_index+i+1
        valid = diffusion.last_curve_masks[i].numpy()
        try:
            image, synthetic, report = project_image(
                native.to(diffusion.betas.device), valid,
                seed=image_seed(folder.name, index), options=options)
        except ValueError as error:
            rejected = folder/'rejected'
            rejected.mkdir(exist_ok=True)
            np.savez_compressed(
                rejected/f'raw_diffusion_{index:03d}.npz',
                native_heatmap=native.numpy(), valid_columns=valid,
                coordinate_json=json.dumps(state))
            report = dict(approved=False, index=index, error=str(error))
        else:
            np.savez_compressed(
                folder/f'raw_diffusion_{index:03d}.npz',
                image=image, native_heatmap=native.numpy(),
                valid_columns=valid, synthetic_columns=synthetic,
                coordinate_json=json.dumps(state),
                geometry_json=json.dumps(report))
            pixels = np.rint((1-(image[0]+1)/2)*255).astype(np.uint8)
            Image.fromarray(pixels).save(folder/f'raw_diffusion_{index:03d}.png')
            report['index'] = index
        with (folder/'geometry_quality.jsonl').open('a') as stream:
            stream.write(json.dumps(
                report, ensure_ascii=False, allow_nan=False)+'\n')
    (folder/'README.txt').write_text(
        'image: continuous geometry projection, not untouched sampler output.\n'
        'native_heatmap: unchanged native DiT output; numerical decoding uses this field.\n'
        'valid_columns: measured region; synthetic_columns: image-only background.\n'
        'Exclude synthetic columns from recognition and numerical metrics.\n'
        'PNG: viewing copy. NPZ: authoritative float32 images and masks.\n')


def install(project, configuration):
    import sys
    sys.path.insert(1, str(project))
    loader = importlib.import_module('src.configuration_loader')
    original_load = loader.load_configuration

    def load(path):
        cfg = original_load(path)
        cfg['_paths']['project_root'] = str(project)
        return cfg

    loader.load_configuration = load
    settings = configuration['d4_38']
    codec = importlib.import_module('src.curve_image_codec')
    original_fit = codec.fit_curve_state

    def fit(**kwargs):
        state = original_fit(**kwargs)
        state['y_min'], state['y_max'] = compact_bounds(
            kwargs['full'], kwargs['bases'], kwargs['masks'],
            kwargs['training_indices'], state['height'], settings['geometry'])
        state['image_geometry'] = settings['geometry']
        state['geometry_version'] = VERSION
        return state

    codec.fit_curve_state = fit
    module = importlib.import_module('src.curve_image_diffusion')
    module.CurveImageGaussianDiffusion2D = diffusion_class(
        module.CurveImageGaussianDiffusion2D,
        configuration['d4_30']['loss'],
        settings['geometry'], settings['localization'])
    importlib.import_module('src.curve_image_outputs').save_curve_batch = save_batch
    log = importlib.import_module('src.training_logger')
    fields = BOUNDARY_FIELDS+FIELDS
    log.COMPONENT_FIELDS = tuple(dict.fromkeys((*log.COMPONENT_FIELDS, *fields)))
    log.TrainingLogger.FIELD_NAMES = tuple(dict.fromkeys((
        *log.TrainingLogger.FIELD_NAMES,
        *(f'{kind}_{name}' for kind in ('training', 'validation') for name in fields))))
