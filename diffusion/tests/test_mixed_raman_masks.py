import numpy as np
import torch

from src.intensity_normalizer import GlobalMinMaxNormalizer
from src.model_builder import build_diffusion_model
from src.spectrum_dataset import SpectrumDataset
from src.spectrum_length_adapter import SpectrumLengthAdapter


def test_union_axis_does_not_extrapolate_short_spectrum() -> None:
    long_axis = np.arange(600.0, 2501.0)
    short_axis = np.arange(600.0, 2001.0)
    adapter = SpectrumLengthAdapter.create(
        raman_shifts=[short_axis, long_axis],
        dimension_multipliers=[1, 2, 4],
        raman_axis_mode="union_with_valid_mask",
    )

    spectra, masks = adapter.interpolate_to_model_axis_with_mask(
        [short_axis * 0.1, long_axis * 0.1],
        [short_axis, long_axis],
    )

    assert adapter.original_length == 1901
    assert adapter.padded_length == 1904
    assert masks[0].sum() == 1401
    assert masks[1].sum() == 1901
    assert np.all(spectra[0, 1401:] == 0.0)
    assert np.all(masks[0, 1401:] == 0.0)
    assert np.all(masks[1] == 1.0)

    padded_masks = adapter.adapt_valid_mask(masks)
    assert padded_masks.shape == (2, 1904)
    assert np.all(padded_masks[:, -3:] == 0.0)


def test_masked_normalizer_ignores_invalid_fill_values() -> None:
    spectra = np.asarray(
        [[1.0, 2.0, -999.0], [3.0, 4.0, -999.0]],
        dtype=np.float32,
    )
    masks = np.asarray(
        [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    normalizer = GlobalMinMaxNormalizer().fit(
        spectra,
        valid_mask=masks,
    )
    normalized = normalizer.transform(
        spectra,
        valid_mask=masks,
    )

    assert normalizer.data_min == 1.0
    assert normalizer.data_max == 4.0
    assert np.all(normalized[:, 2] == 0.0)


def test_spectrum_dataset_returns_valid_mask() -> None:
    spectra = np.zeros((2, 8), dtype=np.float32)
    masks = np.ones((2, 8), dtype=np.float32)
    masks[0, 4:] = 0.0
    item = SpectrumDataset(spectra, valid_masks=masks)[0]

    assert isinstance(item, dict)
    assert item["spectrum"].shape == (1, 8)
    assert item["valid_mask"].shape == (1, 8)
    assert torch.equal(
        item["valid_mask"][0, 4:],
        torch.zeros(4),
    )


def _masked_model_configuration() -> dict:
    return {
        "data": {"raman_axis_mode": "union_with_valid_mask"},
        "normalization": {"enabled": True},
        "prior_residual": {"enabled": False},
        "broad_local_residual": {"enabled": False},
        "physics_constraints": {"enabled": False},
        "diversity_constraints": {"enabled": False},
        "local_peak_distribution_constraints": {"enabled": False},
        "peak_derivative_constraints": {"enabled": False},
        "relative_peak_intensity_constraints": {"enabled": False},
        "peak_parameter_constraints": {"enabled": False},
        "model": {
            "channels": 1,
            "base_dimension": 8,
            "dimension_multipliers": [1, 2],
            "self_condition": False,
            "dropout": 0.0,
        },
        "diffusion": {
            "diffusion_timesteps": 10,
            "sampling_timesteps": 2,
            "objective": "pred_x0",
            "beta_schedule": "cosine",
            "ddim_sampling_eta": 0.0,
            "auto_normalize": False,
            "loss_weighting": "uniform",
            "residual_aware_loss": {"enabled": False},
        },
    }


def test_masked_diffusion_loss_ignores_invalid_tail() -> None:
    _, diffusion = build_diffusion_model(
        _masked_model_configuration(),
        sequence_length=16,
    )
    first = torch.rand(2, 1, 16)
    second = first.clone()
    second[0, :, 8:] = 1000.0
    mask = torch.ones_like(first)
    mask[0, :, 8:] = 0.0
    timestep = torch.tensor([2, 7], dtype=torch.long)
    noise = torch.randn_like(first)

    loss_first = diffusion.p_losses(
        first,
        timestep,
        noise=noise,
        valid_mask=mask,
    )
    loss_second = diffusion.p_losses(
        second,
        timestep,
        noise=noise,
        valid_mask=mask,
    )
    assert torch.allclose(loss_first, loss_second)
    components = diffusion.get_latest_loss_components()
    assert torch.isfinite(components["ddpm_uniform_loss"])
    assert torch.allclose(components["ddpm_uniform_loss"], loss_second)


def test_masked_ddim_sampling_keeps_invalid_tail_zero() -> None:
    _, diffusion = build_diffusion_model(
        _masked_model_configuration(),
        sequence_length=16,
    )
    mask = torch.ones(2, 1, 16)
    mask[0, :, 8:] = 0.0
    generated = diffusion.sample(
        batch_size=2,
        valid_mask=mask,
    )

    assert generated.shape == (2, 1, 16)
    assert torch.all(generated[0, :, 8:] == 0.0)
    assert torch.isfinite(generated).all()
