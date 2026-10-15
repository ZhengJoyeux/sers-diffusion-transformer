from __future__ import annotations

import numpy as np
import pytest
import torch

from src.model_builder import build_diffusion_model
from src.spectrum_conditioning import (
    CONDITION_VECTOR_SIZE,
    build_conditioning_metadata,
    parse_condition_from_source_file,
    parse_condition_text,
)
from src.spectrum_dataset import SpectrumDataset


def test_filename_parser_canonicalizes_pesticide_order() -> None:
    parsed = parse_condition_from_source_file(
        "CHL_TEB/CHL-H_TEB-S_water.xlsx"
    )
    assert parsed.condition_id == "TEB-S_CHL-H_water"
    assert parsed.levels == ("0", "S", "H")
    vector = parsed.to_vector()
    assert vector.shape == (CONDITION_VECTOR_SIZE,)
    assert vector.sum() == 4.0
    assert vector[0] == 1.0  # DEL absent
    assert vector[5] == 1.0  # TEB-S
    assert vector[11] == 1.0  # CHL-H
    assert vector[12] == 1.0  # water


def test_parser_rejects_unknown_or_duplicate_tokens() -> None:
    with pytest.raises(ValueError):
        parse_condition_text("ABC-H_water")
    with pytest.raises(ValueError):
        parse_condition_text("DEL-H_DEL-M_soil")
    with pytest.raises(ValueError):
        parse_condition_text("DEL-H_air")


def test_condition_metadata_rejects_duplicate_condition_files() -> None:
    with pytest.raises(ValueError):
        build_conditioning_metadata(
            ["DEL/a/DEL-H_water.xlsx", "DEL/b/DEL-H_water.xlsx"]
        )


def test_dataset_returns_condition_with_spectrum_and_mask() -> None:
    spectra = np.zeros((2, 16), dtype=np.float32)
    masks = np.ones_like(spectra)
    conditions = np.zeros((2, 14), dtype=np.float32)
    conditions[:, [0, 4, 8, 12]] = 1.0
    item = SpectrumDataset(
        spectra,
        valid_masks=masks,
        conditions=conditions,
    )[0]
    assert item["spectrum"].shape == (1, 16)
    assert item["valid_mask"].shape == (1, 16)
    assert item["condition"].shape == (14,)


def _configuration() -> dict:
    return {
        "data": {"raman_axis_mode": "union_with_valid_mask"},
        "conditioning": {
            "enabled": True,
            "vector_size": 14,
            "embedding_dimension": 4,
        },
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


def test_conditioned_diffusion_requires_condition_and_samples_masked() -> None:
    unet, diffusion = build_diffusion_model(_configuration(), sequence_length=16)
    assert unet.condition_dimension == 14
    assert unet.network.init_conv.in_channels == 6  # spectrum + mask + 4 embed
    spectra = torch.randn(2, 1, 16)
    masks = torch.ones_like(spectra)
    masks[0, :, 8:] = 0.0
    conditions = torch.zeros(2, 14)
    conditions[:, [0, 4, 8, 12]] = 1.0
    with pytest.raises(ValueError):
        diffusion(spectra, valid_mask=masks)
    loss = diffusion(spectra, valid_mask=masks, condition=conditions)
    assert loss.ndim == 0 and torch.isfinite(loss)
    generated = diffusion.sample(
        batch_size=2,
        valid_mask=masks,
        condition=conditions[0],
    )
    assert generated.shape == (2, 1, 16)
    assert torch.all(generated[0, :, 8:] == 0.0)
    assert torch.isfinite(generated).all()


def test_d4_2_enables_prior_and_injects_condition_into_all_resnet_blocks() -> None:
    configuration = _configuration()
    configuration["conditioning"]["injection"] = (
        "input_and_all_resnet_blocks_film"
    )
    configuration["prior_residual"] = {
        "enabled": True,
        "prior_method": "pca_reconstruction",
    }
    configuration["broad_local_residual"] = {"enabled": True}
    unet, diffusion = build_diffusion_model(configuration, sequence_length=16)
    assert len(unet.conditioned_resnet_blocks) > 0
    assert diffusion.configured_condition_injection == (
        "input_and_all_resnet_blocks_film"
    )
    assert any(
        "condition_film" in name for name in diffusion.state_dict()
    )
