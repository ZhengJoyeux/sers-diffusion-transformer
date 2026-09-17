from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.conditional_prior_residual import ConditionalPriorResidualBank
from src.spectrum_generator import generate_spectra
from src.spectrum_dataset import SpectrumDataset
from src.spectrum_length_adapter import SpectrumLengthAdapter


def _configuration() -> tuple[dict, dict]:
    prior = {
        "enabled": True,
        "prior_method": "pca_reconstruction",
        "pca_explained_variance_ratio": 0.95,
        "pca_max_components": 2,
        "pca_sampling_strategy": "independent_truncated_gaussian_scores",
        "pca_score_clip_standard_deviations": 1.5,
        "residual_normalization": "robust_asinh",
        "residual_quantile": 99.0,
        "target_abs_max": 1.0,
        "epsilon": 1.0e-8,
    }
    broad = {
        "enabled": True,
        "broad_filter": {"method": "gaussian", "sigma_cm1": 1.5, "truncate": 3.0},
        "broad_prior": {
            "method": "pca",
            "pca_explained_variance_ratio": 0.95,
            "pca_max_components": 2,
            "sampling_strategy": "independent_truncated_gaussian_scores",
            "score_clip_standard_deviations": 1.5,
        },
        "local_normalization": {
            "method": "robust_asinh",
            "residual_quantile": 99.0,
            "target_abs_max": 1.0,
        },
        "epsilon": 1.0e-8,
    }
    return prior, broad


def _data() -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    generator = np.random.default_rng(42)
    axis = np.arange(600.0, 616.0)
    conditions = ["DEL-H_water"] * 8 + ["CHL-H_soil"] * 8
    masks = np.ones((16, 16), dtype=np.float32)
    masks[:8, 12:] = 0.0
    spectra = np.zeros((16, 16), dtype=np.float32)
    for index in range(16):
        length = 12 if index < 8 else 16
        center = 4.0 if index < 8 else 10.0
        x = np.arange(length, dtype=np.float32)
        spectra[index, :length] = (
            0.1 * x
            + np.exp(-0.5 * ((x - center) / 1.2) ** 2)
            + generator.normal(0.0, 0.03, size=length)
        )
    spectra[:8, 12:] = 9999.0  # must never enter the short-axis prior
    train = np.asarray([0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13])
    return spectra, masks, conditions, train, axis


def test_condition_bank_fits_all_conditions_and_masks_invalid_tail() -> None:
    spectra, masks, conditions, train, axis = _data()
    prior, broad = _configuration()
    bank = ConditionalPriorResidualBank(
        prior_configuration=prior,
        broad_local_configuration=broad,
    ).fit(
        spectra,
        valid_masks=masks,
        condition_ids=conditions,
        training_indices=train,
        raman_shift=axis,
    )
    transformed = bank.transform(
        spectra,
        valid_masks=masks,
        condition_ids=conditions,
    )
    assert transformed.shape == spectra.shape
    assert np.all(transformed[:8, 12:] == 0.0)
    assert np.isfinite(transformed).all()
    summary = bank.summary()
    assert summary["number_of_conditions"] == 2
    assert summary["total_training_spectra"] == 12
    assert summary["valid_lengths"] == {"12": 1, "16": 1}


def test_condition_bank_checkpoint_round_trip_and_generation() -> None:
    spectra, masks, conditions, train, axis = _data()
    prior, broad = _configuration()
    fitted = ConditionalPriorResidualBank(
        prior_configuration=prior,
        broad_local_configuration=broad,
    ).fit(
        spectra,
        valid_masks=masks,
        condition_ids=conditions,
        training_indices=train,
        raman_shift=axis,
    )
    restored = ConditionalPriorResidualBank.from_state_dict(fitted.state_dict())
    generated = restored.reconstruct_generated(
        np.zeros((3, 16), dtype=np.float32),
        condition_id="DEL-H_water",
        prior_random_generator=np.random.default_rng(100),
        broad_random_generator=np.random.default_rng(200),
    )
    assert generated.shape == (3, 12)
    assert np.isfinite(generated).all()
    assert restored.valid_length("CHL-H_soil") == 16


def test_transform_conditioning_round_trip_uses_matching_base() -> None:
    spectra, masks, conditions, train, axis = _data()
    prior, broad = _configuration()
    bank = ConditionalPriorResidualBank(
        prior_configuration=prior,
        broad_local_configuration=broad,
    ).fit(
        spectra,
        valid_masks=masks,
        condition_ids=conditions,
        training_indices=train,
        raman_shift=axis,
    )
    residual, conditioning = bank.transform_with_conditioning(
        spectra,
        valid_masks=masks,
        condition_ids=conditions,
    )
    restored = bank.reconstruct_generated_with_conditioning(
        residual[:8],
        prior_conditioning=conditioning[:8],
        condition_id="DEL-H_water",
    )
    np.testing.assert_allclose(restored, spectra[:8, :12], atol=1.0e-5)
    assert np.all(conditioning[:8, 12:] == 0.0)

    dataset = SpectrumDataset(
        residual,
        valid_masks=masks,
        conditions=np.zeros((residual.shape[0], 14), dtype=np.float32),
        prior_conditionings=conditioning,
    )
    item = dataset[0]
    assert isinstance(item, dict)
    assert item["prior_conditioning"].shape == item["spectrum"].shape


def test_runtime_score_clip_override_updates_outer_and_broad_only_in_memory() -> None:
    spectra, masks, conditions, train, axis = _data()
    prior, broad = _configuration()
    fitted = ConditionalPriorResidualBank(
        prior_configuration=prior,
        broad_local_configuration=broad,
    ).fit(
        spectra,
        valid_masks=masks,
        condition_ids=conditions,
        training_indices=train,
        raman_shift=axis,
    )
    checkpoint_state = fitted.state_dict()
    restored = ConditionalPriorResidualBank.from_state_dict(checkpoint_state)

    information = restored.apply_score_clip_runtime_override(2.5)

    assert information["active"]
    assert information["overridden"]
    assert information["checkpoint_outer_values"] == [1.5]
    assert information["checkpoint_broad_values"] == [1.5]
    assert information["runtime_value"] == 2.5
    for entry in restored.entries.values():
        assert entry.outer.pca_score_clip_standard_deviations == 2.5
        assert entry.broad_local.broad_score_clip_standard_deviations == 2.5

    # The serialized checkpoint dictionary passed to from_state_dict is not mutated.
    first = next(iter(checkpoint_state["conditions"].values()))
    assert first["prior_residual_state"]["pca_prior"][
        "score_clip_standard_deviations"
    ] == 1.5
    assert first["broad_local_residual_state"]["broad_prior"][
        "score_clip_standard_deviations"
    ] == 1.5


def test_diagnostic_transform_round_trip_restores_selected_real_spectrum() -> None:
    spectra, masks, conditions, train, axis = _data()
    prior, broad = _configuration()
    bank = ConditionalPriorResidualBank(
        prior_configuration=prior,
        broad_local_configuration=broad,
    ).fit(
        spectra,
        valid_masks=masks,
        condition_ids=conditions,
        training_indices=train,
        raman_shift=axis,
    )

    model_domain, base = bank.diagnostic_transform(
        spectra[[0]],
        valid_mask=masks[[0]],
        condition_id="DEL-H_water",
    )
    restored = bank.restore_diagnostic_prediction(
        model_domain,
        deterministic_base=base,
        condition_id="DEL-H_water",
    )

    assert np.allclose(restored[:, :12], spectra[[0], :12], atol=1.0e-5)
    assert np.all(restored[:, 12:] == 0.0)


class _ZeroMaskedDiffusion(nn.Module):
    supports_valid_mask = True
    supports_condition = True
    supports_prior_conditioning = True
    configured_prior_conditioning_enabled = True

    def sample(
        self,
        batch_size: int,
        *,
        valid_mask,
        condition,
        prior_conditioning,
    ):
        assert condition.shape == (14,)
        assert prior_conditioning.shape == (batch_size, 16)
        assert torch.isfinite(prior_conditioning).all()
        mask = valid_mask.reshape(1, 1, -1).expand(batch_size, 1, -1)
        return torch.zeros_like(mask)


def test_generator_restores_short_condition_to_its_original_axis() -> None:
    spectra, masks, conditions, train, axis = _data()
    prior, broad = _configuration()
    bank = ConditionalPriorResidualBank(
        prior_configuration=prior,
        broad_local_configuration=broad,
    ).fit(
        spectra,
        valid_masks=masks,
        condition_ids=conditions,
        training_indices=train,
        raman_shift=axis,
    )
    short_axis = axis[:12]
    adapter = SpectrumLengthAdapter.create(
        raman_shifts=[short_axis, axis],
        dimension_multipliers=[1, 2],
        raman_axis_mode="union_with_valid_mask",
    )
    generated = generate_spectra(
        diffusion=_ZeroMaskedDiffusion(),
        number_of_spectra=3,
        generation_batch_size=2,
        device=torch.device("cpu"),
        length_adapter=adapter,
        output_raman_shifts=short_axis,
        condition_vector=np.zeros(14, dtype=np.float32),
        condition_id="DEL-H_water",
        conditional_prior_residual_bank=bank,
        prior_random_seed=2026,
    )
    assert generated.shape == (3, 12)
    assert np.isfinite(generated).all()

def test_d4_3_2_14_cross_fit_disabled_preserves_legacy_transform() -> None:
    spectra, masks, conditions, train, axis = _data()
    prior, broad = _configuration()
    prior = dict(prior)
    prior["training_cross_fit"] = {"enabled": False, "method": "leave_one_out"}
    bank = ConditionalPriorResidualBank(
        prior_configuration=prior, broad_local_configuration=broad,
    ).fit(spectra, valid_masks=masks, condition_ids=conditions,
          training_indices=train, raman_shift=axis)
    legacy_r, legacy_b = bank.transform_with_conditioning(
        spectra, valid_masks=masks, condition_ids=conditions)
    aware_r, aware_b = bank.transform_training_aware_with_conditioning(
        spectra, valid_masks=masks, condition_ids=conditions, training_indices=train)
    np.testing.assert_allclose(aware_r, legacy_r, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(aware_b, legacy_b, rtol=0.0, atol=0.0)


def test_d4_3_2_14_leave_one_out_training_transform_is_out_of_fold() -> None:
    spectra, masks, conditions, train, axis = _data()
    prior, broad = _configuration()
    prior = dict(prior)
    prior["training_cross_fit"] = {"enabled": True, "method": "leave_one_out"}
    bank = ConditionalPriorResidualBank(
        prior_configuration=prior, broad_local_configuration=broad,
    ).fit(spectra, valid_masks=masks, condition_ids=conditions,
          training_indices=train, raman_shift=axis)
    final_r, final_b = bank.transform_with_conditioning(
        spectra, valid_masks=masks, condition_ids=conditions)
    aware_r, aware_b = bank.transform_training_aware_with_conditioning(
        spectra, valid_masks=masks, condition_ids=conditions, training_indices=train)
    train = np.asarray(train, dtype=np.int64)
    non_train = np.asarray(sorted(set(range(spectra.shape[0])) - set(train.tolist())), dtype=np.int64)
    if non_train.size:
        np.testing.assert_allclose(aware_r[non_train], final_r[non_train], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(aware_b[non_train], final_b[non_train], rtol=0.0, atol=0.0)
    assert float(np.max(np.abs(aware_b[train] - final_b[train]))) > 1.0e-7
    for index in train:
        entry = bank._entry(str(conditions[index]))
        n = entry.valid_length
        local = entry.broad_local.inverse_local_transform(aware_r[index:index+1, :n])
        restored = aware_b[index:index+1, :n] + local
        np.testing.assert_allclose(restored, spectra[index:index+1, :n], rtol=2e-5, atol=2e-6)


def test_d4_3_2_14_checkpoint_final_prior_and_legacy_compatibility() -> None:
    spectra, masks, conditions, train, axis = _data()
    prior, broad = _configuration()
    prior = dict(prior)
    prior["training_cross_fit"] = {"enabled": True, "method": "leave_one_out"}
    bank = ConditionalPriorResidualBank(
        prior_configuration=prior, broad_local_configuration=broad,
    ).fit(spectra, valid_masks=masks, condition_ids=conditions,
          training_indices=train, raman_shift=axis)
    state = bank.state_dict()
    assert state["training_cross_fit"] == {"enabled": True, "method": "leave_one_out"}
    assert "_cross_fitted_reference_priors" not in state
    restored = ConditionalPriorResidualBank.from_state_dict(state)
    assert restored.training_cross_fit_metadata()["enabled"] is True
    sampled = restored.sample_generation_conditioning(
        3, condition_id=str(conditions[int(train[0])]),
        prior_random_generator=np.random.default_rng(2026),
        broad_random_generator=np.random.default_rng(2027))
    assert sampled.shape[0] == 3 and np.isfinite(sampled).all()
    legacy_state = dict(state)
    legacy_state.pop("training_cross_fit")
    legacy = ConditionalPriorResidualBank.from_state_dict(legacy_state)
    assert legacy.training_cross_fit_metadata() == {"enabled": False, "method": "leave_one_out"}

