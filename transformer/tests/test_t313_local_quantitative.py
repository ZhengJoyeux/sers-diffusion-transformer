import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from Dataset import MODEL_RAMAN_AXIS, SampleRef, collect_real_training_intensities
from Inference import _ordinal_level_metrics, _ordinal_strata_diagnostics
from Model_v2 import (
    LocalQuantitativeBranch, TransformerClassifyRegress_sep,
    decode_ordinal_numpy, ordinal_class_probabilities_numpy,
    present_only_corn_ordinal_loss,
)
from SERSFormer_Training import _ordinal_metrics_from_probabilities


CENTERS = ((1000.0, 1600.0), (2230.0,), (1090.0, 1597.0))


def _spectra(number=24):
    axis = torch.from_numpy(MODEL_RAMAN_AXIS)
    rows = []
    for index in range(number):
        values = 2.0 + 0.001 * (axis - 600.0)
        for position in (1000.0, 1090.0, 1597.0, 1600.0, 2230.0):
            values = values + (10.0 + index) * torch.exp(-0.5 * ((axis - position) / (3.0 + index % 3)) ** 2)
        rows.append(values)
    intensity = torch.stack(rows).unsqueeze(1)
    return intensity, torch.ones(number, 1901, dtype=torch.bool)


def _model(enabled=False):
    return TransformerClassifyRegress_sep(
        dim_model=8, attn_head=2, dim_ff=16, drop=0.0, encoder_layers=1,
        concentration_head_mode="ordinal", use_mixture_aware_query_fusion=True,
        use_local_quantitative_branch=enabled, local_hidden=4,
    )


def _batch(number=4):
    intensity, mask = _spectra(number)
    raw = intensity / intensity.amax(dim=-1, keepdim=True)
    return {"raw": raw, "percentile": raw.repeat(1, 4, 1), "smoothed": raw,
            "raw_intensity": intensity, "valid_mask": mask}


def test_disabled_branch_preserves_legacy_state_keys_and_strict_load():
    first = _model()
    assert not any(key.startswith("local_quantitative_branch.") for key in first.state_dict())
    second = _model()
    second.load_state_dict(first.state_dict(), strict=True)
    first.eval()
    second.eval()
    with torch.no_grad():
        expected = first(_batch(), return_ordinal=True)
        actual = second(_batch(), return_ordinal=True)
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)


def test_zero_initialized_correction_preserves_t310_forward():
    torch.manual_seed(313)
    old = _model()
    torch.manual_seed(313)
    new = _model(True)
    for name, value in old.state_dict().items():
        torch.testing.assert_close(new.state_dict()[name], value, rtol=0.0, atol=0.0)
    intensity, mask = _spectra()
    new.local_quantitative_branch.fit_normalization(intensity, mask)
    old.eval()
    new.eval()
    with torch.no_grad():
        expected = old(_batch(), return_ordinal=True)
        actual = new(_batch(), return_ordinal=True)
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)


def test_branch_requires_original_intensity_and_fitted_training_state():
    new = _model(True)
    with pytest.raises(RuntimeError, match="Fit local normalization"):
        new(_batch())
    batch = _batch()
    new.local_quantitative_branch.fit_normalization(*_spectra())
    del batch["raw_intensity"]
    with pytest.raises(ValueError, match="raw_intensity"):
        new(batch)


def test_amplitude_and_peak_width_are_retained_in_local_measurements():
    branch = LocalQuantitativeBranch(16, 1901, CENTERS, hidden=4)
    intensity, mask = _spectra()
    _, original, _ = branch._measure(intensity, mask)
    _, doubled, _ = branch._measure(2.0 * intensity, mask)
    assert torch.all(doubled[:, 0, :, 0] > original[:, 0, :, 0])
    assert torch.all(doubled[:, 0, :, 1] > original[:, 0, :, 1])
    torch.testing.assert_close(doubled[:, 0, :, 2:5], original[:, 0, :, 2:5])
    assert original[:, 0, 0, 3].std() > 0.01


def test_invalid_window_cannot_contribute_a_correction():
    branch = LocalQuantitativeBranch(16, 1901, CENTERS, hidden=4)
    intensity, mask = _spectra()
    branch.fit_normalization(intensity, mask)
    for head in branch.projections:
        torch.nn.init.normal_(head[-1].weight, std=0.02)
        torch.nn.init.constant_(head[-1].bias, 0.5)
    mask[:, 1401:] = False
    correction = branch(intensity, mask)
    torch.testing.assert_close(correction[:, 1], torch.zeros_like(correction[:, 1]), rtol=0, atol=0)
    changed = intensity.clone()
    changed[:, :, 1401:] = 1e5
    torch.testing.assert_close(branch(changed, mask), correction, rtol=0, atol=0)


def test_normalization_buffers_survive_checkpoint_without_refitting(tmp_path):
    model = _model(True)
    model.local_quantitative_branch.fit_normalization(*_spectra())
    path = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), path)
    restored = _model(True)
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    assert bool(restored.local_quantitative_branch.normalization_fitted)
    model.eval()
    restored.eval()
    with torch.no_grad():
        for left, right in zip(model(_batch(), return_ordinal=True), restored(_batch(), return_ordinal=True)):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="already fitted"):
        restored.local_quantitative_branch.fit_normalization(*_spectra())


def test_training_updates_local_path_and_keeps_probabilities_ordered():
    model = _model(True)
    model.local_quantitative_branch.fit_normalization(*_spectra())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    initial = model.local_quantitative_branch.projections[0][-1].weight.detach().clone()
    targets = torch.tensor([[1., 2., 3.], [2., 3., 1.], [3., 1., 2.], [1., 3., 2.]])
    for _ in range(3):
        optimizer.zero_grad()
        classification, _, probabilities, logits = model(_batch(), return_ordinal=True, return_ordinal_logits=True)
        loss = present_only_corn_ordinal_loss(logits, targets, torch.ones_like(targets))
        loss = loss + torch.nn.functional.binary_cross_entropy(classification, torch.ones_like(classification))
        loss.backward()
        assert torch.isfinite(loss)
        for parameter in model.parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all()
        optimizer.step()
        assert torch.all(probabilities[..., 1] <= probabilities[..., 0])
    assert not torch.equal(initial, model.local_quantitative_branch.projections[0][-1].weight)
    assert model.local_quantitative_branch.window_cnn[0].weight.grad.abs().sum() > 0


def _repository_fixture():
    intensity, _ = _spectra(20)
    records = {
        name: {"axis": MODEL_RAMAN_AXIS, "spectra": intensity[:, 0, :].numpy().T.copy()}
        for name in ("DEL-S_water", "TEB-H_soil")
    }
    refs = [SampleRef("real", name, index) for name in records for index in range(12)]
    refs.append(SampleRef("generated", "DEL-S_water", 19))
    return SimpleNamespace(split="train", repository=SimpleNamespace(real_data=records), samples=refs)


def test_fit_input_excludes_generated_validation_and_test_rows():
    dataset = _repository_fixture()
    first, masks, audit = collect_real_training_intensities(dataset)
    for record in dataset.repository.real_data.values():
        record["spectra"][:, 12:] = 1e8
    second, other_masks, _ = collect_real_training_intensities(dataset)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(masks, other_masks)
    assert audit["real_training_rows"] == 24
    assert audit["generated_rows_used"] == audit["validation_rows_used"] == audit["test_rows_used"] == 0
    dataset.split = "validation"
    with pytest.raises(ValueError, match="training dataset only"):
        collect_real_training_intensities(dataset)


@pytest.mark.parametrize("bad_row", [12, 16, 19])
def test_fit_input_rejects_nontraining_real_indices(bad_row):
    dataset = _repository_fixture()
    dataset.samples.append(SampleRef("real", "DEL-S_water", bad_row))
    with pytest.raises(RuntimeError, match="non-training"):
        collect_real_training_intensities(dataset)


@pytest.mark.parametrize("mode,expected", [("median", 2), ("map", 3)])
def test_decode_policy_matches_training_and_evaluation(mode, expected):
    probabilities = np.tile([0.65, 0.4875], (4, 3, 1))
    classes = ordinal_class_probabilities_numpy(probabilities)
    np.testing.assert_allclose(classes.sum(axis=-1), 1.0)
    np.testing.assert_array_equal(decode_ordinal_numpy(probabilities, mode=mode), expected)
    truth = np.full((4, 3), 3.0)
    present = np.ones((4, 3))
    frame = pd.DataFrame({"matrix": ["water", "soil", "water", "soil"]})
    train = _ordinal_metrics_from_probabilities(truth, present, present, probabilities, decoding=mode)
    evaluation = _ordinal_level_metrics(truth, probabilities, present, present, frame, decoding=mode)[0]
    assert train["ordinal_accuracy_present"] == evaluation["present_target_accuracy"]
    assert train["ordinal_exact_profile_accuracy"] == evaluation["full_0_S_M_H_profile_exact_accuracy"]
    diagnostics = _ordinal_strata_diagnostics(truth, probabilities, present, frame, mode)
    assert set(diagnostics["group"]) == {"ternary"}
    assert int(diagnostics["H_to_M"].sum()) == (12 if mode == "median" else 0)


def test_decoder_rejects_inconsistent_cumulative_probabilities():
    with pytest.raises(ValueError, match="must not exceed"):
        decode_ordinal_numpy(np.array([0.2, 0.8]), mode="map")


def test_branch_rejects_boundary_specific_fusion_in_first_ablation():
    with pytest.raises(ValueError, match="disable boundary-specific"):
        TransformerClassifyRegress_sep(
            concentration_head_mode="ordinal", use_mixture_aware_query_fusion=True,
            use_adaptive_mixture_gate=True, use_boundary_specific_mixture_gate=True,
            use_local_quantitative_branch=True,
        )
