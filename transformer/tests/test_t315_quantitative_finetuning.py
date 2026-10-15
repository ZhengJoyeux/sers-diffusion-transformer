import copy
from types import SimpleNamespace

import pytest
import torch

from Model_v2 import TransformerClassifyRegress_sep
from Train_T3_15 import (anchor_config, configure_finetuning, set_finetuning_mode,
                        check_frozen_state, check_classification, run_finetuning_epoch)
from Train_T3_14 import save_checkpoint


def make_model(full=False):
    return TransformerClassifyRegress_sep(
        dim_model=32 if full else 8, attn_head=4 if full else 2,
        dim_ff=64 if full else 16, encoder_layers=4 if full else 1,
        drop=0.2, concentration_head_mode="ordinal", ordinal_head_hidden=64,
        use_mixture_aware_query_fusion=True)


def make_batch(n=6):
    generator = torch.Generator().manual_seed(315)
    raw = torch.rand(n, 1, 1901, generator=generator)
    return {"raw": raw, "smoothed": raw,
            "percentile": raw.repeat(1, 4, 1), "raw_intensity": raw,
            "valid_mask": torch.ones(n, 1901, dtype=torch.bool),
            "class_target": torch.ones(n, 3),
            "concentration_target": torch.tensor([[1., 2., 3.]]).repeat(n, 1),
            "source": ["real", "generated"] * (n // 2)}


class Batches(list):
    dataset = SimpleNamespace(split="train")


def updated_model():
    torch.manual_seed(315)
    model = make_model()
    anchor = copy.deepcopy(model).requires_grad_(False).eval()
    counts, frozen = configure_finetuning(model)
    old = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    batch = make_batch()
    metrics, _ = run_finetuning_epoch(model, anchor, Batches([batch] * 3),
                                     torch.device("cpu"), optimizer=optimizer)
    return model, anchor, batch, counts, frozen, old, optimizer, metrics


def test_actual_t310_parameter_scope_and_unchanged_architecture():
    model = make_model(full=True)
    keys = set(model.state_dict())
    counts, _ = configure_finetuning(model)
    assert counts == {"mixture_aware_query_fusion": 33219, "ordinal_heads": 12870}
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 46089
    assert sum(p.numel() for p in model.parameters() if not p.requires_grad) == 197862
    assert set(model.state_dict()) == keys
    assert all(n.startswith(("mixture_aware_query_fusion.", "ordinal_heads."))
               for n, p in model.named_parameters() if p.requires_grad)


def test_epoch_zero_outputs_exactly_match_original_and_modes_are_separate():
    model = make_model()
    anchor = copy.deepcopy(model).eval()
    configure_finetuning(model)
    batch = make_batch()
    with torch.no_grad():
        for actual, expected in zip(model(batch, return_ordinal=True), anchor(batch, return_ordinal=True)):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
    for training in (True, False, True):
        set_finetuning_mode(model, training)
        assert model.mixture_aware_query_fusion.training == training
        assert model.ordinal_heads.training == training
        assert not model.cnn.training and not model.transformer_encoder.training
        assert not model.classification_heads.training
        assert all(not module.training for name, module in model.named_modules()
                   if isinstance(module, torch.nn.BatchNorm1d))


def test_optimizer_changes_both_quantitative_modules_but_no_original_state_or_classification():
    model, anchor, batch, _, frozen, old, _, metrics = updated_model()
    check_frozen_state(model, frozen)
    for prefix in ("mixture_aware_query_fusion.", "ordinal_heads."):
        assert any(not torch.equal(value, old[name]) for name, value in model.state_dict().items()
                   if name.startswith(prefix))
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert metrics["processed_rows"] == 18
    assert metrics["real_rows_processed"] == metrics["generated_rows_processed"] == 9
    assert metrics["classification_max_abs_error_to_anchor"] == 0
    set_finetuning_mode(model, False)
    with torch.no_grad():
        actual = model(batch, return_ordinal=True, return_ordinal_logits=True)
        original = anchor(batch, return_ordinal=True, return_ordinal_logits=True)
    torch.testing.assert_close(actual[0], original[0], atol=1e-6, rtol=0)
    assert not torch.equal(actual[3], original[3])
    assert torch.all(actual[2][..., 1] <= actual[2][..., 0])


def test_trained_checkpoint_reload_needs_no_architecture_change_or_normalization_refit(tmp_path):
    model, _, batch, _, frozen, _, optimizer, metrics = updated_model()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    config = dict(dim_model=8, attn_head=2, dim_ff=16, encoder_layers=1, drop=.2,
                  concentration_head_mode="ordinal", ordinal_head_hidden=64,
                  use_mixture_aware_query_fusion=True)
    path = tmp_path / "t315.pt"
    save_checkpoint(path, model, optimizer, scheduler, 3, config,
                    {"ordinal_loss_mode": "corn", "training_phase": "t315_quantitative_branch_finetune"},
                    {}, metrics)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    loaded = TransformerClassifyRegress_sep(**saved["model_config"])
    loaded.load_state_dict(saved["model_state_dict"], strict=True)
    loaded.eval(); model.eval()
    check_frozen_state(loaded, frozen)
    with torch.no_grad():
        for left, right in zip(loaded(batch, return_ordinal=True), model(batch, return_ordinal=True)):
            torch.testing.assert_close(left, right, atol=1e-6, rtol=0)


@pytest.mark.parametrize("buffer", [False, True])
def test_frozen_integrity_check_detects_parameter_and_batchnorm_buffer_mutation(buffer):
    model = make_model()
    _, frozen = configure_finetuning(model)
    state = model.state_dict()
    key = next(k for k in frozen if ("running_mean" in k if buffer else k.endswith("weight")))
    state[key].add_(1)
    with pytest.raises(RuntimeError, match="state changed"):
        check_frozen_state(model, frozen)


@pytest.mark.parametrize("value", [1e-3, float("nan"), float("inf")])
def test_classification_guard_rejects_drift_and_nonfinite_predictions(value):
    with pytest.raises(RuntimeError, match="Classification output drifted"):
        check_classification(torch.full((2, 3), value), torch.zeros(2, 3))


@pytest.mark.parametrize("flag", ["use_adaptive_mixture_gate", "use_boundary_specific_mixture_gate",
                                 "use_local_quantitative_branch", "use_anchored_local_calibration"])
def test_wrong_architecture_is_rejected_before_training(flag):
    checkpoint = {"model_config": {"concentration_head_mode": "ordinal",
                    "use_mixture_aware_query_fusion": True, flag: True},
                  "training_config": {"ordinal_loss_mode": "corn"}}
    with pytest.raises(ValueError, match="original verified"):
        anchor_config(checkpoint)


def test_a_previous_finetuned_candidate_cannot_become_the_original_teacher():
    checkpoint = {"model_config": {"concentration_head_mode": "ordinal", "use_mixture_aware_query_fusion": True},
                  "training_config": {"ordinal_loss_mode": "corn", "training_phase": "t315_quantitative_branch_finetune"}}
    with pytest.raises(ValueError, match="original verified"):
        anchor_config(checkpoint)


def test_validation_only_pass_does_not_update_state_or_accept_unknown_sources():
    model = make_model()
    anchor = copy.deepcopy(model).requires_grad_(False).eval()
    configure_finetuning(model)
    before = copy.deepcopy(model.state_dict())
    run_finetuning_epoch(model, anchor, Batches([make_batch()]), torch.device("cpu"))
    for name, value in model.state_dict().items():
        assert torch.equal(before[name], value)
    bad = make_batch(); bad["source"][0] = "test"
    with pytest.raises(ValueError, match="Unexpected data source"):
        run_finetuning_epoch(model, anchor, Batches([bad]), torch.device("cpu"))
