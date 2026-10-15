from types import SimpleNamespace

import numpy as np
import pytest
import torch

from Model_v2 import TransformerClassifyRegress_sep, present_only_corn_ordinal_loss
from T3_14_Calibration import (AnchoredLocalCalibrator, cumulative_probabilities, anchor_kl_loss,
                              weighted_corn_loss, guard_candidate, selection_score, warmup_cosine_factor)
from Train_T3_14 import anchor_model_config, load_anchor_weights, check_frozen_state


CENTERS = ((1000.0, 1600.0), (2230.0,), (1090.0, 1597.0))


def spectra(n=12):
    axis = torch.arange(600, 2501, dtype=torch.float32)
    rows = []
    for i in range(n):
        x = torch.full_like(axis, 2.0)
        for center in (1000.0, 1600.0, 2230.0, 1090.0, 1597.0):
            x += (10+i) * torch.exp(-0.5 * ((axis-center)/(3+i%3))**2)
        rows.append(x)
    raw = torch.stack(rows)[:, None]
    mask = torch.ones(n, 1901, dtype=torch.bool)
    normalized = raw / raw.amax(dim=-1, keepdim=True)
    return {"raw": normalized, "smoothed": normalized, "percentile": normalized.repeat(1,4,1),
            "raw_intensity": raw, "valid_mask": mask}


def model(enabled=False):
    return TransformerClassifyRegress_sep(dim_model=8, attn_head=2, dim_ff=16, encoder_layers=1,
            drop=0.2, concentration_head_mode="ordinal", use_mixture_aware_query_fusion=True,
            use_anchored_local_calibration=enabled, calibration_hidden=4)


def fitted_module():
    module = AnchoredLocalCalibrator(CENTERS, hidden=4)
    batch = spectra()
    module.local.fit_normalization(batch["raw_intensity"], batch["valid_mask"])
    return module, batch


def test_zero_calibration_exactly_reproduces_loaded_anchor_and_old_model_keys():
    old, new = model(), model(True)
    snapshot = load_anchor_weights(new, {"model_state_dict": old.state_dict()})
    batch = spectra()
    new.anchored_local_calibration.local.fit_normalization(batch["raw_intensity"], batch["valid_mask"])
    old.eval()
    new.train()
    assert not new.cnn.training and not new.transformer_encoder.training
    assert new.anchored_local_calibration.training
    with torch.no_grad():
        left, right = old(batch, return_ordinal=True), new(batch, return_ordinal=True)
    for a, b in zip(left, right):
        # CPU/CUDA attention kernels can differ by one float32 ULP when
        # requires_grad flags differ. Frozen state is still checked exactly.
        torch.testing.assert_close(a, b, rtol=0, atol=1e-6)
    check_frozen_state(new, snapshot)
    assert all(name.startswith("anchored_local_calibration.") for name,p in new.named_parameters() if p.requires_grad)
    assert not any("anchored_local_calibration" in key for key in old.state_dict())


def test_three_updates_only_change_local_parameters_and_leave_classification_identical():
    old, new = model(), model(True)
    frozen = load_anchor_weights(new, {"model_state_dict": old.state_dict()})
    batch = spectra()
    new.anchored_local_calibration.local.fit_normalization(batch["raw_intensity"], batch["valid_mask"])
    old.eval()
    with torch.no_grad():
        reference = old(batch)[0].detach()
    parameters = [p for p in new.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=1e-2)
    initial = new.anchored_local_calibration.local.projections[0][-1].weight.detach().clone()
    targets = torch.ones(12,3)
    for _ in range(3):
        new.train()
        optimizer.zero_grad()
        classification, _, probability, logits = new(batch, return_ordinal=True, return_ordinal_logits=True)
        torch.testing.assert_close(classification, reference, rtol=0, atol=0)
        loss = weighted_corn_loss(logits, targets, torch.ones_like(targets), torch.ones(12))
        loss += .1 * anchor_kl_loss(new.anchored_local_calibration.last_base_probabilities,
                                   probability, torch.ones_like(targets), torch.ones(12))
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None)
        optimizer.step()
        check_frozen_state(new, frozen)
        assert torch.all(probability[...,1] <= probability[...,0])
    assert not torch.equal(initial, new.anchored_local_calibration.local.projections[0][-1].weight)
    assert new.anchored_local_calibration.local.window_cnn[0].weight.grad.abs().sum() > 0


def test_checkpoint_restores_local_normalization_without_refit(tmp_path):
    new = model(True)
    batch = spectra()
    new.anchored_local_calibration.local.fit_normalization(batch["raw_intensity"], batch["valid_mask"])
    with torch.no_grad():
        for head in new.anchored_local_calibration.local.projections:
            head[-1].bias.fill_(0.8)
    path = tmp_path / "state.pt"
    torch.save(new.state_dict(), path)
    loaded = model(True)
    loaded.load_state_dict(torch.load(path, weights_only=True), strict=True)
    new.eval(); loaded.eval()
    with torch.no_grad():
        for a,b in zip(new(batch,return_ordinal=True), loaded(batch,return_ordinal=True)):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
    with pytest.raises(RuntimeError, match="already fitted"):
        loaded.anchored_local_calibration.local.fit_normalization(batch["raw_intensity"], batch["valid_mask"])


def test_bounded_shift_is_shared_and_confident_anchor_is_unchanged():
    module, batch = fitted_module()
    for head in module.local.projections:
        torch.nn.init.constant_(head[-1].bias, 1e4)
    base = torch.zeros(12,3,2)
    corrected = module(batch["raw_intensity"],batch["valid_mask"],base)
    delta = corrected - base
    assert delta.abs().max() <= .5
    assert (delta > 0).any()
    torch.testing.assert_close(delta[...,0],delta[...,1],rtol=0,atol=0)
    confident = torch.full_like(base,10.)
    torch.testing.assert_close(module(batch["raw_intensity"],batch["valid_mask"],confident),confident,rtol=0,atol=0)
    confident = torch.full_like(base,-10.)
    torch.testing.assert_close(module(batch["raw_intensity"],batch["valid_mask"],confident),confident,rtol=0,atol=0)


def test_invalid_chl_window_and_zero_signal_have_zero_correction():
    module,batch = fitted_module()
    for head in module.local.projections:
        torch.nn.init.constant_(head[-1].bias,2.)
    batch["valid_mask"][:,1401:] = False
    result = module(batch["raw_intensity"],batch["valid_mask"],torch.zeros(12,3,2))
    assert torch.equal(result[:,1],torch.zeros_like(result[:,1]))
    constant = torch.full_like(batch["raw_intensity"],1.)
    result = module(constant,torch.ones_like(batch["valid_mask"]),torch.zeros(12,3,2))
    torch.testing.assert_close(result,torch.zeros_like(result),rtol=0,atol=1e-5)


def test_no_label_or_matrix_metadata_is_required_by_forward():
    new = model(True)
    batch=spectra()
    new.anchored_local_calibration.local.fit_normalization(batch["raw_intensity"],batch["valid_mask"])
    new(batch,return_ordinal=True)
    batch.update(matrix_target=torch.full((12,),-999),class_target=torch.zeros(12,3),concentration_target=torch.full((12,3),1e5))
    with torch.no_grad():
        output=new(batch,return_ordinal=True)
        del batch["matrix_target"];del batch["class_target"];del batch["concentration_target"]
        plain=new(batch,return_ordinal=True)
    for a,b in zip(output,plain):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_equal_weight_corn_matches_legacy_loss_and_zero_kl():
    torch.manual_seed(14)
    logits=torch.randn(7,3,2,requires_grad=True)
    targets=torch.randint(1,4,(7,3)).float()
    presence=torch.randint(0,2,(7,3)).float()
    torch.testing.assert_close(weighted_corn_loss(logits,targets,presence,torch.ones(7)),
                               present_only_corn_ordinal_loss(logits,targets,presence))
    base=cumulative_probabilities(logits)
    assert float(anchor_kl_loss(base,base,presence,torch.ones(7)).detach()) == 0.
    absent=weighted_corn_loss(logits,targets,torch.zeros_like(presence),torch.ones(7))
    assert float(absent.detach())==0.


def metrics():
    return {"class_subset_accuracy":1.,"ordinal_accuracy_present":.86,"ordinal_exact_ternary_water":.61,
            "ordinal_exact_ternary_soil":.55,"ordinal_accuracy_CHL_ternary":.89,
            "ordinal_accuracy_CHL_ternary_water":.91,
            "ordinal_exact_ternary":.58,"ordinal_exact_profile_accuracy":.75,"ordinal_severe_error_rate":.003}


@pytest.mark.parametrize("key",["class_subset_accuracy","ordinal_accuracy_present","ordinal_exact_ternary_water",
                               "ordinal_exact_ternary_soil","ordinal_accuracy_CHL_ternary","ordinal_accuracy_CHL_ternary_water"])
def test_guard_rejects_each_predeclared_regression(key):
    old=metrics();new=metrics();new[key]-=.01
    new["ordinal_exact_ternary"]+=.05
    assert guard_candidate(new,old)[0] is False


def test_guard_accepts_nonregression_but_selection_needs_real_score_improvement():
    old=metrics();new=metrics()
    assert guard_candidate(new,old)[0]
    assert not selection_score(new)>selection_score(old)
    new["ordinal_exact_ternary"]+=.01
    assert selection_score(new)>selection_score(old)
    del new["ordinal_exact_ternary_water"]
    assert guard_candidate(new,old)[0] is False


def test_guard_rejects_more_severe_errors_even_if_ternary_accuracy_improves():
    old=metrics();new=metrics();new["ordinal_exact_ternary"]+=.03
    new["ordinal_severe_error_rate"]+=.001
    assert not guard_candidate(new,old)[0]


@pytest.mark.parametrize("flag",["use_adaptive_mixture_gate","use_boundary_specific_mixture_gate","use_local_quantitative_branch"])
def test_wrong_reference_checkpoint_is_rejected(flag):
    cfg={"concentration_head_mode":"ordinal","use_mixture_aware_query_fusion":True,flag:True}
    args=SimpleNamespace(logit_bound=.5,uncertainty_band=.15,local_hidden=4,local_half_width=24)
    with pytest.raises(ValueError,match="original ordinal T3.10"):
        anchor_model_config({"model_config":cfg,"training_config":{"ordinal_loss_mode":"corn"}},args)


def test_incompatible_anchor_state_is_not_silently_loaded():
    old=model();state=dict(old.state_dict());state.pop(next(iter(state)))
    with pytest.raises(ValueError,match="incompatible"):
        load_anchor_weights(model(True),{"model_state_dict":state})


def test_warmup_cosine_update_schedule_has_correct_endpoints_and_no_restart():
    factors=np.array([warmup_cosine_factor(i,total_steps=100,warmup_steps=20,minimum_ratio=.01) for i in range(105)])
    assert factors[0]==pytest.approx(.1)
    assert factors[19]==pytest.approx(1.) and factors[20]==pytest.approx(1.)
    assert np.all(np.diff(factors[:20])>=0)
    assert np.all(np.diff(factors[20:])<=1e-12)
    assert factors[99]==pytest.approx(.01) and factors[104]==pytest.approx(.01)
    parameter=torch.nn.Parameter(torch.tensor(1.))
    opt=torch.optim.Adam([parameter],lr=1e-4)
    schedule=torch.optim.lr_scheduler.LambdaLR(opt,lambda i:warmup_cosine_factor(i,total_steps=100,warmup_steps=20,minimum_ratio=.01))
    used=[]
    for _ in range(100):
        used.append(opt.param_groups[0]["lr"])
        opt.zero_grad();parameter.square().backward();opt.step();schedule.step()
    np.testing.assert_allclose(used,1e-4*factors[:100])
    assert schedule.last_epoch==100


@pytest.mark.parametrize("total,warm",[(1,0),(10,10),(10,-1)])
def test_warmup_cosine_rejects_invalid_step_budgets(total,warm):
    with pytest.raises(ValueError):
        warmup_cosine_factor(0,total_steps=total,warmup_steps=warm,minimum_ratio=.01)
