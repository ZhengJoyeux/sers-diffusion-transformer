from __future__ import annotations

import copy
import numpy as np
import torch

from src.conditional_diversity_constraints import (
    fit_condition_aware_diversity_constraint_state,
)
from src.model_builder import build_diffusion_model


def _configuration() -> dict:
    return {
        "data": {"raman_axis_mode": "union_with_valid_mask"},
        "conditioning": {
            "enabled": True,
            "vector_size": 14,
            "embedding_dimension": 8,
            "injection": "input_and_all_resnet_blocks_film",
        },
        "normalization": {"enabled": True},
        "prior_residual": {"enabled": False},
        "broad_local_residual": {"enabled": False},
        "physics_constraints": {"enabled": False},
        "diversity_constraints": {
            "enabled": True,
            "total_weight": 0.02,
            "maximum_total_ratio_to_ddpm": 0.02,
            "condition_grouping": {
                "samples_per_condition": 4,
                "shared_timestep": True,
            },
            "low_noise_gate": {
                "minimum_alpha_cumprod": 0.5,
                "minimum_samples": 4,
            },
            "pairwise_distance": {
                "minimum_distance_ratio": 0.8,
                "maximum_distance_ratio": 1.25,
            },
            "pointwise_variance_floor": {
                "minimum_std_ratio": 0.8,
                "maximum_std_ratio": 1.25,
            },
            "peak_morphology": {"enabled": False},
        },
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


def test_d4_3_runs_inside_masked_diffusion_without_state_dict_changes() -> None:
    generator = np.random.default_rng(2026)
    training = generator.normal(0.0, 0.1, size=(8, 16)).astype(np.float32)
    masks = np.ones_like(training)
    conditions = np.zeros((8, 14), dtype=np.float32)
    conditions[:, 0] = 1.0
    state = fit_condition_aware_diversity_constraint_state(
        training_scaled_residuals=training,
        training_valid_masks=masks,
        training_condition_vectors=conditions,
        configuration=_configuration()["diversity_constraints"],
    )

    _, diffusion = build_diffusion_model(_configuration(), sequence_length=16)
    keys_before = set(diffusion.state_dict())
    diffusion.configure_diversity_constraints(
        diversity_constraint_state=state
    )
    keys_after = set(diffusion.state_dict())
    assert keys_before == keys_after

    spectrum = torch.as_tensor(training[:4]).unsqueeze(1)
    mask = torch.ones_like(spectrum)
    condition = torch.as_tensor(conditions[:4])
    loss = diffusion.p_losses(
        spectrum,
        torch.zeros(4, dtype=torch.long),
        noise=torch.zeros_like(spectrum),
        valid_mask=mask,
        condition=condition,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in diffusion.parameters()
    )
    components = diffusion.get_latest_loss_components()
    assert components["conditional_diversity_active_groups"] == 1
    assert components["diversity_loss"] <= 0.02 * components["ddpm_loss"] + 1.0e-8


def test_conditioned_model_predictions_supports_explicit_diagnostic_context() -> None:
    _, diffusion = build_diffusion_model(_configuration(), sequence_length=16)
    spectrum = torch.randn(1, 1, 16)
    timestep = torch.tensor([9], dtype=torch.long)
    mask = torch.ones_like(spectrum)
    mask[..., 12:] = 0.0
    condition = torch.zeros(1, 14)
    condition[:, 0] = 1.0

    prediction = diffusion.model_predictions(
        spectrum,
        timestep,
        clip_x_start=False,
        valid_mask=mask,
        condition=condition,
    )

    assert prediction.pred_x_start.shape == spectrum.shape
    assert prediction.pred_noise.shape == spectrum.shape
    assert torch.all(prediction.pred_x_start[..., 12:] == 0.0)
    assert torch.all(prediction.pred_noise[..., 12:] == 0.0)


def test_prior_conditioned_residual_requires_and_uses_matching_prior() -> None:
    configuration = copy.deepcopy(_configuration())
    configuration["diversity_constraints"]["enabled"] = False
    configuration["prior_residual"]["enabled"] = True
    configuration["broad_local_residual"]["enabled"] = True
    configuration["conditioning"]["prior_spectrum"] = {
        "enabled": True,
        "source": "condition_reconstruction_base",
        "injection": "input_channel",
    }
    unet, diffusion = build_diffusion_model(configuration, sequence_length=16)
    assert unet.prior_conditioning_enabled is True
    assert diffusion.configured_prior_conditioning_enabled is True

    spectrum = torch.randn(2, 1, 16)
    mask = torch.ones_like(spectrum)
    mask[..., 12:] = 0.0
    condition = torch.zeros(2, 14)
    condition[:, 0] = 1.0
    prior = torch.randn_like(spectrum) * mask

    with torch.no_grad():
        try:
            diffusion.p_losses(
                spectrum,
                torch.zeros(2, dtype=torch.long),
                noise=torch.zeros_like(spectrum),
                valid_mask=mask,
                condition=condition,
            )
        except ValueError as error:
            assert "prior_conditioning" in str(error)
        else:
            raise AssertionError("缺少prior_conditioning时应拒绝训练。")

    loss = diffusion.p_losses(
        spectrum,
        torch.zeros(2, dtype=torch.long),
        noise=torch.zeros_like(spectrum),
        valid_mask=mask,
        condition=condition,
        prior_conditioning=prior,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in diffusion.parameters()
    )


def _quality_budget_configuration(
    *,
    independent_enabled: bool,
) -> dict:
    """Build a small deterministic D4.3.2 quality-budget test config."""
    configuration = copy.deepcopy(_configuration())

    # Disable diversity here so the quality-budget accounting can be
    # checked independently.
    configuration["diversity_constraints"]["enabled"] = False

    configuration["diversity_constraints"]["quality_fidelity"] = {
        "enabled": True,
        "total_weight": 0.1,
        "maximum_total_ratio_to_ddpm": 0.1,
        "independent_full_spectrum_budget": {
            "enabled": independent_enabled,
            "total_weight": 0.1,
            "maximum_total_ratio_to_ddpm": 0.05,
        },
        "full_spectrum_tail_distribution": {
            "enabled": True,
            "piecewise_probabilities": [
                0.01,
                0.10,
                0.50,
                0.90,
                0.99,
            ],
            "segment_weights": [
                1.5,
                0.5,
                0.5,
                1.5,
            ],
            "minimum_segment_span_ratio": 0.9,
            "maximum_segment_span_ratio": 1.1,
            "maximum_timestep_fraction": 0.5,
            "minimum_group_size": 4,
            "pointwise_scale_floor_quantile": 10.0,
            "reconstruction_smooth_l1_beta": 0.01,
            "span_smooth_l1_beta": 0.02,
            "reconstruction_weight": 0.75,
            "span_weight": 1.0,
            "short_axis_maximum_points": 1500,
            "short_axis_weight_multiplier": 1.4,
            "epsilon": 1.0e-6,
        },
    }

    return configuration


def _install_fixed_quality_losses(diffusion) -> None:
    """Replace only the internal quality calculation with fixed scalars.

    This leaves p_losses and the budget/cap implementation untouched,
    allowing the test to verify the accounting deterministically.
    """

    def fixed_quality_losses(**kwargs):
        prediction = kwargs["prediction"]
        scalar = prediction.new_tensor

        # Weighted full-spectrum raw term:
        #
        #   0.75 * 200 + 1.00 * 200 = 350
        #
        # Total quality raw = 1000, therefore:
        #
        #   legacy raw = 1000 - 350 = 650
        #
        # With total_weight=0.1:
        #
        #   legacy uncapped = 65
        #   full uncapped   = 35
        #
        # These deliberately large values force both caps to activate.
        return {
            "quality_fidelity_raw_loss": scalar(1000.0),
            "full_spectrum_reconstruction_loss": scalar(200.0),
            "full_spectrum_tail_span_loss": scalar(200.0),
        }

    diffusion._quality_fidelity_losses = fixed_quality_losses


def _run_quality_budget_loss(diffusion):
    spectrum = torch.zeros(4, 1, 16)
    mask = torch.ones_like(spectrum)

    condition = torch.zeros(4, 14)
    condition[:, 0] = 1.0

    loss = diffusion.p_losses(
        spectrum,
        torch.zeros(4, dtype=torch.long),
        noise=torch.zeros_like(spectrum),
        valid_mask=mask,
        condition=condition,
    )

    return loss, diffusion.get_latest_loss_components()


def test_independent_full_spectrum_budget_disabled_keeps_legacy_cap():
    """D4.3.2.11 behavior must remain unchanged when new budget is off."""
    configuration = _quality_budget_configuration(
        independent_enabled=False,
    )

    _, diffusion = build_diffusion_model(
        configuration,
        sequence_length=16,
    )
    _install_fixed_quality_losses(diffusion)

    loss, components = _run_quality_budget_loss(diffusion)

    ddpm = components["ddpm_loss"]
    uncapped = components["quality_fidelity_uncapped_loss"]

    expected_quality = torch.minimum(
        uncapped,
        0.10 * ddpm,
    )

    torch.testing.assert_close(
        components["quality_fidelity_loss"],
        expected_quality,
    )

    # Backward-compatible branch: everything is still one quality pool.
    torch.testing.assert_close(
        components["legacy_quality_loss"],
        components["quality_fidelity_loss"],
    )
    torch.testing.assert_close(
        components["legacy_quality_uncapped_loss"],
        components["quality_fidelity_uncapped_loss"],
    )

    assert (
        float(components["independent_full_spectrum_loss"])
        == 0.0
    )
    assert (
        float(
            components[
                "independent_full_spectrum_uncapped_loss"
            ]
        )
        == 0.0
    )

    torch.testing.assert_close(
        loss.detach(),
        ddpm + components["quality_fidelity_loss"],
    )


def test_independent_full_spectrum_budget_separates_quality_caps():
    """D4.3.2.13 must apply independent legacy/full-spectrum caps."""
    configuration = _quality_budget_configuration(
        independent_enabled=True,
    )

    _, diffusion = build_diffusion_model(
        configuration,
        sequence_length=16,
    )
    _install_fixed_quality_losses(diffusion)

    loss, components = _run_quality_budget_loss(diffusion)

    ddpm = components["ddpm_loss"]

    # From the deterministic fake quality values:
    #
    # full raw:
    #   0.75 * 200 + 1.00 * 200 = 350
    #
    # legacy raw:
    #   1000 - 350 = 650
    #
    # total_weight=0.1 for both branches.
    torch.testing.assert_close(
        components["independent_full_spectrum_raw_loss"],
        ddpm.new_tensor(350.0),
    )
    torch.testing.assert_close(
        components["independent_full_spectrum_uncapped_loss"],
        ddpm.new_tensor(35.0),
    )
    torch.testing.assert_close(
        components["legacy_quality_uncapped_loss"],
        ddpm.new_tensor(65.0),
    )

    expected_legacy = torch.minimum(
        components["legacy_quality_uncapped_loss"],
        0.10 * ddpm,
    )
    expected_full = torch.minimum(
        components[
            "independent_full_spectrum_uncapped_loss"
        ],
        0.05 * ddpm,
    )

    torch.testing.assert_close(
        components["legacy_quality_loss"],
        expected_legacy,
    )
    torch.testing.assert_close(
        components["independent_full_spectrum_loss"],
        expected_full,
    )

    # Total quality must be the sum of the two independently capped
    # branches.
    torch.testing.assert_close(
        components["quality_fidelity_loss"],
        expected_legacy + expected_full,
    )

    # Diversity is disabled in this isolated accounting test.
    torch.testing.assert_close(
        components["diversity_loss"],
        ddpm.new_tensor(0.0),
    )

    # Therefore total training loss must be exactly:
    #
    # DDPM + legacy quality + independent full-spectrum quality.
    torch.testing.assert_close(
        loss.detach(),
        ddpm + expected_legacy + expected_full,
    )

    # The deliberately large raw losses should force both caps to engage.
    assert float(components["legacy_quality_scale"]) < 1.0
    assert (
        float(
            components["independent_full_spectrum_scale"]
        )
        < 1.0
    )

    # Explicitly verify the maximum contribution ratios.
    assert (
        float(components["legacy_quality_loss"])
        <= 0.10 * float(ddpm) + 1.0e-7
    )
    assert (
        float(components["independent_full_spectrum_loss"])
        <= 0.05 * float(ddpm) + 1.0e-7
    )



# ============================================================
# D4.3.2.17 integration tests
# ============================================================


def _d417_integration_configuration() -> dict:
    configuration = copy.deepcopy(
        _configuration()
    )

    quality = {
        "enabled": True,
        "total_weight": 0.10,
        "maximum_total_ratio_to_ddpm": 0.10,

        "independent_full_spectrum_budget": {
            "enabled": True,
            "total_weight": 0.10,
            "maximum_total_ratio_to_ddpm": 0.05,
        },

        # D4.3.2.16明确关闭。
        "condition_full_spectrum_variance_profile": {
            "enabled": False,
        },

        # 只开启D4.3.2.17。
        "condition_full_spectrum_group_variance": {
            "enabled": True,
            "minimum_group_size": 4,
            "maximum_timestep_fraction": 0.50,

            # 集成测试设为0，
            # 只要predicted std与reference std不完全相同，
            # 就必须产生非零约束。
            "overdispersion_deadband": 0.0,
            "underdispersion_deadband": 0.0,

            "smooth_l1_beta": 0.05,
            "overdispersion_weight": 1.0,
            "underdispersion_weight": 1.0,
            "weight": 0.35,
            "epsilon": 1.0e-6,
        },
    }

    configuration[
        "diversity_constraints"
    ][
        "quality_fidelity"
    ] = quality

    return configuration


def test_d417_group_variance_runs_through_masked_diffusion():
    """D4.3.2.17必须真正进入p_losses，而不只是独立class可运行。"""

    generator = np.random.default_rng(
        2026
    )

    # 8条training residual。
    training = generator.normal(
        0.0,
        0.05,
        size=(8, 16),
    ).astype(
        np.float32
    )

    masks = np.ones_like(
        training,
        dtype=np.float32,
    )

    conditions = np.zeros(
        (8, 14),
        dtype=np.float32,
    )
    conditions[:, 0] = 1.0

    # 故意构造training full-spectrum variance：
    # 前4条=0，后4条=10。
    #
    # 因此8条training reference std明显大于0，
    # 但当前训练batch前4条本身几乎没有这种full-spectrum变化，
    # 保证group variance约束被激活。
    training_full = np.zeros(
        (8, 16),
        dtype=np.float32,
    )
    training_full[4:, :] = 10.0

    configuration = (
        _d417_integration_configuration()
    )

    state = (
        fit_condition_aware_diversity_constraint_state(
            training_scaled_residuals=training,
            training_valid_masks=masks,
            training_condition_vectors=conditions,
            configuration=configuration[
                "diversity_constraints"
            ],
            training_full_spectra=training_full,
        )
    )

    # D4.3.2.16关闭时，
    # D4.3.2.17仍必须建立train-only reference。
    assert state[
        "full_spectrum_variance_profile_enabled"
    ]

    assert (
        state["training_counts"]
        == [8]
    )

    _, diffusion = build_diffusion_model(
        configuration,
        sequence_length=16,
    )

    keys_before = set(
        diffusion.state_dict()
    )

    diffusion.configure_diversity_constraints(
        diversity_constraint_state=state
    )

    keys_after = set(
        diffusion.state_dict()
    )

    # train-only reference buffer使用persistent=False，
    # 不改变模型checkpoint parameter keys。
    assert keys_before == keys_after

    spectrum = torch.as_tensor(
        training[:4]
    ).unsqueeze(
        1
    )

    mask = torch.ones_like(
        spectrum
    )

    condition = torch.as_tensor(
        conditions[:4]
    )

    full_target = torch.as_tensor(
        training_full[:4]
    ).unsqueeze(
        1
    )

    inverse_slope = torch.ones_like(
        spectrum
    )

    loss = diffusion.p_losses(
        spectrum,
        torch.zeros(
            4,
            dtype=torch.long,
        ),
        noise=torch.zeros_like(
            spectrum
        ),
        valid_mask=mask,
        condition=condition,
        full_spectrum_target=full_target,
        local_inverse_slope=inverse_slope,
    )

    assert torch.isfinite(
        loss
    )

    components = (
        diffusion.get_latest_loss_components()
    )

    assert (
        components[
            "full_spectrum_group_variance_active_groups"
        ].item()
        == 1.0
    )

    assert (
        components[
            "full_spectrum_group_variance_loss"
        ].item()
        > 0.0
    )

    assert torch.isfinite(
        components[
            "full_spectrum_group_variance_mean_abs_z"
        ]
    )

    assert (
        components[
            "independent_full_spectrum_raw_loss"
        ].item()
        > 0.0
    )

    # 本测试没有开启full-tail或D4.3.2.16，
    # 所以independent raw应只来自：
    #
    # 0.35 * D4.3.2.17 group variance loss
    expected_independent_raw = (
        0.35
        * components[
            "full_spectrum_group_variance_loss"
        ]
    )

    torch.testing.assert_close(
        components[
            "independent_full_spectrum_raw_loss"
        ],
        expected_independent_raw,
    )

    loss.backward()

    finite_nonzero_gradient = False

    for parameter in diffusion.parameters():
        if parameter.grad is None:
            continue

        assert torch.isfinite(
            parameter.grad
        ).all()

        if torch.any(
            parameter.grad != 0
        ):
            finite_nonzero_gradient = True

    assert finite_nonzero_gradient


def test_d417_group_variance_is_counted_in_independent_budget():
    """确定性检查D4.3.2.17是否进入独立full-spectrum预算。"""

    configuration = (
        _quality_budget_configuration(
            independent_enabled=True,
        )
    )

    quality_configuration = (
        configuration[
            "diversity_constraints"
        ][
            "quality_fidelity"
        ]
    )

    quality_configuration[
        "condition_full_spectrum_variance_profile"
    ] = {
        "enabled": False,
    }

    quality_configuration[
        "condition_full_spectrum_group_variance"
    ] = {
        "enabled": True,
        "minimum_group_size": 4,
        "maximum_timestep_fraction": 0.50,
        "overdispersion_deadband": 0.25,
        "underdispersion_deadband": 0.20,
        "smooth_l1_beta": 0.05,
        "overdispersion_weight": 1.0,
        "underdispersion_weight": 1.0,
        "weight": 0.35,
        "epsilon": 1.0e-6,
    }

    _, diffusion = build_diffusion_model(
        configuration,
        sequence_length=16,
    )

    def fixed_quality_losses(**kwargs):
        prediction = kwargs[
            "prediction"
        ]

        scalar = prediction.new_tensor

        # full-tail:
        #   0.75*200 + 1.00*200 = 350
        #
        # D4.3.2.17:
        #   0.35*200 = 70
        #
        # independent raw = 420
        #
        # 保持legacy raw仍为650：
        # total raw = 650 + 420 = 1070
        return {
            "quality_fidelity_raw_loss": scalar(
                1070.0
            ),
            "full_spectrum_reconstruction_loss": scalar(
                200.0
            ),
            "full_spectrum_tail_span_loss": scalar(
                200.0
            ),
            "full_spectrum_group_variance_loss": scalar(
                200.0
            ),
        }

    diffusion._quality_fidelity_losses = (
        fixed_quality_losses
    )

    loss, components = (
        _run_quality_budget_loss(
            diffusion
        )
    )

    ddpm = components[
        "ddpm_loss"
    ]

    torch.testing.assert_close(
        components[
            "independent_full_spectrum_raw_loss"
        ],
        ddpm.new_tensor(
            420.0
        ),
    )

    # independent total_weight = 0.1
    torch.testing.assert_close(
        components[
            "independent_full_spectrum_uncapped_loss"
        ],
        ddpm.new_tensor(
            42.0
        ),
    )

    # legacy raw应保持：
    #
    # 1070 - 420 = 650
    torch.testing.assert_close(
        components[
            "legacy_quality_uncapped_loss"
        ],
        ddpm.new_tensor(
            65.0
        ),
    )

    assert torch.isfinite(
        loss
    )
