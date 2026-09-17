import numpy as np
import torch
from torch import nn

from src.model_builder import (
    build_diffusion_model,
)
from src.prior_residual import (
    PriorResidualTransformer,
)
from src.spectrum_generator import (
    _protect_pairwise_diversity_after_calibration,
    apply_condition_intensity_envelope_guard,
    apply_condition_mean_fidelity_calibration,
    apply_condition_negative_floor_guard,
    apply_condition_oracle_tail_calibration,
    apply_condition_pca_spread_calibration,
    resolve_condition_tail_calibration_configuration,
    generate_spectra,
)
from src.masked_diffusion import MaskedGaussianDiffusion1D
from src.spectrum_length_adapter import (
    SpectrumLengthAdapter,
)
from src.sers_sampling_calibrator import (
    SersSamplingCalibrator,
)


class FixedSampleDiffusion(nn.Module):
    """
    测试专用扩散模型。

    不执行随机采样，而是按照顺序返回预先设置好的
    缩放残差，用于准确验证D2生成恢复流程。
    """

    def __init__(
        self,
        generated_samples: np.ndarray,
    ) -> None:
        super().__init__()

        samples = torch.as_tensor(
            generated_samples,
            dtype=torch.float32,
        )

        if samples.ndim != 2:
            raise ValueError(
                "generated_samples必须是二维数组[N,L]。"
            )

        self.register_buffer(
            "generated_samples",
            samples[
                :,
                None,
                :,
            ],
        )

        self.sample_offset = 0

    def sample(
        self,
        batch_size: int,
    ) -> torch.Tensor:
        """返回指定数量的固定生成结果。"""

        batch_size = int(
            batch_size
        )

        start = (
            self.sample_offset
        )

        end = (
            start
            + batch_size
        )

        if (
            end
            > self.generated_samples.shape[0]
        ):
            raise RuntimeError(
                "测试扩散模型中的固定样本数量不足。"
            )

        batch = (
            self.generated_samples[
                start:end
            ]
        )

        self.sample_offset = (
            end
        )

        return batch


def test_negative_floor_guard_repairs_extreme_valley_and_keeps_normal_noise():
    training = np.asarray(
        [
            [-18.0, 98.0, 202.0, 101.0, -12.0],
            [-15.0, 102.0, 210.0, 99.0, -14.0],
            [-20.0, 95.0, 198.0, 104.0, -10.0],
            [-16.0, 100.0, 205.0, 97.0, -13.0],
        ],
        dtype=np.float32,
    )
    generated = np.repeat(
        np.asarray([[-18.0, 99.0, 204.0, 100.0, -11.0]], dtype=np.float32),
        40,
        axis=0,
    )
    generated[:, 0] += np.linspace(-1.0, 1.0, generated.shape[0])
    generated[7, 2] = -211.0

    corrected, diagnostics = apply_condition_negative_floor_guard(
        generated,
        training,
        configuration={
            "training_lower_quantile": 0.05,
            "training_iqr_margin": 0.5,
            "training_extrema_margin_iqr_fraction": 0.10,
            "minimum_extrema_margin_intensity": 2.0,
            "generated_lower_quantile": 0.025,
            "minimum_generated_count": 20,
            "activation_maximum_intensity": -18.0,
            "minimum_deficit_intensity": 5.0,
            "softness_iqr_fraction": 0.05,
            "minimum_softness_intensity": 0.25,
            "maximum_softness_intensity": 2.0,
            "enforce_training_global_minimum": True,
            "maximum_modified_fraction": 0.02,
        },
    )

    np.testing.assert_allclose(corrected[:, 0], generated[:, 0])
    assert corrected[7, 2] >= float(np.min(training))
    assert corrected[7, 2] <= 0.0
    assert diagnostics["modified_spectrum_count"] == 1
    assert diagnostics["modified_point_count"] == 1
    assert diagnostics["minimum_before"] == -211.0
    assert diagnostics["minimum_after"] >= float(np.min(training))


def test_intensity_envelope_repairs_both_deep_valley_and_excess_peak():
    training = np.asarray(
        [
            [-18.0, 98.0, 202.0, 101.0, -12.0],
            [-15.0, 102.0, 210.0, 99.0, -14.0],
            [-20.0, 95.0, 198.0, 104.0, -10.0],
            [-16.0, 100.0, 205.0, 97.0, -13.0],
        ],
        dtype=np.float32,
    )
    generated = np.repeat(training[[0]], 40, axis=0)
    generated[3, 2] = -211.0
    generated[9, 2] = 900.0
    corrected, diagnostics = apply_condition_intensity_envelope_guard(
        generated,
        training,
        configuration={
            "training_lower_quantile": 0.05,
            "training_upper_quantile": 0.95,
            "training_iqr_margin": 0.5,
            "training_extrema_margin_iqr_fraction": 0.10,
            "minimum_extrema_margin_intensity": 2.0,
            "generated_lower_quantile": 0.025,
            "generated_upper_quantile": 0.975,
            "minimum_generated_count": 20,
            "activation_maximum_intensity": -18.0,
            "minimum_deficit_intensity": 5.0,
            "minimum_excess_intensity": 5.0,
            "softness_iqr_fraction": 0.05,
            "minimum_softness_intensity": 0.25,
            "maximum_softness_intensity": 2.0,
            "enforce_training_global_extrema": False,
            "maximum_modified_fraction": 0.05,
        },
    )
    assert corrected[3, 2] > -50.0
    assert corrected[9, 2] < 300.0
    np.testing.assert_allclose(corrected[0], generated[0])
    assert diagnostics["lower_modified_point_count"] == 1
    assert diagnostics["upper_modified_point_count"] == 1


def test_generated_outliers_cannot_loosen_training_owned_envelope():
    training = np.repeat(
        np.asarray([[-18.0, 100.0, 200.0]], dtype=np.float32),
        12,
        axis=0,
    )
    generated = np.repeat(training[[0]], 40, axis=0)
    generated[:4, 2] = -200.0
    corrected, diagnostics = apply_condition_intensity_envelope_guard(
        generated,
        training,
        configuration={
            "training_lower_quantile": 0.05,
            "training_upper_quantile": 0.95,
            "training_iqr_margin": 0.5,
            "training_extrema_margin_iqr_fraction": 0.10,
            "minimum_extrema_margin_intensity": 2.0,
            "generated_lower_quantile": 0.025,
            "generated_upper_quantile": 0.975,
            "minimum_generated_count": 20,
            "activation_maximum_intensity": -18.0,
            "minimum_deficit_intensity": 5.0,
            "minimum_excess_intensity": 5.0,
            "softness_iqr_fraction": 0.05,
            "minimum_softness_intensity": 0.25,
            "maximum_softness_intensity": 2.0,
            "enforce_training_global_extrema": False,
            "maximum_modified_fraction": 0.05,
        },
    )
    assert np.all(corrected[:4, 2] > -30.0)
    assert diagnostics["lower_modified_point_count"] == 4
    assert diagnostics["generated_batch_is_rank_gate_only"] is True


def test_condition_negative_background_controls_coherent_valley_without_global_floor():
    random_generator = np.random.default_rng(2041)
    training_shallow = random_generator.uniform(-20.0, -8.0, size=(12, 100))
    training_deep = random_generator.uniform(-105.0, -75.0, size=(12, 100))
    generated_shallow = np.repeat(training_shallow[[0]], 20, axis=0)
    generated_deep = np.repeat(training_deep[[0]], 20, axis=0)
    generated_shallow[2, 40:51] = -72.0
    generated_shallow[3, 20] = -18.0
    generated_deep[2, 40:51] = -95.0
    configuration = {
        "training_lower_quantile": 0.05,
        "training_upper_quantile": 0.95,
        "training_iqr_margin": 0.5,
        "training_extrema_margin_iqr_fraction": 0.1,
        "minimum_extrema_margin_intensity": 2.0,
        "generated_lower_quantile": 0.025,
        "generated_upper_quantile": 0.975,
        "minimum_generated_count": 20,
        "activation_maximum_intensity": -18.0,
        "minimum_deficit_intensity": 5.0,
        "minimum_excess_intensity": 1.0e9,
        "softness_iqr_fraction": 0.05,
        "minimum_softness_intensity": 0.25,
        "maximum_softness_intensity": 2.0,
        "enforce_training_global_extrema": False,
        "maximum_modified_fraction": 0.20,
        "condition_negative_background_quantile": 0.01,
        "condition_negative_background_iqr_margin": 0.5,
        "minimum_condition_negative_count": 64,
        "local_negative_support_quantile": 0.1,
        "minimum_negative_valley_contiguous_points": 3,
        "severe_negative_depth_iqr_multiplier": 3.0,
    }

    corrected_shallow, shallow_diagnostics = (
        apply_condition_intensity_envelope_guard(
            generated_shallow,
            training_shallow,
            configuration=configuration,
        )
    )
    corrected_deep, deep_diagnostics = apply_condition_intensity_envelope_guard(
        generated_deep,
        training_deep,
        configuration=configuration,
    )

    assert np.min(corrected_shallow[2, 40:51]) > -30.0
    assert corrected_shallow[3, 20] == generated_shallow[3, 20]
    np.testing.assert_allclose(corrected_deep, generated_deep)
    assert shallow_diagnostics["condition_negative_background_floor"] > -30.0
    assert deep_diagnostics["condition_negative_background_floor"] < -75.0


def test_full_spectrum_tail_loss_penalizes_reconstructed_qq_overrun():
    diffusion = object.__new__(MaskedGaussianDiffusion1D)
    nn.Module.__init__(diffusion)
    diffusion.num_timesteps = 200
    full_target = torch.tensor([-1.0, -0.5, 0.5, 1.0]).reshape(4, 1, 1)
    full_target = full_target.expand(-1, -1, 64).clone()
    target_residual = torch.zeros_like(full_target)
    prediction_residual = full_target.clone().requires_grad_(True)
    mask = torch.ones_like(full_target)
    condition = torch.tensor([[1.0, 0.0]]).repeat(4, 1)
    configuration = {
        "piecewise_probabilities": [0.01, 0.10, 0.50, 0.90, 0.99],
        "segment_weights": [1.5, 0.5, 0.5, 1.5],
        "minimum_segment_span_ratio": 0.90,
        "maximum_segment_span_ratio": 1.10,
        "maximum_timestep_fraction": 0.50,
        "minimum_group_size": 4,
        "pointwise_scale_floor_quantile": 10.0,
        "reconstruction_smooth_l1_beta": 0.01,
        "span_smooth_l1_beta": 0.02,
        "short_axis_maximum_points": 1500,
        "short_axis_weight_multiplier": 1.4,
        "epsilon": 1.0e-6,
    }
    reconstruction, span, active = diffusion._full_spectrum_tail_losses(
        prediction=prediction_residual,
        target=target_residual,
        full_spectrum_target=full_target,
        local_inverse_slope=torch.ones_like(full_target),
        valid_mask=mask,
        condition=condition,
        timesteps=torch.full((4,), 20, dtype=torch.long),
        configuration=configuration,
    )
    loss = reconstruction + span
    assert reconstruction.item() > 0.0
    assert span.item() > 0.0
    assert active.item() == 1.0
    loss.backward()
    assert prediction_residual.grad is not None
    assert torch.isfinite(prediction_residual.grad).all()


def test_mean_fidelity_calibration_reduces_bias_without_changing_pairwise_mse():
    random_generator = np.random.default_rng(2026)
    training = random_generator.normal(10.0, 2.0, size=(12, 80))
    generated = random_generator.normal(16.0, 2.0, size=(40, 80))
    pairwise_before = np.mean(np.square(generated[0] - generated[1]))
    corrected, diagnostics = apply_condition_mean_fidelity_calibration(
        generated,
        training,
        configuration={
            "standard_error_multiplier": 1.96,
            "correction_strength": 0.35,
            "maximum_correction_iqr_fraction": 0.25,
            "minimum_maximum_correction_intensity": 1.0,
            "maximum_modified_raman_fraction": 1.0,
        },
    )
    pairwise_after = np.mean(np.square(corrected[0] - corrected[1]))
    assert (
        diagnostics["mean_bias_rmse_after"]
        < diagnostics["mean_bias_rmse_before"]
    )
    np.testing.assert_allclose(pairwise_after, pairwise_before, rtol=1.0e-6)


def test_pca_spread_calibration_restores_supported_variation_and_keeps_mean():
    random_generator = np.random.default_rng(2026)
    axis = np.linspace(-1.0, 1.0, 120)
    first_mode = np.sin(np.pi * axis)
    second_mode = np.exp(-0.5 * np.square(axis / 0.18))
    train_scores = random_generator.normal(size=(12, 2))
    training = (
        20.0
        + 5.0 * train_scores[:, [0]] * first_mode[None, :]
        + 8.0 * train_scores[:, [1]] * second_mode[None, :]
    )
    generated_scores = random_generator.normal(scale=0.35, size=(80, 2))
    generated = (
        22.0
        + 5.0 * generated_scores[:, [0]] * first_mode[None, :]
        + 8.0 * generated_scores[:, [1]] * second_mode[None, :]
    )
    mean_before = np.mean(generated, axis=0)
    corrected, diagnostics = apply_condition_pca_spread_calibration(
        generated,
        training,
        configuration={
            "maximum_components": 6,
            "explained_variance_ratio": 0.95,
            "minimum_component_variance_fraction": 0.01,
            "minimum_component_std_ratio": 0.85,
            "maximum_scale_factor": 1.35,
            "maximum_correction_range_fraction": 0.25,
            "minimum_generated_count": 20,
        },
    )
    np.testing.assert_allclose(
        np.mean(corrected, axis=0), mean_before, rtol=0.0, atol=2.0e-5
    )
    assert diagnostics["expanded_component_count"] >= 1
    assert (
        diagnostics["component_std_ratio_median_after"]
        > diagnostics["component_std_ratio_median_before"]
    )
    assert (
        diagnostics["mean_pairwise_mse_ratio_after"]
        > diagnostics["mean_pairwise_mse_ratio_before"]
    )


def test_pca_spread_calibration_does_not_force_global_floor_without_deficient_pc():
    """D4.3.2.12: overall diversity floor must not expand adequate PCs."""
    random_generator = np.random.default_rng(2029)
    axis = np.linspace(-1.0, 1.0, 160)
    modes = np.stack(
        [
            np.sin(np.pi * axis),
            np.exp(-0.5 * np.square(axis / 0.20)),
            np.cos(2.0 * np.pi * axis),
        ],
        axis=0,
    )
    training_scores = random_generator.normal(size=(12, 3))
    generated_scores = random_generator.normal(scale=0.70, size=(80, 3))
    training = 10.0 + training_scores @ modes
    generated = 14.0 + generated_scores @ modes
    mean_before = np.mean(generated, axis=0)

    corrected, diagnostics = apply_condition_pca_spread_calibration(
        generated,
        training,
        configuration={
            "maximum_components": 6,
            "explained_variance_ratio": 0.95,
            "minimum_component_variance_fraction": 0.01,
            # All PCs are above this floor in this deterministic example.
            # D4.3.2.12 therefore must not use a global pairwise target
            # to expand otherwise adequate train-supported PCs.
            "minimum_component_std_ratio": 0.50,
            "maximum_scale_factor": 1.45,
            "minimum_pairwise_mse_ratio": 0.90,
            "maximum_global_scale_factor": 1.60,
            "maximum_component_std_ratio_after_global": 1.0,
            "maximum_correction_range_fraction": 0.25,
            "minimum_generated_count": 20,
        },
    )

    np.testing.assert_allclose(
        np.mean(corrected, axis=0),
        mean_before,
        rtol=0.0,
        atol=2.0e-5,
    )
    np.testing.assert_allclose(
        corrected,
        generated,
        rtol=0.0,
        atol=2.0e-5,
    )

    assert diagnostics["expanded_component_count"] == 0
    assert diagnostics["global_pairwise_spread_activated"] is False
    assert diagnostics["global_pairwise_scale_factor"] == 1.0
    assert diagnostics["effective_maximum_global_scale_factor"] == 1.0
    assert diagnostics["global_pairwise_target_met"] is False

    assert (
        diagnostics["mean_pairwise_mse_ratio_after"]
        == diagnostics["mean_pairwise_mse_ratio_before"]
    )
    assert diagnostics["pairwise_mse_statistic"].startswith("median")


def test_pca_spread_calibration_global_stage_uses_only_deficient_pcs():
    """D4.3.2.12: global recovery may continue only deficient PCA directions."""
    random_generator = np.random.default_rng(2029)
    axis = np.linspace(-1.0, 1.0, 160)
    modes = np.stack(
        [
            np.sin(np.pi * axis),
            np.exp(-0.5 * np.square(axis / 0.20)),
            np.cos(2.0 * np.pi * axis),
        ],
        axis=0,
    )
    training_scores = random_generator.normal(size=(12, 3))
    generated_scores = random_generator.normal(scale=0.70, size=(80, 3))
    training = 10.0 + training_scores @ modes
    generated = 14.0 + generated_scores @ modes
    mean_before = np.mean(generated, axis=0)

    corrected, diagnostics = apply_condition_pca_spread_calibration(
        generated,
        training,
        configuration={
            "maximum_components": 6,
            "explained_variance_ratio": 0.95,
            "minimum_component_variance_fraction": 0.01,
            # The first PCA component is deficient for this deterministic
            # example, while the remaining components already have enough
            # spread.
            "minimum_component_std_ratio": 0.88,
            "maximum_scale_factor": 1.45,
            "minimum_pairwise_mse_ratio": 0.90,
            "maximum_global_scale_factor": 1.60,
            "maximum_component_std_ratio_after_global": 1.0,
            "maximum_correction_range_fraction": 0.25,
            "minimum_generated_count": 20,
        },
    )

    np.testing.assert_allclose(
        np.mean(corrected, axis=0),
        mean_before,
        rtol=0.0,
        atol=2.0e-5,
    )

    before_ratios = np.asarray(
        diagnostics["component_std_ratio_before"],
        dtype=np.float64,
    )
    after_ratios = np.asarray(
        diagnostics["component_std_ratio_after"],
        dtype=np.float64,
    )
    first_stage_scales = np.asarray(
        diagnostics["component_scale_factors"],
        dtype=np.float64,
    )

    deficient = first_stage_scales > 1.0 + 1.0e-12
    nondeficient = ~deficient

    assert diagnostics["expanded_component_count"] >= 1
    assert np.any(deficient)
    assert diagnostics["global_pairwise_spread_activated"] is True
    assert diagnostics["global_pairwise_scale_factor"] > 1.0

    # Deficient PCs must gain spread.
    assert np.all(after_ratios[deficient] > before_ratios[deficient])

    # The D4.3.2.12 global stage must not expand PCs that were already
    # adequate during the first-stage component check.
    np.testing.assert_allclose(
        after_ratios[nondeficient],
        before_ratios[nondeficient],
        rtol=0.0,
        atol=1.0e-8,
    )

    # No globally expanded deficient PC may exceed the training spread.
    assert np.all(after_ratios[deficient] <= 1.0 + 1.0e-8)

    assert (
        diagnostics["mean_pairwise_mse_ratio_after"]
        > diagnostics["mean_pairwise_mse_ratio_before"]
    )
    assert diagnostics["pairwise_mse_statistic"].startswith("median")


def test_tail_calibration_profile_separates_axis_and_analyte_count():
    configuration = {
        "correction_strength": 0.60,
        "maximum_absolute_correction_z": 0.40,
        "axis_profiles": {
            "1401": {
                "correction_strength_multiplier": 1.25,
                "maximum_absolute_correction_z_multiplier": 1.50,
            },
            "1901": {"correction_strength_multiplier": 0.75},
        },
        "analyte_count_profiles": {
            "2": {"correction_strength_multiplier": 1.15},
            "3": {"correction_strength_multiplier": 0.90},
        },
    }
    two_1401, diagnostic_1401 = (
        resolve_condition_tail_calibration_configuration(
            configuration,
            point_count=1401,
            condition_name="DEL-M_TEB-H_water",
        )
    )
    three_1901, diagnostic_1901 = (
        resolve_condition_tail_calibration_configuration(
            configuration,
            point_count=1901,
            condition_name="DEL-S_TEB-M_CHL-H_soil",
        )
    )
    assert abs(two_1401["correction_strength"] - 0.8625) < 1.0e-12
    assert abs(two_1401["maximum_absolute_correction_z"] - 0.60) < 1.0e-12
    assert abs(three_1901["correction_strength"] - 0.405) < 1.0e-12
    assert diagnostic_1401["analyte_count"] == 2
    assert diagnostic_1901["selected_profiles"] == "axis:1901,analytes:3"


def test_oracle_tail_calibration_keeps_middle_and_shrinks_only_excess_tails():
    random_generator = np.random.default_rng(2026)
    training = random_generator.normal(0.0, 1.0, size=(12, 160))
    generated = random_generator.normal(0.0, 2.2, size=(200, 160))
    center = np.mean(training, axis=0)
    scale = np.std(training, axis=0, ddof=1)
    scale = np.maximum(scale, np.percentile(scale[scale > 1.0e-12], 10.0))
    generated_z = (generated - center[None, :]) / scale[None, :]
    lower_anchor, upper_anchor = np.quantile(generated_z, [0.05, 0.95])
    middle = np.logical_and(
        generated_z >= lower_anchor, generated_z <= upper_anchor
    )

    corrected, diagnostics = apply_condition_oracle_tail_calibration(
        generated,
        training,
        configuration={
            "lower_probability": 0.01,
            "upper_probability": 0.99,
            "lower_anchor_probability": 0.05,
            "upper_anchor_probability": 0.95,
            "oracle_reference_count": 4,
            "oracle_bootstrap_repeats": 64,
            "oracle_confidence": 0.975,
            "minimum_allowed_tail_ratio": 1.15,
            "minimum_trigger_tail_ratio": 1.50,
            "minimum_compression_factor": 0.25,
            "maximum_modified_fraction": 0.10,
            "random_seed": 2026,
        },
    )
    np.testing.assert_allclose(corrected[middle], generated[middle], atol=1.0e-6)
    assert diagnostics["lower_compression_factor"] < 1.0
    assert diagnostics["upper_compression_factor"] < 1.0
    assert diagnostics["lower_tail_calibration_activated"] is True
    assert diagnostics["upper_tail_calibration_activated"] is True
    assert (
        diagnostics["lower_tail_ratio_after"]
        < diagnostics["lower_tail_ratio_before"]
    )
    assert (
        diagnostics["upper_tail_ratio_after"]
        < diagnostics["upper_tail_ratio_before"]
    )


def test_oracle_tail_calibration_leaves_nonsevere_tail_unchanged():
    random_generator = np.random.default_rng(2027)
    training = random_generator.normal(0.0, 1.0, size=(12, 160))
    generated = random_generator.normal(0.0, 1.10, size=(200, 160))
    corrected, diagnostics = apply_condition_oracle_tail_calibration(
        generated,
        training,
        configuration={
            "lower_probability": 0.01,
            "upper_probability": 0.99,
            "lower_anchor_probability": 0.05,
            "upper_anchor_probability": 0.95,
            "oracle_reference_count": 4,
            "oracle_bootstrap_repeats": 64,
            "oracle_confidence": 0.975,
            "minimum_allowed_tail_ratio": 1.15,
            "minimum_trigger_tail_ratio": 1.50,
            "minimum_compression_factor": 0.25,
            "maximum_modified_fraction": 0.10,
            "random_seed": 2026,
        },
    )
    np.testing.assert_allclose(corrected, generated, atol=1.0e-6)
    assert diagnostics["lower_tail_calibration_activated"] is False
    assert diagnostics["upper_tail_calibration_activated"] is False


def test_oracle_piecewise_quantile_calibration_reduces_s_shaped_qq_error():
    random_generator = np.random.default_rng(2031)
    training = random_generator.normal(0.0, 1.0, size=(12, 240))
    center = np.mean(training, axis=0)
    scale = np.std(training, axis=0, ddof=1)
    scale = np.maximum(scale, np.percentile(scale[scale > 1.0e-12], 10.0))
    latent = random_generator.normal(0.0, 1.0, size=(200, 240))
    magnitude = np.abs(latent)
    distorted = np.sign(latent) * np.where(
        magnitude <= 1.0,
        1.30 * magnitude,
        1.30 + 0.55 * (magnitude - 1.0),
    )
    generated = center[None, :] + scale[None, :] * distorted

    corrected, diagnostics = apply_condition_oracle_tail_calibration(
        generated,
        training,
        configuration={
            "strategy": "oracle_piecewise_quantile",
            "piecewise_probabilities": [0.01, 0.10, 0.50, 0.90, 0.99],
            "oracle_reference_count": 4,
            "oracle_bootstrap_repeats": 64,
            "oracle_confidence": 0.95,
            "minimum_segment_span_ratio": 0.90,
            "maximum_segment_span_ratio": 1.10,
            "correction_strength": 0.80,
            "maximum_absolute_correction_z": 0.60,
            "maximum_modified_fraction": 1.0,
            "random_seed": 2031,
        },
    )

    before = np.asarray(diagnostics["segment_span_ratio_before"])
    after = np.asarray(diagnostics["segment_span_ratio_after"])
    assert diagnostics["active_segment_count"] >= 1
    assert np.mean(np.abs(after - 1.0)) < np.mean(np.abs(before - 1.0))
    assert diagnostics["maximum_absolute_correction_z_used"] <= 0.60 + 1.0e-12
    assert np.isfinite(corrected).all()
    generated_z = (generated - center[None, :]) / scale[None, :]
    corrected_z = (corrected - center[None, :]) / scale[None, :]
    assert abs(np.median(corrected_z) - np.median(generated_z)) < 1.0e-4


def test_postcalibration_diversity_guard_limits_pairwise_mse_collapse():
    random_generator = np.random.default_rng(2032)
    training = random_generator.normal(0.0, 1.0, size=(12, 160))
    original = random_generator.normal(0.0, 1.0, size=(20, 160))
    collapsed = np.mean(original, axis=0, keepdims=True) + 0.20 * (
        original - np.mean(original, axis=0, keepdims=True)
    )

    protected, diagnostics = _protect_pairwise_diversity_after_calibration(
        original,
        collapsed,
        training,
        minimum_ratio=0.82,
        search_iterations=24,
    )

    assert diagnostics["diversity_guard_activated"] is True
    assert 0.0 < diagnostics["calibration_blend_factor"] < 1.0
    assert (
        diagnostics["pairwise_mse_ratio_after"]
        >= diagnostics["pairwise_mse_ratio_required"] - 1.0e-6
    )
    assert (
        diagnostics["pairwise_mse_ratio_after"]
        > diagnostics["pairwise_mse_ratio_unguarded"]
    )
    assert np.isfinite(protected).all()


def test_quality_fidelity_losses_detect_shape_mean_and_envelope_errors():
    diffusion = object.__new__(MaskedGaussianDiffusion1D)
    nn.Module.__init__(diffusion)
    diffusion.quality_fidelity_configuration = {
        "enabled": True,
        "first_derivative": {
            "enabled": True,
            "smooth_l1_beta": 0.05,
            "weight": 0.5,
        },
        "multiscale_shape": {
            "enabled": True,
            "kernel_sizes": [3, 5],
            "smooth_l1_beta": 0.05,
            "weight": 0.5,
        },
        "condition_mean": {
            "enabled": True,
            "smooth_l1_beta": 0.05,
            "minimum_group_size": 4,
            "weight": 0.5,
        },
        "condition_pointwise_envelope": {
            "enabled": True,
            "lower_quantile": 0.05,
            "upper_quantile": 0.95,
            "iqr_margin": 0.5,
            "smooth_l1_beta": 0.05,
            "minimum_group_size": 4,
            "weight": 1.0,
        },
    }
    target = torch.zeros((8, 1, 12), dtype=torch.float32)
    prediction = target.clone()
    prediction[:4, :, 5] = 2.0
    prediction[4:, :, 7] = -2.0
    mask = torch.ones_like(target)
    condition = torch.vstack(
        [
            torch.tensor([[1.0, 0.0]]).repeat(4, 1),
            torch.tensor([[0.0, 1.0]]).repeat(4, 1),
        ]
    )
    losses = diffusion._quality_fidelity_losses(
        prediction=prediction,
        target=target,
        valid_mask=mask,
        condition=condition,
    )
    assert losses["first_derivative_fidelity_loss"].item() > 0.0
    assert losses["multiscale_shape_loss"].item() > 0.0
    assert losses["condition_mean_fidelity_loss"].item() > 0.0
    assert losses["condition_envelope_loss"].item() > 0.0
    assert losses["quality_fidelity_active_groups"].item() == 2.0


def test_condition_multiscale_variance_floor_detects_collapsed_group():
    diffusion = object.__new__(MaskedGaussianDiffusion1D)
    nn.Module.__init__(diffusion)
    diffusion.num_timesteps = 200
    diffusion.quality_fidelity_configuration = {
        "enabled": True,
        "condition_multiscale_variance_floor": {
            "enabled": True,
            "kernel_sizes": [3, 5],
            "minimum_group_size": 4,
            "minimum_std_ratio": 0.85,
            "active_std_quantile": 20.0,
            "maximum_timestep_fraction": 0.50,
            "smooth_l1_beta": 0.05,
            "weight": 1.0,
        },
    }
    target = torch.zeros((4, 1, 21), dtype=torch.float32)
    target[0, 0, 8:13] = 1.0
    target[1, 0, 9:14] = 1.3
    target[2, 0, 7:12] = 0.7
    target[3, 0, 10:15] = 1.1
    prediction = target.mean(dim=0, keepdim=True).repeat(4, 1, 1)
    losses = diffusion._quality_fidelity_losses(
        prediction=prediction,
        target=target,
        valid_mask=torch.ones_like(target),
        condition=torch.tensor([[1.0, 0.0]]).repeat(4, 1),
        timesteps=torch.tensor([20, 20, 20, 20]),
    )
    assert losses["condition_multiscale_variance_floor_loss"].item() > 0.0
    assert losses["quality_fidelity_raw_loss"].item() > 0.0


def test_condition_derivative_variance_floor_detects_frozen_peak_positions():
    diffusion = object.__new__(MaskedGaussianDiffusion1D)
    nn.Module.__init__(diffusion)
    diffusion.num_timesteps = 200
    diffusion.quality_fidelity_configuration = {
        "enabled": True,
        "condition_derivative_variance_floor": {
            "enabled": True,
            "smoothing_kernel_sizes": [3, 5],
            "minimum_group_size": 4,
            "minimum_std_ratio": 0.90,
            "active_std_quantile": 30.0,
            "maximum_timestep_fraction": 0.50,
            "smooth_l1_beta": 0.05,
            "weight": 1.0,
        },
    }
    target = torch.zeros((4, 1, 31), dtype=torch.float32)
    target[0, 0, 10:15] = 1.0
    target[1, 0, 11:16] = 1.0
    target[2, 0, 12:17] = 1.0
    target[3, 0, 13:18] = 1.0
    prediction = target.mean(dim=0, keepdim=True).repeat(4, 1, 1)
    losses = diffusion._quality_fidelity_losses(
        prediction=prediction,
        target=target,
        valid_mask=torch.ones_like(target),
        condition=torch.tensor([[1.0, 0.0]]).repeat(4, 1),
        timesteps=torch.tensor([20, 20, 20, 20]),
    )
    assert losses["condition_derivative_variance_floor_loss"].item() > 0.0
    assert losses["quality_fidelity_raw_loss"].item() > 0.0


def test_high_noise_stratified_sampling_preserves_full_range_and_tail_coverage():
    diffusion = object.__new__(MaskedGaussianDiffusion1D)
    nn.Module.__init__(diffusion)
    diffusion.num_timesteps = 200
    diffusion.quality_fidelity_enabled = True
    diffusion.quality_fidelity_configuration = {
        "high_noise_sampling": {
            "enabled": True,
            "sampling_probability": 0.25,
            "minimum_timestep_fraction": 0.75,
        }
    }
    torch.manual_seed(2026)
    timesteps = diffusion._sample_training_timesteps(
        20_000,
        torch.device("cpu"),
    )
    high_fraction = float((timesteps >= 149).to(torch.float32).mean())
    assert int(timesteps.min()) < 20
    assert int(timesteps.max()) >= 195
    assert 0.40 < high_fraction < 0.48


def test_spectrum_generation_shape():
    """验证普通DDPM能够生成正确形状的张量。"""

    model_configuration = {
        "channels": 1,
        "base_dimension": 8,
        "dimension_multipliers": [
            1,
            2,
        ],
        "self_condition": False,
        "dropout": 0.0,
        "diffusion_timesteps": 10,
        "sampling_timesteps": 5,
        "objective": "pred_noise",
        "beta_schedule": "cosine",
        "ddim_sampling_eta": 0.0,
        "auto_normalize": True,
    }

    (
        _,
        diffusion,
    ) = build_diffusion_model(
        model_configuration=(
            model_configuration
        ),
        sequence_length=16,
    )

    with torch.inference_mode():
        generated = diffusion.sample(
            batch_size=2
        )

    assert (
        generated.shape
        == (
            2,
            1,
            16,
        )
    )


def test_d2_generation_restores_prior_residual():
    """
    验证D2生成端能够：

    1. 删除U-Net末尾补齐点；
    2. 恢复checkpoint中的先验残差变换器；
    3. 取消残差缩放；
    4. 加回训练集逐点中位数先验；
    5. 插值恢复到目标拉曼位移轴。
    """

    model_axis = np.asarray(
        [
            100.0,
            200.0,
            300.0,
            400.0,
            500.0,
        ],
        dtype=np.float64,
    )

    training_spectra = np.asarray(
        [
            [
                -0.8,
                -0.4,
                0.0,
                0.4,
                0.8,
            ],
            [
                -0.6,
                -0.2,
                0.2,
                0.6,
                0.6,
            ],
            [
                -0.4,
                0.0,
                0.4,
                0.8,
                0.4,
            ],
        ],
        dtype=np.float32,
    )

    expected_model_axis_spectra = np.asarray(
        [
            [
                -0.55,
                -0.10,
                0.25,
                0.70,
                0.50,
            ],
            [
                -0.65,
                -0.30,
                0.10,
                0.50,
                0.70,
            ],
        ],
        dtype=np.float32,
    )

    fitted_transformer = (
        PriorResidualTransformer(
            target_abs_max=0.8,
        )
        .fit(
            training_spectra
        )
    )

    prior_residual_state = (
        fitted_transformer.state_dict()
    )

    restored_transformer = (
        PriorResidualTransformer
        .from_state_dict(
            prior_residual_state
        )
    )

    scaled_residuals = (
        restored_transformer.transform(
            expected_model_axis_spectra
        )
    )

    length_adapter = (
        SpectrumLengthAdapter.create(
            dimension_multipliers=[
                1,
                2,
                4,
            ],
            raman_shifts=[
                model_axis
            ],
            model_length="auto",
            padding_mode=(
                "right_zero_padding"
            ),
            padding_value=0.0,
            raman_range_tolerance=1.0,
        )
    )

    assert (
        length_adapter.original_length
        == 5
    )

    assert (
        length_adapter.padded_length
        == 8
    )

    assert (
        length_adapter.padding_size
        == 3
    )

    padded_scaled_residuals = (
        length_adapter.adapt(
            scaled_residuals
        )
    )

    assert (
        padded_scaled_residuals.shape
        == (
            2,
            8,
        )
    )

    fixed_diffusion = (
        FixedSampleDiffusion(
            padded_scaled_residuals
        )
    )

    output_axis = np.asarray(
        [
            100.0,
            300.0,
            500.0,
        ],
        dtype=np.float64,
    )

    generated = generate_spectra(
        diffusion=(
            fixed_diffusion
        ),
        number_of_spectra=2,
        generation_batch_size=1,
        device=torch.device(
            "cpu"
        ),
        length_adapter=(
            length_adapter
        ),
        output_raman_shifts=(
            output_axis
        ),
        prior_residual_transformer=(
            restored_transformer
        ),
    )

    expected_output_spectra = (
        expected_model_axis_spectra[
            :,
            [
                0,
                2,
                4,
            ],
        ]
    )

    assert (
        generated.shape
        == (
            2,
            3,
        )
    )

    np.testing.assert_allclose(
        generated,
        expected_output_spectra,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_d2_2_generation_uses_sampled_pca_priors():
    model_axis = np.arange(
        5,
        dtype=np.float64,
    )

    training_spectra = np.asarray(
        [
            [
                -0.8,
                -0.4,
                0.0,
                0.4,
                0.8,
            ],
            [
                -0.5,
                -0.2,
                0.2,
                0.5,
                0.7,
            ],
            [
                -0.3,
                0.1,
                0.4,
                0.7,
                0.6,
            ],
            [
                -0.6,
                -0.1,
                0.1,
                0.6,
                0.5,
            ],
        ],
        dtype=np.float32,
    )

    transformer = (
        PriorResidualTransformer(
            prior_method=(
                "pca_reconstruction"
            ),
            normalization_method=(
                "pointwise_mad_asinh"
            ),
            pca_explained_variance_ratio=0.90,
            pca_max_components=2,
        )
        .fit(
            training_spectra
        )
    )

    adapter = (
        SpectrumLengthAdapter.create(
            dimension_multipliers=[
                1,
                2,
                4,
            ],
            raman_shifts=[
                model_axis
            ],
            model_length="auto",
            padding_mode=(
                "right_zero_padding"
            ),
            padding_value=0.0,
            raman_range_tolerance=1.0,
        )
    )

    fixed_diffusion = (
        FixedSampleDiffusion(
            np.zeros(
                (
                    3,
                    adapter.padded_length,
                ),
                dtype=np.float32,
            )
        )
    )

    expected = (
        transformer.sample_reference_priors(
            3,
            random_generator=(
                np.random.default_rng(
                    2026
                )
            ),
        )
    )

    generated = generate_spectra(
        diffusion=(
            fixed_diffusion
        ),
        number_of_spectra=3,
        generation_batch_size=2,
        device=torch.device(
            "cpu"
        ),
        length_adapter=(
            adapter
        ),
        prior_residual_transformer=(
            transformer
        ),
        prior_random_seed=2026,
    )

    np.testing.assert_allclose(
        generated,
        expected,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_d2_3_generation_restores_low_frequency_prior():
    """验证D2.3 fixed low-frequency prior完整生成恢复链。"""

    model_axis = np.asarray(
        [
            100.0,
            170.0,
            300.0,
            410.0,
            500.0,
        ],
        dtype=np.float64,
    )

    # 刻意使用非严格等间距Raman轴，
    # 验证D2.3以真实cm^-1轴构造低频prior。
    training_spectra = np.asarray(
        [
            [
                -0.75,
                -0.42,
                0.10,
                0.45,
                0.70,
            ],
            [
                -0.65,
                -0.30,
                0.22,
                0.58,
                0.62,
            ],
            [
                -0.55,
                -0.18,
                0.35,
                0.68,
                0.54,
            ],
            [
                -0.72,
                -0.24,
                0.18,
                0.52,
                0.66,
            ],
            [
                -0.60,
                -0.36,
                0.30,
                0.63,
                0.58,
            ],
        ],
        dtype=np.float32,
    )

    transformer = (
        PriorResidualTransformer(
            prior_method=(
                "training_low_frequency_median"
            ),
            normalization_method=(
                "pointwise_mad_asinh"
            ),
            low_frequency_sigma_cm1=90.0,
            low_frequency_truncate=4.0,
        )
        .fit(
            training_spectra,
            raman_shift=(
                model_axis
            ),
        )
    )

    state = (
        transformer.state_dict()
    )

    assert (
        state[
            "schema_version"
        ]
        == 5
    )

    restored_transformer = (
        PriorResidualTransformer
        .from_state_dict(
            state
        )
    )

    expected_model_axis_spectra = np.asarray(
        [
            [
                -0.67,
                -0.28,
                0.26,
                0.56,
                0.61,
            ],
            [
                -0.58,
                -0.34,
                0.16,
                0.64,
                0.57,
            ],
        ],
        dtype=np.float32,
    )

    scaled_residuals = (
        restored_transformer.transform(
            expected_model_axis_spectra
        )
    )

    adapter = (
        SpectrumLengthAdapter.create(
            dimension_multipliers=[
                1,
                2,
                4,
            ],
            raman_shifts=[
                model_axis
            ],
            model_length="auto",
            padding_mode=(
                "right_zero_padding"
            ),
            padding_value=0.0,
            raman_range_tolerance=1.0,
        )
    )

    padded_scaled_residuals = (
        adapter.adapt(
            scaled_residuals
        )
    )

    fixed_diffusion = (
        FixedSampleDiffusion(
            padded_scaled_residuals
        )
    )

    generated = generate_spectra(
        diffusion=(
            fixed_diffusion
        ),
        number_of_spectra=2,
        generation_batch_size=1,
        device=torch.device(
            "cpu"
        ),
        length_adapter=(
            adapter
        ),
        output_raman_shifts=(
            model_axis
        ),
        prior_residual_transformer=(
            restored_transformer
        ),
        prior_random_seed=2026,
    )

    assert (
        generated.shape
        == expected_model_axis_spectra.shape
    )

    np.testing.assert_allclose(
        generated,
        expected_model_axis_spectra,
        rtol=2.0e-5,
        atol=2.0e-6,
    )


def test_d2_5_pca3_robust_asinh_generation_is_reproducible():
    """验证D2.5 checkpoint恢复、PCA prior采样和生成反变换。"""

    model_axis = np.arange(
        16,
        dtype=np.float64,
    )

    random_generator = np.random.default_rng(
        9157
    )

    training_spectra = random_generator.normal(
        loc=0.0,
        scale=0.25,
        size=(16, model_axis.size),
    ).astype(np.float32)

    fitted_transformer = (
        PriorResidualTransformer(
            prior_method="pca_reconstruction",
            normalization_method="robust_asinh",
            residual_quantile=99.5,
            target_abs_max=1.0,
            pca_explained_variance_ratio=0.95,
            pca_max_components=3,
            pca_sampling_strategy=(
                "independent_truncated_gaussian_scores"
            ),
            pca_score_clip_standard_deviations=1.5,
        )
        .fit(training_spectra)
    )

    restored_transformer = (
        PriorResidualTransformer.from_state_dict(
            fitted_transformer.state_dict()
        )
    )

    adapter = SpectrumLengthAdapter.create(
        dimension_multipliers=[
            1,
            2,
            4,
        ],
        raman_shifts=[
            model_axis
        ],
        model_length="auto",
        padding_mode="right_zero_padding",
        padding_value=0.0,
        raman_range_tolerance=1.0,
    )

    zero_scaled_residuals = np.zeros(
        (
            7,
            adapter.padded_length,
        ),
        dtype=np.float32,
    )

    expected_priors = (
        restored_transformer.sample_reference_priors(
            7,
            random_generator=np.random.default_rng(
                2026
            ),
        )
    )

    generated_batch_3 = generate_spectra(
        diffusion=FixedSampleDiffusion(
            zero_scaled_residuals
        ),
        number_of_spectra=7,
        generation_batch_size=3,
        device=torch.device("cpu"),
        length_adapter=adapter,
        output_raman_shifts=model_axis,
        prior_residual_transformer=(
            restored_transformer
        ),
        prior_random_seed=2026,
    )

    generated_batch_1 = generate_spectra(
        diffusion=FixedSampleDiffusion(
            zero_scaled_residuals
        ),
        number_of_spectra=7,
        generation_batch_size=1,
        device=torch.device("cpu"),
        length_adapter=adapter,
        output_raman_shifts=model_axis,
        prior_residual_transformer=(
            restored_transformer
        ),
        prior_random_seed=2026,
    )

    np.testing.assert_allclose(
        generated_batch_3,
        expected_priors,
        rtol=1.0e-6,
        atol=1.0e-6,
    )

    np.testing.assert_allclose(
        generated_batch_1,
        expected_priors,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_d2_5_generation_variation_scale_contracts_around_pca_mean():
    """验证0.90校准只围绕checkpoint中的PCA均值谱收缩。"""

    model_axis = np.arange(
        16,
        dtype=np.float64,
    )

    random_generator = np.random.default_rng(
        9157
    )

    training_spectra = random_generator.normal(
        loc=0.0,
        scale=0.25,
        size=(16, model_axis.size),
    ).astype(np.float32)

    transformer = (
        PriorResidualTransformer(
            prior_method="pca_reconstruction",
            normalization_method="robust_asinh",
            residual_quantile=99.5,
            target_abs_max=1.0,
            pca_explained_variance_ratio=0.95,
            pca_max_components=3,
            pca_sampling_strategy=(
                "independent_truncated_gaussian_scores"
            ),
            pca_score_clip_standard_deviations=1.5,
        )
        .fit(training_spectra)
    )

    restored_transformer = (
        PriorResidualTransformer.from_state_dict(
            transformer.state_dict()
        )
    )

    adapter = SpectrumLengthAdapter.create(
        dimension_multipliers=[
            1,
            2,
            4,
        ],
        raman_shifts=[
            model_axis
        ],
        model_length="auto",
        padding_mode="right_zero_padding",
        padding_value=0.0,
        raman_range_tolerance=1.0,
    )

    zero_scaled_residuals = np.zeros(
        (
            7,
            adapter.padded_length,
        ),
        dtype=np.float32,
    )

    sampled_priors = (
        restored_transformer.sample_reference_priors(
            7,
            random_generator=np.random.default_rng(
                2026
            ),
        )
    )

    variation_scale = 0.90
    variation_center = np.asarray(
        restored_transformer.pca_mean,
        dtype=np.float32,
    )[np.newaxis, :]
    expected = (
        variation_center
        + variation_scale
        * (
            sampled_priors
            - variation_center
        )
    )

    generated = generate_spectra(
        diffusion=FixedSampleDiffusion(
            zero_scaled_residuals
        ),
        number_of_spectra=7,
        generation_batch_size=3,
        device=torch.device("cpu"),
        length_adapter=adapter,
        output_raman_shifts=model_axis,
        prior_residual_transformer=(
            restored_transformer
        ),
        prior_random_seed=2026,
        variation_scale=variation_scale,
    )

    np.testing.assert_allclose(
        generated,
        expected,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def _sampling_calibration_reference():
    axis = np.arange(600.0, 801.0, 1.0, dtype=np.float64)
    first_peak = np.exp(-0.5 * ((axis - 660.0) / 4.0) ** 2)
    second_peak = 0.75 * np.exp(-0.5 * ((axis - 742.0) / 5.0) ** 2)
    reference = 0.1 + first_peak + second_peak
    return axis, reference.astype(np.float32)


def test_sampling_calibration_reduces_only_non_peak_wide_noise():
    """验证宽杂讯被定向压缩，而自动峰区中心保持不变。"""

    axis, reference = _sampling_calibration_reference()
    phases = np.linspace(0.0, np.pi, 8, endpoint=False)
    spectra = np.stack(
        [
            reference
            + 0.12 * np.sin((axis - 600.0) / 4.0 + phase)
            + 0.025 * np.sin((axis - 600.0) * 2.1 + phase)
            for phase in phases
        ]
    ).astype(np.float32)
    configuration = {
        "enabled": True,
        "non_peak_noise": {
            "enabled": True,
            "reference_smoothing_sigma_cm1": 24.0,
            "minimum_relative_prominence": 0.05,
            "minimum_peak_distance_cm1": 20.0,
            "maximum_peak_count": 6,
            "peak_protection_half_width_cm1": 10.0,
            "transition_width_cm1": 4.0,
            "fine_sigma_cm1": 1.5,
            "broad_sigma_cm1": 7.0,
            "middle_component_scale": 0.40,
            "fine_component_scale": 0.85,
        },
        "raman_shift_jitter": {"enabled": False},
    }
    calibrator = SersSamplingCalibrator(
        raman_shift=axis,
        reference_spectrum=reference,
        configuration=configuration,
        random_seed=2026,
    )
    calibrated = calibrator.apply(spectra)

    non_peak = (
        (np.abs(axis - 660.0) > 16.0)
        & (np.abs(axis - 742.0) > 16.0)
    )
    before_width = np.median(
        np.percentile(spectra[:, non_peak], 97.5, axis=1)
        - np.percentile(spectra[:, non_peak], 2.5, axis=1)
    )
    after_width = np.median(
        np.percentile(calibrated[:, non_peak], 97.5, axis=1)
        - np.percentile(calibrated[:, non_peak], 2.5, axis=1)
    )
    assert after_width < 0.75 * before_width

    for peak_position in (660.0, 742.0):
        protected = np.abs(axis - peak_position) <= 8.0
        np.testing.assert_allclose(
            calibrated[:, protected],
            spectra[:, protected],
            rtol=0.0,
            atol=1.0e-6,
        )

    summary = calibrator.summary()
    assert summary["detected_peak_count"] == 2


def test_sampling_calibration_adds_coherent_bounded_peak_shift():
    """验证整谱峰位共同漂移约±5 cm^-1且峰间距不被破坏。"""

    axis, reference = _sampling_calibration_reference()
    spectra = np.repeat(reference[np.newaxis, :], 100, axis=0)
    configuration = {
        "enabled": True,
        "non_peak_noise": {"enabled": False},
        "raman_shift_jitter": {
            "enabled": True,
            "distribution": "truncated_normal",
            "standard_deviation_cm1": 3.0,
            "maximum_absolute_shift_cm1": 5.0,
        },
    }
    first_calibrator = SersSamplingCalibrator(
        raman_shift=axis,
        reference_spectrum=reference,
        configuration=configuration,
        random_seed=2026,
    )
    shifted = first_calibrator.apply(spectra)
    summary = first_calibrator.summary()

    assert -5.0 <= summary["sampled_shift_min_cm1"] < -4.0
    assert 4.0 < summary["sampled_shift_max_cm1"] <= 5.0
    assert 1.5 < summary["sampled_shift_std_cm1"] < 3.0

    first_window = (axis >= 650.0) & (axis <= 670.0)
    second_window = (axis >= 732.0) & (axis <= 752.0)
    first_indices = np.flatnonzero(first_window)
    second_indices = np.flatnonzero(second_window)
    first_centers = axis[
        first_indices[np.argmax(shifted[:, first_window], axis=1)]
    ]
    second_centers = axis[
        second_indices[np.argmax(shifted[:, second_window], axis=1)]
    ]
    np.testing.assert_allclose(
        second_centers - first_centers,
        82.0,
        rtol=0.0,
        atol=1.0,
    )

    second_calibrator = SersSamplingCalibrator(
        raman_shift=axis,
        reference_spectrum=reference,
        configuration=configuration,
        random_seed=2026,
    )
    np.testing.assert_allclose(
        shifted,
        second_calibrator.apply(spectra),
        rtol=0.0,
        atol=0.0,
    )


def test_sampling_calibration_compresses_peak_height_variation_and_skips_weak_peaks():
    """验证自动主峰峰高离散度被压缩，弱峰/肩峰不参与压缩。"""

    axis, _ = _sampling_calibration_reference()
    strong_one = np.exp(-0.5 * ((axis - 660.0) / 4.0) ** 2)
    strong_two = 0.8 * np.exp(-0.5 * ((axis - 742.0) / 5.0) ** 2)
    weak_shoulder = 0.03 * np.exp(-0.5 * ((axis - 700.0) / 3.0) ** 2)
    reference = (0.1 + strong_one + strong_two + weak_shoulder).astype(
        np.float32
    )
    amplitudes = np.linspace(0.55, 1.65, 40)
    spectra = np.stack(
        [
            0.1
            + amplitude * strong_one
            + (1.15 - 0.35 * amplitude) * strong_two
            + weak_shoulder
            for amplitude in amplitudes
        ]
    ).astype(np.float32)
    configuration = {
        "enabled": True,
        "non_peak_noise": {"enabled": False},
        "raman_shift_jitter": {"enabled": False},
        "peak_height_compression": {
            "enabled": True,
            "reference_smoothing_sigma_cm1": 24.0,
            "reference_peak_smoothing_sigma_cm1": 1.5,
            "minimum_relative_prominence": 0.02,
            "minimum_peak_distance_cm1": 20.0,
            "maximum_peak_count": 4,
            "peak_half_width_cm1": 8.0,
            "transition_width_cm1": 3.0,
            "height_variation_scale": 0.50,
            "minimum_batch_peak_median_height_fraction": 0.08,
            "minimum_valid_height": 1.0e-8,
            "minimum_factor": 0.40,
            "maximum_factor": 1.80,
        },
    }
    calibrator = SersSamplingCalibrator(
        raman_shift=axis,
        reference_spectrum=reference,
        configuration=configuration,
        random_seed=2026,
    )
    calibrated = calibrator.apply(spectra)
    summary = calibrator.summary()

    assert summary["peak_height_compression_enabled"]
    assert summary["height_compression_peak_count"] >= 2
    assert summary["height_compression_active_peak_count"] == 2
    assert summary["height_cv_after"] < 0.75 * summary["height_cv_before"]

    for center in (660.0, 742.0):
        window = np.abs(axis - center) <= 8.0
        before_heights = np.max(spectra[:, window] - 0.1, axis=1)
        after_heights = np.max(calibrated[:, window] - 0.1, axis=1)
        assert np.std(after_heights) < 0.75 * np.std(before_heights)

    weak_window = np.abs(axis - 700.0) <= 5.0
    np.testing.assert_allclose(
        calibrated[:, weak_window],
        spectra[:, weak_window],
        rtol=0.0,
        atol=2.0e-2,
    )


def test_pca_score_clip_runtime_override_uses_cli_and_preserves_checkpoint_state():
    # CLI只修改恢复后的内存transformer，不修改原checkpoint state。

    from argparse import Namespace

    from scripts.generate_spectra import (
        apply_pca_score_clip_runtime_override,
    )

    rng = np.random.default_rng(20260814)

    training_spectra = rng.normal(
        loc=0.0,
        scale=0.25,
        size=(16, 32),
    ).astype(np.float32)

    fitted = PriorResidualTransformer(
        prior_method="pca_reconstruction",
        normalization_method="robust_asinh",
        residual_quantile=99.5,
        target_abs_max=1.0,
        pca_explained_variance_ratio=0.95,
        pca_max_components=6,
        pca_sampling_strategy=(
            "independent_truncated_gaussian_scores"
        ),
        pca_score_clip_standard_deviations=1.5,
    ).fit(training_spectra)

    checkpoint_state = fitted.state_dict()

    assert (
        checkpoint_state["pca_prior"][
            "score_clip_standard_deviations"
        ]
        == 1.5
    )

    restored = PriorResidualTransformer.from_state_dict(
        checkpoint_state
    )

    arguments = Namespace(
        pca_score_clip_standard_deviations=2.5,
    )

    information = (
        apply_pca_score_clip_runtime_override(
            arguments=arguments,
            generation_config={
                "pca_score_clip_override_standard_deviations": 2.0,
            },
            prior_residual_transformer=restored,
        )
    )

    assert information["active"]
    assert information["source"] == "command_line"
    assert information["overridden"]
    assert information["checkpoint_value"] == 1.5
    assert information["runtime_value"] == 2.5

    assert (
        restored.pca_score_clip_standard_deviations
        == 2.5
    )

    assert (
        checkpoint_state["pca_prior"][
            "score_clip_standard_deviations"
        ]
        == 1.5
    )


def test_pca_score_clip_2_5_restores_more_prior_score_dispersion_than_1_5():
    # 相同PCA状态下，2.5σ应比1.5σ恢复更多score方差。

    rng = np.random.default_rng(20260815)

    training_spectra = rng.normal(
        loc=0.0,
        scale=0.25,
        size=(16, 32),
    ).astype(np.float32)

    fitted = PriorResidualTransformer(
        prior_method="pca_reconstruction",
        normalization_method="robust_asinh",
        residual_quantile=99.5,
        target_abs_max=1.0,
        pca_explained_variance_ratio=0.95,
        pca_max_components=6,
        pca_sampling_strategy=(
            "independent_truncated_gaussian_scores"
        ),
        pca_score_clip_standard_deviations=1.5,
    ).fit(training_spectra)

    state = fitted.state_dict()

    transformer_1_5 = (
        PriorResidualTransformer.from_state_dict(
            state
        )
    )

    transformer_2_5 = (
        PriorResidualTransformer.from_state_dict(
            state
        )
    )

    transformer_2_5.pca_score_clip_standard_deviations = (
        2.5
    )

    prior_1_5 = (
        transformer_1_5.sample_reference_priors(
            12000,
            random_generator=np.random.default_rng(
                2026
            ),
        )
    )

    prior_2_5 = (
        transformer_2_5.sample_reference_priors(
            12000,
            random_generator=np.random.default_rng(
                2026
            ),
        )
    )

    def standardized_score_std(
        transformer,
        priors,
    ):
        centered = (
            priors.astype(np.float64)
            - transformer.pca_mean[
                np.newaxis,
                :
            ]
        )

        scores = (
            centered
            @ transformer.pca_components.T
        )

        standardized = (
            scores
            - transformer.pca_training_score_mean[
                np.newaxis,
                :
            ]
        ) / transformer.pca_score_standard_deviation[
            np.newaxis,
            :
        ]

        return float(
            np.mean(
                np.std(
                    standardized,
                    axis=0,
                    ddof=0,
                )
            )
        )

    std_1_5 = standardized_score_std(
        transformer_1_5,
        prior_1_5,
    )

    std_2_5 = standardized_score_std(
        transformer_2_5,
        prior_2_5,
    )

    assert 0.70 < std_1_5 < 0.79
    assert 0.91 < std_2_5 < 0.99
    assert std_2_5 > 1.20 * std_1_5

def test_d4_3_2_14a_local_residual_guard_repairs_unsupported_coherent_valley():
    import numpy as np
    from src.spectrum_generator import (
        apply_condition_intensity_envelope_guard,
    )

    axis = np.arange(
        1000.0,
        1100.0,
        1.0,
    )
    x = np.arange(
        axis.size,
        dtype=np.float64,
    )

    rows = []

    for index in range(12):
        # 模拟同condition不同mapping光谱存在明显整体强度差异，
        # 但局部噪声/残差幅度较小。
        broad = (
            35.0
            + 4.0 * index
            + 0.005 * (x - 50.0) ** 2
        )

        local = (
            1.5
            * np.sin(
                2.0 * np.pi * x / 17.0
                + 0.2 * index
            )
        )

        rows.append(
            broad + local
        )

    training = np.asarray(
        rows,
        dtype=np.float64,
    )

    generated = np.repeat(
        training[[5]],
        20,
        axis=0,
    )

    # --------------------------------------------------------
    # 模拟真实项目里发现的unsupported连续深负谷：
    # 原本约 +55，减130后约 -75。
    # --------------------------------------------------------
    generated[3, 45:52] -= 130.0

    # 单点小负波动：
    # 只产生轻微负值，不能因为它而修整整条谱。
    generated[4, 20] -= 70.0

    before = generated.copy()

    configuration = {
        "mode":
            "local_residual_negative_valley",

        "local_residual_baseline_sigma_cm1":
            30.0,

        "local_residual_training_lower_quantile":
            0.10,

        "local_residual_training_iqr_margin":
            0.50,

        "local_residual_minimum_scale_intensity":
            2.0,

        "activation_maximum_intensity":
            -18.0,

        "minimum_deficit_intensity":
            5.0,

        "minimum_negative_valley_contiguous_points":
            3,

        "local_residual_softness_scale_fraction":
            0.20,

        "minimum_softness_intensity":
            0.25,

        "maximum_softness_intensity":
            2.0,

        "maximum_modified_fraction":
            0.05,
    }

    corrected, diagnostics = (
        apply_condition_intensity_envelope_guard(
            generated,
            training,
            configuration=configuration,
            raman_shift=axis,
        )
    )

    assert diagnostics["mode"] == (
        "local_residual_negative_valley"
    )

    assert (
        diagnostics[
            "negative_valley_retained_point_count"
        ]
        >= 3
    )

    assert (
        diagnostics[
            "modified_spectrum_count"
        ]
        == 1
    )

    # 深负谷应该被明显拉高。
    assert (
        float(
            np.min(
                corrected[3, 45:52]
            )
        )
        >
        float(
            np.min(
                before[3, 45:52]
            )
        )
        + 20.0
    )

    # 普通光谱不能发生有意义的变化。
    np.testing.assert_allclose(
        corrected[0],
        before[0],
        rtol=0.0,
        atol=1.0e-5,
    )

    # 单点小负波动不属于连续深谷，也必须保留。
    assert np.isclose(
        corrected[4, 20],
        before[4, 20],
        rtol=0.0,
        atol=1.0e-5,
    )

    # 深谷之外的谱形不能被动。
    np.testing.assert_allclose(
        corrected[3, :40],
        before[3, :40],
        rtol=0.0,
        atol=1.0e-5,
    )

    np.testing.assert_allclose(
        corrected[3, 60:],
        before[3, 60:],
        rtol=0.0,
        atol=1.0e-5,
    )


def test_d4_3_2_14a_local_residual_guard_preserves_training_supported_valley():
    import numpy as np
    from src.spectrum_generator import (
        apply_condition_intensity_envelope_guard,
    )

    axis = np.arange(
        1000.0,
        1100.0,
        1.0,
    )

    x = np.arange(
        axis.size,
        dtype=np.float64,
    )

    rows = []

    for index in range(12):

        broad = (
            25.0
            + 2.0 * index
            + 0.005 * (x - 50.0) ** 2
        )

        row = broad.copy()

        # training自己就具有连续真实负谷。
        # 谷值大约在 -50 ~ -30，
        # 因此生成相似负谷不能被错误拉平。
        row[45:52] -= (
            75.0
            + 0.5 * index
        )

        rows.append(row)

    training = np.asarray(
        rows,
        dtype=np.float64,
    )

    generated = np.repeat(
        training[[6]],
        20,
        axis=0,
    )

    before = generated.copy()

    configuration = {
        "mode":
            "local_residual_negative_valley",

        "local_residual_baseline_sigma_cm1":
            30.0,

        "local_residual_training_lower_quantile":
            0.10,

        "local_residual_training_iqr_margin":
            0.50,

        "local_residual_minimum_scale_intensity":
            2.0,

        "activation_maximum_intensity":
            -18.0,

        "minimum_deficit_intensity":
            5.0,

        "minimum_negative_valley_contiguous_points":
            3,

        "local_residual_softness_scale_fraction":
            0.20,

        "minimum_softness_intensity":
            0.25,

        "maximum_softness_intensity":
            2.0,

        "maximum_modified_fraction":
            0.05,
    }

    corrected, diagnostics = (
        apply_condition_intensity_envelope_guard(
            generated,
            training,
            configuration=configuration,
            raman_shift=axis,
        )
    )

    # training已经支持这个负谷，因此不能修改。
    assert (
        diagnostics[
            "modified_point_count"
        ]
        == 0
    )

    np.testing.assert_allclose(
        corrected,
        before,
        rtol=0.0,
        atol=1.0e-5,
    )

