import numpy as np
import pandas as pd

from scripts.evaluate_d4_2_generation import (
    _center_metrics,
    _comparison_summary_fields,
    _diversity_fields,
    _metric_summary_fields,
    _pointwise_wasserstein,
    _peak_morphology_diversity,
    _qq_agreement_metrics,
    _qq_table,
    _quality_stratified_summary,
    _training_supported_intensity_metrics,
)
from src.sers_generation_evaluator import (
    broad_local_distribution_evaluation,
    build_nonpeak_mask,
    diversity_summary,
    nearest_reference_metrics,
    pca_distribution_evaluation,
    row_pearson,
    training_nearest_other_metrics,
)


def _synthetic_sers(
    *,
    number: int,
    seed: int,
    variation_scale: float = 1.0,
):
    axis = np.linspace(
        600.0,
        900.0,
        301,
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    rows = []

    for _ in range(number):
        spectrum = np.zeros_like(axis)

        for center, amplitude, width in (
            (700.0, 100.0, 5.0),
            (800.0, 70.0, 7.0),
        ):
            shifted_center = (
                center
                + rng.normal(
                    0.0,
                    0.7 * variation_scale,
                )
            )
            shifted_amplitude = (
                amplitude
                * (
                    1.0
                    + rng.normal(
                        0.0,
                        0.04 * variation_scale,
                    )
                )
            )
            shifted_width = (
                width
                * (
                    1.0
                    + rng.normal(
                        0.0,
                        0.03 * variation_scale,
                    )
                )
            )

            spectrum += (
                shifted_amplitude
                * np.exp(
                    -0.5
                    * (
                        (
                            axis
                            - shifted_center
                        )
                        / shifted_width
                    )
                    ** 2
                )
            )

        spectrum += rng.normal(
            0.0,
            0.5,
            size=axis.size,
        )

        rows.append(spectrum)

    return axis, np.stack(rows, axis=0)


def test_nearest_metrics_are_finite():
    _, training = _synthetic_sers(
        number=16,
        seed=2026,
    )
    _, generated = _synthetic_sers(
        number=32,
        seed=2027,
    )

    generated_nearest = (
        nearest_reference_metrics(
            query_spectra=generated,
            reference_spectra=training,
        )
    )

    training_nearest = (
        training_nearest_other_metrics(
            training
        )
    )

    assert generated_nearest.shape[0] == 32
    assert training_nearest.shape[0] == 16
    assert np.isfinite(
        generated_nearest[
            "standardized_shape_rmse"
        ].to_numpy()
    ).all()
    assert np.isfinite(
        generated_nearest[
            "pearson"
        ].to_numpy()
    ).all()
    context = _comparison_summary_fields(
        generated_nearest,
        "generated_to_training",
    )
    assert context["generated_to_training_mse_mean"] >= 0.0
    assert -1.0 <= context["generated_to_training_cosine_mean"] <= 1.0
    assert -1.0 <= context["generated_to_training_pearson_mean"] <= 1.0


def test_identical_spectra_have_unit_pearson():
    _, spectra = _synthetic_sers(
        number=4,
        seed=2026,
    )

    result = row_pearson(
        spectra,
        spectra,
    )

    assert np.allclose(
        result,
        1.0,
        atol=1.0e-10,
    )


def test_pca_distribution_evaluation_runs():
    _, training = _synthetic_sers(
        number=16,
        seed=2026,
    )
    _, generated = _synthetic_sers(
        number=40,
        seed=2027,
    )
    _, heldout = _synthetic_sers(
        number=4,
        seed=2028,
    )

    (
        model,
        pca_summary,
        pca_scores,
        distribution_summary,
        reconstruction_summary,
    ) = pca_distribution_evaluation(
        training_spectra=training,
        generated_spectra=generated,
        heldout_spectra=heldout,
        pca_components=6,
        bootstrap_repeats=10,
        random_seed=2026,
    )

    assert model.components.shape[0] == 6
    assert pca_summary.shape[0] == 6
    assert set(
        pca_scores["dataset"].unique()
    ) == {
        "training",
        "generated",
        "heldout",
    }
    assert distribution_summary.shape[0] == 3
    assert set(
        reconstruction_summary["dataset"]
    ) == {
        "training",
        "generated",
        "heldout",
    }


def test_broad_local_matched_bootstrap_runs():
    axis, training = _synthetic_sers(
        number=16,
        seed=2026,
    )
    _, generated = _synthetic_sers(
        number=40,
        seed=2027,
    )

    mask = build_nonpeak_mask(
        raman_shift=axis,
        peak_centers_cm1=[
            700.0,
            800.0,
        ],
        half_width_cm1=15.0,
    )

    full_summary, bootstrap_summary = (
        broad_local_distribution_evaluation(
            training_spectra=training,
            generated_spectra=generated,
            raman_shift=axis,
            nonpeak_mask=mask,
            broad_sigma_cm1=7.0,
            bootstrap_repeats=10,
            random_seed=2026,
        )
    )

    assert full_summary.shape[0] == 9
    assert bootstrap_summary.shape[0] == 9
    assert np.isfinite(
        bootstrap_summary[
            "ratio_median"
        ].to_numpy()
    ).all()


def test_diversity_summary_uses_all_unique_pairs_when_request_is_larger():
    _, training = _synthetic_sers(
        number=16,
        seed=2026,
    )
    _, generated = _synthetic_sers(
        number=32,
        seed=2027,
    )

    (
        summary,
        training_pairs,
        generated_pairs,
    ) = diversity_summary(
        training_spectra=training,
        generated_spectra=generated,
        generated_pair_count=500,
        random_seed=2026,
    )

    assert training_pairs.shape[0] == 120
    assert generated_pairs.shape[0] == 496
    assert summary.shape[0] == 4


def test_core_generation_metrics_and_qq_are_finite():
    _, training = _synthetic_sers(number=12, seed=2026)
    _, reference = _synthetic_sers(number=4, seed=2027)
    _, generated = _synthetic_sers(number=32, seed=2028)

    center = _center_metrics(reference, generated)
    wasserstein = _pointwise_wasserstein(
        training=training,
        reference=reference,
        generated=generated,
    )
    diversity, training_pairs, generated_pairs = _diversity_fields(
        training=training,
        generated=generated,
        pair_count=200,
        random_seed=2026,
    )
    qq, qq_metrics = _qq_table(
        condition="DEL-S_water",
        training=training,
        reference=reference,
        generated=generated,
        quantile_count=51,
        oracle_repeats=64,
        oracle_confidence=0.975,
        random_seed=2026,
    )

    assert center["mean_spectrum_mse"] >= 0.0
    assert -1.0 <= center["mean_spectrum_cosine"] <= 1.0
    assert -1.0 <= center["mean_spectrum_pearson"] <= 1.0
    assert wasserstein["wasserstein_raw_pointwise_median"] >= 0.0
    assert diversity["generated_pairwise_mse_median"] >= 0.0
    assert diversity["diversity_pearson_distance_ratio"] >= 0.0
    assert training_pairs.shape[0] == 66
    assert generated_pairs.shape[0] == 200
    assert qq.shape == (51, 14)
    assert np.isfinite(qq.select_dtypes(include=[np.number]).to_numpy()).all()
    assert qq_metrics["qq_lower_tail_span_ratio"] > 0.0
    assert qq_metrics["qq_upper_tail_span_ratio"] > 0.0
    assert np.isfinite(qq_metrics["qq_model_normalized_r2"])
    assert np.isfinite(qq_metrics["qq_model_normalized_rmse"])
    assert qq_metrics["qq_lower_outer_01_10_span_ratio"] > 0.0
    assert qq_metrics["qq_lower_inner_10_50_span_ratio"] > 0.0
    assert qq_metrics["qq_upper_inner_50_90_span_ratio"] > 0.0
    assert qq_metrics["qq_upper_outer_90_99_span_ratio"] > 0.0
    assert qq_metrics["qq_lower_tail_curve_rmse"] >= 0.0
    assert qq_metrics["qq_upper_tail_curve_rmse"] >= 0.0
    assert 0.0 <= qq_metrics["qq_oracle_outside_fraction"] <= 1.0
    assert (
        qq_metrics["qq_lower_tail_oracle_lower"]
        < qq_metrics["qq_lower_tail_oracle_upper"]
    )
    assert (
        qq_metrics["qq_upper_tail_oracle_lower"]
        < qq_metrics["qq_upper_tail_oracle_upper"]
    )
    assert isinstance(qq_metrics["qq_any_tail_overrun"], bool)
    assert isinstance(qq_metrics["qq_any_tail_underrun"], bool)
    assert isinstance(qq_metrics["qq_any_tail_mismatch"], bool)


def test_peak_morphology_diversity_detects_frozen_peak_positions():
    axis, training = _synthetic_sers(
        number=12, seed=2026, variation_scale=1.0
    )
    _, generated = _synthetic_sers(
        number=80, seed=2027, variation_scale=0.08
    )
    metrics, details = _peak_morphology_diversity(
        condition="CHL-H_water",
        axis=axis,
        training=training,
        generated=generated,
        configuration={
            "baseline_sigma_cm1": 30.0,
            "smoothing_sigma_cm1": 2.0,
            "minimum_relative_prominence": 0.06,
            "minimum_peak_distance_cm1": 12.0,
            "maximum_peak_count": 10,
            "edge_exclusion_cm1": 24.0,
            "measurement_half_width_cm1": 12.0,
            "minimum_position_std_cm1": 0.20,
        },
    )
    assert metrics["peak_diversity_peak_count"] == 2
    assert metrics["peak_position_active_count"] >= 1
    assert metrics["peak_position_std_ratio_median"] < 0.5
    assert metrics["peak_position_frozen_fraction"] > 0.0
    assert details.shape[0] == 2
    assert set(
        [
            "peak_position_std_ratio",
            "peak_height_iqr_ratio",
            "peak_width_iqr_ratio",
        ]
    ).issubset(details.columns)


def test_qq_agreement_metrics_separate_fit_from_identity_error():
    reference = np.linspace(-2.0, 2.0, 101)
    generated = 1.5 * reference + 0.25
    metrics = _qq_agreement_metrics(reference, generated, "qq")
    assert np.isclose(metrics["qq_r2"], 1.0)
    assert metrics["qq_rmse"] > 0.5


def test_metric_summary_includes_raw_and_standardized_rmse():
    frame = pd.DataFrame(
        {
            "raw_mse": [4.0, 9.0],
            "raw_rmse": [2.0, 3.0],
            "standardized_shape_mse": [0.04, 0.09],
            "standardized_shape_rmse": [0.2, 0.3],
            "cosine": [0.9, 0.8],
            "pearson": [0.8, 0.7],
        }
    )
    summary = _metric_summary_fields(frame)
    assert summary["nearest_reference_rmse_mean"] == 2.5
    assert summary["nearest_reference_standardized_shape_mse_mean"] == 0.065
    assert summary["nearest_reference_standardized_shape_rmse_mean"] == 0.25


def test_training_supported_intensity_metrics_detect_both_sides():
    training = np.repeat(
        np.asarray([[-18.0, 100.0, 200.0]], dtype=np.float64), 12, axis=0
    )
    generated = np.repeat(training[[0]], 20, axis=0)
    generated[0, 1] = -200.0
    generated[1, 2] = 900.0
    metrics = _training_supported_intensity_metrics(
        training=training,
        generated=generated,
        configuration={
            "training_lower_quantile": 0.05,
            "training_upper_quantile": 0.95,
            "training_iqr_margin": 0.5,
            "training_extrema_margin_iqr_fraction": 0.10,
            "minimum_extrema_margin_intensity": 2.0,
            "activation_maximum_intensity": -18.0,
            "minimum_deficit_intensity": 5.0,
            "minimum_excess_intensity": 5.0,
        },
    )
    assert metrics["generated_below_training_envelope_point_fraction"] > 0.0
    assert metrics["generated_above_training_envelope_point_fraction"] > 0.0
    assert metrics["training_minimum_intensity"] == -18.0
    assert metrics["generated_maximum_intensity"] == 900.0


def test_identical_distributions_have_zero_pointwise_wasserstein():
    _, spectra = _synthetic_sers(number=12, seed=2026)
    metrics = _pointwise_wasserstein(
        training=spectra,
        reference=spectra,
        generated=spectra.copy(),
    )
    assert metrics["wasserstein_raw_pointwise_mean"] == 0.0
    assert metrics["wasserstein_normalized_pointwise_mean"] == 0.0


def test_quality_stratified_summary_covers_all_condition_families():
    rows = []
    for condition, points in (
        ("DEL-S_water", 1401),
        ("CHL-H_soil", 1901),
        ("DEL-M_TEB-H_water", 1401),
        ("DEL-H_TEB-M_CHL-S_soil", 1901),
    ):
        rows.append(
            {
                "condition": condition,
                "point_count": points,
                "nearest_reference_mse_mean": 100.0,
                "nearest_reference_rmse_mean": 10.0,
                "nearest_reference_standardized_shape_mse_mean": 0.04,
                "nearest_reference_standardized_shape_rmse_mean": 0.2,
                "nearest_reference_model_normalized_mse_mean": 0.01,
                "nearest_reference_model_normalized_rmse_mean": 0.1,
                "nearest_reference_model_normalized_cosine_mean": 0.97,
                "nearest_reference_model_normalized_pearson_mean": 0.96,
                "nearest_reference_cosine_mean": 0.98,
                "nearest_reference_pearson_mean": 0.97,
                "mean_spectrum_mse": 20.0,
                "wasserstein_raw_pointwise_mean": 5.0,
                "wasserstein_normalized_pointwise_mean": 0.03,
                "wasserstein_model_normalized_pointwise_mean": 0.02,
                "wasserstein_model_normalized_global_flattened": 0.018,
                "generated_to_training_mse_ratio_vs_heldout": 0.8,
                "diversity_pairwise_mse_ratio": 0.85,
                "diversity_pearson_distance_ratio": 0.83,
                "peak_position_std_ratio_median": 0.9,
                "peak_position_wasserstein_cm1_mean": 0.5,
                "peak_position_frozen_fraction": 0.1,
                "peak_height_iqr_ratio_median": 0.88,
                "peak_width_iqr_ratio_median": 0.92,
                "qq_lower_tail_span_ratio": 0.9,
                "qq_central_10_90_span_ratio": 0.95,
                "qq_upper_tail_span_ratio": 0.9,
                "qq_raw_r2": 0.99,
                "qq_raw_rmse": 2.0,
                "qq_model_normalized_r2": 0.985,
                "qq_model_normalized_rmse": 0.08,
                "qq_training_standardized_r2": 0.98,
                "qq_training_standardized_rmse": 0.1,
                "generated_below_training_envelope_point_fraction": 0.0,
                "generated_above_training_envelope_point_fraction": 0.0,
            }
        )
    summary = _quality_stratified_summary(pd.DataFrame(rows))
    assert {
        "overall",
        "matrix",
        "analyte_count",
        "analyte_signature",
        "matrix_x_analyte_count",
        "axis_group",
    }.issubset(set(summary["stratum_level"]))
    overall = summary[summary["stratum_level"] == "overall"]
    assert set(overall["condition_count"]) == {4}
    assert "peak_position_std_ratio_median" in set(overall["metric"])
