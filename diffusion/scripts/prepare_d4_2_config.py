"""Create the formal D4.2 mixed-axis conditional prior-residual config."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml


def _disable(configuration: dict, name: str) -> None:
    section = configuration.get(name, {}) or {}
    if not isinstance(section, dict):
        section = {}
    section["enabled"] = False
    configuration[name] = section


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从已验证的D4.1配置生成D4.2正式配置。"
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", default="d4_2_conditional_prior_residual_formal")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()

    source = Path(arguments.source).expanduser().resolve()
    output = Path(arguments.output).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"找不到源配置：{source}")
    if output.exists() and not arguments.force:
        raise FileExistsError(f"输出已存在：{output}；确认覆盖时增加--force。")
    if arguments.epochs <= 0:
        raise ValueError("epochs必须大于0。")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("源YAML顶层必须为字典。")
    configuration = deepcopy(raw)

    configuration.setdefault("project", {})["name"] = arguments.name
    data = configuration.setdefault("data", {})
    data.update(
        {
            "recursive": True,
            "require_same_raman_shift_axis": False,
            "original_spectrum_length": "auto",
            "model_spectrum_length": "auto",
            "raman_axis_mode": "union_with_valid_mask",
            "length_adaptation": "raman_axis_interpolation",
            "padding_mode": "right_zero_padding",
            "padding_value": 0.0,
            "split_unit": "fixed_spectra_within_source_file",
            "fixed_spectrum_split": {
                "train_count": 12,
                "validation_count": 4,
                "test_count": 4,
            },
            "train_ratio": 0.6,
            "validation_ratio": 0.2,
            "test_ratio": 0.2,
            "shuffle": False,
        }
    )
    normalization = configuration.setdefault("normalization", {})
    normalization.update(
        {
            "enabled": True,
            "method": "global_minmax",
            "fit_on": "train_only",
            "target_min": -1.0,
            "target_max": 1.0,
            "save_in_checkpoint": True,
        }
    )
    configuration["conditioning"] = {
        "enabled": True,
        "vector_size": 14,
        "embedding_dimension": 32,
        "injection": "input_and_all_resnet_blocks_film",
    }

    prior = configuration.get("prior_residual", {}) or {}
    if not isinstance(prior, dict):
        prior = {}
    prior.update(
        {
            "enabled": True,
            "prior_method": "pca_reconstruction",
            "pca_explained_variance_ratio": 0.95,
            "pca_max_components": 6,
            "pca_sampling_strategy": "independent_truncated_gaussian_scores",
            "pca_score_clip_standard_deviations": 1.5,
            "residual_normalization": "robust_asinh",
            "residual_quantile": 99.5,
            "target_abs_max": 1.0,
            "epsilon": 1.0e-8,
        }
    )
    configuration["prior_residual"] = prior

    broad_local = configuration.get("broad_local_residual", {}) or {}
    if not isinstance(broad_local, dict):
        broad_local = {}
    broad_local.update(
        {
            "enabled": True,
            "broad_filter": {
                "method": "gaussian",
                "sigma_cm1": 7.0,
                "truncate": 4.0,
            },
            "broad_prior": {
                "method": "pca",
                "pca_explained_variance_ratio": 0.95,
                "pca_max_components": 6,
                "sampling_strategy": "independent_truncated_gaussian_scores",
                "score_clip_standard_deviations": 1.5,
            },
            "local_normalization": {
                "method": "robust_asinh",
                "residual_quantile": 99.5,
                "target_abs_max": 1.0,
            },
            "epsilon": 1.0e-8,
        }
    )
    configuration["broad_local_residual"] = broad_local

    for name in (
        "physics_constraints",
        "peak_derivative_constraints",
        "relative_peak_intensity_constraints",
        "peak_parameter_constraints",
        "diversity_constraints",
        "feature_peak_residual_limiter",
        "local_peak_distribution_constraints",
    ):
        _disable(configuration, name)
    configuration.setdefault("model", {})["self_condition"] = False
    configuration["model"]["auto_normalize"] = False
    diffusion = configuration.setdefault("diffusion", {})
    diffusion["auto_normalize"] = False
    diffusion["objective"] = "pred_x0"
    diffusion["loss_weighting"] = "uniform"
    residual_loss = diffusion.get("residual_aware_loss", {}) or {}
    if not isinstance(residual_loss, dict):
        residual_loss = {}
    residual_loss["enabled"] = False
    diffusion["residual_aware_loss"] = residual_loss

    training = configuration.setdefault("training", {})
    training["number_of_epochs"] = int(arguments.epochs)
    training["validate_every_epochs"] = 1
    training["save_every_epochs"] = 1

    root = f"outputs/experiments/{arguments.name}"
    output_config = configuration.setdefault("output", {})
    output_config.update(
        {
            "output_directory": root,
            "checkpoint_directory": f"{root}/checkpoints",
            "log_directory": f"{root}/logs",
            "generated_spectrum_directory": f"{root}/generated",
            "preview_plot_directory": f"{root}/plots",
        }
    )
    generation = configuration.setdefault("generation", {})
    generation["model_source"] = "ema"
    generation.setdefault("number_of_spectra", 20)
    generation.setdefault("batch_size", 10)
    generation["inverse_normalize"] = True
    generation.setdefault("output_formats", ["xlsx", "png"])

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(configuration, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"已生成D4.2配置：{output}")
    print(
        f"实验名：{arguments.name}；epoch：{arguments.epochs}；"
        "共享条件DDPM＋126套train-only条件先验。"
    )
    print("源配置未被修改。")


if __name__ == "__main__":
    main()
