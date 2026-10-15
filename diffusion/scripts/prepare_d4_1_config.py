"""Create a D4.1 baseline YAML from the user's verified D4.0 YAML."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成D4.1条件掩码基线配置。")
    parser.add_argument("--source", required=True, help="已验证的D4.0配置。")
    parser.add_argument("--output", required=True, help="新D4.1配置路径。")
    parser.add_argument("--name", default="d4_1_conditional_masked_baseline_smoke")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _disabled(configuration: dict, section_name: str) -> None:
    section = configuration.get(section_name, {}) or {}
    if not isinstance(section, dict):
        section = {}
    section["enabled"] = False
    configuration[section_name] = section


def main() -> None:
    arguments = parse_arguments()
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
        }
    )
    configuration["conditioning"] = {
        "enabled": True,
        "vector_size": 14,
        "embedding_dimension": 8,
    }
    for section_name in (
        "prior_residual",
        "broad_local_residual",
        "physics_constraints",
        "peak_derivative_constraints",
        "relative_peak_intensity_constraints",
        "peak_parameter_constraints",
        "diversity_constraints",
        "feature_peak_residual_limiter",
        "local_peak_distribution_constraints",
    ):
        _disabled(configuration, section_name)
    configuration.setdefault("model", {})["self_condition"] = False
    configuration["model"]["auto_normalize"] = False
    diffusion = configuration.setdefault("diffusion", {})
    diffusion["auto_normalize"] = False
    residual = diffusion.get("residual_aware_loss", {}) or {}
    if not isinstance(residual, dict):
        residual = {}
    residual["enabled"] = False
    diffusion["residual_aware_loss"] = residual

    training = configuration.setdefault("training", {})
    training["number_of_epochs"] = int(arguments.epochs)
    training["validate_every_epochs"] = 1
    training["save_every_epochs"] = 1

    experiment_root = f"outputs/experiments/{arguments.name}"
    output_config = configuration.setdefault("output", {})
    output_config.update(
        {
            "output_directory": experiment_root,
            "checkpoint_directory": f"{experiment_root}/checkpoints",
            "log_directory": f"{experiment_root}/logs",
            "generated_spectrum_directory": f"{experiment_root}/generated",
            "preview_plot_directory": f"{experiment_root}/plots",
        }
    )
    generation = configuration.setdefault("generation", {})
    generation.setdefault("model_source", "ema")
    generation.setdefault("number_of_spectra", 20)
    generation.setdefault("batch_size", 10)
    generation.setdefault("inverse_normalize", True)
    generation.setdefault("output_formats", ["xlsx", "png"])

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(configuration, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"已生成D4.1配置：{output}")
    print(f"实验名：{arguments.name}；epoch：{arguments.epochs}")
    print("源配置未被修改。")


if __name__ == "__main__":
    main()
