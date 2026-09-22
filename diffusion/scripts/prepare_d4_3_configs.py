"""Create D4.3 smoke/formal configs from the frozen D4.2 formal config."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = PROJECT_ROOT / "config" / "ddpm_training_d4_2_formal.yaml"


def _diversity_configuration() -> dict:
    return {
        "enabled": True,
        "strategy": "low_noise_pairwise_residual_geometry",
        "total_weight": 0.02,
        "maximum_total_ratio_to_ddpm": 0.02,
        "epsilon": 1.0e-8,
        "condition_grouping": {
            "samples_per_condition": 4,
            "shared_timestep": True,
        },
        "low_noise_gate": {
            "minimum_alpha_cumprod": 0.50,
            "minimum_samples": 4,
        },
        "high_frequency_filter": {
            "smoothing_sigma_points": 1.0,
            "kernel_truncate": 3.0,
            "distance_reference_quantile": 50.0,
            "correlation_reference_quantile": 90.0,
        },
        "pairwise_distance": {
            "enabled": True,
            "metric": "whitened_l2",
            "penalty_mode": "floor_only",
            "minimum_distance_ratio": 0.80,
            "maximum_distance_ratio": 1.25,
            "transition_fraction": 0.20,
            "weight": 1.0,
        },
        "pairwise_correlation": {
            "enabled": True,
            "maximum_excess_correlation": 0.02,
            "transition_fraction": 0.20,
            "weight": 0.5,
        },
        "pointwise_variance_floor": {
            "enabled": True,
            "penalty_mode": "floor_only",
            "active_std_quantile": 20.0,
            "minimum_std_ratio": 0.80,
            "maximum_std_ratio": 1.25,
            "reference_std_floor_fraction": 0.25,
            "weight": 0.5,
        },
        "peak_morphology": {"enabled": False},
    }


def _set_output(config: dict, experiment_name: str) -> None:
    root = f"outputs/experiments/{experiment_name}"
    config["project"]["name"] = experiment_name
    config["output"]["output_directory"] = root
    config["output"]["checkpoint_directory"] = f"{root}/checkpoints"
    config["output"]["log_directory"] = f"{root}/logs"
    config["output"]["generated_spectrum_directory"] = f"{root}/generated"
    config["output"]["preview_plot_directory"] = f"{root}/plots"


def _write(config: dict, file_name: str) -> Path:
    path = PROJECT_ROOT / "config" / file_name
    path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def main() -> None:
    if not BASE_PATH.is_file():
        raise FileNotFoundError(f"找不到D4.2正式配置：{BASE_PATH}")
    base = yaml.safe_load(BASE_PATH.read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        raise TypeError("D4.2正式配置必须是YAML字典。")
    if base.get("data", {}).get("raman_axis_mode") != "union_with_valid_mask":
        raise ValueError("D4.3必须从D4.2 union_with_valid_mask配置派生。")
    if not bool(base.get("conditioning", {}).get("enabled", False)):
        raise ValueError("D4.3必须从已启用conditioning的D4.2配置派生。")

    smoke = deepcopy(base)
    smoke["diversity_constraints"] = _diversity_configuration()
    smoke["training"]["number_of_epochs"] = 2
    smoke["training"]["validate_every_epochs"] = 1
    smoke["training"]["save_every_epochs"] = 1
    _set_output(smoke, "d4_3_1_floor_only_diversity_smoke")

    formal = deepcopy(base)
    formal["diversity_constraints"] = _diversity_configuration()
    formal["training"]["number_of_epochs"] = 100
    _set_output(formal, "d4_3_1_floor_only_diversity_formal")

    paths = [
        _write(smoke, "ddpm_training_d4_3_smoke.yaml"),
        _write(formal, "ddpm_training_d4_3_formal.yaml"),
    ]
    print("D4.3配置已生成：")
    for path in paths:
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
