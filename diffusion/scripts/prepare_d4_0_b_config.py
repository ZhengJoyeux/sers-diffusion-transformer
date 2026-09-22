"""从当前配置生成独立的D4.0-B混合轴掩码烟雾实验配置。"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml


DISABLED_SECTIONS = (
    "prior_residual",
    "broad_local_residual",
    "physics_constraints",
    "peak_derivative_constraints",
    "relative_peak_intensity_constraints",
    "peak_parameter_constraints",
    "diversity_constraints",
    "feature_peak_residual_limiter",
    "local_peak_distribution_constraints",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--name",
        default="d4_0_b_mixed_axis_mask_smoke",
    )
    parser.add_argument("--epochs", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    source = Path(arguments.source).expanduser().resolve()
    output = Path(arguments.output).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"找不到源配置：{source}")
    if output.exists():
        raise FileExistsError(
            f"输出配置已存在，为避免覆盖已停止：{output}"
        )
    if arguments.epochs <= 0:
        raise ValueError("--epochs必须大于0。")

    with source.open("r", encoding="utf-8") as file:
        original = yaml.safe_load(file)
    if not isinstance(original, dict):
        raise ValueError("源配置顶层必须是字典。")
    config = deepcopy(original)

    config.setdefault("project", {})["name"] = arguments.name
    data = config.setdefault("data", {})
    data["input_directory"] = "data/input"
    data["recursive"] = True
    data["raman_axis_mode"] = "union_with_valid_mask"
    data["split_unit"] = "fixed_spectra_within_source_file"
    data["fixed_spectrum_split"] = {
        "train_count": 12,
        "validation_count": 4,
        "test_count": 4,
    }
    data["train_ratio"] = 0.6
    data["validation_ratio"] = 0.2
    data["test_ratio"] = 0.2
    data["shuffle"] = False

    for section_name in DISABLED_SECTIONS:
        section = config.setdefault(section_name, {})
        if not isinstance(section, dict):
            raise TypeError(f"{section_name}必须是字典。")
        section["enabled"] = False

    model = config.setdefault("model", {})
    model["objective"] = "pred_x0"
    model["auto_normalize"] = False
    diffusion = config.setdefault("diffusion", {})
    diffusion["objective"] = "pred_x0"
    diffusion["auto_normalize"] = False
    diffusion.setdefault("residual_aware_loss", {})["enabled"] = False

    training = config.setdefault("training", {})
    training["number_of_epochs"] = int(arguments.epochs)
    training["validate_every_epochs"] = 1
    training["save_every_epochs"] = 1
    training["log_every_batches"] = 20

    root = f"outputs/experiments/{arguments.name}"
    output_config = config.setdefault("output", {})
    output_config["output_directory"] = root
    output_config["checkpoint_directory"] = f"{root}/checkpoints"
    output_config["log_directory"] = f"{root}/logs"
    output_config["generated_spectrum_directory"] = f"{root}/generated"
    output_config["preview_plot_directory"] = f"{root}/plots"

    generation = config.setdefault("generation", {})
    generation["number_of_spectra"] = 20
    generation["batch_size"] = 10
    generation["variation_scale"] = 1.0
    calibration = generation.setdefault("sampling_calibration", {})
    if isinstance(calibration, dict):
        calibration["enabled"] = False

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        yaml.safe_dump(
            config,
            file,
            allow_unicode=True,
            sort_keys=False,
        )

    print(f"已生成D4.0-B独立配置：{output}")
    print(f"烟雾训练epoch：{arguments.epochs}")
    print("原始配置未被修改。")


if __name__ == "__main__":
    main()
