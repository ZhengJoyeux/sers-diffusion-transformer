"""Generate spectra for one or all D4.1 chemical conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.checkpoint_manager import load_checkpoint_file
from src.conditional_diversity_constraints import (
    evaluate_condition_aware_diversity_residuals,
)
from src.conditional_prior_residual import ConditionalPriorResidualBank
from src.configuration_loader import load_configuration, resolve_project_path
from src.intensity_normalizer import GlobalMinMaxNormalizer
from src.model_builder import build_diffusion_model
from src.random_seed_manager import set_random_seed
from src.spectrum_conditioning import (
    resolve_source_named_condition_from_metadata,
    source_named_conditions_from_metadata,
)
from src.spectrum_exporter import export_generated_spectra
from src.spectrum_file_reader import read_spectrum_collection
from src.spectrum_generator import (
    apply_condition_intensity_envelope_guard,
    apply_condition_mean_fidelity_calibration,
    apply_condition_oracle_tail_calibration,
    apply_condition_pca_spread_calibration,
    generate_spectra,
    resolve_condition_tail_calibration_configuration,
)
from src.spectrum_length_adapter import SpectrumLengthAdapter


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用D4.1/D4.2条件扩散模型生成指定组合的SERS光谱。"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--condition",
        required=True,
        help=(
            "必须填写输入表格的原始文件名（不含扩展名），例如"
            "CHL-H_TEB-M_water；使用all生成检查点内全部条件。"
        ),
    )
    parser.add_argument("--number", type=int, default=None)
    parser.add_argument("--model-source", choices=("raw", "ema"), default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output-directory",
        default=None,
        help=(
            "可选的本次生成结果根目录。用于A/B实验时写入新目录，"
            "避免覆盖配置中的旧生成结果。"
        ),
    )
    parser.add_argument(
        "--pca-score-clip-standard-deviations",
        type=float,
        default=None,
        help=(
            "生成时同时覆盖指定条件的外层PCA先验和broad残差先验"
            "采样截断范围；不修改checkpoint。未填写时读取generation配置。"
        ),
    )
    return parser.parse_args()


def _checkpoint_path(configuration: dict, value: str | None) -> Path:
    if value is None:
        value = str(
            Path(configuration["output"]["checkpoint_directory"])
            / str(configuration["output"].get("latest_checkpoint_name", "latest.pt"))
        )
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = resolve_project_path(configuration, path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到检查点：{path}")
    return path


def _device(configuration: dict, override: str | None) -> torch.device:
    text = str(
        override
        if override is not None
        else configuration["training"].get("device", "cpu")
    ).strip().lower()
    if text.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("要求使用CUDA，但PyTorch没有检测到可用GPU。")
    return torch.device(text)


def _load_model_state(
    diffusion: torch.nn.Module,
    checkpoint: dict,
    source: str,
) -> None:
    if source == "raw":
        state = checkpoint.get("diffusion_state")
    else:
        ema = checkpoint.get("ema_state")
        state = ema.get("ema_model") if isinstance(ema, dict) else None
    if not isinstance(state, dict):
        raise RuntimeError(f"检查点不包含有效的{source}模型状态。")
    diffusion.load_state_dict(state)


def _axis_for_source(metadata: dict, source_file: str) -> tuple[np.ndarray, str]:
    axis_metadata = metadata.get("axis_metadata")
    sources = (
        axis_metadata.get("source_files")
        if isinstance(axis_metadata, dict)
        else None
    )
    if not isinstance(sources, dict) or source_file not in sources:
        raise RuntimeError(f"检查点中找不到源文件{source_file}的Raman轴。")
    record = sources[source_file]
    axis = np.asarray(record.get("raman_shift"), dtype=np.float64).reshape(-1)
    if axis.size < 2 or not np.isfinite(axis).all() or not np.all(np.diff(axis) > 0):
        raise RuntimeError(f"源文件{source_file}的Raman轴无效。")
    return axis, str(record.get("profile_id", "unknown"))


def _condition_training_spectra(
    *,
    collection,
    relative_source_files: np.ndarray,
    training_indices: set[int],
    source_file: str,
    output_axis: np.ndarray,
) -> np.ndarray:
    source_indices = np.flatnonzero(relative_source_files == source_file)
    selected = [
        int(index) for index in source_indices if int(index) in training_indices
    ]
    if len(selected) != 12:
        raise RuntimeError(
            f"条件源文件{source_file}不是固定12条训练光谱：{len(selected)}。"
        )
    rows: list[np.ndarray] = []
    for index in selected:
        source_axis = np.asarray(collection.raman_shifts[index], dtype=np.float64)
        if (
            output_axis[0] < source_axis[0] - 1.0e-6
            or output_axis[-1] > source_axis[-1] + 1.0e-6
        ):
            raise RuntimeError(f"条件校准禁止对源文件{source_file}进行Raman轴外推。")
        rows.append(
            np.interp(
                output_axis,
                source_axis,
                np.asarray(collection.spectra[index], dtype=np.float64),
            )
        )
    return np.stack(rows, axis=0)


def main() -> None:
    arguments = parse_arguments()
    configuration = load_configuration(arguments.config)
    generation = configuration["generation"]
    output = configuration["output"]
    checkpoint_path = _checkpoint_path(configuration, arguments.checkpoint)
    checkpoint = load_checkpoint_file(checkpoint_path, map_location="cpu")
    metadata = checkpoint.get("metadata")
    checkpoint_configuration = checkpoint.get("configuration")
    if not isinstance(metadata, dict) or not isinstance(
        checkpoint_configuration, dict
    ):
        raise RuntimeError("检查点缺少metadata或configuration。")
    conditioning = checkpoint_configuration.get("conditioning", {}) or {}
    if not bool(conditioning.get("enabled", False)):
        raise RuntimeError("该检查点不是D4.1条件生成检查点。")

    conditional_prior_bank = None
    prior_enabled = bool(
        (checkpoint_configuration.get("prior_residual", {}) or {}).get(
            "enabled", False
        )
    )
    if prior_enabled:
        state = metadata.get("conditional_prior_residual_state")
        if not isinstance(state, dict):
            raise RuntimeError(
                "检查点启用了条件先验残差，但缺少"
                "conditional_prior_residual_state。"
            )
        conditional_prior_bank = ConditionalPriorResidualBank.from_state_dict(
            state
        )

    score_clip_override = arguments.pca_score_clip_standard_deviations
    score_clip_source = "command_line"
    if score_clip_override is None:
        score_clip_override = generation.get(
            "pca_score_clip_override_standard_deviations"
        )
        score_clip_source = "configuration"
    score_clip_information = None
    if conditional_prior_bank is not None:
        score_clip_information = (
            conditional_prior_bank.apply_score_clip_runtime_override(
                score_clip_override
            )
        )
        score_clip_information["source"] = (
            score_clip_source if score_clip_override is not None else "checkpoint"
        )

    length_adapter = SpectrumLengthAdapter.from_metadata(metadata)
    _, diffusion = build_diffusion_model(
        model_configuration=checkpoint_configuration,
        sequence_length=length_adapter.padded_length,
    )
    diversity_enabled = bool(
        (checkpoint_configuration.get("diversity_constraints", {}) or {}).get(
            "enabled", False
        )
    )
    diversity_state = None
    if diversity_enabled:
        diversity_state = metadata.get("diversity_constraint_state")
        if not isinstance(diversity_state, dict):
            raise RuntimeError(
                "检查点启用了D4.3多样性约束，但缺少"
                "diversity_constraint_state。"
            )
        configure_diversity = getattr(
            diffusion, "configure_diversity_constraints", None
        )
        if not callable(configure_diversity):
            raise RuntimeError("当前扩散模型缺少D4.3多样性配置接口。")
        configure_diversity(
            diversity_constraint_state=diversity_state
        )
    model_source = str(
        arguments.model_source
        if arguments.model_source is not None
        else generation.get("model_source", "ema")
    ).strip().lower()
    if model_source not in {"raw", "ema"}:
        raise ValueError("model_source必须为raw或ema。")
    _load_model_state(diffusion, checkpoint, model_source)
    device = _device(configuration, arguments.device)
    diffusion = diffusion.to(device).eval()

    number = int(
        arguments.number
        if arguments.number is not None
        else generation["number_of_spectra"]
    )
    if number <= 0:
        raise ValueError("生成数量必须大于0。")
    batch_size = int(generation["batch_size"])
    if batch_size <= 0:
        raise ValueError("generation.batch_size必须大于0。")

    conditioning_metadata = metadata.get("conditioning_metadata")
    if not isinstance(conditioning_metadata, dict):
        raise RuntimeError("检查点缺少conditioning_metadata。")
    source_named_conditions = source_named_conditions_from_metadata(
        conditioning_metadata
    )
    requested = str(arguments.condition).strip()
    condition_queries = (
        [record[0] for record in source_named_conditions]
        if requested.lower() == "all"
        else [Path(requested).stem]
    )

    normalizer = None
    if bool(generation.get("inverse_normalize", True)):
        state = metadata.get("normalization_state")
        if not isinstance(state, dict):
            raise RuntimeError("检查点缺少normalization_state。")
        normalizer = GlobalMinMaxNormalizer.from_state_dict(state)

    mean_configuration = generation.get(
        "mean_fidelity_calibration", {}
    ) or {}
    spread_configuration = generation.get(
        "pca_spread_calibration", {}
    ) or {}
    tail_configuration = generation.get("tail_calibration", {}) or {}
    envelope_configuration = generation.get(
        "intensity_envelope_guard", {}
    ) or {}
    mean_enabled = bool(mean_configuration.get("enabled", False))
    spread_enabled = bool(spread_configuration.get("enabled", False))
    tail_enabled = bool(tail_configuration.get("enabled", False))
    envelope_enabled = bool(envelope_configuration.get("enabled", False))
    condition_calibration_enabled = (
        mean_enabled or spread_enabled or tail_enabled or envelope_enabled
    )
    training_collection = None
    relative_source_files = None
    training_indices: set[int] = set()
    if condition_calibration_enabled:
        data = configuration["data"]
        training_collection = read_spectrum_collection(
            resolve_project_path(configuration, data["input_directory"]), data
        )
        relative_source_files = np.asarray(
            [str(value) for value in training_collection.relative_source_files],
            dtype=object,
        )
        if relative_source_files.tolist() != metadata.get("relative_source_files"):
            raise RuntimeError(
                "当前输入文件顺序与checkpoint训练时不一致，禁止拟合条件强度边界。"
            )
        training_indices = {
            int(value) for value in metadata.get("training_indices", [])
        }

    base_seed = int(
        configuration.get("random", {}).get(
            "seed", configuration.get("project", {}).get("random_seed", 2026)
        )
    )
    checkpoint_step = int(checkpoint.get("step", 0))
    formats = list(generation.get("output_formats", ["xlsx", "png"]))
    if arguments.output_directory is None:
        spectrum_root = resolve_project_path(
            configuration, output["generated_spectrum_directory"]
        )
        plot_root = resolve_project_path(
            configuration, output["preview_plot_directory"]
        )
    else:
        spectrum_root = Path(arguments.output_directory).expanduser()
        if not spectrum_root.is_absolute():
            spectrum_root = resolve_project_path(configuration, spectrum_root)
        spectrum_root = spectrum_root.resolve()
        plot_root = spectrum_root.parent / f"{spectrum_root.name}_plots"

    prior_conditioning_enabled = bool(
        (
            (checkpoint_configuration.get("conditioning", {}) or {}).get(
                "prior_spectrum", {}
            )
            or {}
        ).get("enabled", False)
    )
    cross_fit_metadata = (
        conditional_prior_bank.training_cross_fit_metadata()
        if conditional_prior_bank is not None
        else {"enabled": False, "method": "leave_one_out"}
    )
    require_cross_fit_checkpoint = bool(
        generation.get("require_training_cross_fit_checkpoint", False)
    )
    if require_cross_fit_checkpoint and not bool(cross_fit_metadata["enabled"]):
        raise RuntimeError(
            "当前生成配置要求D4.3.2.14 cross-fit checkpoint，"
            "但所加载checkpoint未启用training cross-fit。"
        )
    if bool(cross_fit_metadata["enabled"]):
        stage_name = "D4.3.2.14交叉拟合PCA先验残差"
    elif prior_conditioning_enabled:
        stage_name = "D4.3.2.10先验条件化残差"
    elif diversity_enabled:
        stage_name = "D4.3条件感知多样性先验残差"
    elif conditional_prior_bank:
        stage_name = "D4.2条件先验残差"
    else:
        stage_name = "D4.1条件基线"
    print(f"\n===== {stage_name}生成 =====")
    print(f"检查点：{checkpoint_path}")
    print(f"检查点step：{checkpoint_step}")
    print(f"模型来源：{model_source}；设备：{device}")
    print(f"本次条件数量：{len(condition_queries)}；每个条件生成：{number}条")
    print(
        "本次实际抽取的先验谱输入U-Net："
        f"{prior_conditioning_enabled}"
    )
    print(
        "training PCA cross-fit checkpoint："
        f"{cross_fit_metadata['enabled']}；"
        f"method={cross_fit_metadata['method']}"
    )

    manifest = {
        "schema_version": "d4.2_source_named_generation_v1",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_step": checkpoint_step,
        "model_source": model_source,
        "number_of_spectra_per_condition": number,
        "requested_condition": requested,
        "prior_spectrum_conditioning_enabled": prior_conditioning_enabled,
        "training_cross_fit": cross_fit_metadata,
        "require_training_cross_fit_checkpoint": require_cross_fit_checkpoint,
        "pca_score_clip_runtime_override": score_clip_information,
        "tail_calibration_configuration": (
            tail_configuration if tail_enabled else None
        ),
        "mean_fidelity_calibration_configuration": (
            mean_configuration if mean_enabled else None
        ),
        "pca_spread_calibration_configuration": (
            spread_configuration if spread_enabled else None
        ),
        "intensity_envelope_guard_configuration": (
            envelope_configuration if envelope_enabled else None
        ),
        "conditions": [],
    }

    if score_clip_information is not None:
        print(
            "PCA先验采样范围："
            f"checkpoint outer={score_clip_information['checkpoint_outer_values']}；"
            f"broad={score_clip_information['checkpoint_broad_values']}；"
            f"runtime={score_clip_information['runtime_value']}；"
            f"来源={score_clip_information['source']}"
        )
    for condition_index, query in enumerate(condition_queries):
        (
            source_condition_name,
            condition_id,
            vector,
            source_file,
        ) = resolve_source_named_condition_from_metadata(
            conditioning_metadata,
            query,
        )
        output_axis, profile_id = _axis_for_source(metadata, source_file)
        set_random_seed(random_seed=base_seed + condition_index)
        scaled_residual_batches = [] if diversity_enabled else None
        spectra = generate_spectra(
            diffusion=diffusion,
            number_of_spectra=number,
            generation_batch_size=batch_size,
            device=device,
            length_adapter=length_adapter,
            output_raman_shifts=output_axis,
            condition_vector=vector,
            condition_id=condition_id,
            conditional_prior_residual_bank=conditional_prior_bank,
            prior_random_seed=base_seed + condition_index,
            generated_scaled_residual_batches=scaled_residual_batches,
        )
        diversity_diagnostics = None
        if scaled_residual_batches is not None:
            if diversity_state is None or not scaled_residual_batches:
                raise RuntimeError("D4.3生成缺少多样性状态或scaled residual。")
            diversity_diagnostics = (
                evaluate_condition_aware_diversity_residuals(
                    generated_scaled_residuals=np.concatenate(
                        scaled_residual_batches,
                        axis=0,
                    ),
                    condition_vector=vector,
                    diversity_constraint_state=diversity_state,
                )
            )
        if normalizer is not None:
            spectra = normalizer.inverse_transform(spectra)
        mean_diagnostics = None
        spread_diagnostics = None
        tail_diagnostics = None
        tail_profile_diagnostics = None
        envelope_diagnostics = None
        if condition_calibration_enabled:
            if training_collection is None or relative_source_files is None:
                raise RuntimeError("条件强度校准缺少训练数据。")
            condition_training = _condition_training_spectra(
                collection=training_collection,
                relative_source_files=relative_source_files,
                training_indices=training_indices,
                source_file=source_file,
                output_axis=output_axis,
            )
        if spread_enabled:
            spectra, spread_diagnostics = (
                apply_condition_pca_spread_calibration(
                    spectra,
                    condition_training,
                    configuration=spread_configuration,
                )
            )
        if tail_enabled:
            (
                condition_tail_configuration,
                tail_profile_diagnostics,
            ) = resolve_condition_tail_calibration_configuration(
                tail_configuration,
                point_count=int(output_axis.size),
                condition_name=source_condition_name,
            )
            condition_tail_configuration["random_seed"] = int(
                tail_configuration.get("random_seed", base_seed)
            ) + condition_index
            spectra, tail_diagnostics = (
                apply_condition_oracle_tail_calibration(
                    spectra,
                    condition_training,
                    configuration=condition_tail_configuration,
                )
            )
        # Shape/spread changes can reintroduce a small pointwise mean bias.
        # Run the bounded translation-only mean correction afterwards; it
        # leaves pairwise differences unchanged and improves final fidelity.
        if mean_enabled:
            spectra, mean_diagnostics = (
                apply_condition_mean_fidelity_calibration(
                    spectra,
                    condition_training,
                    configuration=mean_configuration,
                )
            )
        if envelope_enabled:
            spectra, envelope_diagnostics = (
                apply_condition_intensity_envelope_guard(
                    spectra,
                    condition_training,
                    configuration=envelope_configuration,
                    raman_shift=output_axis,
                )
            )
        condition_spectrum_directory = spectrum_root / source_condition_name
        condition_plot_directory = plot_root / source_condition_name
        paths = export_generated_spectra(
            raman_shift=output_axis,
            spectra=np.asarray(spectra, dtype=np.float32),
            spectrum_output_directory=condition_spectrum_directory,
            plot_output_directory=condition_plot_directory,
            base_name=f"{source_condition_name}_generated",
            output_formats=formats,
        )
        condition_record = {
            "source_condition_name": source_condition_name,
            "internal_condition_id": condition_id,
            "relative_source_file": source_file,
            "raman_axis_profile_id": profile_id,
            "raman_start_cm1": float(output_axis[0]),
            "raman_end_cm1": float(output_axis[-1]),
            "point_count": int(output_axis.size),
            "output_files": [str(Path(path).resolve()) for path in paths],
        }
        if diversity_diagnostics is not None:
            condition_record["d4_3_diversity_diagnostics"] = (
                diversity_diagnostics
            )
        if mean_diagnostics is not None:
            condition_record["mean_fidelity_calibration"] = mean_diagnostics
        if spread_diagnostics is not None:
            condition_record["pca_spread_calibration"] = spread_diagnostics
        if tail_diagnostics is not None:
            condition_record["oracle_tail_calibration"] = tail_diagnostics
        if tail_profile_diagnostics is not None:
            condition_record["tail_calibration_profile"] = (
                tail_profile_diagnostics
            )
        if envelope_diagnostics is not None:
            condition_record["intensity_envelope_guard"] = envelope_diagnostics
        manifest["conditions"].append(condition_record)
        print(
            f"[{condition_index + 1}/{len(condition_queries)}] "
            f"{source_condition_name}："
            f"{output_axis[0]:.0f}–{output_axis[-1]:.0f} cm^-1，"
            f"{output_axis.size}点，axis={profile_id}"
        )
        if source_condition_name != condition_id:
            print(f"  模型内部条件ID：{condition_id}")
        if diversity_diagnostics is not None:
            print(
                "  D4.3同域诊断：distance_ratio="
                f"{diversity_diagnostics['pairwise_distance_ratio']:.4f}；"
                "active_std_ratio="
                f"{diversity_diagnostics['active_pointwise_std_ratio_median']:.4f}；"
                "corr_excess="
                f"{diversity_diagnostics['pairwise_correlation_excess']:.4f}"
            )
        if mean_diagnostics is not None:
            print(
                "  条件均值校准：bias RMSE="
                f"{mean_diagnostics['mean_bias_rmse_before']:.4f} -> "
                f"{mean_diagnostics['mean_bias_rmse_after']:.4f}；"
                "修正Raman点比例="
                f"{mean_diagnostics['modified_raman_fraction']:.3%}"
            )
        if spread_diagnostics is not None:
            print(
                "  training-PCA分布宽度恢复：component std ratio="
                f"{spread_diagnostics['component_std_ratio_median_before']:.4f} -> "
                f"{spread_diagnostics['component_std_ratio_median_after']:.4f}；"
                "mean-pairwise-MSE ratio="
                f"{spread_diagnostics['mean_pairwise_mse_ratio_before']:.4f} -> "
                f"{spread_diagnostics['mean_pairwise_mse_ratio_after']:.4f}；"
                "扩展PC数="
                f"{spread_diagnostics['expanded_component_count']}；"
                "修正RMSE="
                f"{spread_diagnostics['correction_rmse']:.4f}"
            )
        if tail_diagnostics is not None:
            if tail_profile_diagnostics is not None:
                print(
                    "  QQ分层配置："
                    f"{tail_profile_diagnostics['selected_profiles']}；"
                    "strength="
                    f"{tail_profile_diagnostics['effective_correction_strength']:.4f}；"
                    "max|delta-z|="
                    f"{tail_profile_diagnostics['effective_maximum_absolute_correction_z']:.4f}"
                )
            print(
                "  QQ尾部校准：lower ratio="
                f"{tail_diagnostics['lower_tail_ratio_before']:.4f} -> "
                f"{tail_diagnostics['lower_tail_ratio_after']:.4f}；"
                "upper ratio="
                f"{tail_diagnostics['upper_tail_ratio_before']:.4f} -> "
                f"{tail_diagnostics['upper_tail_ratio_after']:.4f}；"
                "lower/upper触发="
                f"{tail_diagnostics['lower_tail_calibration_activated']}/"
                f"{tail_diagnostics['upper_tail_calibration_activated']}；"
                "修改点比例="
                f"{tail_diagnostics['modified_point_fraction']:.4%}"
            )
            if tail_diagnostics.get("strategy") == "oracle_piecewise_quantile":
                before = tail_diagnostics["segment_span_ratio_before"]
                after = tail_diagnostics["segment_span_ratio_after"]
                active = tail_diagnostics["segment_calibration_activated"]
                print(
                    "  QQ四段比率[1-10,10-50,50-90,90-99]："
                    f"{[round(float(value), 4) for value in before]} -> "
                    f"{[round(float(value), 4) for value in after]}；"
                    f"触发={active}；"
                    "最大|delta-z|="
                    f"{tail_diagnostics['maximum_absolute_correction_z_used']:.4f}"
                )
                print(
                    "  QQ后多样性保护：pairwise-MSE ratio="
                    f"{tail_diagnostics['pairwise_mse_ratio_before']:.4f} -> "
                    f"{tail_diagnostics['pairwise_mse_ratio_unguarded']:.4f}"
                    "(未保护) -> "
                    f"{tail_diagnostics['pairwise_mse_ratio_after']:.4f}；"
                    "要求≥"
                    f"{tail_diagnostics['pairwise_mse_ratio_required']:.4f}；"
                    "QQ修正保留系数="
                    f"{tail_diagnostics['calibration_blend_factor']:.4f}；"
                    "触发="
                    f"{tail_diagnostics['diversity_guard_activated']}"
                )
        if envelope_diagnostics is not None:
            envelope_mode = str(
                envelope_diagnostics.get(
                    "mode",
                    "legacy_intensity_envelope",
                )
            )

            if envelope_mode == "local_residual_negative_valley":
                print(
                    "  D4.3.2.14a局部残差负谷保护："
                    "候选/实际回拉点="
                    f"{envelope_diagnostics['negative_valley_candidate_point_count']}/"
                    f"{envelope_diagnostics['negative_valley_retained_point_count']}；"
                    "修改光谱数="
                    f"{envelope_diagnostics['modified_spectrum_count']}；"
                    "最小值="
                    f"{envelope_diagnostics['minimum_before']:.3f} -> "
                    f"{envelope_diagnostics['minimum_after']:.3f}；"
                    "local floor(min/median)="
                    f"{envelope_diagnostics['local_floor_minimum']:.3f}/"
                    f"{envelope_diagnostics['local_floor_median']:.3f}；"
                    "最小连续点="
                    f"{envelope_diagnostics['minimum_negative_valley_contiguous_points']}"
                )

            elif (
                envelope_mode
                == "pointwise_shape_preserving_negative_valley"
            ):
                print(
                    "  D4.3.2.15a逐Raman保形负谷保护："
                    "候选/实际回拉点="
                    f"{envelope_diagnostics['negative_valley_candidate_point_count']}/"
                    f"{envelope_diagnostics['negative_valley_retained_point_count']}；"
                    "修改光谱数="
                    f"{envelope_diagnostics['modified_spectrum_count']}；"
                    "修改点比例="
                    f"{envelope_diagnostics['modified_point_fraction']:.4%}；"
                    "最小值="
                    f"{envelope_diagnostics['minimum_before']:.3f} -> "
                    f"{envelope_diagnostics['minimum_after']:.3f}；"
                    "pointwise floor(min/median)="
                    f"{envelope_diagnostics['pointwise_floor_minimum']:.3f}/"
                    f"{envelope_diagnostics['pointwise_floor_median']:.3f}；"
                    "raw保留比例="
                    f"{envelope_diagnostics['shape_preserving_raw_retention']:.2f}；"
                    "最大修正="
                    f"{envelope_diagnostics['maximum_absolute_correction']:.3f}；"
                    "最小连续点="
                    f"{envelope_diagnostics['minimum_negative_valley_contiguous_points']}"
                )

            else:
                print(
                    "  逐点强度包络：lower/upper修改点="
                    f"{envelope_diagnostics['lower_modified_point_count']}/"
                    f"{envelope_diagnostics['upper_modified_point_count']}；"
                    "最小值="
                    f"{envelope_diagnostics['minimum_before']:.3f} -> "
                    f"{envelope_diagnostics['minimum_after']:.3f}；"
                    "最大值="
                    f"{envelope_diagnostics['maximum_before']:.3f} -> "
                    f"{envelope_diagnostics['maximum_after']:.3f}"
                )
                print(
                    "  条件负背景深谷保护：training背景下限="
                    f"{envelope_diagnostics['condition_negative_background_floor']:.3f}；"
                    "候选/实际回拉点="
                    f"{envelope_diagnostics['negative_valley_candidate_point_count']}/"
                    f"{envelope_diagnostics['negative_valley_retained_point_count']}；"
                    "最小连续点="
                    f"{envelope_diagnostics['minimum_negative_valley_contiguous_points']}"
                )
        for path in paths:
            print(f"  已写出：{path}")

    spectrum_root.mkdir(parents=True, exist_ok=True)
    manifest_path = spectrum_root / "generation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"生成清单：{manifest_path}")
    print(f"===== {stage_name}生成完成 =====")


if __name__ == "__main__":
    main()
