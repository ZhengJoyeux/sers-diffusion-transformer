"""Evaluate D4.1 test loss by Raman range and condition permutation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.checkpoint_manager import load_checkpoint_file
from src.conditional_prior_residual import ConditionalPriorResidualBank
from src.configuration_loader import load_configuration, resolve_project_path
from src.intensity_normalizer import GlobalMinMaxNormalizer
from src.model_builder import build_diffusion_model
from src.spectrum_conditioning import (
    condition_ids_from_source_files,
    encode_source_file_conditions,
)
from src.spectrum_file_reader import read_spectrum_collection
from src.spectrum_length_adapter import SpectrumLengthAdapter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按Raman范围报告D4.1固定测试集损失和条件置乱对照。"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-source", choices=("raw", "ema"), default="ema")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=202641)
    arguments = parser.parse_args()

    configuration = load_configuration(arguments.config)
    checkpoint_path = Path(arguments.checkpoint).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = resolve_project_path(configuration, checkpoint_path)
    checkpoint = load_checkpoint_file(checkpoint_path, map_location="cpu")
    metadata = checkpoint.get("metadata")
    checkpoint_configuration = checkpoint.get("configuration")
    if not isinstance(metadata, dict) or not isinstance(
        checkpoint_configuration, dict
    ):
        raise RuntimeError("检查点缺少metadata或configuration。")
    condition_meta = metadata.get("conditioning_metadata")
    if not isinstance(condition_meta, dict):
        raise RuntimeError("检查点不是D4.1条件检查点。")
    if len(condition_meta.get("conditions", {})) != 126:
        raise RuntimeError("检查点不包含预期的126个条件。")

    data = configuration["data"]
    collection = read_spectrum_collection(
        resolve_project_path(configuration, data["input_directory"]), data
    )
    if [str(v) for v in collection.relative_source_files] != metadata.get(
        "relative_source_files"
    ):
        raise RuntimeError("当前输入文件顺序与检查点训练时不一致。")
    adapter = SpectrumLengthAdapter.from_metadata(metadata)
    spectra, masks = adapter.interpolate_to_model_axis_with_mask(
        collection.spectra, collection.raman_shifts
    )
    normalizer_state = metadata.get("normalization_state")
    if not isinstance(normalizer_state, dict):
        raise RuntimeError("检查点缺少normalization_state。")
    normalizer = GlobalMinMaxNormalizer.from_state_dict(normalizer_state)
    spectra = normalizer.transform(spectra, valid_mask=masks)
    conditional_prior_state = metadata.get("conditional_prior_residual_state")
    stage_name = "D4.1"
    if isinstance(conditional_prior_state, dict):
        stage_name = "D4.2"
        bank = ConditionalPriorResidualBank.from_state_dict(
            conditional_prior_state
        )
        spectra = bank.transform(
            spectra,
            valid_masks=masks,
            condition_ids=condition_ids_from_source_files(
                collection.relative_source_files
            ),
        )
    spectra = adapter.adapt(spectra).astype(np.float32, copy=False)
    masks = adapter.adapt_valid_mask(masks).astype(np.float32, copy=False)
    spectra *= masks
    conditions = encode_source_file_conditions(collection.relative_source_files)

    test_indices = np.asarray(metadata.get("test_indices"), dtype=np.int64)
    if test_indices.size != 504 or np.unique(test_indices).size != 504:
        raise RuntimeError("检查点固定测试集不是预期的504条。")
    test_spectra = spectra[test_indices]
    test_masks = masks[test_indices]
    test_conditions = conditions[test_indices]
    # 每个源文件有连续4条测试谱，滚动4条可确保换成另一组合条件。
    permuted_conditions = np.roll(test_conditions, shift=4, axis=0)

    _, diffusion = build_diffusion_model(
        checkpoint_configuration, adapter.padded_length
    )
    if arguments.model_source == "raw":
        state = checkpoint.get("diffusion_state")
    else:
        ema = checkpoint.get("ema_state")
        state = ema.get("ema_model") if isinstance(ema, dict) else None
    if not isinstance(state, dict):
        raise RuntimeError("检查点缺少所选模型状态。")
    diffusion.load_state_dict(state)
    device_text = arguments.device or configuration["training"].get("device", "cpu")
    device = torch.device(str(device_text).strip().lower())
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("要求使用CUDA，但当前不可用。")
    diffusion = diffusion.to(device).eval()

    torch.manual_seed(arguments.seed)
    correct_parts: list[np.ndarray] = []
    permuted_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, test_indices.size, arguments.batch_size):
            stop = min(start + arguments.batch_size, test_indices.size)
            x = torch.from_numpy(test_spectra[start:stop]).unsqueeze(1).to(device)
            mask = torch.from_numpy(test_masks[start:stop]).unsqueeze(1).to(device)
            condition = torch.from_numpy(test_conditions[start:stop]).to(device)
            wrong = torch.from_numpy(permuted_conditions[start:stop].copy()).to(device)
            timestep = torch.randint(
                0, diffusion.num_timesteps, (stop - start,), device=device
            )
            noise = torch.randn_like(x) * mask
            correct = diffusion.p_losses(
                x,
                timestep,
                noise=noise,
                valid_mask=mask,
                condition=condition,
                loss_reduction="none",
            )
            permuted = diffusion.p_losses(
                x,
                timestep,
                noise=noise,
                valid_mask=mask,
                condition=wrong,
                loss_reduction="none",
            )
            correct_parts.append(correct.cpu().numpy())
            permuted_parts.append(permuted.cpu().numpy())

    correct_loss = np.concatenate(correct_parts)
    permuted_loss = np.concatenate(permuted_parts)
    valid_counts = test_masks.sum(axis=1).astype(np.int64)
    if not np.isfinite(correct_loss).all() or not np.isfinite(permuted_loss).all():
        raise RuntimeError("测试损失出现NaN或无穷值。")

    print(f"===== {stage_name} 固定测试集评估 =====")
    print(f"检查点：{checkpoint_path}；step={int(checkpoint.get('step', 0))}")
    print(f"模型：{arguments.model_source}；测试光谱：{test_indices.size}")
    for valid_count in sorted(np.unique(valid_counts)):
        selected = valid_counts == valid_count
        print(
            f"有效点数{valid_count}（{selected.sum()}条）："
            f"正确条件loss={correct_loss[selected].mean():.8g}；"
            f"置乱条件loss={permuted_loss[selected].mean():.8g}"
        )
    correct_mean = float(correct_loss.mean())
    permuted_mean = float(permuted_loss.mean())
    gap = (permuted_mean - correct_mean) / max(correct_mean, 1.0e-12) * 100.0
    print(f"全部：正确条件loss={correct_mean:.8g}")
    print(f"全部：置乱条件loss={permuted_mean:.8g}")
    print(f"条件置乱相对损失增幅：{gap:.3f}%")
    print("解释：两种有效点数loss均有限，证明两种Raman范围都进入同一模型。")
    print("若置乱条件loss明显更高，说明模型确实在利用条件而非忽略条件。")


if __name__ == "__main__":
    main()
