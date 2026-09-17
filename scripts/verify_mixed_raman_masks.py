"""只读检查联合Raman轴、有效区掩码和固定划分。"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from src.configuration_loader import load_configuration, resolve_project_path
from src.dataset_splitter import split_spectrum_collection
from src.spectrum_file_reader import read_spectrum_collection
from src.spectrum_length_adapter import SpectrumLengthAdapter


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检查混合Raman范围是否得到正确有效区掩码，不训练模型。"
    )
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def _profile_counts(
    masks: np.ndarray,
    indices: np.ndarray | None = None,
) -> str:
    selected = masks if indices is None else masks[indices]
    counts = Counter(
        int(value)
        for value in selected.sum(axis=1).astype(np.int64).tolist()
    )
    return "，".join(
        f"{length}个有效点×{counts[length]}条"
        for length in sorted(counts)
    )


def main() -> None:
    arguments = parse_arguments()
    configuration = load_configuration(arguments.config)
    data_config = configuration["data"]

    if data_config.get("raman_axis_mode") != "union_with_valid_mask":
        raise ValueError(
            "该脚本要求data.raman_axis_mode=union_with_valid_mask。"
        )

    collection = read_spectrum_collection(
        input_directory=resolve_project_path(
            configuration,
            data_config["input_directory"],
        ),
        data_config=data_config,
    )
    dataset_split = split_spectrum_collection(
        collection=collection,
        data_config=data_config,
        random_seed=int(configuration["project"]["random_seed"]),
    )
    adapter = SpectrumLengthAdapter.create(
        raman_shifts=collection.raman_shifts,
        dimension_multipliers=configuration["model"][
            "dimension_multipliers"
        ],
        model_length=data_config.get("model_spectrum_length", "auto"),
        padding_mode=data_config.get(
            "padding_mode", "right_zero_padding"
        ),
        padding_value=float(data_config.get("padding_value", 0.0)),
        raman_range_tolerance=float(
            data_config.get("raman_range_tolerance", 1.0)
        ),
        raman_axis_mode=data_config["raman_axis_mode"],
    )
    spectra, masks = adapter.interpolate_to_model_axis_with_mask(
        collection.spectra,
        collection.raman_shifts,
    )
    padded_masks = adapter.adapt_valid_mask(masks)

    if spectra.shape != masks.shape:
        raise RuntimeError("插值光谱和掩码形状不一致。")
    if not np.logical_or(masks == 0.0, masks == 1.0).all():
        raise RuntimeError("掩码不是严格的0/1数组。")
    if not np.all(spectra[masks == 0.0] == float(adapter.padding_value)):
        raise RuntimeError("发现无效区被插值或填入了真实强度。")
    if padded_masks.shape[1] != adapter.padded_length:
        raise RuntimeError("补齐掩码长度不正确。")
    if adapter.padding_size and not np.all(
        padded_masks[:, -adapter.padding_size :] == 0.0
    ):
        raise RuntimeError("网络补齐位置没有被掩码为0。")

    train_indices = np.asarray(dataset_split.train.indices, dtype=np.int64)
    validation_indices = np.asarray(
        dataset_split.validation.indices, dtype=np.int64
    )
    test_indices = np.asarray(dataset_split.test.indices, dtype=np.int64)

    print("===== D4.0-B 混合Raman轴掩码检查 =====")
    print(
        "联合物理轴："
        f"{adapter.model_axis[0]:.3f}–"
        f"{adapter.model_axis[-1]:.3f} cm^-1"
    )
    print(f"联合物理轴点数：{adapter.original_length}")
    print(f"U-Net输入长度：{adapter.padded_length}")
    print(f"网络右侧补齐点数：{adapter.padding_size}")
    print(f"全部数据：{_profile_counts(masks)}")
    print(f"训练集：{_profile_counts(masks, train_indices)}")
    print(f"验证集：{_profile_counts(masks, validation_indices)}")
    print(f"测试集：{_profile_counts(masks, test_indices)}")
    print("无效区强度：全部为padding_value，未发生尾部外推")
    print("网络补齐区：全部掩码为0")
    print("===== 混合Raman轴掩码检查通过 =====")


if __name__ == "__main__":
    main()
