"""Verify the experimentally fixed per-source train/validation/test split."""

from __future__ import annotations

import argparse

import numpy as np

from src.configuration_loader import (
    load_configuration,
    resolve_project_path,
)
from src.dataset_splitter import split_spectrum_collection
from src.spectrum_file_reader import read_spectrum_collection


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "检查每个源文件是否严格按照前12/中4/后4条光谱"
            "固定划分，不开始模型训练。"
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="YAML配置文件路径。",
    )
    return parser.parse_args()


def _count_unique(values: np.ndarray) -> int:
    return len({str(value) for value in values.tolist()})


def main() -> None:
    arguments = parse_arguments()
    configuration = load_configuration(arguments.config)
    data_config = configuration["data"]

    split_unit = str(data_config.get("split_unit", ""))
    if split_unit != "fixed_spectra_within_source_file":
        raise ValueError(
            "该检查脚本要求data.split_unit="
            "fixed_spectra_within_source_file，"
            f"当前为{split_unit!r}。"
        )

    input_directory = resolve_project_path(
        configuration,
        data_config["input_directory"],
    )
    collection = read_spectrum_collection(
        input_directory=input_directory,
        data_config=data_config,
    )
    dataset_split = split_spectrum_collection(
        collection=collection,
        data_config=data_config,
        random_seed=int(
            configuration.get("project", {}).get(
                "random_seed",
                2026,
            )
        ),
    )

    train_count = len(dataset_split.train.indices)
    validation_count = len(dataset_split.validation.indices)
    test_count = len(dataset_split.test.indices)
    total_count = len(collection.spectra)
    source_count = _count_unique(
        np.asarray(collection.source_files, dtype=object)
    )

    fixed = data_config["fixed_spectrum_split"]
    expected_train = source_count * int(fixed["train_count"])
    expected_validation = source_count * int(
        fixed["validation_count"]
    )
    expected_test = source_count * int(fixed["test_count"])

    actual_counts = (
        train_count,
        validation_count,
        test_count,
    )
    expected_counts = (
        expected_train,
        expected_validation,
        expected_test,
    )

    if actual_counts != expected_counts:
        raise RuntimeError(
            "固定划分数量与配置不一致："
            f"实际={actual_counts}，期望={expected_counts}。"
        )

    print("===== D4.0-A 固定数据集划分检查 =====")
    print(f"输入目录：{input_directory}")
    print(f"源文件数量：{source_count}")
    print(f"总光谱数量：{total_count}")
    print(f"训练集光谱数量：{train_count}")
    print(f"验证集光谱数量：{validation_count}")
    print(f"测试集光谱数量：{test_count}")
    print("划分规则：每个源文件前12/中4/后4（按配置计数）")
    print("随机种子和shuffle不会改变三个子集的成员。")
    print("===== 固定划分检查通过 =====")


if __name__ == "__main__":
    main()
