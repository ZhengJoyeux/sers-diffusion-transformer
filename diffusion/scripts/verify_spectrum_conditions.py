"""Audit filename-derived D4.1 conditions and fixed 12/4/4 membership."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import numpy as np

from src.configuration_loader import load_configuration, resolve_project_path
from src.dataset_splitter import split_spectrum_collection
from src.spectrum_conditioning import (
    CONDITION_VECTOR_SIZE,
    build_conditioning_metadata,
    encode_source_file_conditions,
    parse_condition_from_source_file,
)
from src.spectrum_file_reader import read_spectrum_collection


def main() -> None:
    parser = argparse.ArgumentParser(description="检查D4.1文件名条件和12/4/4划分。")
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-files", type=int, default=126)
    parser.add_argument("--spectra-per-file", type=int, default=20)
    arguments = parser.parse_args()

    configuration = load_configuration(arguments.config)
    data = configuration["data"]
    collection = read_spectrum_collection(
        resolve_project_path(configuration, data["input_directory"]), data
    )
    metadata = build_conditioning_metadata(collection.relative_source_files)
    vectors = encode_source_file_conditions(collection.relative_source_files)
    split = split_spectrum_collection(
        collection=collection,
        data_config=data,
        random_seed=int(configuration["project"].get("random_seed", 2026)),
    )

    source_counts = Counter(str(value) for value in collection.relative_source_files)
    if len(source_counts) != arguments.expected_files:
        raise RuntimeError(
            f"源文件数为{len(source_counts)}，期望{arguments.expected_files}。"
        )
    bad_counts = {
        key: value
        for key, value in source_counts.items()
        if value != arguments.spectra_per_file
    }
    if bad_counts:
        raise RuntimeError(f"以下源文件光谱数不正确：{bad_counts}")
    if vectors.shape != (len(collection.spectra), CONDITION_VECTOR_SIZE):
        raise RuntimeError(f"条件矩阵形状错误：{vectors.shape}")

    subset_names = {
        "train": np.asarray(split.train.indices, dtype=np.int64),
        "validation": np.asarray(split.validation.indices, dtype=np.int64),
        "test": np.asarray(split.test.indices, dtype=np.int64),
    }
    expected_subset_counts = {"train": 12, "validation": 4, "test": 4}
    for name, indices in subset_names.items():
        per_source = Counter(
            str(collection.relative_source_files[index]) for index in indices
        )
        if set(per_source) != set(source_counts) or any(
            count != expected_subset_counts[name] for count in per_source.values()
        ):
            raise RuntimeError(f"{name}没有对每个条件保持固定数量。")

    design_counts = Counter()
    matrix_counts = Counter()
    axis_by_condition: dict[str, set[tuple[float, float, int]]] = defaultdict(set)
    for index, source in enumerate(collection.relative_source_files):
        condition = parse_condition_from_source_file(source)
        present = sum(level != "0" for level in condition.levels)
        design_counts[present] += 1
        matrix_counts[condition.matrix] += 1
        axis = np.asarray(collection.raman_shifts[index], dtype=np.float64)
        axis_by_condition[condition.condition_id].add(
            (float(axis[0]), float(axis[-1]), int(axis.size))
        )
    if any(len(profiles) != 1 for profiles in axis_by_condition.values()):
        raise RuntimeError("同一条件内部出现多个Raman轴。")

    print("===== D4.1 条件与固定划分检查 =====")
    print(f"源文件/唯一条件：{len(source_counts)}/{len(metadata['conditions'])}")
    print(f"总光谱/条件矩阵：{len(collection.spectra)}/{vectors.shape}")
    print(
        "训练/验证/测试："
        f"{len(subset_names['train'])}/{len(subset_names['validation'])}/"
        f"{len(subset_names['test'])}"
    )
    print("每个条件固定为12/4/4：通过")
    print(
        "单组分/双组分/三组分条件数："
        f"{design_counts[1] // arguments.spectra_per_file}/"
        f"{design_counts[2] // arguments.spectra_per_file}/"
        f"{design_counts[3] // arguments.spectra_per_file}"
    )
    print(
        "water/soil条件数："
        f"{matrix_counts['water'] // arguments.spectra_per_file}/"
        f"{matrix_counts['soil'] // arguments.spectra_per_file}"
    )
    print("文件名解析、14维编码、条件覆盖和轴映射全部通过。")


if __name__ == "__main__":
    main()
