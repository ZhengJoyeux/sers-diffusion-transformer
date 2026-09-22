"""Verify that D4.2 output names exactly preserve input file stems."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="严格检查D4.2输入文件名、输出文件夹和输出Excel文件名。"
    )
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--generated-directory", type=Path, required=True)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--expected-step", type=int)
    return parser.parse_args()


def input_condition_names(root: Path) -> set[str]:
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".xlsx", ".xls"}
    )
    names: set[str] = set()
    duplicates: list[str] = []
    for path in paths:
        if path.stem in names:
            duplicates.append(path.stem)
        names.add(path.stem)
    if duplicates:
        raise ValueError(f"输入文件名不唯一：{sorted(set(duplicates))}")
    if not names:
        raise ValueError(f"输入目录没有Excel文件：{root}")
    return names


def verify_generated_names(
    input_directory: Path,
    generated_directory: Path,
    expected_count: int | None = None,
    expected_step: int | None = None,
) -> tuple[int, int]:
    input_directory = input_directory.resolve()
    generated_directory = generated_directory.resolve()
    names = input_condition_names(input_directory)
    if expected_count is not None and len(names) != expected_count:
        raise ValueError(
            f"输入条件数应为{expected_count}，实际为{len(names)}。"
        )

    expected_files = {
        (generated_directory / name / f"{name}_generated.xlsx").resolve()
        for name in names
    }
    actual_files = {
        path.resolve()
        for path in generated_directory.rglob("*.xlsx")
        if path.is_file()
    }
    missing = sorted(expected_files.difference(actual_files))
    unexpected = sorted(actual_files.difference(expected_files))
    if missing or unexpected:
        details = []
        if missing:
            details.append(
                "缺少文件：\n" + "\n".join(str(path) for path in missing)
            )
        if unexpected:
            details.append(
                "额外或错误命名文件：\n"
                + "\n".join(str(path) for path in unexpected)
            )
        raise ValueError("\n\n".join(details))

    manifest_path = generated_directory / "generation_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"缺少生成清单：{manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_step = int(manifest["checkpoint_step"])
        records = manifest["conditions"]
        manifest_names = {
            str(record["source_condition_name"])
            for record in records
        }
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        raise ValueError(f"生成清单无效：{manifest_path}；{error}") from error
    if expected_step is not None and manifest_step != expected_step:
        raise ValueError(
            f"生成清单step应为{expected_step}，实际为{manifest_step}。"
        )
    if manifest_names != names:
        missing_manifest = sorted(names.difference(manifest_names))
        extra_manifest = sorted(manifest_names.difference(names))
        raise ValueError(
            f"生成清单条件名不完整；缺少={missing_manifest}；"
            f"额外={extra_manifest}。"
        )
    if len(records) != len(names):
        raise ValueError(
            f"生成清单记录数应为{len(names)}，实际为{len(records)}。"
        )
    return len(names), manifest_step


def main() -> None:
    arguments = parse_arguments()
    count, step = verify_generated_names(
        arguments.input_directory,
        arguments.generated_directory,
        expected_count=arguments.expected_count,
        expected_step=arguments.expected_step,
    )
    print("===== D4.2生成命名检查通过 =====")
    print(f"输入条件/严格匹配输出：{count}/{count}")
    print("命名规则：<输入文件名>/<输入文件名>_generated.xlsx")
    print("农药顺序：与输入文件名完全一致，未重排")
    print(f"生成检查点step：{step}")


if __name__ == "__main__":
    main()
