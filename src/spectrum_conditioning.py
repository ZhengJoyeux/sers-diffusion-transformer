"""D4.1 chemical-condition parsing and 14-dimensional encoding.

The condition is derived only from the source filename.  Folder names are
kept as descriptive labels, but are never used as the generation condition.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PESTICIDE_ORDER = ("DEL", "TEB", "CHL")
CONCENTRATION_STATES = ("0", "S", "M", "H")
MATRIX_ORDER = ("water", "soil")
CONDITION_VECTOR_SIZE = (
    len(PESTICIDE_ORDER) * len(CONCENTRATION_STATES)
    + len(MATRIX_ORDER)
)


@dataclass(frozen=True)
class SpectrumCondition:
    """One strictly parsed experimental condition."""

    levels: tuple[str, str, str]
    matrix: str

    @property
    def condition_id(self) -> str:
        parts = [
            f"{pesticide}-{level}"
            for pesticide, level in zip(
                PESTICIDE_ORDER,
                self.levels,
                strict=True,
            )
            if level != "0"
        ]
        parts.append(self.matrix)
        return "_".join(parts)

    def to_vector(self) -> np.ndarray:
        vector = np.zeros(CONDITION_VECTOR_SIZE, dtype=np.float32)
        for pesticide_index, level in enumerate(self.levels):
            state_index = CONCENTRATION_STATES.index(level)
            offset = pesticide_index * len(CONCENTRATION_STATES)
            vector[offset + state_index] = 1.0
        matrix_offset = len(PESTICIDE_ORDER) * len(
            CONCENTRATION_STATES
        )
        vector[matrix_offset + MATRIX_ORDER.index(self.matrix)] = 1.0
        return vector

    def to_metadata(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "levels": {
                pesticide: level
                for pesticide, level in zip(
                    PESTICIDE_ORDER,
                    self.levels,
                    strict=True,
                )
            },
            "matrix": self.matrix,
            "vector": self.to_vector().tolist(),
        }


def parse_condition_text(value: str) -> SpectrumCondition:
    """Parse ``DEL-H_TEB-S_CHL-M_soil`` in any pesticide order."""

    text = str(value).strip()
    if not text:
        raise ValueError("条件文本不能为空。")

    stem = Path(text).stem
    tokens = stem.split("_")
    if len(tokens) < 2:
        raise ValueError(
            f"无法从{value!r}解析条件；需要农药-浓度和末尾基质。"
        )

    matrix = tokens[-1].strip().lower()
    if matrix not in MATRIX_ORDER:
        raise ValueError(
            f"{value!r}的末尾基质必须是water或soil。"
        )

    parsed_levels: dict[str, str] = {}
    for token in tokens[:-1]:
        pieces = token.strip().upper().split("-")
        if len(pieces) != 2:
            raise ValueError(
                f"{value!r}中的{token!r}必须写成农药-浓度，例如DEL-H。"
            )
        pesticide, level = pieces
        if pesticide not in PESTICIDE_ORDER:
            raise ValueError(
                f"{value!r}含未知农药{pesticide!r}；"
                f"只允许{PESTICIDE_ORDER}。"
            )
        if level not in CONCENTRATION_STATES[1:]:
            raise ValueError(
                f"{value!r}中{pesticide}的浓度必须是S、M或H。"
            )
        if pesticide in parsed_levels:
            raise ValueError(f"{value!r}重复定义了{pesticide}。")
        parsed_levels[pesticide] = level

    if not parsed_levels:
        raise ValueError(f"{value!r}没有任何农药条件。")

    return SpectrumCondition(
        levels=tuple(
            parsed_levels.get(pesticide, "0")
            for pesticide in PESTICIDE_ORDER
        ),
        matrix=matrix,
    )


def parse_condition_from_source_file(
    relative_source_file: str | Path,
) -> SpectrumCondition:
    """Parse a condition from a source file basename, never its folder."""

    return parse_condition_text(Path(relative_source_file).name)


def encode_source_file_conditions(
    relative_source_files: Sequence[str | Path],
) -> np.ndarray:
    conditions = [
        parse_condition_from_source_file(value).to_vector()
        for value in relative_source_files
    ]
    if not conditions:
        raise ValueError("没有可编码的源文件条件。")
    return np.stack(conditions, axis=0).astype(np.float32, copy=False)


def condition_ids_from_source_files(
    relative_source_files: Sequence[str | Path],
) -> list[str]:
    """Return the canonical condition id for every spectrum/source record."""

    return [
        parse_condition_from_source_file(value).condition_id
        for value in relative_source_files
    ]


def build_conditioning_metadata(
    relative_source_files: Sequence[str | Path],
) -> dict[str, Any]:
    """Build one auditable record per unique source file/condition."""

    source_records: dict[str, dict[str, Any]] = {}
    condition_records: dict[str, dict[str, Any]] = {}

    for value in relative_source_files:
        source_file = Path(value).as_posix()
        condition = parse_condition_from_source_file(source_file)
        condition_id = condition.condition_id

        old_source = source_records.get(source_file)
        if old_source is not None:
            if old_source["condition_id"] != condition_id:
                raise ValueError(f"源文件{source_file}出现不一致条件。")
            continue

        if condition_id in condition_records:
            other = condition_records[condition_id]["relative_source_file"]
            raise ValueError(
                f"条件{condition_id}同时对应{other}和{source_file}；"
                "D4.1要求每个组合条件恰好对应一个源文件。"
            )

        record = condition.to_metadata()
        record["relative_source_file"] = source_file
        source_records[source_file] = {
            "condition_id": condition_id,
            "vector": record["vector"],
        }
        condition_records[condition_id] = record

    if not condition_records:
        raise ValueError("没有生成任何条件元数据。")

    return {
        "schema_version": "d4.1_condition_v1",
        "pesticide_order": list(PESTICIDE_ORDER),
        "concentration_states": list(CONCENTRATION_STATES),
        "matrix_order": list(MATRIX_ORDER),
        "vector_size": CONDITION_VECTOR_SIZE,
        "conditions": {
            key: condition_records[key]
            for key in sorted(condition_records)
        },
        "source_files": {
            key: source_records[key]
            for key in sorted(source_records)
        },
    }


def source_named_conditions_from_metadata(
    conditioning_metadata: dict[str, Any],
) -> list[tuple[str, str, str]]:
    """Return ``(source stem, internal id, source file)`` records.

    The source-file stem is the only public condition name.  The canonical
    condition id remains an internal key for condition vectors and priors.
    """

    if not isinstance(conditioning_metadata, dict):
        raise RuntimeError("检查点缺少有效的conditioning_metadata。")
    conditions = conditioning_metadata.get("conditions")
    if not isinstance(conditions, dict) or not conditions:
        raise RuntimeError("检查点中的conditions为空。")

    records: list[tuple[str, str, str]] = []
    seen_source_names: dict[str, str] = {}
    for condition_id, record in conditions.items():
        if not isinstance(record, dict):
            raise RuntimeError(f"条件{condition_id!r}的元数据无效。")
        source_file = str(record.get("relative_source_file", "")).strip()
        if not source_file:
            raise RuntimeError(f"条件{condition_id!r}缺少relative_source_file。")
        source_name = Path(source_file).stem.strip()
        if not source_name:
            raise RuntimeError(f"源文件{source_file!r}没有有效文件名。")
        parsed_id = parse_condition_text(source_name).condition_id
        if parsed_id != condition_id:
            raise RuntimeError(
                f"源文件名{source_name!r}解析为{parsed_id!r}，"
                f"但检查点记录为{condition_id!r}。"
            )
        previous = seen_source_names.get(source_name)
        if previous is not None:
            raise RuntimeError(
                f"原始条件名{source_name!r}同时对应{previous!r}和"
                f"{condition_id!r}。"
            )
        seen_source_names[source_name] = condition_id
        records.append((source_name, condition_id, source_file))
    return sorted(records, key=lambda item: item[0])


def resolve_source_named_condition_from_metadata(
    conditioning_metadata: dict[str, Any],
    requested_condition: str,
) -> tuple[str, str, np.ndarray, str]:
    """Resolve one condition by its exact original source-file stem."""

    requested_name = Path(str(requested_condition).strip()).stem
    records = source_named_conditions_from_metadata(conditioning_metadata)
    by_source_name = {
        source_name: (condition_id, source_file)
        for source_name, condition_id, source_file in records
    }
    if requested_name not in by_source_name:
        preview = "、".join(sorted(by_source_name)[:12])
        raise ValueError(
            f"检查点中不存在原始条件名{requested_name!r}。"
            f"前12个可用原始条件名：{preview}"
        )
    condition_id, source_file = by_source_name[requested_name]
    resolved_id, vector, resolved_source_file = resolve_condition_from_metadata(
        conditioning_metadata,
        condition_id,
    )
    if resolved_source_file != source_file:
        raise RuntimeError(
            f"条件{requested_name!r}的源文件映射前后不一致。"
        )
    return requested_name, resolved_id, vector, source_file


def resolve_condition_from_metadata(
    conditioning_metadata: dict[str, Any],
    requested_condition: str,
) -> tuple[str, np.ndarray, str]:
    """Return canonical id, vector and source file for generation."""

    if not isinstance(conditioning_metadata, dict):
        raise RuntimeError("检查点缺少有效的conditioning_metadata。")
    conditions = conditioning_metadata.get("conditions")
    if not isinstance(conditions, dict) or not conditions:
        raise RuntimeError("检查点中的conditions为空。")

    canonical = parse_condition_text(requested_condition).condition_id
    if canonical not in conditions:
        preview = "、".join(sorted(conditions)[:12])
        raise ValueError(
            f"检查点中不存在条件{canonical!r}。前12个可用条件：{preview}"
        )
    record = conditions[canonical]
    vector = np.asarray(record.get("vector"), dtype=np.float32).reshape(-1)
    expected_size = int(
        conditioning_metadata.get("vector_size", CONDITION_VECTOR_SIZE)
    )
    if vector.size != expected_size or not np.isfinite(vector).all():
        raise RuntimeError(f"条件{canonical}的向量无效。")
    source_file = str(record.get("relative_source_file", "")).strip()
    if not source_file:
        raise RuntimeError(f"条件{canonical}缺少relative_source_file。")
    return canonical, vector, source_file
