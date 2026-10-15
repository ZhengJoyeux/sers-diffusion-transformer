"""Regression tests for exact source-name generation output."""

from __future__ import annotations

import pytest

from src.spectrum_conditioning import (
    build_conditioning_metadata,
    resolve_source_named_condition_from_metadata,
    source_named_conditions_from_metadata,
)


def _metadata() -> dict:
    return build_conditioning_metadata(
        [
            "CHL_TEB/CHL-H_TEB-M_water.xlsx",
            "DEL/DEL-S_soil.xlsx",
        ]
    )


def test_source_name_is_public_and_canonical_id_stays_internal() -> None:
    metadata = _metadata()
    records = source_named_conditions_from_metadata(metadata)
    assert [record[0] for record in records] == [
        "CHL-H_TEB-M_water",
        "DEL-S_soil",
    ]

    source_name, internal_id, vector, source_file = (
        resolve_source_named_condition_from_metadata(
            metadata,
            "CHL-H_TEB-M_water",
        )
    )
    assert source_name == "CHL-H_TEB-M_water"
    assert internal_id == "TEB-M_CHL-H_water"
    assert vector.shape == (14,)
    assert source_file == "CHL_TEB/CHL-H_TEB-M_water.xlsx"


def test_canonical_alias_cannot_replace_original_source_name() -> None:
    with pytest.raises(ValueError, match="不存在原始条件名"):
        resolve_source_named_condition_from_metadata(
            _metadata(),
            "TEB-M_CHL-H_water",
        )
