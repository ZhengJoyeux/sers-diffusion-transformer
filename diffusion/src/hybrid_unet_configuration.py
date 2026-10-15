"""Pure-Python validation for the optional D4.26 bottleneck adapter."""
from __future__ import annotations

import math


def normalize_hybrid_configuration(raw=None):
    if raw is None:
        return {"enabled": False}
    if not isinstance(raw, dict):
        raise TypeError("model.bottleneck_transformer必须是字典。")
    allowed = {"enabled", "cross_attention", "depth", "num_heads", "ff_multiplier",
               "dropout", "position_frequencies", "raman_start_cm1", "raman_step_cm1",
               "raman_span_cm1", "schema_version"}
    if set(raw) - allowed:
        raise ValueError(f"未知bottleneck_transformer配置: {sorted(set(raw)-allowed)}")
    for key in ("enabled", "cross_attention"):
        if key in raw and not isinstance(raw[key], bool):
            raise TypeError(f"bottleneck_transformer.{key}必须是YAML布尔值。")
    if not raw.get("enabled", False):
        return {"enabled": False}
    result = {"enabled": True, "cross_attention": True, "depth": 1, "num_heads": 4,
              "ff_multiplier": 2.0, "dropout": 0.05, "position_frequencies": 8,
              "raman_start_cm1": 600.0, "raman_step_cm1": 1.0,
              "raman_span_cm1": 1900.0, "schema_version": 1}
    result.update(raw)
    for key, low, high in (("depth", 1, 4), ("num_heads", 1, 16),
                           ("position_frequencies", 1, 12), ("schema_version", 1, 1)):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"bottleneck_transformer.{key}必须为[{low},{high}]内整数。")
    for key in ("ff_multiplier", "dropout", "raman_start_cm1", "raman_step_cm1", "raman_span_cm1"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"bottleneck_transformer.{key}必须为有限数值。")
        result[key] = float(value)
    if not 1 <= result["ff_multiplier"] <= 4:
        raise ValueError("ff_multiplier必须在[1,4]。")
    if not 0 <= result["dropout"] < 1:
        raise ValueError("dropout必须在[0,1)。")
    if result["raman_step_cm1"] <= 0 or result["raman_span_cm1"] <= 0:
        raise ValueError("Raman步长和编码范围必须为正数。")
    return result


def validate_hybrid_context(configuration):
    architecture = configuration.get("model", configuration)
    hybrid = normalize_hybrid_configuration(architecture.get("bottleneck_transformer"))
    if not hybrid["enabled"]:
        return hybrid
    condition = configuration.get("conditioning", {}) or {}
    data = configuration.get("data", {}) or {}
    if data.get("raman_axis_mode") != "union_with_valid_mask" or not condition.get("enabled", False):
        raise ValueError("瓶颈Transformer要求化学条件和union_with_valid_mask。")
    if hybrid["cross_attention"] and not (condition.get("prior_spectrum", {}) or {}).get("enabled", False):
        raise ValueError("底谱交叉注意力要求conditioning.prior_spectrum.enabled=true。")
    dim = int(architecture.get("base_dimension", architecture.get("model_dimension", 0)))
    mults = architecture.get("dimension_multipliers", ())
    if not mults or dim <= 0 or dim * int(mults[-1]) % hybrid["num_heads"]:
        raise ValueError("瓶颈通道数必须能被num_heads整除。")
    return hybrid
