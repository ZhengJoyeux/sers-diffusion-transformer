"""Train-only soft support at the export boundary (generation-only).

No generated-rank gate, contiguous-run gate, torch state, or learned parameter.
The caller supplies the same condition's 12 training spectra on the output axis.
"""
from __future__ import annotations

import numpy as np


def _arrays(generated, training):
    generated = np.asarray(generated, dtype=np.float64)
    training = np.asarray(training, dtype=np.float64)
    if (generated.ndim != 2 or training.ndim != 2
            or generated.shape[0] < 1 or training.shape[0] != 12
            or generated.shape[1] != training.shape[1]
            or generated.shape[1] < 2):
        raise ValueError("final delivery需要[N,L]生成谱和同轴[12,L]训练谱。")
    if not np.isfinite(generated).all() or not np.isfinite(training).all():
        raise ValueError("final delivery输入含NaN/Inf，禁止导出。")
    return generated, training


def _bounds(training, configuration):
    defaults = {
        "training_lower_quantile": 0.05,
        "training_upper_quantile": 0.95,
        "training_iqr_margin": 0.5,
        "training_extrema_margin_iqr_fraction": 0.1,
        "minimum_extrema_margin_intensity": 2.0,
        "activation_maximum_intensity": -18.0,
        "softness_iqr_fraction": 0.05,
        "minimum_softness_intensity": 0.25,
        "maximum_softness_intensity": 2.0,
        "global_extrema_margin_intensity": 2.0,
        "maximum_modified_fraction": 0.05,
    }
    cfg = {k: float(configuration.get(k, v)) for k, v in defaults.items()}
    if not all(np.isfinite(v) for v in cfg.values()):
        raise ValueError("final delivery参数必须有限。")
    if not (0 <= cfg["training_lower_quantile"] < 0.5
            < cfg["training_upper_quantile"] <= 1):
        raise ValueError("final delivery分位数配置无效。")
    for k in ("training_iqr_margin", "training_extrema_margin_iqr_fraction",
              "minimum_extrema_margin_intensity", "softness_iqr_fraction",
              "global_extrema_margin_intensity"):
        if cfg[k] < 0:
            raise ValueError(f"final delivery.{k}不能为负。")
    if not (0 < cfg["minimum_softness_intensity"]
            <= cfg["maximum_softness_intensity"]):
        raise ValueError("final delivery softness配置无效。")
    if not 0 < cfg["maximum_modified_fraction"] <= 0.2:
        raise ValueError("final delivery最大修改比例必须在(0,0.2]。")
    q25, q75 = np.quantile(training, [0.25, 0.75], axis=0)
    iqr = np.maximum(q75 - q25, 0)
    margin = np.maximum(cfg["training_extrema_margin_iqr_fraction"] * iqr,
                        cfg["minimum_extrema_margin_intensity"])
    lower = np.minimum(
        np.quantile(training, cfg["training_lower_quantile"], axis=0)
        - cfg["training_iqr_margin"] * iqr, training.min(axis=0) - margin)
    upper = np.maximum(
        np.quantile(training, cfg["training_upper_quantile"], axis=0)
        + cfg["training_iqr_margin"] * iqr, training.max(axis=0) + margin)
    use_global = bool(configuration.get("enforce_training_global_extrema", True))
    if use_global:
        # An explicit raw-intensity margin prevents one high-IQR peak from
        # legitimizing a very deep negative value in the terminal bounds.
        extra = cfg["global_extrema_margin_intensity"]
        lower = np.maximum(lower, training.min() - extra)
        upper = np.minimum(upper, training.max() + extra)
    # Lower-side protection must not lift low positive peaks or ordinary -18
    # background even if this training condition contains no negative values.
    lower = np.minimum(lower, cfg["activation_maximum_intensity"]
                       - cfg["minimum_softness_intensity"])
    width = np.clip(cfg["softness_iqr_fraction"] * iqr,
                    cfg["minimum_softness_intensity"],
                    cfg["maximum_softness_intensity"])
    if np.any(lower >= upper):
        raise ValueError("final delivery上下界不合法。")
    return lower, upper, width, cfg, use_global


def apply_final_delivery_guard(generated_spectra, training_spectra, *,
                               configuration, raman_shift=None):
    """Leave interior points exact; bound all exterior points with tanh tails.

    The limits are [lower-width, upper+width], in physical intensity units.
    All exterior points are inspected, regardless of generated-batch rank.
    Excessive changes stop export rather than silently flattening a bad batch.
    """
    if not isinstance(configuration, dict):
        raise ValueError("final delivery配置必须为字典。")
    generated, training = _arrays(generated_spectra, training_spectra)
    lower, upper, width, cfg, use_global = _bounds(training, configuration)
    axis = None if raman_shift is None else np.asarray(raman_shift, dtype=np.float64)
    if axis is not None and (axis.ndim != 1 or len(axis) != generated.shape[1]
                             or not np.isfinite(axis).all()
                             or not np.all(np.diff(axis) > 0)):
        raise ValueError("final delivery Raman轴无效。")
    below = generated < lower[None, :]
    above = generated > upper[None, :]
    candidate = below | above
    fraction = float(candidate.mean())
    if fraction > cfg["maximum_modified_fraction"]:
        raise RuntimeError(
            f"final delivery需修改{fraction:.3%}点，超过"
            f"{cfg['maximum_modified_fraction']:.3%}；停止导出，不能自动放宽预算。")
    # Compute exterior entries only: safe even for exceptionally large finite x.
    result = generated.copy()
    rows, points = np.nonzero(below)
    with np.errstate(over="ignore"):
        delta = (lower[points] - generated[rows, points]) / width[points]
    result[rows, points] = lower[points] - width[points] * np.tanh(delta)
    rows, points = np.nonzero(above)
    with np.errstate(over="ignore"):
        delta = (generated[rows, points] - upper[points]) / width[points]
    result[rows, points] = upper[points] + width[points] * np.tanh(delta)
    with np.errstate(over="ignore"):
        delivered = result.astype(np.float32)
    if not np.isfinite(delivered).all():
        raise RuntimeError("final delivery出现NaN/Inf，禁止导出。")
    delivered64 = delivered.astype(np.float64)
    lower_limit, upper_limit = lower - width, upper + width
    tolerance = 8 * np.finfo(np.float32).eps * np.maximum(
        1.0, np.maximum(np.abs(lower_limit), np.abs(upper_limit)))
    remaining = ((delivered64 < (lower_limit - tolerance)[None, :])
                 | (delivered64 > (upper_limit + tolerance)[None, :]))
    if remaining.any():
        raise RuntimeError("final delivery终点检查失败，禁止导出。")
    def mean_pairwise_mse(values):
        if len(values) < 2:
            return 0.0
        center = values - values.mean(axis=0)
        return float(2 * np.mean(center ** 2) * len(values) / (len(values) - 1))
    before_mse = mean_pairwise_mse(generated)
    after_mse = mean_pairwise_mse(delivered64)
    changed = delivered64 - generated
    changed_locations = np.argwhere(candidate)
    locations = []
    if len(changed_locations):
        sizes = np.abs(changed[candidate])
        for k in np.argsort(sizes)[-10:][::-1]:
            row, point = map(int, changed_locations[k])
            locations.append({
                "spectrum_index_zero_based": row, "point_index_zero_based": point,
                "raman_shift_cm1": None if axis is None else float(axis[point]),
                "before": float(generated[row, point]),
                "after": float(delivered64[row, point]),
                "lower": float(lower[point]), "upper": float(upper[point]),
                "soft_width": float(width[point]),
            })
    return delivered, {
        "version": "d4_25_terminal_delivery_v1", "fit_on": "train_only",
        "training_count": int(len(training)), "generated_count": int(len(generated)),
        "point_count": int(generated.shape[1]),
        "uses_generated_rank_gate": False, "uses_contiguous_run_gate": False,
        "uses_global_extrema": use_global,
        "global_extrema_margin_intensity": cfg["global_extrema_margin_intensity"],
        "training_minimum": float(training.min()), "training_maximum": float(training.max()),
        "minimum_before": float(generated.min()), "minimum_after": float(delivered64.min()),
        "maximum_before": float(generated.max()), "maximum_after": float(delivered64.max()),
        "lower_modified_point_count": int(below.sum()),
        "upper_modified_point_count": int(above.sum()),
        "modified_point_count": int(candidate.sum()),
        "modified_spectrum_count": int(candidate.any(axis=1).sum()),
        "modified_point_fraction": fraction,
        "correction_rmse": float(np.sqrt(np.mean(changed ** 2))),
        "pairwise_mse_statistic": "mean_all_pairs",
        "pairwise_mse_before": before_mse, "pairwise_mse_after": after_mse,
        "pairwise_mse_retained_fraction": None if before_mse <= 1e-12 else after_mse / before_mse,
        "remaining_violation_point_count": int(remaining.sum()),
        "largest_corrections": locations,
    }
