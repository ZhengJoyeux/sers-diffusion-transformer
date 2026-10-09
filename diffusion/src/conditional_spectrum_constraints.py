"""D4.25 train-only full-spectrum reconstruction and soft intensity support.

No learnable parameters and no persistent torch buffers are added.  The small
reference state is stored inside the existing diversity checkpoint metadata.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from torch.nn import functional as F


def normalize_spectrum_configuration(source: dict[str, Any]) -> dict[str, Any]:
    raw = source.get("full_spectrum_support", {}) or {}
    if not isinstance(raw, dict):
        raise ValueError("quality_fidelity.full_spectrum_support必须是字典。")
    result = {"enabled": bool(raw.get("enabled", False))}
    if not result["enabled"]:
        return result
    defaults = {
        "lower_quantile": 0.05, "upper_quantile": 0.95,
        "iqr_margin": 0.5, "scale_floor_quantile": 0.1,
        "soft_extension_scale": 0.5, "minimum_alpha_cumprod": 0.5,
        "envelope_weight": 1.0, "peak_derivative_weight": 0.25,
        "total_weight": 0.1, "maximum_total_ratio_to_ddpm": 0.05,
        "smooth_l1_beta": 0.25, "numerical_argument_limit": 20.0,
        "peak_half_width_cm1": 18.0, "epsilon": 1.0e-6,
    }
    result.update({key: float(raw.get(key, value)) for key, value in defaults.items()})
    result["sampling_soft_guard"] = bool(raw.get("sampling_soft_guard", True))
    if not all(np.isfinite(value) for key, value in result.items() if key not in {"enabled", "sampling_soft_guard"}):
        raise ValueError("full_spectrum_support参数必须有限。")
    if not 0 <= result["lower_quantile"] < 0.5 < result["upper_quantile"] <= 1:
        raise ValueError("full_spectrum_support quantile要求0<=lower<0.5<upper<=1。")
    if not 0 <= result["scale_floor_quantile"] <= 0.5:
        raise ValueError("scale_floor_quantile必须位于[0,0.5]。")
    if not 0 < result["minimum_alpha_cumprod"] <= 1:
        raise ValueError("minimum_alpha_cumprod必须位于(0,1]。")
    if not 0 < result["maximum_total_ratio_to_ddpm"] <= 0.25:
        raise ValueError("maximum_total_ratio_to_ddpm必须位于(0,0.25]。")
    for key in ("soft_extension_scale", "total_weight", "smooth_l1_beta", "peak_half_width_cm1", "epsilon"):
        if result[key] <= 0:
            raise ValueError(f"full_spectrum_support.{key}必须大于0。")
    for key in ("iqr_margin", "envelope_weight", "peak_derivative_weight"):
        if result[key] < 0:
            raise ValueError(f"full_spectrum_support.{key}不能为负。")
    if not 8 <= result["numerical_argument_limit"] <= 30:
        raise ValueError("numerical_argument_limit必须位于[8,30]。")
    return result


def fit_spectrum_support_state(
    *, bank: Any, training_full_spectra: np.ndarray,
    training_scaled_residuals: np.ndarray, training_valid_masks: np.ndarray,
    training_condition_vectors: np.ndarray, training_condition_ids: Sequence[str],
    configuration: dict[str, Any],
) -> dict[str, Any]:
    """Inputs are already sliced to training_indices by the training entry."""
    cfg = normalize_spectrum_configuration(configuration)
    if not cfg["enabled"]:
        raise ValueError("full_spectrum_support未启用。")
    full = np.asarray(training_full_spectra, dtype=np.float64)
    scaled = np.asarray(training_scaled_residuals, dtype=np.float64)
    masks = np.asarray(training_valid_masks, dtype=np.float64)
    vectors = np.asarray(training_condition_vectors, dtype=np.float64)
    ids = np.asarray(training_condition_ids, dtype=str)
    if full.ndim != 2 or full.shape != scaled.shape or full.shape != masks.shape:
        raise ValueError("support训练光谱、残差和mask形状必须一致。")
    if vectors.ndim != 2 or len(vectors) != len(full) or len(ids) != len(full):
        raise ValueError("support条件数量与训练光谱不一致。")
    if not all(np.isfinite(x).all() for x in (full, scaled, masks, vectors)):
        raise ValueError("support训练数据必须有限。")
    fields: dict[str, list] = {key: [] for key in (
        "condition_vectors", "valid_masks", "inverse_rate", "inverse_scale",
        "lower", "upper", "scale", "peak_mask", "training_counts")}
    for cid in sorted(set(ids)):
        selected = ids == cid
        if selected.sum() < 4:
            raise ValueError("support每条件至少需要4条训练光谱。")
        entry = bank.entries[cid]
        length = entry.valid_length
        mask = masks[selected][0]
        if not np.array_equal(masks[selected], np.broadcast_to(mask, masks[selected].shape)):
            raise ValueError("同条件support mask必须一致。")
        expected = np.zeros_like(mask); expected[:length] = 1
        if not np.array_equal(mask, expected) or not np.array_equal(vectors[selected], np.broadcast_to(vectors[selected][0], vectors[selected].shape)):
            raise ValueError("support mask/条件与先验状态不一致。")
        local = entry.broad_local
        multiplier = entry.raman_variance_multiplier
        if multiplier is None:
            multiplier = np.ones(length)
        rate = float(local.local_asinh_normalizer) / float(local.local_target_abs_max) / multiplier
        if np.max(np.abs(scaled[selected, :length] * rate)) >= cfg["numerical_argument_limit"]:
            raise ValueError("训练残差超出数值安全范围；不能静默改变训练反变换。")
        values = full[selected, :length]
        q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
        iqr = q75 - q25
        scale = np.std(values, axis=0, ddof=1)
        positive = scale[scale > cfg["epsilon"]]
        floor = max(float(np.quantile(positive, cfg["scale_floor_quantile"])) if positive.size else cfg["epsilon"], cfg["epsilon"])
        scale = np.maximum(scale, floor)
        margin = cfg["iqr_margin"] * np.maximum(iqr, floor)
        # All actual training extrema remain inside support, including negatives.
        lower = np.minimum(np.quantile(values, cfg["lower_quantile"], axis=0) - margin, values.min(axis=0))
        upper = np.maximum(np.quantile(values, cfg["upper_quantile"], axis=0) + margin, values.max(axis=0))
        axis = np.asarray(bank.model_axis[:length])
        spacing = float(np.median(np.diff(axis)))
        mean = values.mean(axis=0)
        profile = gaussian_filter1d(mean, 2.0 / spacing) - gaussian_filter1d(mean, 30.0 / spacing)
        peaks, _ = find_peaks(profile, prominence=max(float(profile.max()) * 0.06, cfg["epsilon"]), distance=max(1, round(12.0 / spacing)))
        peak_mask = np.zeros(length)
        for p in peaks:
            if axis[p] - axis[0] < 24 or axis[-1] - axis[p] < 24:
                continue
            peak_mask[np.abs(axis - axis[p]) <= cfg["peak_half_width_cm1"]] = 1
        def pad(value: np.ndarray, fill: float = 0.0) -> list:
            result = np.full(full.shape[1], fill, dtype=np.float64); result[:length] = value
            return result.tolist()
        fields["condition_vectors"].append(vectors[selected][0].tolist())
        fields["valid_masks"].append(mask.tolist())
        fields["inverse_rate"].append(pad(rate, 1))
        fields["inverse_scale"].append(float(local.local_residual_scale))
        fields["lower"].append(pad(lower)); fields["upper"].append(pad(upper))
        fields["scale"].append(pad(scale, 1)); fields["peak_mask"].append(pad(peak_mask))
        fields["training_counts"].append(int(selected.sum()))
    return {"version": "d4.25_full_spectrum_support_v1", "fit_on": "train_only", "configuration": cfg, **fields}


class ConditionalSpectrumSupport(torch.nn.Module):
    def __init__(self, state: dict[str, Any], padded_length: int):
        super().__init__()
        if state.get("version") != "d4.25_full_spectrum_support_v1" or state.get("fit_on") != "train_only":
            raise ValueError("D4.25 support状态版本或数据来源无效。")
        self.configuration = dict(state["configuration"])
        self.training_counts = tuple(int(value) for value in state["training_counts"])
        # Non-persistent buffers preserve legacy checkpoint tensor keys.
        for key in ("condition_vectors", "valid_masks", "inverse_rate", "inverse_scale", "lower", "upper", "scale", "peak_mask"):
            value = torch.as_tensor(state[key], dtype=torch.float32)
            if not torch.isfinite(value).all():
                raise ValueError(f"support.{key}必须有限。")
            if key not in {"condition_vectors", "inverse_scale"}:
                if value.ndim != 2 or value.shape[-1] > padded_length:
                    raise ValueError(f"support.{key}长度无效。")
                value = F.pad(value, (0, padded_length - value.shape[-1]), value=1.0 if key in {"inverse_rate", "scale"} else 0.0)
            self.register_buffer(key, value, persistent=False)
        if torch.any(self.inverse_rate <= 0) or torch.any(self.inverse_scale <= 0) or torch.any(self.scale <= 0) or torch.any(self.lower > self.upper):
            raise ValueError("support scale/rate/envelope状态无效。")

    def indices(self, condition: torch.Tensor) -> torch.Tensor:
        references = self.condition_vectors.to(condition.device)
        matches = torch.isclose(condition.detach()[:, None], references[None], atol=1e-6, rtol=0).all(-1)
        if not torch.all(matches.sum(1) == 1):
            raise ValueError("D4.25条件无法唯一匹配训练reference。")
        return matches.long().argmax(1)

    def field(self, name: str, index: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        value = getattr(self, name).to(device=like.device, dtype=torch.float32)[index]
        if name == "inverse_scale":
            return value[:, None, None]
        return value[:, None, :like.shape[-1]]

    def decode(self, scaled: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        index = self.indices(condition)
        argument = scaled.float() * self.field("inverse_rate", index, scaled)
        limit = self.configuration["numerical_argument_limit"]
        safe = argument.clamp(-limit, limit)
        # Exact sinh in the useful domain; linear continuation for overflow
        # prevention preserves a corrective gradient on pathological predictions.
        decoded = torch.sinh(safe) + (argument - safe) * float(np.cosh(limit))
        return self.field("inverse_scale", index, scaled) * decoded

    def reconstruct(self, prediction: torch.Tensor, target: torch.Tensor,
                    full_target: torch.Tensor, condition: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (full_target.float() + self.decode(prediction, condition) - self.decode(target.detach(), condition)) * mask

    def losses(self, full_prediction: torch.Tensor, full_target: torch.Tensor,
               condition: torch.Tensor, mask: torch.Tensor, alpha: torch.Tensor) -> dict[str, torch.Tensor]:
        index = self.indices(condition)
        scale = self.field("scale", index, full_prediction)
        lower = self.field("lower", index, full_prediction)
        upper = self.field("upper", index, full_prediction)
        active = (alpha >= self.configuration["minimum_alpha_cumprod"]).float()[:, None, None]
        effective = mask * active
        beta = self.configuration["smooth_l1_beta"]
        excess = (F.relu(lower - full_prediction) + F.relu(full_prediction - upper)) / scale
        per_point = F.smooth_l1_loss(excess, torch.zeros_like(excess), reduction="none", beta=beta)
        envelope = (per_point * effective).sum() / effective.sum().clamp_min(1)
        # Compare derivatives in actual reconstructed intensity units, after a
        # small mask-normalized filter. Invalid tails never enter the filter.
        def smooth(x: torch.Tensor) -> torch.Tensor:
            numerator = F.avg_pool1d(x * mask, 5, 1, 2)
            denominator = F.avg_pool1d(mask, 5, 1, 2)
            return numerator / denominator.clamp_min(1e-6)
        error = torch.diff(smooth(full_prediction) - smooth(full_target.detach()), dim=-1)
        derivative_scale = ((scale[..., 1:] + scale[..., :-1]) * 0.5).clamp_min(1e-6)
        peak = self.field("peak_mask", index, full_prediction)
        derivative_mask = effective[..., 1:] * mask[..., :-1] * peak[..., 1:] * peak[..., :-1]
        penalty = F.smooth_l1_loss(error / derivative_scale, torch.zeros_like(error), reduction="none", beta=beta)
        derivative = (penalty * derivative_mask).sum() / derivative_mask.sum().clamp_min(1)
        raw = self.configuration["envelope_weight"] * envelope + self.configuration["peak_derivative_weight"] * derivative
        return {"raw_loss": raw, "envelope_loss": envelope, "peak_derivative_loss": derivative, "active_fraction": active.mean()}

    def soft_guard(self, scaled: torch.Tensor, base: torch.Tensor,
                   condition: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        index = self.indices(condition)
        full = base.float() + self.decode(scaled, condition)
        corrected = self.bound_full_spectrum(full, condition, mask)
        local = (corrected - base.float()) / self.field("inverse_scale", index, full)
        restored = torch.asinh(local) / self.field("inverse_rate", index, full)
        return restored.to(scaled.dtype) * mask

    def bound_full_spectrum(self, full: torch.Tensor, condition: torch.Tensor,
                            mask: torch.Tensor) -> torch.Tensor:
        index = self.indices(condition)
        lower = self.field("lower", index, full)
        upper = self.field("upper", index, full)
        extension = self.field("scale", index, full) * self.configuration["soft_extension_scale"]
        below = F.relu(lower - full); above = F.relu(full - upper)
        # Identity inside support; monotonic C1 tails with no boundary pile-up.
        corrected = torch.where(full < lower, lower - extension * (-torch.expm1(-below / extension)), full)
        corrected = torch.where(full > upper, upper + extension * (-torch.expm1(-above / extension)), corrected)
        return corrected * mask
