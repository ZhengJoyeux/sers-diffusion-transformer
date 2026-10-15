"""
Peak-guided SERSFormer for the DEL/CHL/TEB multi-task SERS project.

Model path:
    three 1-D CNN branches
        raw spectrum / smoothed spectrum / percentile features
    -> token-level feature fusion
    -> shared Transformer encoder
    -> pesticide-specific peak-guided query attention
    -> one classification head and one concentration head per pesticide

Key design points:
- DEL, CHL and TEB each own a learnable query vector.
- Every query can attend to every *valid* Raman token.
- Chemistry-core + training-derived auxiliary priors add a positive soft bias
  to attention logits; they never delete non-peak regions.
- valid_mask remains authoritative for any unavailable Raman positions.
- The formal downstream model uses the complete 600-2500 cm^-1 input axis,
  with the Dataset adapter's existing policy for shorter source spectra.
- The default public forward interface remains [classification_probability, concentration].
- T3.1 optionally uses pesticide-specific ordered S/M/H concentration heads while
  preserving the T1/T2 continuous-regression checkpoint path.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


DEFAULT_MODEL_LENGTH = 1901


def ordinal_class_probabilities_numpy(probabilities: np.ndarray) -> np.ndarray:
    """Convert ordered cumulative probabilities to [S, M, H] probabilities."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape[-1:] != (2,) or not np.isfinite(values).all():
        raise ValueError("Expected finite cumulative probabilities ending in dimension 2")
    if (values < -1e-7).any() or (values > 1.0 + 1e-7).any():
        raise ValueError("Cumulative probabilities must be in [0, 1]")
    if (values[..., 1] > values[..., 0] + 1e-7).any():
        raise ValueError("P(level>=H) must not exceed P(level>=M)")
    values = np.clip(values, 0.0, 1.0)
    return np.maximum(np.stack(
        (1.0 - values[..., 0], values[..., 0] - values[..., 1], values[..., 1]),
        axis=-1,
    ), 0.0)


def decode_ordinal_numpy(
    probabilities: np.ndarray, *, mode: str = "median", threshold: float = 0.5,
) -> np.ndarray:
    """One shared decoder for training metrics, inference tables and CSV rows.

    median preserves the historical cumulative-threshold policy. map chooses
    the most probable S/M/H class; exact ties choose the lower level.
    """
    classes = ordinal_class_probabilities_numpy(probabilities)
    if mode == "map":
        return 1 + classes.argmax(axis=-1)
    if mode != "median":
        raise ValueError("ordinal decoding must be median or map")
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("ordinal threshold must be in (0, 1)")
    return 1 + (np.asarray(probabilities) >= float(threshold)).sum(axis=-1)


class LocalQuantitativeBranch(nn.Module):
    """T3.13: native-resolution windows plus train-normalized intensity features.

    The window CNN has no strided pooling. A linear baseline is estimated
    from the two window flanks; signed local residuals are retained. Features
    describe positive area/height, centroid, effective width, asymmetry,
    baseline and flank noise. They are effective window measurements, not
    chemically resolved individual-peak fits.

    The final projections start at zero: this path initially contributes no
    concentration correction. All normalization buffers are checkpointed.
    """

    feature_names = (
        "log_height", "log_area", "relative_centroid", "relative_width",
        "asymmetry", "log_rms", "signed_log_baseline", "log_flank_noise",
    )

    def __init__(
        self, d_model: int, model_length: int, centers: Sequence[Sequence[float]],
        half_width: int = 24, hidden: int = 16, strength: float = 0.25,
        projection_hidden: int | None = None,
    ) -> None:
        super().__init__()
        if model_length != DEFAULT_MODEL_LENGTH:
            raise ValueError("T3.13 local windows require the project's 1901-point axis")
        if half_width < 8 or half_width > 60 or hidden < 4:
            raise ValueError("local half-width must be 8..60 and local hidden must be >=4")
        if not 0.0 < float(strength) <= 1.0:
            raise ValueError("local correction strength must be in (0,1]")
        if len(centers) != 3 or any(not group for group in centers):
            raise ValueError("Provide nonempty DEL/CHL/TEB window-center groups")
        self.half_width = int(half_width)
        self.strength = float(strength)
        self.window_count = max(len(group) for group in centers)
        center_array = torch.zeros(3, self.window_count, dtype=torch.float32)
        active = torch.zeros(3, self.window_count, dtype=torch.bool)
        for pesticide, group in enumerate(centers):
            for window, center in enumerate(group):
                if not math.isfinite(float(center)):
                    raise ValueError("local window center must be finite")
                index = round(float(center) - 600.0)
                if index - half_width < 0 or index + half_width >= model_length:
                    raise ValueError("local window must lie inside the measured model axis")
                center_array[pesticide, window] = float(center)
                active[pesticide, window] = True
        # Inactive padded slots use a safe index and are explicitly zeroed.
        safe_centers = torch.where(active, center_array, torch.full_like(center_array, 1000.0))
        offsets = torch.arange(-half_width, half_width + 1, dtype=torch.float32)
        indices = (safe_centers - 600.0).round().long().unsqueeze(-1) + offsets.long()
        self.register_buffer("centers_cm1", center_array)
        self.register_buffer("window_active", active)
        self.register_buffer("window_indices", indices)
        self.register_buffer("offsets", offsets)
        self.register_buffer("signal_scale", torch.ones(3, self.window_count))
        self.register_buffer("feature_mean", torch.zeros(3, self.window_count, 8))
        self.register_buffer("feature_std", torch.ones(3, self.window_count, 8))
        self.register_buffer("normalization_fitted", torch.tensor(False))
        self.window_cnn = nn.Sequential(
            nn.Conv1d(3, hidden, kernel_size=5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(4), nn.Flatten(),
        )
        feature_dim = self.window_count * (hidden * 4 + 8) + self.window_count
        projection_dim = d_model if projection_hidden is None else int(projection_hidden)
        if projection_dim < 1:
            raise ValueError("local projection hidden dimension must be positive")
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(feature_dim, projection_dim), nn.GELU(), nn.Linear(projection_dim, d_model))
            for _ in range(3)
        ])
        for projection in self.projections:
            nn.init.zeros_(projection[-1].weight)
            nn.init.zeros_(projection[-1].bias)

    def _measure(self, intensity: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if intensity.ndim != 3 or intensity.shape[1:] != (1, DEFAULT_MODEL_LENGTH):
            raise ValueError("raw_intensity must have shape [B,1,1901]")
        if valid_mask.shape != (intensity.shape[0], DEFAULT_MODEL_LENGTH):
            raise ValueError("local branch valid_mask shape mismatch")
        if not torch.isfinite(intensity).all():
            raise ValueError("raw_intensity contains NaN or Inf")
        windows = intensity[:, 0, self.window_indices]
        masks = valid_mask[:, self.window_indices].bool()
        available = masks.all(dim=-1) & self.window_active.unsqueeze(0)
        flank = max(3, windows.shape[-1] // 6)
        left = windows[..., :flank].mean(dim=-1, keepdim=True)
        right = windows[..., -flank:].mean(dim=-1, keepdim=True)
        # Anchor the baseline at flank midpoints rather than the outermost points.
        coordinate = torch.arange(windows.shape[-1], device=windows.device, dtype=windows.dtype)
        fraction = (coordinate - (flank - 1) / 2.0) / (windows.shape[-1] - flank)
        baseline = left + (right - left) * fraction
        corrected = windows - baseline
        corrected = corrected * available.unsqueeze(-1)
        positive = corrected.clamp_min(0.0)
        area = positive.sum(dim=-1)
        height = positive.amax(dim=-1)
        mass = area.clamp_min(1e-8)
        centroid = (positive * self.offsets).sum(dim=-1) / mass
        width = torch.sqrt(((positive * (self.offsets - centroid.unsqueeze(-1)).square()).sum(dim=-1) / mass).clamp_min(0.0))
        middle = self.half_width
        asymmetry = (positive[..., middle + 1:].sum(dim=-1) - positive[..., :middle].sum(dim=-1)) / mass
        rms = corrected.square().mean(dim=-1).sqrt()
        noise = torch.cat((corrected[..., :flank], corrected[..., -flank:]), dim=-1).std(dim=-1, unbiased=False)
        background = baseline.mean(dim=-1)
        features = torch.stack((
            torch.log1p(height), torch.log1p(area), centroid / self.half_width,
            width / self.half_width, asymmetry, torch.log1p(rms),
            background.sign() * torch.log1p(background.abs()), torch.log1p(noise),
        ), dim=-1)
        features = features * available.unsqueeze(-1)
        signed_signal = corrected.sign() * torch.log1p(corrected.abs())
        return signed_signal, features, available

    @torch.no_grad()
    def fit_normalization(self, intensity: Tensor, valid_mask: Tensor) -> dict[str, object]:
        """Called once by the training entry with real training rows only."""
        if bool(self.normalization_fitted):
            raise RuntimeError("Local normalization is already fitted; do not refit on validation/test")
        signal, features, available = self._measure(intensity.float(), valid_mask.bool())
        counts = []
        for pesticide in range(3):
            group_counts = []
            for window in range(self.window_count):
                if not bool(self.window_active[pesticide, window]):
                    group_counts.append(0)
                    continue
                rows = available[:, pesticide, window]
                count = int(rows.sum())
                if count < 8:
                    raise ValueError("Every active quantitative window needs >=8 real training rows")
                values = features[rows, pesticide, window]
                self.feature_mean[pesticide, window].copy_(values.mean(dim=0))
                self.feature_std[pesticide, window].copy_(values.std(dim=0, unbiased=False).clamp_min(0.05))
                self.signal_scale[pesticide, window].copy_(
                    torch.quantile(signal[rows, pesticide, window].abs().flatten(), 0.95).clamp_min(0.1)
                )
                group_counts.append(count)
            counts.append(group_counts)
        self.normalization_fitted.fill_(True)
        return {
            "centers_cm1": self.centers_cm1.tolist(), "window_active": self.window_active.tolist(),
            "half_width_cm1": self.half_width, "feature_names": list(self.feature_names),
            "fit_rows_per_window": counts, "fit_scope": "real training rows only",
            "signal_scale": self.signal_scale.tolist(),
            "feature_mean": self.feature_mean.tolist(), "feature_std": self.feature_std.tolist(),
        }

    def forward(self, intensity: Tensor, valid_mask: Tensor) -> Tensor:
        if not bool(self.normalization_fitted):
            raise RuntimeError("Fit local normalization on real training spectra before forward")
        signal, features, available = self._measure(intensity, valid_mask)
        amplitude = signal / self.signal_scale.unsqueeze(0).unsqueeze(-1)
        shape = signal / signal.abs().amax(dim=-1, keepdim=True).clamp_min(0.1)
        mask_channel = available.unsqueeze(-1).expand_as(signal).to(signal.dtype)
        cnn_input = torch.stack((amplitude.clamp(-8.0, 8.0), shape, mask_channel), dim=-2)
        batch_size = intensity.shape[0]
        local = self.window_cnn(cnn_input.reshape(-1, 3, signal.shape[-1]))
        local = local.reshape(batch_size, 3, self.window_count, -1) * available.unsqueeze(-1)
        numerical = ((features - self.feature_mean) / self.feature_std).clamp(-8.0, 8.0)
        numerical = numerical * available.unsqueeze(-1)
        # log(1+area) differences approximate log-area ratios without dividing by noise.
        pair_available = available & available[..., :1]
        area_ratios = (features[..., 1] - features[..., :1, 1]) * pair_available
        pesticide_features = torch.cat((local.flatten(2), numerical.flatten(2), area_ratios), dim=-1)
        result = torch.stack([
            head(pesticide_features[:, pesticide]) for pesticide, head in enumerate(self.projections)
        ], dim=1)
        return self.strength * result * available.any(dim=-1).unsqueeze(-1)


def _conv_pool_output_length(length: int) -> int:
    """Two [Conv1d(k=3,s=1) -> MaxPool1d(k=3,s=3)] stages."""
    for _ in range(2):
        length = length - 3 + 1
        length = (length - 3) // 3 + 1
    if length <= 0:
        raise ValueError("Input length is too short for the CNN frontend")
    return int(length)


def _downsample_valid_mask(valid_mask: Tensor) -> Tensor:
    """
    Conservatively propagate a point-level boolean mask through the two
    Conv(k=3) + Pool(k=3,s=3) stages.

    A token remains valid only when every contributing input position is valid.
    For a 1901-point input with the first 1401 points valid, this yields 154
    valid tokens followed by padding tokens; a fully valid 1901-point input
    yields 210 valid tokens.
    """
    if valid_mask.ndim != 2:
        raise ValueError(
            f"valid_mask must have shape [B, L], got {tuple(valid_mask.shape)}"
        )

    mask = valid_mask.to(dtype=torch.float32).unsqueeze(1)
    for _ in range(2):
        mask = F.avg_pool1d(mask, kernel_size=3, stride=1)
        mask = (mask >= (1.0 - 1e-6)).to(dtype=torch.float32)
        mask = F.avg_pool1d(mask, kernel_size=3, stride=3)
        mask = (mask >= (1.0 - 1e-6)).to(dtype=torch.float32)
    return mask.squeeze(1).bool()


def _downsample_peak_prior(peak_prior: Tensor) -> Tensor:
    """
    Map point-level peak priors [Q, L] onto CNN/Transformer tokens [Q, T].

    Average pooling models the Conv1d receptive field and max pooling preserves
    a local peak hint inside each pooling cell. The result is re-normalized to
    [0, 1] independently for every pesticide query.
    """
    if peak_prior.ndim != 2:
        raise ValueError(
            f"peak_prior must have shape [Q, L], got {tuple(peak_prior.shape)}"
        )

    prior = peak_prior.to(dtype=torch.float32).unsqueeze(1)
    for _ in range(2):
        prior = F.avg_pool1d(prior, kernel_size=3, stride=1)
        prior = F.max_pool1d(prior, kernel_size=3, stride=3)
    prior = prior.squeeze(1)

    maximum = prior.amax(dim=-1, keepdim=True).clamp_min(1e-8)
    prior = torch.where(maximum > 1e-8, prior / maximum, prior)
    return prior.clamp_(0.0, 1.0)


def _sinusoidal_position_encoding(length: int, dimension: int) -> Tensor:
    """Standard fixed positional encoding for the Raman-token sequence."""
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dimension, 2, dtype=torch.float32)
        * (-math.log(10000.0) / float(dimension))
    )
    encoding = torch.zeros(length, dimension, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * div_term)
    if dimension > 1:
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
    return encoding.unsqueeze(0)


def level_codes_to_ordinal_targets(concentration_target: Tensor) -> Tensor:
    """Convert 0/S/M/H codes to cumulative ordinal targets.

    Present levels are encoded as S=1->[0,0], M=2->[1,0], H=3->[1,1].
    Absent entries (0) are intentionally handled by a separate presence mask.
    """
    if concentration_target.ndim != 2:
        raise ValueError(
            "concentration_target must be [B,P], got "
            f"{tuple(concentration_target.shape)}"
        )
    ge_m = (concentration_target >= 2.0).to(concentration_target.dtype)
    ge_h = (concentration_target >= 3.0).to(concentration_target.dtype)
    return torch.stack((ge_m, ge_h), dim=-1)


def present_only_ordinal_bce(
    ordinal_probability: Tensor,
    concentration_target: Tensor,
    presence_target: Tensor,
) -> Tensor:
    """Cumulative BCE evaluated only where the pesticide is truly present."""
    if ordinal_probability.ndim != 3 or ordinal_probability.shape[-1] != 2:
        raise ValueError(
            "ordinal_probability must be [B,P,2], got "
            f"{tuple(ordinal_probability.shape)}"
        )
    if concentration_target.shape != ordinal_probability.shape[:2]:
        raise ValueError("concentration_target shape does not match ordinal probabilities")
    if presence_target.shape != ordinal_probability.shape[:2]:
        raise ValueError("presence_target shape does not match ordinal probabilities")

    target = level_codes_to_ordinal_targets(concentration_target)
    present = presence_target > 0.5
    if not bool(present.any()):
        return ordinal_probability.sum() * 0.0
    return F.binary_cross_entropy(ordinal_probability[present], target[present])


def present_only_corn_ordinal_loss(
    ordinal_logits: Tensor,
    concentration_target: Tensor,
    presence_target: Tensor,
) -> Tensor:
    """CORN-style conditional ordinal loss for present pesticides only.

    For S/M/H level codes 1/2/3:

    task 0:
        P(level >= M)
        trained on all present S/M/H targets.

    task 1:
        P(level >= H | level >= M)
        trained only on present M/H targets.

    The loss is normalized by the total number of valid conditional
    training entries, following the CORN conditional-training-set
    formulation.
    """
    if ordinal_logits.ndim != 3 or ordinal_logits.shape[-1] != 2:
        raise ValueError(
            "ordinal_logits must be [B,P,2], got "
            f"{tuple(ordinal_logits.shape)}"
        )

    if concentration_target.shape != ordinal_logits.shape[:2]:
        raise ValueError(
            "concentration_target shape does not match ordinal logits"
        )

    if presence_target.shape != ordinal_logits.shape[:2]:
        raise ValueError(
            "presence_target shape does not match ordinal logits"
        )

    present = presence_target > 0.5

    if not bool(present.any()):
        return ordinal_logits.sum() * 0.0

    # --------------------------------------------------------
    # Conditional task 0:
    # S versus {M,H}
    #
    # All truly present pesticides participate.
    # --------------------------------------------------------
    target_ge_m = (
        concentration_target >= 2.0
    ).to(ordinal_logits.dtype)

    first_logits = ordinal_logits[..., 0][present]
    first_target = target_ge_m[present]

    loss_sum = F.binary_cross_entropy_with_logits(
        first_logits,
        first_target,
        reduction="sum",
    )

    valid_count = int(first_logits.numel())

    # --------------------------------------------------------
    # Conditional task 1:
    # M versus H
    #
    # S samples must NOT train this conditional boundary.
    # --------------------------------------------------------
    second_mask = (
        present
        & (concentration_target >= 2.0)
    )

    if bool(second_mask.any()):
        target_h_given_ge_m = (
            concentration_target >= 3.0
        ).to(ordinal_logits.dtype)

        second_logits = ordinal_logits[..., 1][second_mask]
        second_target = target_h_given_ge_m[second_mask]

        loss_sum = (
            loss_sum
            + F.binary_cross_entropy_with_logits(
                second_logits,
                second_target,
                reduction="sum",
            )
        )

        valid_count += int(second_logits.numel())

    return loss_sum / float(valid_count)


class regression_features(nn.Module):
    """Percentile-feature 1-D CNN branch."""

    def __init__(self, dim_model: int = 32, channels: int = 4) -> None:
        super().__init__()
        self.cnn = nn.Conv1d(channels, dim_model, kernel_size=3)
        self.cnn2 = nn.Conv1d(dim_model, dim_model * 2, kernel_size=3)
        self.cnn1_bn = nn.BatchNorm1d(dim_model)
        self.max_pool = nn.MaxPool1d(kernel_size=3)
        self.dropout = nn.Dropout1d(p=0.1)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.cnn(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.cnn1_bn(x)
        x = self.cnn2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.dropout(x)
        return x


class PeakGuidedQueryAttention(nn.Module):
    """
    Pesticide-specific query attention with a *soft* training-data peak prior.

    The prior is additive in attention-logit space:

        logits = QK^T / sqrt(d) + positive_strength * peak_prior

    Therefore every valid token remains visible. The model can also learn to
    reduce the positive guidance strength toward zero when the prior is not
    useful for a pesticide or matrix.
    """

    def __init__(
        self,
        d_model: int,
        n_queries: int,
        n_heads: int,
        model_length: int,
        dropout: float = 0.1,
        guidance_strength_init: float = 1.5,
        use_peak_guidance: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by query n_heads={n_heads}"
            )
        if guidance_strength_init <= 0.0:
            raise ValueError("guidance_strength_init must be positive")

        self.d_model = int(d_model)
        self.n_queries = int(n_queries)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.model_length = int(model_length)
        self.use_peak_guidance = bool(use_peak_guidance)

        self.query_embedding = nn.Parameter(torch.empty(self.n_queries, self.d_model))
        nn.init.normal_(self.query_embedding, mean=0.0, std=0.02)

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.attention_dropout = nn.Dropout(dropout)

        inverse_softplus = math.log(math.expm1(float(guidance_strength_init)))
        self.guidance_log_strength = nn.Parameter(
            torch.full((self.n_queries,), inverse_softplus, dtype=torch.float32)
        )

        self.output_norm = nn.LayerNorm(self.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(self.d_model)

        # Stored in the checkpoint. It is fitted from real training spectra only.
        self.register_buffer(
            "peak_prior_points",
            torch.zeros(self.n_queries, self.model_length, dtype=torch.float32),
            persistent=True,
        )

    @property
    def guidance_strength(self) -> Tensor:
        """Positive learnable strength stored for the optional peak-guidance bias."""
        return F.softplus(self.guidance_log_strength)

    @property
    def effective_guidance_strength(self) -> Tensor:
        """Strength actually used in forward; zero in query-only ablation mode."""
        if self.use_peak_guidance:
            return self.guidance_strength
        return torch.zeros_like(self.guidance_log_strength)

    def set_peak_prior(self, peak_prior: Tensor) -> None:
        peak_prior = torch.as_tensor(peak_prior, dtype=torch.float32)
        expected = (self.n_queries, self.model_length)
        if tuple(peak_prior.shape) != expected:
            raise ValueError(
                f"peak_prior must have shape {expected}, got {tuple(peak_prior.shape)}"
            )
        if not torch.isfinite(peak_prior).all():
            raise ValueError("peak_prior contains NaN or Inf")
        if (peak_prior < 0.0).any():
            raise ValueError("peak_prior must be non-negative")

        maximum = peak_prior.amax(dim=-1, keepdim=True)
        normalized = torch.where(
            maximum > 0.0,
            peak_prior / maximum.clamp_min(1e-8),
            peak_prior,
        ).clamp(0.0, 1.0)
        self.peak_prior_points.copy_(normalized.to(self.peak_prior_points.device))

    def forward(
        self,
        tokens: Tensor,
        token_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if tokens.ndim != 3:
            raise ValueError(f"tokens must be [B,T,D], got {tuple(tokens.shape)}")
        if token_mask.shape != tokens.shape[:2]:
            raise ValueError(
                f"token_mask must be {tuple(tokens.shape[:2])}, "
                f"got {tuple(token_mask.shape)}"
            )
        if not token_mask.any(dim=1).all():
            raise ValueError("Every sample must contain at least one valid Raman token")

        batch_size, token_length, _ = tokens.shape
        queries = self.query_embedding.unsqueeze(0).expand(batch_size, -1, -1)

        q = self.q_proj(queries).view(
            batch_size, self.n_queries, self.n_heads, self.head_dim
        )
        q = q.permute(0, 2, 1, 3)  # [B,H,Q,Dh]

        k = self.k_proj(tokens).view(
            batch_size, token_length, self.n_heads, self.head_dim
        )
        k = k.permute(0, 2, 1, 3)  # [B,H,T,Dh]

        v = self.v_proj(tokens).view(
            batch_size, token_length, self.n_heads, self.head_dim
        )
        v = v.permute(0, 2, 1, 3)  # [B,H,T,Dh]

        logits = torch.einsum("bhqd,bhtd->bhqt", q, k)
        logits = logits / math.sqrt(float(self.head_dim))

        peak_prior_tokens = _downsample_peak_prior(self.peak_prior_points)
        if peak_prior_tokens.shape != (self.n_queries, token_length):
            raise RuntimeError(
                "Peak prior token length does not match Transformer token length: "
                f"{tuple(peak_prior_tokens.shape)} vs {(self.n_queries, token_length)}"
            )

        # Optional positive additive bias only. In query-only mode this block is
        # skipped entirely, so the three pesticide queries learn where to attend
        # from the full valid spectrum without any hand/data-derived peak hint.
        # In peak-guided mode non-peak regions still remain visible because the
        # prior is an additive bias rather than a hard mask.
        if self.use_peak_guidance:
            guidance = self.guidance_strength.view(1, 1, self.n_queries, 1)
            logits = logits + guidance * peak_prior_tokens.view(
                1, 1, self.n_queries, token_length
            ).to(dtype=logits.dtype, device=logits.device)

        invalid = ~token_mask[:, None, None, :]
        logits = logits.masked_fill(invalid, torch.finfo(logits.dtype).min)

        attention = torch.softmax(logits.float(), dim=-1).to(dtype=logits.dtype)
        attention = attention.masked_fill(invalid, 0.0)
        attention_for_output = attention
        attention = self.attention_dropout(attention)

        context = torch.einsum("bhqt,bhtd->bhqd", attention, v)
        context = context.permute(0, 2, 1, 3).contiguous().view(
            batch_size, self.n_queries, self.d_model
        )
        context = self.out_proj(context)

        query_features = self.output_norm(context + queries)
        query_features = self.ffn_norm(query_features + self.ffn(query_features))

        # Average heads only for diagnostics/visualization. This does not affect
        # the model prediction path.
        attention_mean = attention_for_output.mean(dim=1)
        return query_features, attention_mean


class MixtureAwareQueryFusion(nn.Module):
    """
    Lightweight cross-pesticide query fusion for ordinal concentration
    prediction.

    query_features:
        [B, P, D], P = DEL / CHL / TEB.

    presence_probability:
        [B, P], pesticide-presence probabilities from the classification
        branch.

    Classification itself is not changed. The fused representation is used
    only by the concentration branch.
    """

    def __init__(
        self,
        d_model: int,
        n_queries: int,
        n_heads: int,
        dropout: float = 0.1,
        gate_init: float = 0.10,
        use_adaptive_gate: bool = False,
        use_boundary_specific_gate: bool = False,
    ) -> None:
        super().__init__()

        if n_queries < 2:
            raise ValueError(
                "MixtureAwareQueryFusion requires at least two queries"
            )

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )

        if not 0.0 < gate_init < 1.0:
            raise ValueError(
                "gate_init must satisfy 0 < gate_init < 1"
            )

        self.d_model = int(d_model)
        self.n_queries = int(n_queries)
        self.n_heads = int(n_heads)

        self.use_adaptive_gate = bool(
            use_adaptive_gate
        )

        self.use_boundary_specific_gate = bool(
            use_boundary_specific_gate
        )

        if (
            self.use_boundary_specific_gate
            and not self.use_adaptive_gate
        ):
            raise ValueError(
                "use_boundary_specific_gate=True requires "
                "use_adaptive_gate=True"
            )

        self.cross_query_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.n_heads,
            dropout=dropout,
            bias=False,
            batch_first=True,
        )

        self.attention_norm = nn.LayerNorm(
            self.d_model
        )

        self.ffn = nn.Sequential(
            nn.Linear(
                self.d_model,
                self.d_model * 2,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                self.d_model * 2,
                self.d_model,
            ),
            nn.Dropout(dropout),
        )

        self.ffn_norm = nn.LayerNorm(
            self.d_model
        )

        gate_logit = math.log(
            float(gate_init)
            / (1.0 - float(gate_init))
        )

        self.gate_logit = nn.Parameter(
            torch.full(
                (self.n_queries,),
                gate_logit,
                dtype=torch.float32,
            )
        )

        # T3.11:
        # The T3.10 gate is one learnable scalar for each pesticide.
        # T3.11 optionally adds a sample-specific correction using:
        #
        #   original pesticide query       [D]
        #   cross-query residual           [D]
        #   presence probability           [1]
        #   mixture strength               [1]
        #
        # The final layer is initialized to zero. Therefore the
        # adaptive correction is initially zero and the model starts
        # from exactly the T3.10 fusion behavior.
        self.adaptive_gate = None
        self.adaptive_gate_sm = None
        self.adaptive_gate_mh = None

        if self.use_adaptive_gate:
            adaptive_input_dim = (
                self.d_model * 2
                + 2
            )

            adaptive_hidden_dim = max(
                16,
                self.d_model // 2,
            )

            def make_adaptive_gate() -> nn.Sequential:
                module = nn.Sequential(
                    nn.Linear(
                        adaptive_input_dim,
                        adaptive_hidden_dim,
                    ),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(
                        adaptive_hidden_dim,
                        1,
                    ),
                )

                # Zero initialization means T3.11/T3.12 start
                # from the same T3.10 base pesticide gate.
                nn.init.zeros_(
                    module[-1].weight
                )
                nn.init.zeros_(
                    module[-1].bias
                )

                return module

            if self.use_boundary_specific_gate:
                self.adaptive_gate_sm = (
                    make_adaptive_gate()
                )

                self.adaptive_gate_mh = (
                    make_adaptive_gate()
                )

            else:
                # Preserve the original T3.11 structure.
                self.adaptive_gate = (
                    make_adaptive_gate()
                )

    @property
    def gate(self) -> Tensor:
        return torch.sigmoid(
            self.gate_logit
        )

    def forward(
        self,
        query_features: Tensor,
        presence_probability: Tensor,
    ) -> Tensor | tuple[Tensor, Tensor]:

        if query_features.ndim != 3:
            raise ValueError(
                "query_features must be [B,P,D], got "
                f"{tuple(query_features.shape)}"
            )

        batch_size = query_features.shape[0]

        expected_query_shape = (
            batch_size,
            self.n_queries,
            self.d_model,
        )

        if tuple(query_features.shape) != expected_query_shape:
            raise ValueError(
                f"query_features must be {expected_query_shape}, "
                f"got {tuple(query_features.shape)}"
            )

        expected_presence_shape = (
            batch_size,
            self.n_queries,
        )

        if tuple(presence_probability.shape) != expected_presence_shape:
            raise ValueError(
                "presence_probability must be "
                f"{expected_presence_shape}, got "
                f"{tuple(presence_probability.shape)}"
            )

        if not torch.isfinite(query_features).all():
            raise ValueError(
                "query_features contains NaN or Inf"
            )

        if not torch.isfinite(presence_probability).all():
            raise ValueError(
                "presence_probability contains NaN or Inf"
            )

        presence_probability = (
            presence_probability.to(
                dtype=query_features.dtype,
                device=query_features.device,
            )
            .clamp(0.0, 1.0)
        )

        # A pesticide predicted absent contributes less K/V context.
        context_source = (
            query_features
            * presence_probability.unsqueeze(-1)
        )

        # Diagonal=True masks self-attention:
        # DEL reads CHL/TEB, CHL reads DEL/TEB, etc.
        cross_query_mask = torch.eye(
            self.n_queries,
            dtype=torch.bool,
            device=query_features.device,
        )

        attended, _ = self.cross_query_attention(
            query=query_features,
            key=context_source,
            value=context_source,
            attn_mask=cross_query_mask,
            need_weights=False,
        )

        contextual = self.attention_norm(
            query_features + attended
        )

        contextual = self.ffn_norm(
            contextual
            + self.ffn(contextual)
        )

        # Soft mixture complexity.
        #
        # For 3 pesticides:
        # single  -> approximately 0
        # binary  -> approximately 0.5
        # ternary -> approximately 1
        mixture_strength = (
            (
                presence_probability.sum(
                    dim=1,
                    keepdim=True,
                )
                - 1.0
            )
            / float(self.n_queries - 1)
        ).clamp(
            0.0,
            1.0,
        )

        mixture_strength = (
            mixture_strength.unsqueeze(-1)
        )

        base_gate_logit = (
            self.gate_logit.view(
                1,
                self.n_queries,
                1,
            )
        )

        cross_query_residual = (
            contextual
            - query_features
        )

        # ====================================================
        # T3.12:
        # S/M and M/H boundaries use separate adaptive gates.
        # ====================================================
        if self.use_boundary_specific_gate:

            mixture_feature = (
                mixture_strength.expand(
                    -1,
                    self.n_queries,
                    -1,
                )
            )

            gate_features = torch.cat(
                (
                    query_features,
                    cross_query_residual,
                    presence_probability.unsqueeze(-1),
                    mixture_feature,
                ),
                dim=-1,
            )

            adjustment_sm = (
                self.adaptive_gate_sm(
                    gate_features
                )
            )

            adjustment_mh = (
                self.adaptive_gate_mh(
                    gate_features
                )
            )

            gate_sm = torch.sigmoid(
                base_gate_logit
                + adjustment_sm
            )

            gate_mh = torch.sigmoid(
                base_gate_logit
                + adjustment_mh
            )

            fused_sm = (
                query_features
                + gate_sm
                * mixture_strength
                * cross_query_residual
            )

            fused_mh = (
                query_features
                + gate_mh
                * mixture_strength
                * cross_query_residual
            )

            return (
                fused_sm,
                fused_mh,
            )

        # ====================================================
        # Legacy T3.10 / T3.11 paths
        # ====================================================
        if self.adaptive_gate is None:
            # Exact T3.10 path.
            gate = torch.sigmoid(
                base_gate_logit
            )

        else:
            # Exact T3.11 path.
            mixture_feature = (
                mixture_strength.expand(
                    -1,
                    self.n_queries,
                    -1,
                )
            )

            gate_features = torch.cat(
                (
                    query_features,
                    cross_query_residual,
                    presence_probability.unsqueeze(-1),
                    mixture_feature,
                ),
                dim=-1,
            )

            gate_adjustment = (
                self.adaptive_gate(
                    gate_features
                )
            )

            gate = torch.sigmoid(
                base_gate_logit
                + gate_adjustment
            )

        effective_gate = (
            gate
            * mixture_strength
        )

        fused = (
            query_features
            + effective_gate
            * cross_query_residual
        )

        return fused


class TransformerClassifyRegress_sep(nn.Module):
    def __init__(
        self,
        dim_model: int = 32,
        attn_head: int = 1,
        dim_ff: int = 64,
        drop: float = 0.1,
        batch_f: bool = True,
        encoder_layers: int = 1,
        n_labels: int = 3,
        model_length: int = DEFAULT_MODEL_LENGTH,
        peak_guidance_strength_init: float = 1.5,
        use_peak_guidance: bool = True,
        concentration_head_mode: str = "continuous",
        ordinal_head_hidden: int | None = None,
        use_mixture_aware_query_fusion: bool = False,
        use_adaptive_mixture_gate: bool = False,
        use_boundary_specific_mixture_gate: bool = False,
        use_local_quantitative_branch: bool = False,
        local_window_centers: Sequence[Sequence[float]] | None = None,
        local_half_width: int = 24,
        local_hidden: int = 16,
        local_strength: float = 0.25,
        use_anchored_local_calibration: bool = False,
        calibration_logit_bound: float = 0.5,
        calibration_uncertainty_band: float = 0.15,
        calibration_hidden: int = 16,
        calibration_half_width: int = 24,
    ) -> None:
        super().__init__()

        if not batch_f:
            raise ValueError("This adapted implementation requires batch_first=True")
        if (dim_model * 2) % attn_head != 0:
            raise ValueError(
                f"d_model={dim_model * 2} must be divisible by nhead={attn_head}"
            )

        self.dim_model = int(dim_model)
        self.n_labels = int(n_labels)
        self.model_length = int(model_length)
        self.concentration_head_mode = str(concentration_head_mode)
        self.use_mixture_aware_query_fusion = bool(
            use_mixture_aware_query_fusion
        )

        self.use_adaptive_mixture_gate = bool(
            use_adaptive_mixture_gate
        )

        self.use_boundary_specific_mixture_gate = bool(
            use_boundary_specific_mixture_gate
        )
        self.use_local_quantitative_branch = bool(use_local_quantitative_branch)

        if (
            self.use_adaptive_mixture_gate
            and not self.use_mixture_aware_query_fusion
        ):
            raise ValueError(
                "use_adaptive_mixture_gate=True requires "
                "use_mixture_aware_query_fusion=True"
            )

        if (
            self.use_boundary_specific_mixture_gate
            and not self.use_adaptive_mixture_gate
        ):
            raise ValueError(
                "use_boundary_specific_mixture_gate=True requires "
                "use_adaptive_mixture_gate=True"
            )

        if self.concentration_head_mode not in {"continuous", "ordinal"}:
            raise ValueError(
                "concentration_head_mode must be 'continuous' or 'ordinal', "
                f"got {self.concentration_head_mode!r}"
            )
        self.token_length = _conv_pool_output_length(self.model_length)
        feature_dim = self.dim_model * 2

        def make_signal_cnn() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(1, self.dim_model, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3),
                nn.BatchNorm1d(self.dim_model),
                nn.Conv1d(self.dim_model, feature_dim, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3),
                nn.Dropout1d(p=0.5),
            )

        # Three 1-D CNN routes: raw, smoothed and percentile features.
        self.cnn = make_signal_cnn()
        self.cnn2 = make_signal_cnn()
        self.reg_feat = regression_features(dim_model=self.dim_model, channels=4)

        self.cnn_fusion = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.LayerNorm(feature_dim),
        )
        self.register_buffer(
            "position_encoding",
            _sinusoidal_position_encoding(self.token_length, feature_dim),
            persistent=False,
        )

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=attn_head,
            dim_feedforward=dim_ff,
            dropout=drop,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.transformer_encoder_layer,
            num_layers=encoder_layers,
        )

        self.peak_guided_attention = PeakGuidedQueryAttention(
            d_model=feature_dim,
            n_queries=self.n_labels,
            n_heads=attn_head,
            model_length=self.model_length,
            dropout=drop,
            guidance_strength_init=peak_guidance_strength_init,
            use_peak_guidance=use_peak_guidance,
        )

        head_hidden = max(
            self.dim_model,
            feature_dim // 2,
        )

        if ordinal_head_hidden is None:
            resolved_ordinal_head_hidden = head_hidden
        else:
            resolved_ordinal_head_hidden = int(
                ordinal_head_hidden
            )

            if resolved_ordinal_head_hidden <= 0:
                raise ValueError(
                    "ordinal_head_hidden must be > 0, "
                    f"got {resolved_ordinal_head_hidden}"
                )

        self.ordinal_head_hidden = (
            resolved_ordinal_head_hidden
        )

        self.classification_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_dim, head_hidden),
                    nn.GELU(),
                    nn.Dropout(drop),
                    nn.Linear(head_hidden, 1),
                    nn.Sigmoid(),
                )
                for _ in range(self.n_labels)
            ]
        )
        if self.concentration_head_mode == "continuous":
            # T2-compatible scalar regression heads. Keeping these keys only in
            # continuous mode preserves strict loading of existing T1/T2 checkpoints.
            self.regression_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(feature_dim, head_hidden),
                        nn.GELU(),
                        nn.Dropout(drop),
                        nn.Linear(head_hidden, 1),
                        nn.ReLU(),
                    )
                    for _ in range(self.n_labels)
                ]
            )
        else:
            # T3.1: one pesticide-specific ordinal head per query. Each head
            # predicts two ordered cumulative events: level >= M and level >= H.
            # The second probability is parameterized conditionally so that
            # P(level>=H) <= P(level>=M) by construction.
            self.ordinal_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(
                            feature_dim,
                            self.ordinal_head_hidden,
                        ),
                        nn.GELU(),
                        nn.Dropout(drop),
                        nn.Linear(
                            self.ordinal_head_hidden,
                            2,
                        ),
                    )
                    for _ in range(self.n_labels)
                ]
            )

        self.mixture_aware_query_fusion = None

        if self.use_mixture_aware_query_fusion:
            if self.concentration_head_mode != "ordinal":
                raise ValueError(
                    "Mixture-aware query fusion currently requires "
                    "concentration_head_mode='ordinal'"
                )

            self.mixture_aware_query_fusion = (
                MixtureAwareQueryFusion(
                    d_model=feature_dim,
                    n_queries=self.n_labels,
                    n_heads=attn_head,
                    dropout=drop,
                    gate_init=0.10,
                    use_adaptive_gate=(
                        self.use_adaptive_mixture_gate
                    ),
                    use_boundary_specific_gate=(
                        self.use_boundary_specific_mixture_gate
                    ),
                )
            )

        # Construct after all legacy parameters: same seed + disabled path keeps
        # the T3.10-T3.12 state dictionary and parameter initialization unchanged.
        self.local_quantitative_branch = None
        if self.use_local_quantitative_branch:
            if self.n_labels != 3 or self.concentration_head_mode != "ordinal":
                raise ValueError("T3.13 local quantitative branch requires three ordinal pesticide heads")
            if self.use_boundary_specific_mixture_gate:
                raise ValueError("First T3.13 ablation uses shared concentration features; disable boundary-specific gate")
            centers = local_window_centers or ((1000.0, 1600.0), (2230.0,), (1090.0, 1597.0))
            self.local_quantitative_branch = LocalQuantitativeBranch(
                feature_dim, self.model_length, centers, half_width=local_half_width,
                hidden=local_hidden, strength=local_strength,
            )

        # T3.14 is a separate, optional anchored calibration path. Default OFF
        # adds no state_dict entries to the historical T3.10/T3.13 models.
        self.anchored_local_calibration = None
        if use_anchored_local_calibration:
            if (self.n_labels != 3 or self.concentration_head_mode != "ordinal"
                    or not self.use_mixture_aware_query_fusion
                    or self.use_local_quantitative_branch
                    or self.use_adaptive_mixture_gate
                    or self.use_boundary_specific_mixture_gate):
                raise ValueError("T3.14 requires the original three-head ordinal T3.10 path; disable T3.13/adaptive/boundary gates")
            from T3_14_Calibration import AnchoredLocalCalibrator
            centers = local_window_centers or ((1000.0, 1600.0), (2230.0,), (1090.0, 1597.0))
            self.anchored_local_calibration = AnchoredLocalCalibrator(
                centers, logit_bound=calibration_logit_bound,
                uncertainty_band=calibration_uncertainty_band,
                hidden=calibration_hidden, half_width=calibration_half_width,
            )
            for name, parameter in self.named_parameters():
                parameter.requires_grad_(name.startswith("anchored_local_calibration."))
            self.train(False)

    def train(self, mode: bool = True):
        if getattr(self, "anchored_local_calibration", None) is not None:
            # requires_grad=False alone does NOT freeze BatchNorm buffers or
            # disable dropout. Keep every anchor child in eval mode explicitly.
            super().train(False)
            self.anchored_local_calibration.train(mode)
            return self
        return super().train(mode)

    def set_peak_prior(self, peak_prior: Tensor) -> None:
        """Install [DEL, CHL, TEB] point-level soft priors into the model."""
        self.peak_guided_attention.set_peak_prior(peak_prior)

    @staticmethod
    def _ordered_probabilities_from_logits(logits: Tensor) -> Tensor:
        """Convert two raw ordinal logits to monotonic cumulative probabilities.

        Output order is [P(level>=M), P(level>=H)]. Multiplicative
        parameterization guarantees the chemically sensible ordering
        P(level>=H) <= P(level>=M) without a hard post-hoc correction.
        """
        if logits.shape[-1] != 2:
            raise ValueError(
                f"ordinal logits must end with dimension 2, got {tuple(logits.shape)}"
            )
        probability_ge_m = torch.sigmoid(logits[..., 0])
        probability_h_given_ge_m = torch.sigmoid(logits[..., 1])
        probability_ge_h = probability_ge_m * probability_h_given_ge_m
        return torch.stack((probability_ge_m, probability_ge_h), dim=-1)

    @staticmethod
    def ordinal_probabilities_to_expected_level(probabilities: Tensor) -> Tensor:
        """Decode cumulative probabilities to a differentiable expected level in [1,3]."""
        if probabilities.shape[-1] != 2:
            raise ValueError(
                "ordinal probabilities must end with dimension 2, "
                f"got {tuple(probabilities.shape)}"
            )
        return 1.0 + probabilities.sum(dim=-1)

    @staticmethod
    def ordinal_probabilities_to_level(
        probabilities: Tensor,
        threshold: float = 0.5,
    ) -> Tensor:
        """Decode cumulative probabilities to discrete S/M/H codes 1/2/3."""
        if probabilities.shape[-1] != 2:
            raise ValueError(
                "ordinal probabilities must end with dimension 2, "
                f"got {tuple(probabilities.shape)}"
            )
        return 1 + (probabilities >= float(threshold)).to(torch.long).sum(dim=-1)

    @staticmethod
    def _unpack_data(
        data: Mapping[str, Tensor] | Sequence[Tensor],
        valid_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        if isinstance(data, Mapping):
            raw = data["raw"]
            percentile = data["percentile"]
            smoothed = data["smoothed"]
            if valid_mask is None:
                valid_mask = data.get("valid_mask")
            return raw, percentile, smoothed, valid_mask

        if len(data) < 3:
            raise ValueError("data must contain raw, percentile and smoothed tensors")
        raw, percentile, smoothed = data[0], data[1], data[2]
        if valid_mask is None and len(data) >= 4:
            valid_mask = data[3]
        return raw, percentile, smoothed, valid_mask

    def _validate_inputs(
        self,
        raw: Tensor,
        percentile: Tensor,
        smoothed: Tensor,
        valid_mask: Tensor | None,
    ) -> Tensor:
        if raw.ndim != 3 or raw.shape[1] != 1:
            raise ValueError(f"raw must be [B,1,L], got {tuple(raw.shape)}")
        if percentile.ndim != 3 or percentile.shape[1] != 4:
            raise ValueError(
                f"percentile must be [B,4,L], got {tuple(percentile.shape)}"
            )
        if smoothed.ndim != 3 or smoothed.shape[1] != 1:
            raise ValueError(
                f"smoothed must be [B,1,L], got {tuple(smoothed.shape)}"
            )
        if not (
            raw.shape[0] == percentile.shape[0] == smoothed.shape[0]
            and raw.shape[-1] == percentile.shape[-1] == smoothed.shape[-1]
        ):
            raise ValueError("All SERSFormer branches must share batch and length")
        if raw.shape[-1] != self.model_length:
            raise ValueError(
                f"Model expects fixed length {self.model_length}, got {raw.shape[-1]}"
            )

        if valid_mask is None:
            valid_mask = torch.ones(
                raw.shape[0], self.model_length, dtype=torch.bool, device=raw.device
            )
        else:
            valid_mask = valid_mask.to(device=raw.device, dtype=torch.bool)
            if valid_mask.shape != (raw.shape[0], self.model_length):
                raise ValueError(
                    f"valid_mask must be {(raw.shape[0], self.model_length)}, "
                    f"got {tuple(valid_mask.shape)}"
                )
        return valid_mask

    @staticmethod
    def _zero_invalid_tokens(features: Tensor, token_mask: Tensor) -> Tensor:
        return features.masked_fill(~token_mask.unsqueeze(-1), 0.0)

    def forward(
        self,
        data: Mapping[str, Tensor] | Sequence[Tensor],
        valid_mask: Tensor | None = None,
        return_attention: bool = False,
        return_ordinal: bool = False,
        return_ordinal_logits: bool = False,
    ) -> list[Tensor]:
        raw, percentile, smoothed, valid_mask = self._unpack_data(data, valid_mask)
        valid_mask = self._validate_inputs(raw, percentile, smoothed, valid_mask)

        token_mask = _downsample_valid_mask(valid_mask)
        if token_mask.shape[1] != self.token_length:
            raise RuntimeError(
                f"Downsampled mask length {token_mask.shape[1]} != expected "
                f"{self.token_length}"
            )
        key_padding_mask = ~token_mask

        # Three CNN routes are aligned token-by-token before the Transformer.
        raw_tokens = self.cnn(raw).transpose(1, 2)
        smooth_tokens = self.cnn2(smoothed).transpose(1, 2)
        percentile_tokens = self.reg_feat(percentile).transpose(1, 2)

        raw_tokens = self._zero_invalid_tokens(raw_tokens, token_mask)
        smooth_tokens = self._zero_invalid_tokens(smooth_tokens, token_mask)
        percentile_tokens = self._zero_invalid_tokens(percentile_tokens, token_mask)

        fused = torch.cat(
            (raw_tokens, smooth_tokens, percentile_tokens), dim=-1
        )
        fused = self.cnn_fusion(fused)
        fused = fused + self.position_encoding.to(dtype=fused.dtype, device=fused.device)
        fused = self._zero_invalid_tokens(fused, token_mask)

        encoded = self.transformer_encoder(
            fused,
            src_key_padding_mask=key_padding_mask,
        )
        encoded = self._zero_invalid_tokens(encoded, token_mask)

        query_features, attention = self.peak_guided_attention(encoded, token_mask)

        class_columns = [
            head(query_features[:, pesticide_index, :])
            for pesticide_index, head in enumerate(self.classification_heads)
        ]
        classify = torch.cat(class_columns, dim=1)

        # Classification remains on the original pesticide-specific query
        # representations. Only concentration prediction receives the
        # mixture-aware cross-query representation.
        concentration_query_features = query_features

        concentration_query_features_sm: Tensor | None = None
        concentration_query_features_mh: Tensor | None = None

        if self.mixture_aware_query_fusion is not None:

            fusion_output = (
                self.mixture_aware_query_fusion(
                    query_features,
                    classify.detach(),
                )
            )

            if self.use_boundary_specific_mixture_gate:

                if (
                    not isinstance(fusion_output, tuple)
                    or len(fusion_output) != 2
                ):
                    raise RuntimeError(
                        "Boundary-specific mixture fusion must return "
                        "(S/M features, M/H features)"
                    )

                (
                    concentration_query_features_sm,
                    concentration_query_features_mh,
                ) = fusion_output

            else:

                if isinstance(fusion_output, tuple):
                    raise RuntimeError(
                        "Legacy mixture fusion unexpectedly returned "
                        "boundary-specific features"
                    )

                concentration_query_features = (
                    fusion_output
                )

        if self.local_quantitative_branch is not None:
            if not isinstance(data, Mapping) or "raw_intensity" not in data:
                raise ValueError("T3.13 requires raw_intensity from the updated Dataset.py")
            local_correction = self.local_quantitative_branch(data["raw_intensity"], valid_mask)
            concentration_query_features = concentration_query_features + local_correction

        ordinal_probabilities: Tensor | None = None
        ordinal_logits: Tensor | None = None

        if self.concentration_head_mode == "continuous":
            regression_columns = [
                head(concentration_query_features[:, pesticide_index, :])
                for pesticide_index, head in enumerate(self.regression_heads)
            ]
            regress = torch.cat(regression_columns, dim=1)
        else:
            if self.use_boundary_specific_mixture_gate:

                if (
                    concentration_query_features_sm is None
                    or concentration_query_features_mh is None
                ):
                    raise RuntimeError(
                        "Boundary-specific ordinal features are unavailable"
                    )

                boundary_logits = []

                for pesticide_index, head in enumerate(
                    self.ordinal_heads
                ):

                    sm_outputs = head(
                        concentration_query_features_sm[
                            :,
                            pesticide_index,
                            :,
                        ]
                    )

                    mh_outputs = head(
                        concentration_query_features_mh[
                            :,
                            pesticide_index,
                            :,
                        ]
                    )

                    boundary_logits.append(
                        torch.stack(
                            (
                                sm_outputs[:, 0],
                                mh_outputs[:, 1],
                            ),
                            dim=-1,
                        )
                    )

                ordinal_logits = torch.stack(
                    boundary_logits,
                    dim=1,
                )

            else:

                ordinal_logits = torch.stack(
                    [
                        head(
                            concentration_query_features[
                                :,
                                pesticide_index,
                                :,
                            ]
                        )
                        for pesticide_index, head in enumerate(
                            self.ordinal_heads
                        )
                    ],
                    dim=1,
                )

            if self.anchored_local_calibration is not None:
                if not isinstance(data, Mapping) or "raw_intensity" not in data:
                    raise ValueError("T3.14 requires raw_intensity from the updated Dataset.py")
                ordinal_logits = self.anchored_local_calibration(
                    data["raw_intensity"], valid_mask, ordinal_logits,
                )

            ordinal_probabilities = self._ordered_probabilities_from_logits(
                ordinal_logits
            )
            # Keep the public concentration tensor [B,3] so the existing
            # evaluation/plotting pipeline can still report level-code MAE/RMSE.
            # This is the conditional expected S/M/H code, not a physical concentration.
            regress = self.ordinal_probabilities_to_expected_level(
                ordinal_probabilities
            )

        if return_ordinal_logits and not return_ordinal:
            raise RuntimeError(
                "return_ordinal_logits=True also requires return_ordinal=True"
            )

        if return_attention and return_ordinal:
            if ordinal_probabilities is None:
                raise RuntimeError(
                    "return_ordinal=True requires concentration_head_mode='ordinal'"
                )

            if return_ordinal_logits:
                if ordinal_logits is None:
                    raise RuntimeError(
                        "ordinal logits are unavailable in continuous mode"
                    )
                return [
                    classify,
                    regress,
                    attention,
                    ordinal_probabilities,
                    ordinal_logits,
                ]

            return [
                classify,
                regress,
                attention,
                ordinal_probabilities,
            ]

        if return_attention:
            return [classify, regress, attention]

        if return_ordinal:
            if ordinal_probabilities is None:
                raise RuntimeError(
                    "return_ordinal=True requires concentration_head_mode='ordinal'"
                )

            if return_ordinal_logits:
                if ordinal_logits is None:
                    raise RuntimeError(
                        "ordinal logits are unavailable in continuous mode"
                    )
                return [
                    classify,
                    regress,
                    ordinal_probabilities,
                    ordinal_logits,
                ]

            return [
                classify,
                regress,
                ordinal_probabilities,
            ]

        return [classify, regress]
