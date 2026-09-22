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
- Training-data-derived peak priors add a positive soft bias to attention logits;
  they never delete non-peak regions.
- valid_mask remains authoritative for any unavailable Raman positions.
- The formal downstream model uses the common 600-2000 cm^-1 range to prevent
  source-axis length from leaking the CHL label.
- The public forward interface remains [classification_probability, concentration].
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


DEFAULT_MODEL_LENGTH = 1401


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

        head_hidden = max(self.dim_model, feature_dim // 2)
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

    def set_peak_prior(self, peak_prior: Tensor) -> None:
        """Install [DEL, CHL, TEB] point-level soft priors into the model."""
        self.peak_guided_attention.set_peak_prior(peak_prior)

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
        regression_columns = [
            head(query_features[:, pesticide_index, :])
            for pesticide_index, head in enumerate(self.regression_heads)
        ]
        classify = torch.cat(class_columns, dim=1)
        regress = torch.cat(regression_columns, dim=1)

        if return_attention:
            return [classify, regress, attention]
        return [classify, regress]
