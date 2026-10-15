"""Existing spectral heads/fusion, extracted unchanged from the verified T3.26 core."""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

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

def _downsample_peak_prior(peak_prior: Tensor, pooling_stride: int = 3) -> Tensor:
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
        prior = F.max_pool1d(prior, kernel_size=3, stride=pooling_stride)
    prior = prior.squeeze(1)

    maximum = prior.amax(dim=-1, keepdim=True).clamp_min(1e-8)
    prior = torch.where(maximum > 1e-8, prior / maximum, prior)
    return prior.clamp_(0.0, 1.0)

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
        pooling_stride: int = 3,
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
        self.pooling_stride = int(pooling_stride)

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

        peak_prior_tokens = _downsample_peak_prior(self.peak_prior_points, self.pooling_stride)
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
