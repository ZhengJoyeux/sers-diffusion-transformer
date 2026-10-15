import pytest
import torch

from Model_v2 import (
    MixtureAwareQueryFusion,
    TransformerClassifyRegress_sep,
)


def test_adaptive_gate_initializes_as_exact_t310_gate():
    torch.manual_seed(2026)

    legacy = MixtureAwareQueryFusion(
        d_model=64,
        n_queries=3,
        n_heads=4,
        dropout=0.0,
        gate_init=0.10,
        use_adaptive_gate=False,
    )

    adaptive = MixtureAwareQueryFusion(
        d_model=64,
        n_queries=3,
        n_heads=4,
        dropout=0.0,
        gate_init=0.10,
        use_adaptive_gate=True,
    )

    result = adaptive.load_state_dict(
        legacy.state_dict(),
        strict=False,
    )

    assert result.unexpected_keys == []

    assert all(
        key.startswith("adaptive_gate.")
        for key in result.missing_keys
    )

    legacy.eval()
    adaptive.eval()

    query = torch.randn(
        5,
        3,
        64,
    )

    presence = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.90, 0.80, 0.20],
            [0.95, 0.90, 0.85],
        ],
        dtype=torch.float32,
    )

    with torch.no_grad():
        legacy_output = legacy(
            query,
            presence,
        )

        adaptive_output = adaptive(
            query,
            presence,
        )

    assert torch.allclose(
        legacy_output,
        adaptive_output,
        atol=1e-6,
        rtol=1e-6,
    )


def test_adaptive_gate_receives_gradient():
    torch.manual_seed(2026)

    module = MixtureAwareQueryFusion(
        d_model=64,
        n_queries=3,
        n_heads=4,
        dropout=0.0,
        gate_init=0.10,
        use_adaptive_gate=True,
    )

    query = torch.randn(
        8,
        3,
        64,
        requires_grad=True,
    )

    presence = torch.tensor(
        [[1.0, 1.0, 1.0]] * 8,
        dtype=torch.float32,
    )

    output = module(
        query,
        presence,
    )

    loss = output.square().mean()
    loss.backward()

    final_linear = module.adaptive_gate[-1]

    assert final_linear.weight.grad is not None

    assert torch.isfinite(
        final_linear.weight.grad
    ).all()

    assert (
        final_linear.weight.grad.abs().sum()
        > 0
    )


def test_adaptive_gate_requires_mixture_fusion():
    with pytest.raises(
        ValueError,
        match="requires",
    ):
        TransformerClassifyRegress_sep(
            concentration_head_mode="ordinal",
            use_mixture_aware_query_fusion=False,
            use_adaptive_mixture_gate=True,
        )


def test_default_t310_path_remains_non_adaptive():
    model = TransformerClassifyRegress_sep(
        concentration_head_mode="ordinal",
        use_mixture_aware_query_fusion=True,
    )

    assert model.use_adaptive_mixture_gate is False

    assert (
        model.mixture_aware_query_fusion
        is not None
    )

    assert (
        model.mixture_aware_query_fusion.adaptive_gate
        is None
    )
