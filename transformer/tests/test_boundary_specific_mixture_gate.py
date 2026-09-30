import pytest
import torch

from Model_v2 import (
    MixtureAwareQueryFusion,
    TransformerClassifyRegress_sep,
)


def test_boundary_specific_fusion_requires_adaptive_gate():
    """T3.12 cannot be enabled without the T3.11 adaptive mechanism."""
    with pytest.raises(
        ValueError,
        match="requires",
    ):
        MixtureAwareQueryFusion(
            d_model=64,
            n_queries=3,
            n_heads=4,
            dropout=0.0,
            use_adaptive_gate=False,
            use_boundary_specific_gate=True,
        )


def test_main_model_boundary_specific_requires_adaptive_gate():
    """Main-model configuration must reject an invalid T3.12 combination."""
    with pytest.raises(
        ValueError,
        match="requires",
    ):
        TransformerClassifyRegress_sep(
            concentration_head_mode="ordinal",
            use_mixture_aware_query_fusion=True,
            use_adaptive_mixture_gate=False,
            use_boundary_specific_mixture_gate=True,
        )


def test_t312_initial_sm_mh_outputs_match_t310_base_gate():
    """
    Both T3.12 adaptive corrections are zero-initialized.

    Therefore, before training:
        S/M fused feature
        M/H fused feature
    must both reduce to the original T3.10 base-gate behavior.
    """
    torch.manual_seed(2026)

    t310 = MixtureAwareQueryFusion(
        d_model=64,
        n_queries=3,
        n_heads=4,
        dropout=0.0,
        gate_init=0.10,
        use_adaptive_gate=False,
        use_boundary_specific_gate=False,
    )

    torch.manual_seed(2026)

    t312 = MixtureAwareQueryFusion(
        d_model=64,
        n_queries=3,
        n_heads=4,
        dropout=0.0,
        gate_init=0.10,
        use_adaptive_gate=True,
        use_boundary_specific_gate=True,
    )

    t310.eval()
    t312.eval()

    query = torch.randn(
        6,
        3,
        64,
    )

    presence = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.95, 0.90, 0.85],
            [0.90, 0.80, 0.20],
            [0.99, 0.98, 0.97],
        ],
        dtype=torch.float32,
    )

    with torch.no_grad():
        base_output = t310(
            query,
            presence,
        )

        sm_output, mh_output = t312(
            query,
            presence,
        )

    assert base_output.shape == (
        6,
        3,
        64,
    )

    assert sm_output.shape == base_output.shape
    assert mh_output.shape == base_output.shape

    assert torch.allclose(
        sm_output,
        base_output,
        atol=1e-6,
        rtol=1e-6,
    )

    assert torch.allclose(
        mh_output,
        base_output,
        atol=1e-6,
        rtol=1e-6,
    )


def test_t312_sm_and_mh_gates_both_receive_gradient():
    """Both ordinal-boundary adaptive gates must participate in backprop."""
    torch.manual_seed(2026)

    module = MixtureAwareQueryFusion(
        d_model=64,
        n_queries=3,
        n_heads=4,
        dropout=0.0,
        gate_init=0.10,
        use_adaptive_gate=True,
        use_boundary_specific_gate=True,
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

    sm_output, mh_output = module(
        query,
        presence,
    )

    # Different objectives prevent the two branches from being
    # artificially forced into exactly identical gradients.
    loss = (
        sm_output.square().mean()
        + 0.7 * mh_output.abs().mean()
    )

    loss.backward()

    sm_final = module.adaptive_gate_sm[-1]
    mh_final = module.adaptive_gate_mh[-1]

    assert sm_final.weight.grad is not None
    assert mh_final.weight.grad is not None

    assert torch.isfinite(
        sm_final.weight.grad
    ).all()

    assert torch.isfinite(
        mh_final.weight.grad
    ).all()

    assert (
        sm_final.weight.grad.abs().sum()
        > 0
    )

    assert (
        mh_final.weight.grad.abs().sum()
        > 0
    )


def test_t312_main_model_builds_two_boundary_gates():
    """The complete T3.12 model must build SM and MH gates, not the T3.11 gate."""
    model = TransformerClassifyRegress_sep(
        dim_model=16,
        attn_head=4,
        dim_ff=32,
        encoder_layers=1,
        concentration_head_mode="ordinal",
        ordinal_head_hidden=32,
        use_mixture_aware_query_fusion=True,
        use_adaptive_mixture_gate=True,
        use_boundary_specific_mixture_gate=True,
    )

    fusion = model.mixture_aware_query_fusion

    assert model.use_boundary_specific_mixture_gate is True

    assert fusion is not None
    assert fusion.use_boundary_specific_gate is True

    # T3.12 uses two distinct adaptive paths.
    assert fusion.adaptive_gate is None
    assert fusion.adaptive_gate_sm is not None
    assert fusion.adaptive_gate_mh is not None

    # They must actually be distinct parameter modules.
    assert (
        fusion.adaptive_gate_sm
        is not fusion.adaptive_gate_mh
    )
