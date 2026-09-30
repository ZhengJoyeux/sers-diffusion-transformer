import torch

from Model_v2 import (
    MixtureAwareQueryFusion,
    TransformerClassifyRegress_sep,
)


def test_single_strength_preserves_query_exactly():
    torch.manual_seed(1)

    fusion = MixtureAwareQueryFusion(
        d_model=8,
        n_queries=3,
        n_heads=2,
        dropout=0.0,
        gate_init=0.10,
    )
    fusion.eval()

    query = torch.randn(
        2,
        3,
        8,
    )

    presence = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    output = fusion(
        query,
        presence,
    )

    assert torch.equal(
        output,
        query,
    )


def test_ternary_fusion_is_differentiable():
    torch.manual_seed(2)

    fusion = MixtureAwareQueryFusion(
        d_model=8,
        n_queries=3,
        n_heads=2,
        dropout=0.0,
        gate_init=0.10,
    )

    query = torch.randn(
        2,
        3,
        8,
        requires_grad=True,
    )

    presence = torch.ones(
        2,
        3,
        dtype=torch.float32,
    )

    output = fusion(
        query,
        presence,
    )

    assert output.shape == query.shape
    assert torch.isfinite(output).all()

    output.square().mean().backward()

    assert query.grad is not None
    assert torch.isfinite(
        query.grad
    ).all()

    assert (
        fusion.gate_logit.grad
        is not None
    )


def test_disabled_mode_preserves_old_checkpoint_structure():
    model = TransformerClassifyRegress_sep(
        dim_model=8,
        attn_head=2,
        dim_ff=16,
        drop=0.0,
        encoder_layers=1,
        n_labels=3,
        model_length=64,
        concentration_head_mode="ordinal",
    )

    assert (
        model.mixture_aware_query_fusion
        is None
    )

    state = model.state_dict()

    assert not any(
        key.startswith(
            "mixture_aware_query_fusion."
        )
        for key in state
    )

    restored = TransformerClassifyRegress_sep(
        dim_model=8,
        attn_head=2,
        dim_ff=16,
        drop=0.0,
        encoder_layers=1,
        n_labels=3,
        model_length=64,
        concentration_head_mode="ordinal",
    )

    restored.load_state_dict(
        state,
        strict=True,
    )


def test_enabled_model_keeps_public_ordinal_interface():
    torch.manual_seed(3)

    model = TransformerClassifyRegress_sep(
        dim_model=8,
        attn_head=2,
        dim_ff=16,
        drop=0.0,
        encoder_layers=1,
        n_labels=3,
        model_length=64,
        concentration_head_mode="ordinal",
        use_mixture_aware_query_fusion=True,
    )

    model.eval()

    batch = {
        "raw": torch.randn(
            2,
            1,
            64,
        ),
        "smoothed": torch.randn(
            2,
            1,
            64,
        ),
        "percentile": torch.randn(
            2,
            4,
            64,
        ),
        "valid_mask": torch.ones(
            2,
            64,
            dtype=torch.bool,
        ),
    }

    with torch.no_grad():
        classify, regress, ordinal = model(
            batch,
            return_ordinal=True,
        )

    assert classify.shape == (
        2,
        3,
    )

    assert regress.shape == (
        2,
        3,
    )

    assert ordinal.shape == (
        2,
        3,
        2,
    )

    assert torch.isfinite(
        classify
    ).all()

    assert torch.isfinite(
        regress
    ).all()

    assert torch.isfinite(
        ordinal
    ).all()

    assert torch.all(
        ordinal[..., 1]
        <= ordinal[..., 0]
        + 1e-7
    )
