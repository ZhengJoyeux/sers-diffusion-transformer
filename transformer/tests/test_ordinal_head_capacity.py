import torch

from Model_v2 import TransformerClassifyRegress_sep


def _build_ordinal_model(
    *,
    ordinal_head_hidden=None,
):
    return TransformerClassifyRegress_sep(
        dim_model=32,
        attn_head=4,
        dim_ff=64,
        encoder_layers=1,
        n_labels=3,
        concentration_head_mode="ordinal",
        ordinal_head_hidden=ordinal_head_hidden,
        use_mixture_aware_query_fusion=False,
    )


def test_default_ordinal_hidden_preserves_legacy_shape():
    model = _build_ordinal_model()

    assert model.ordinal_head_hidden == 32

    for head in model.ordinal_heads:
        assert head[0].in_features == 64
        assert head[0].out_features == 32
        assert head[3].in_features == 32
        assert head[3].out_features == 2


def test_t3_6_expands_only_ordinal_hidden_dimension():
    model = _build_ordinal_model(
        ordinal_head_hidden=64,
    )

    assert model.ordinal_head_hidden == 64

    for head in model.ordinal_heads:
        assert head[0].in_features == 64
        assert head[0].out_features == 64
        assert head[3].in_features == 64
        assert head[3].out_features == 2

    # Classification heads must remain unchanged:
    # 64 -> 32 -> 1.
    for head in model.classification_heads:
        assert head[0].in_features == 64
        assert head[0].out_features == 32
        assert head[3].in_features == 32
        assert head[3].out_features == 1


def test_legacy_config_without_new_key_strict_loads():
    legacy_config = {
        "dim_model": 32,
        "attn_head": 4,
        "dim_ff": 64,
        "encoder_layers": 1,
        "n_labels": 3,
        "concentration_head_mode": "ordinal",
        "use_mixture_aware_query_fusion": False,
    }

    original = TransformerClassifyRegress_sep(
        **legacy_config
    )

    state = original.state_dict()

    reconstructed = TransformerClassifyRegress_sep(
        **legacy_config
    )

    reconstructed.load_state_dict(
        state,
        strict=True,
    )

    assert reconstructed.ordinal_head_hidden == 32


def test_t3_6_config_strict_roundtrip():
    config = {
        "dim_model": 32,
        "attn_head": 4,
        "dim_ff": 64,
        "encoder_layers": 1,
        "n_labels": 3,
        "concentration_head_mode": "ordinal",
        "ordinal_head_hidden": 64,
        "use_mixture_aware_query_fusion": False,
    }

    original = TransformerClassifyRegress_sep(
        **config
    )

    reconstructed = TransformerClassifyRegress_sep(
        **config
    )

    reconstructed.load_state_dict(
        original.state_dict(),
        strict=True,
    )

    assert reconstructed.ordinal_head_hidden == 64


def test_invalid_ordinal_hidden_is_rejected():
    try:
        _build_ordinal_model(
            ordinal_head_hidden=0,
        )
    except ValueError as exc:
        assert "ordinal_head_hidden" in str(exc)
    else:
        raise AssertionError(
            "Expected ordinal_head_hidden=0 "
            "to raise ValueError"
        )
