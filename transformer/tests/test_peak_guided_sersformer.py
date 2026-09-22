import torch

from Model_v2 import TransformerClassifyRegress_sep, _downsample_valid_mask


def test_peak_guided_attention_respects_valid_mask_and_keeps_nonpeak_visible():
    torch.manual_seed(2026)
    model = TransformerClassifyRegress_sep(
        dim_model=16,
        attn_head=4,
        dim_ff=64,
        encoder_layers=1,
        n_labels=3,
        model_length=1901,
        peak_guidance_strength_init=1.5,
    )

    axis = torch.arange(600.0, 2501.0)
    centers = torch.tensor([1000.0, 2235.0, 1090.0])[:, None]
    prior = torch.exp(-0.5 * ((axis[None, :] - centers) / 10.0) ** 2)
    model.set_peak_prior(prior)

    valid_mask = torch.ones(2, 1901, dtype=torch.bool)
    valid_mask[1, 1401:] = False
    batch = {
        "raw": torch.rand(2, 1, 1901),
        "percentile": torch.rand(2, 4, 1901),
        "smoothed": torch.rand(2, 1, 1901),
        "valid_mask": valid_mask,
    }

    class_pred, reg_pred, attention = model(batch, return_attention=True)
    token_mask = _downsample_valid_mask(valid_mask)

    assert class_pred.shape == (2, 3)
    assert reg_pred.shape == (2, 3)
    assert attention.shape == (2, 3, 210)
    assert int(token_mask[0].sum()) == 210
    assert int(token_mask[1].sum()) == 154

    # A 1401-point spectrum must never attend to padded 2000-2500 cm^-1 tokens.
    assert torch.count_nonzero(attention[1, :, ~token_mask[1]]) == 0

    # Peak guidance is soft: a valid non-peak token is still visible.
    assert torch.all(attention[1, :, 5] > 0.0)

    (class_pred.mean() + reg_pred.mean()).backward()
    assert model.peak_guided_attention.query_embedding.grad is not None
    assert model.peak_guided_attention.guidance_log_strength.grad is not None


def test_peak_prior_is_saved_in_model_state_dict():
    model = TransformerClassifyRegress_sep(
        dim_model=8,
        attn_head=2,
        dim_ff=32,
        encoder_layers=1,
        n_labels=3,
        model_length=1901,
    )
    prior = torch.zeros(3, 1901)
    prior[0, 400] = 1.0
    prior[1, 1635] = 1.0
    prior[2, 490] = 1.0
    model.set_peak_prior(prior)

    state = model.state_dict()
    key = "peak_guided_attention.peak_prior_points"
    assert key in state
    assert torch.equal(state[key].cpu(), prior)
