from pathlib import Path

import numpy as np
import torch

from Inference import _save_query_attention_profiles
from Model_v2 import TransformerClassifyRegress_sep


def _batch(length: int = 1401):
    return {
        "raw": torch.rand(2, 1, length),
        "percentile": torch.rand(2, 4, length),
        "smoothed": torch.rand(2, 1, length),
        "valid_mask": torch.ones(2, length, dtype=torch.bool),
    }


def test_query_only_mode_is_independent_of_peak_prior():
    torch.manual_seed(2026)
    model = TransformerClassifyRegress_sep(
        dim_model=8,
        attn_head=2,
        dim_ff=32,
        encoder_layers=1,
        n_labels=3,
        model_length=1401,
        use_peak_guidance=False,
    )
    model.eval()
    batch = _batch()

    with torch.no_grad():
        class_before, reg_before, attention_before = model(
            batch, return_attention=True
        )

    prior = torch.zeros(3, 1401)
    prior[0, 400] = 1.0
    prior[1, 700] = 1.0
    prior[2, 1000] = 1.0
    model.set_peak_prior(prior)

    with torch.no_grad():
        class_after, reg_after, attention_after = model(
            batch, return_attention=True
        )

    torch.testing.assert_close(class_before, class_after)
    torch.testing.assert_close(reg_before, reg_after)
    torch.testing.assert_close(attention_before, attention_after)
    assert torch.count_nonzero(
        model.peak_guided_attention.effective_guidance_strength
    ) == 0


def test_query_attention_profile_exports(tmp_path: Path):
    rng = np.random.default_rng(2026)
    attention = rng.random((6, 3, 154))
    attention /= attention.sum(axis=-1, keepdims=True)
    y_true = np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 1],
            [1, 1, 1],
            [0, 0, 1],
            [1, 0, 1],
        ],
        dtype=np.int64,
    )

    _save_query_attention_profiles(attention, y_true, tmp_path)

    assert (tmp_path / "query_attention_profiles.csv").exists()
    assert (tmp_path / "query_attention_top_regions.csv").exists()
    for pesticide in ("DEL", "CHL", "TEB"):
        assert (tmp_path / f"{pesticide}_query_attention.png").exists()
