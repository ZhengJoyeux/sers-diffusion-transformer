import numpy as np
import torch

import Dataset
from Model_v2 import TransformerClassifyRegress_sep


def test_full_raman_axis_is_600_to_2500() -> None:
    assert Dataset.MODEL_LENGTH == 1901
    assert float(Dataset.MODEL_RAMAN_AXIS[0]) == 600.0
    assert float(Dataset.MODEL_RAMAN_AXIS[-1]) == 2500.0


def test_chemistry_core_peak_centers_are_fixed() -> None:
    expected = {
        "DEL": [1000.0, 1600.0],
        "CHL": [2230.0],
        "TEB": [1090.0, 1597.0],
    }
    for pesticide, centers in expected.items():
        observed = [
            float(item["center_cm-1"])
            for item in Dataset.CHEMISTRY_CORE_PEAKS[pesticide]
        ]
        assert observed == centers


def test_chl_2230_core_prior_has_plus_minus_7_support() -> None:
    prior = Dataset._truncated_core_gaussian(
        center_cm1=2230.0,
        half_width_cm1=7.0,
        weight=1.0,
    )
    axis = Dataset.MODEL_RAMAN_AXIS
    center_index = int(np.where(axis == 2230.0)[0][0])
    assert prior[center_index] == 1.0
    assert np.all(prior[(axis < 2223.0) | (axis > 2237.0)] == 0.0)
    assert np.all(prior[(axis >= 2223.0) & (axis <= 2237.0)] > 0.0)


def test_del_teb_shared_core_windows_overlap() -> None:
    half_width = 7.0
    del_interval = (1600.0 - half_width, 1600.0 + half_width)
    teb_interval = (1597.0 - half_width, 1597.0 + half_width)
    overlap_lower = max(del_interval[0], teb_interval[0])
    overlap_upper = min(del_interval[1], teb_interval[1])
    assert overlap_lower == 1593.0
    assert overlap_upper == 1604.0
    assert overlap_lower <= overlap_upper


def test_chemistry_core_intervals_are_excluded_from_auxiliary_search() -> None:
    intervals = Dataset._chemistry_core_intervals(7.0)
    for center in (1000.0, 1600.0, 2230.0, 1090.0, 1597.0):
        assert any(lower <= center <= upper for lower, upper in intervals)


def test_1901_point_model_forward_ordinal_mode() -> None:
    model = TransformerClassifyRegress_sep(
        dim_model=32,
        attn_head=4,
        dim_ff=64,
        drop=0.1,
        batch_f=True,
        encoder_layers=1,
        n_labels=3,
        model_length=1901,
        peak_guidance_strength_init=1.5,
        use_peak_guidance=True,
        concentration_head_mode="ordinal",
    )

    prior = np.zeros((3, 1901), dtype=np.float32)
    prior[0] = Dataset._truncated_core_gaussian(1000.0, 7.0, 1.0)
    prior[1] = Dataset._truncated_core_gaussian(2230.0, 7.0, 1.0)
    prior[2] = Dataset._truncated_core_gaussian(1090.0, 7.0, 1.0)
    model.set_peak_prior(torch.from_numpy(prior))

    batch_size = 2
    batch = {
        "raw": torch.rand(batch_size, 1, 1901),
        "percentile": torch.rand(batch_size, 4, 1901),
        "smoothed": torch.rand(batch_size, 1, 1901),
        "valid_mask": torch.ones(batch_size, 1901, dtype=torch.bool),
    }

    with torch.no_grad():
        classification, concentration, attention, ordinal = model(
            batch,
            return_attention=True,
            return_ordinal=True,
        )

    assert tuple(classification.shape) == (2, 3)
    assert tuple(concentration.shape) == (2, 3)
    assert tuple(ordinal.shape) == (2, 3, 2)
    assert attention.shape[0] == 2
    assert attention.shape[1] == 3
