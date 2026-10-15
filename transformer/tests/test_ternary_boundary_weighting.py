import torch

from SERSFormer_Training import (
    present_only_boundary_margin_loss,
)


def test_unit_weights_preserve_original_boundary_mean():
    probability = torch.tensor(
        [
            [[0.40, 0.10]],
            [[0.90, 0.10]],
        ],
        dtype=torch.float32,
    )

    concentration = torch.tensor(
        [
            [1.0],
            [1.0],
        ],
        dtype=torch.float32,
    )

    presence = torch.ones(
        2,
        1,
        dtype=torch.float32,
    )

    unweighted = present_only_boundary_margin_loss(
        probability,
        concentration,
        presence,
        margin=0.15,
    )

    weighted_with_ones = (
        present_only_boundary_margin_loss(
            probability,
            concentration,
            presence,
            margin=0.15,
            sample_weight=torch.ones(2),
        )
    )

    assert torch.allclose(
        unweighted,
        weighted_with_ones,
        atol=1e-8,
        rtol=0.0,
    )


def test_hard_sample_receives_more_boundary_weight():
    # S target:
    # low boundary = 0.35.
    #
    # sample 0: p>=M = 0.40 -> small penalty
    # sample 1: p>=M = 0.90 -> large penalty
    probability = torch.tensor(
        [
            [[0.40, 0.10]],
            [[0.90, 0.10]],
        ],
        dtype=torch.float32,
    )

    concentration = torch.tensor(
        [
            [1.0],
            [1.0],
        ],
        dtype=torch.float32,
    )

    presence = torch.ones(
        2,
        1,
        dtype=torch.float32,
    )

    baseline = present_only_boundary_margin_loss(
        probability,
        concentration,
        presence,
        margin=0.15,
    )

    weighted = present_only_boundary_margin_loss(
        probability,
        concentration,
        presence,
        margin=0.15,
        sample_weight=torch.tensor(
            [1.0, 1.5],
            dtype=torch.float32,
        ),
    )

    assert weighted > baseline


def test_invalid_sample_weight_shape_is_rejected():
    probability = torch.tensor(
        [[[0.50, 0.20]]],
        dtype=torch.float32,
    )

    concentration = torch.tensor(
        [[2.0]],
        dtype=torch.float32,
    )

    presence = torch.ones(
        1,
        1,
        dtype=torch.float32,
    )

    try:
        present_only_boundary_margin_loss(
            probability,
            concentration,
            presence,
            sample_weight=torch.ones(
                1,
                1,
            ),
        )
    except ValueError as exc:
        assert "sample_weight" in str(exc)
    else:
        raise AssertionError(
            "Expected invalid sample_weight shape "
            "to raise ValueError"
        )
