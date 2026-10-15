import torch

from SERSFormer_Training import present_only_boundary_margin_loss


def test_boundary_loss_zero_for_well_separated_levels():
    probability = torch.tensor(
        [[
            [0.20, 0.05],
            [0.80, 0.20],
            [0.90, 0.80],
        ]],
        dtype=torch.float32,
    )
    target = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    presence = torch.ones_like(target)

    loss = present_only_boundary_margin_loss(
        probability,
        target,
        presence,
        margin=0.15,
    )

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-8)


def test_boundary_loss_penalizes_middle_collapse():
    probability = torch.tensor(
        [[
            [0.60, 0.20],
            [0.55, 0.50],
            [0.80, 0.40],
        ]],
        dtype=torch.float32,
        requires_grad=True,
    )
    target = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    presence = torch.ones_like(target)

    loss = present_only_boundary_margin_loss(
        probability,
        target,
        presence,
        margin=0.15,
    )

    assert float(loss.detach()) > 0.0
    loss.backward()
    assert probability.grad is not None
    assert torch.isfinite(probability.grad).all()


def test_boundary_loss_ignores_absent_targets():
    probability = torch.tensor(
        [[[0.99, 0.99]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    target = torch.tensor([[0.0]], dtype=torch.float32)
    presence = torch.tensor([[0.0]], dtype=torch.float32)

    loss = present_only_boundary_margin_loss(
        probability,
        target,
        presence,
        margin=0.15,
    )

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-8)


def test_boundary_loss_rejects_invalid_margin():
    probability = torch.zeros(1, 1, 2, dtype=torch.float32)
    target = torch.ones(1, 1, dtype=torch.float32)
    presence = torch.ones_like(target)

    try:
        present_only_boundary_margin_loss(
            probability,
            target,
            presence,
            margin=0.5,
        )
    except ValueError:
        return

    raise AssertionError("Expected ValueError for margin >= 0.5")
