import math

import torch

from Model_v2 import present_only_corn_ordinal_loss


def test_corn_zero_logits_have_log2_loss():
    logits = torch.zeros(
        (3, 1, 2),
        dtype=torch.float32,
        requires_grad=True,
    )

    concentration = torch.tensor(
        [[1.0], [2.0], [3.0]],
        dtype=torch.float32,
    )

    presence = torch.ones(
        (3, 1),
        dtype=torch.float32,
    )

    loss = present_only_corn_ordinal_loss(
        logits,
        concentration,
        presence,
    )

    expected = torch.tensor(
        math.log(2.0),
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        loss.detach(),
        expected,
    )


def test_corn_does_not_train_second_boundary_on_S():
    logits = torch.zeros(
        (3, 1, 2),
        dtype=torch.float32,
        requires_grad=True,
    )

    concentration = torch.tensor(
        [[1.0], [2.0], [3.0]],
        dtype=torch.float32,
    )

    presence = torch.ones(
        (3, 1),
        dtype=torch.float32,
    )

    loss = present_only_corn_ordinal_loss(
        logits,
        concentration,
        presence,
    )

    loss.backward()

    assert logits.grad is not None

    # S participates in S/M, but not in conditional M/H.
    assert float(logits.grad[0, 0, 0]) != 0.0
    assert float(logits.grad[0, 0, 1]) == 0.0

    # M and H both participate in the M/H conditional task.
    assert float(logits.grad[1, 0, 1]) != 0.0
    assert float(logits.grad[2, 0, 1]) != 0.0


def test_corn_ignores_absent_pesticides():
    logits = torch.randn(
        (4, 3, 2),
        dtype=torch.float32,
        requires_grad=True,
    )

    concentration = torch.zeros(
        (4, 3),
        dtype=torch.float32,
    )

    presence = torch.zeros(
        (4, 3),
        dtype=torch.float32,
    )

    loss = present_only_corn_ordinal_loss(
        logits,
        concentration,
        presence,
    )

    assert float(loss.detach()) == 0.0

    loss.backward()

    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad).item() == 0
