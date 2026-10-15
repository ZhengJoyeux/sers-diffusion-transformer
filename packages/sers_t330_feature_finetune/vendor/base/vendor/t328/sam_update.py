"""SAM on the new joint head; encoder remains frozen and runs once per batch."""
import torch
from base_bridge import rt
from ternary_model import training_loss


def backward_sam(model, data, classes, levels, optimizer, rho, marginal_weight):
    if model.arm != 'joint_sam' or model.student is not None:
        raise ValueError('This SAM experiment perturbs the new head only')
    before_dropout = rt.capture_rng()
    output = model(data)
    loss, _ = training_loss(model, output, classes, levels, marginal_weight, 0., 0.)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite SAM base loss')
    loss.backward()
    parameters = [p for p in model.parameters() if p.requires_grad]
    gradient_norm = torch.stack([p.grad.norm() for p in parameters if p.grad is not None]).norm()
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError('Nonfinite SAM perturbation gradient')
    original = [p.detach().clone() for p in parameters]
    try:
        with torch.no_grad():
            for p in parameters:
                if p.grad is not None:
                    p.add_(p.grad * (rho / gradient_norm.clamp_min(1e-12)))
        optimizer.zero_grad(set_to_none=True)
        # Reuse this batch's frozen features, not a persistent feature cache.
        # Replay the same dropout mask so weight perturbation is isolated.
        rt.restore_rng(before_dropout)
        second = dict(output, scores=model.second_scores(output))
        second_loss, parts = training_loss(model, second, classes, levels, marginal_weight, 0., 0.)
        if not bool(torch.isfinite(second_loss)):
            raise FloatingPointError('Nonfinite SAM perturbed loss')
        second_loss.backward()
    finally:
        with torch.no_grad():
            for p, old in zip(parameters, original):
                p.copy_(old)
    return second_loss.detach(), parts
