"""Check delayed gradient flow caused by official DiT zero initialization."""
from copy import deepcopy
import torch


def check_dit_gradient_flow(model, spectrum, mask, condition, base):
    # Use a disposable model: pipeline-check never updates the formal model.
    probe = deepcopy(model).train()
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    timestep = torch.ones(spectrum.shape[0], dtype=torch.long, device=spectrum.device)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        prediction = probe(spectrum, timestep, valid_mask=mask, condition=condition, prior_conditioning=base)
        loss = (((prediction - spectrum) * mask).square().sum() / mask.sum())
        if not torch.isfinite(loss):
            raise RuntimeError('DiT gradient probe loss is nonfinite')
        loss.backward()
        optimizer.step()
    for prefix in ('patch_embed.', 'time_embed.', 'condition_encoder.', 'blocks.0.qkv.',
                   'blocks.0.adaLN_modulation.', 'final_modulation.', 'final_linear.'):
        gradients = [p.grad for name, p in probe.named_parameters() if name.startswith(prefix) and p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients) or not any(g.abs().sum() > 0 for g in gradients):
            raise RuntimeError(f'DiT gradient flow failed: {prefix}')
    print('DiT adaLN-Zero: disposable-copy 3-step gradient check passed; formal model unchanged.')
