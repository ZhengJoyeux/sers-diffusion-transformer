"""Retain the existing residual Adapter and its layer hook."""
import torch
from torch import nn

class ResidualAdapter(nn.Module):
    def __init__(self, dimension, bottleneck):
        super().__init__()
        if not 1 <= bottleneck <= dimension:
            raise ValueError('Invalid Adapter bottleneck')
        self.norm = nn.LayerNorm(dimension)
        self.down = nn.Linear(dimension, bottleneck)
        self.up = nn.Linear(bottleneck, dimension)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, value):
        return value + self.up(nn.functional.gelu(self.down(self.norm(value))))

def _adapter_hook(module, args, output):
    return module.t327_adapter(output)
