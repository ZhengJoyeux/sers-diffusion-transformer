"""Real Q/V weight adaptation, including fused PyTorch MultiheadAttention."""
import math
import torch
from torch import nn
from torch.nn.utils import parametrize

VARIANTS = ('adam_control', 'adamw', 'layerwise', 'lora', 'adapter')
TASK_PREFIXES = ('classification_projection.', 'classification_heads.',
                 'concentration_projection.', 'ordinal_heads.',
                 'mixture_aware_query_fusion.')
UNUSED_PREFIXES = ('transformer_encoder_layer.', 'relation_heads.')


class LowRank(nn.Module):
    def __init__(self, output, input_dim, rank, alpha):
        super().__init__()
        if not 1 <= rank <= min(output, input_dim) or alpha <= 0:
            raise ValueError('Invalid LoRA rank/alpha')
        self.A = nn.Parameter(torch.empty(rank, input_dim))
        self.B = nn.Parameter(torch.zeros(output, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, weight):
        return weight + (self.B @ self.A) * self.scale


class QVLowRank(nn.Module):
    def __init__(self, dimension, rank, alpha):
        super().__init__()
        self.q = LowRank(dimension, dimension, rank, alpha)
        self.v = LowRank(dimension, dimension, rank, alpha)

    def forward(self, weight):
        q, k, v = weight.chunk(3, dim=0)
        return torch.cat((self.q(q), k, self.v(v)), dim=0)


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


def configure(model, spec):
    variant = spec['variant']
    if variant not in VARIANTS:
        raise ValueError('Unknown variant')
    peft = variant in ('lora', 'adapter')
    model.requires_grad_(not peft)
    # This registered template is not used in forward; relation loss is disabled
    # in EVERY arm of this continuation experiment.
    for name, parameter in model.named_parameters():
        if name.startswith(UNUSED_PREFIXES):
            parameter.requires_grad_(False)
        elif peft and name.startswith(TASK_PREFIXES):
            parameter.requires_grad_(True)
    model.transformer_encoder.enable_nested_tensor = False
    model.transformer_encoder.use_nested_tensor = False
    if variant == 'lora':
        for layer in model.transformer_encoder.layers:
            attention = layer.self_attn
            if attention.in_proj_weight is None:
                raise ValueError('Expected fused self-attention Q/K/V')
            parametrize.register_parametrization(attention, 'in_proj_weight',
                QVLowRank(attention.embed_dim, spec['rank'], spec['alpha']))
        for name in ('q_proj', 'v_proj'):
            linear = getattr(model.peak_guided_attention, name)
            parametrize.register_parametrization(linear, 'weight',
                LowRank(linear.out_features, linear.in_features, spec['rank'], spec['alpha']))
    elif variant == 'adapter':
        for layer in model.transformer_encoder.layers:
            layer.add_module('t327_adapter', ResidualAdapter(
                layer.self_attn.embed_dim, spec['bottleneck']))
            layer.register_forward_hook(_adapter_hook)
    return model


def train_mode(model):
    model.train()
    # Identical in all arms: preserve archived normalization at batch size 16.
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def is_new(name):
    return ('t327_adapter.' in name or
            ('.parametrizations.' in name and '.original' not in name))


def optimizer_groups(model, spec):
    decay_ids = set()
    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention) and module.in_proj_weight is not None and not parametrize.is_parametrized(module, 'in_proj_weight'):
            decay_ids.add(id(module.in_proj_weight))
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # Parametrized LoRA A/B are handled below; its frozen original is excluded.
            weight = module._parameters.get('weight')
            if weight is not None:
                decay_ids.add(id(weight))
    groups = {}
    depth = len(model.transformer_encoder.layers)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if is_new(name):
            maximum = spec['peft_lr']
            decay = spec['weight_decay'] if parameter.ndim == 2 and not name.endswith('bias') and '.norm.' not in name else 0.
            family = 'peft'
        elif name.startswith(TASK_PREFIXES):
            maximum, family = spec['head_lr'], 'task'
            decay = spec['weight_decay'] if id(parameter) in decay_ids else 0.
        elif name.startswith('multi_peak_evidence.'):
            maximum, family = spec['evidence_lr'], 'multipeak'
            decay = spec['weight_decay'] if id(parameter) in decay_ids else 0.
        else:
            maximum, family = spec['backbone_lr'], 'backbone'
            if spec['variant'] == 'layerwise':
                if name.startswith('transformer_encoder.layers.'):
                    index = int(name.split('.')[2])
                    factor = spec['layer_decay'] ** (depth - 1 - index)
                    family = f'encoder_{index}'
                elif name.startswith(('cnn.', 'cnn2.', 'reg_feat.', 'cnn_fusion.', 'token_expansion.')):
                    factor = spec['layer_decay'] ** depth
                    family = 'input_features'
                else:
                    factor = 1.
                maximum *= factor
            decay = spec['weight_decay'] if id(parameter) in decay_ids else 0.
        key = (family, maximum, decay)
        if key not in groups:
            groups[key] = dict(params=[], names=[], family=family, lr=maximum,
                maximum_lr=maximum, minimum_lr=maximum * spec['minimum_ratio'], weight_decay=decay)
        groups[key]['params'].append(parameter)
        groups[key]['names'].append(name)
    result = list(groups.values())
    ids = [id(p) for group in result for p in group['params']]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise RuntimeError('Optimizer must cover trainable parameters exactly once')
    return result


def build_optimizer(model, spec):
    cls = torch.optim.Adam if spec['variant'] == 'adam_control' else torch.optim.AdamW
    return cls(optimizer_groups(model, spec), betas=(.9, .999), eps=1e-8)
