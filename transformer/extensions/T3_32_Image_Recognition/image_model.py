"""2-D line-plot CNN -> spectral tokens -> existing query/fusion/CORN designs."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from spectral_core import PeakGuidedQueryAttention, MixtureAwareQueryFusion
from intensity_evidence import MultiPeakEvidence
from adapter import ResidualAdapter, _adapter_hook

VERSION = 'T3.32_line_image_recognition_v1'


class ImageQueryAttention(PeakGuidedQueryAttention):
    """Project a point-level chemical prior to the actual image-column token bins."""
    def forward(self, tokens, token_mask):
        if token_mask.shape != tokens.shape[:2] or not token_mask.any(1).all():
            raise ValueError('Each spectrum needs valid image tokens')
        b, t, d = tokens.shape
        queries = self.query_embedding[None].expand(b, -1, -1)
        q = self.q_proj(queries).view(b, self.n_queries, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(tokens).view(b, t, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(tokens).view(b, t, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        logits = torch.einsum('bhqd,bhtd->bhqt', q, k)/math.sqrt(self.head_dim)
        if self.use_peak_guidance:
            prior = F.adaptive_max_pool1d(self.peak_prior_points[None], t)[0]
            logits = logits + self.guidance_strength[None, None, :, None]*prior[None, None]
        invalid = ~token_mask[:, None, None]
        logits = logits.masked_fill(invalid, torch.finfo(logits.dtype).min)
        attention = logits.float().softmax(-1).to(logits.dtype).masked_fill(invalid, 0.)
        context = torch.einsum('bhqt,bhtd->bhqd', self.attention_dropout(attention), v)
        context = context.permute(0, 2, 1, 3).reshape(b, self.n_queries, d)
        features = self.output_norm(self.out_proj(context)+queries)
        features = self.ffn_norm(features+self.ffn(features))
        return features, attention.mean(1)


class LineImageCNN(nn.Module):
    """Vertical bands and explicit curve moments preserve absolute peak height."""
    def __init__(self, dimension=128, tokens=210, bands=8):
        super().__init__()
        self.tokens, self.bands = int(tokens), int(bands)
        layers = []
        previous = 3  # foreground, horizontal coordinate, vertical coordinate
        for channels, stride in [(8, (4, 2)), (16, (2, 2)), (24, (2, 2))]:
            layers.extend([nn.Conv2d(previous, channels, 3, stride=stride, padding=1),
                           nn.GroupNorm(4, channels), nn.GELU()])
            previous = channels
        self.convolutions = nn.Sequential(*layers)
        self.projection = nn.Sequential(nn.Linear(24*bands+2, dimension), nn.GELU(), nn.LayerNorm(dimension))

    def forward(self, image, columns):
        if image.ndim != 4 or image.shape[1] != 1 or columns.shape != (len(image), image.shape[-1]):
            raise ValueError('Expected image [B,1,H,W] and measured columns [B,W]')
        if not torch.isfinite(image).all() or image.amin() < 0 or image.amax() > 1:
            raise ValueError('Image must contain finite grayscale values in [0,1]')
        b, _, h, w = image.shape
        columns = columns.bool()
        valid = columns[:, None, None, :]
        foreground = (1-image)*valid
        vertical = torch.linspace(0, 1, h, device=image.device, dtype=image.dtype)[None, None, :, None]
        horizontal = torch.linspace(0, 1, w, device=image.device, dtype=image.dtype)[None, None, None, :]
        coordinates = torch.cat((foreground, horizontal.expand(b, 1, h, w)*valid,
                                 vertical.expand(b, 1, h, w)*valid), 1)
        features = self.convolutions(coordinates)
        # Do not globally average vertical position: each band has distinct weights.
        features = F.adaptive_avg_pool2d(features, (self.bands, self.tokens))
        features = features.permute(0, 3, 2, 1).flatten(2)
        mass = foreground.sum(2).clamp_min(1e-6)
        mean = (foreground*vertical).sum(2)/mass
        second = (foreground*vertical.square()).sum(2)/mass
        moments = F.adaptive_avg_pool1d(torch.cat((mean, second), 1), self.tokens).transpose(1, 2)
        # Conservative receptive-field exclusion around unmeasured boundaries.
        mask = F.avg_pool1d(columns.float()[:, None], 15, stride=1, padding=7,
                            count_include_pad=False) >= 1-1e-6
        mask = (F.adaptive_avg_pool1d(mask.float(), self.tokens)[:, 0] >= 1-1e-6)
        if not mask.any(1).all():
            raise ValueError('No fully measured image token remains')
        return self.projection(torch.cat((features, moments), -1)).masked_fill(~mask[..., None], 0.), mask


def position_encoding(length, dimension):
    coordinate = torch.linspace(0, 1900/9., length)
    frequency = torch.exp(torch.arange(0, dimension, 2)*(-math.log(10000.)/dimension))
    result = torch.zeros(length, dimension)
    result[:, 0::2] = torch.sin(coordinate[:, None]*frequency)
    result[:, 1::2] = torch.cos(coordinate[:, None]*frequency)
    return result


class ImageSERSFormer(nn.Module):
    def __init__(self, attention_dim=128, heads=8, layers=6, dim_ff=256, dropout=.1,
                 tokens=210, numeric_evidence=True, adapter_bottleneck=16,
                 adaptive_gate=False, peak_guidance=True):
        super().__init__()
        if attention_dim % heads or attention_dim % 2 or tokens < 8:
            raise ValueError('Invalid Transformer dimension/head/token configuration')
        self.config = dict(attention_dim=attention_dim, heads=heads, layers=layers, dim_ff=dim_ff,
                           dropout=dropout, tokens=tokens, numeric_evidence=numeric_evidence,
                           adapter_bottleneck=adapter_bottleneck, adaptive_gate=adaptive_gate,
                           peak_guidance=peak_guidance)
        self.image_frontend = LineImageCNN(attention_dim, tokens)
        layer = nn.TransformerEncoderLayer(attention_dim, heads, dim_ff, dropout,
                                            activation='gelu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        if adapter_bottleneck:
            for block in self.transformer_encoder.layers:
                block.add_module('t327_adapter', ResidualAdapter(attention_dim, adapter_bottleneck))
                block.register_forward_hook(_adapter_hook)
        self.register_buffer('position_encoding', position_encoding(tokens, attention_dim), persistent=False)
        self.peak_guided_attention = ImageQueryAttention(attention_dim, 3, heads, 1901, dropout,
                                                        1.5, peak_guidance)
        self.multi_peak_evidence = MultiPeakEvidence(attention_dim, heads) if numeric_evidence else None
        self.classification_projection = nn.Linear(attention_dim, 64)
        self.concentration_projection = nn.Linear(attention_dim, 64)
        self.classification_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1), nn.Sigmoid())
            for _ in range(3)])
        self.mixture_aware_query_fusion = MixtureAwareQueryFusion(
            attention_dim, 3, heads, dropout, use_adaptive_gate=adaptive_gate)
        self.ordinal_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 2))
            for _ in range(3)])
        self.relation_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 16)) for _ in range(3)])

    def forward(self, data):
        expected = {'image', 'image_valid_columns', 'raw_intensity', 'measured_mask'}
        if set(data) != expected:
            raise ValueError('Forward accepts spectrum-derived tensors only; no labels/condition IDs')
        tokens, mask = self.image_frontend(data['image'], data['image_valid_columns'])
        tokens = (tokens+self.position_encoding).masked_fill(~mask[..., None], 0.)
        encoded = self.transformer_encoder(tokens, src_key_padding_mask=~mask).masked_fill(~mask[..., None], 0.)
        queries, attention = self.peak_guided_attention(encoded, mask)
        if self.multi_peak_evidence is not None:
            queries = self.multi_peak_evidence(data['raw_intensity'], data['measured_mask'], queries)
        features = self.classification_projection(queries)
        cp = torch.cat([head(features[:, j]) for j, head in enumerate(self.classification_heads)], 1)
        concentration = self.concentration_projection(self.mixture_aware_query_fusion(queries, cp.detach()))
        logits = torch.stack([head(concentration[:, j]) for j, head in enumerate(self.ordinal_heads)], 1)
        ge_m = logits[..., 0].sigmoid()
        op = torch.stack((ge_m, ge_m*logits[..., 1].sigmoid()), -1)
        relation = torch.stack([head(concentration[:, j]) for j, head in enumerate(self.relation_heads)], 1)
        return dict(cp=cp, op=op, logits=logits, attention=attention, relation=relation)


def warm_start(model, path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    version = payload.get('checkpoint_version')
    if version != 'T3.30_feature_finetune_v1':
        raise ValueError('Warm start expects the actual T3.30 checkpoint; use --from-scratch otherwise')
    raw = payload['model_state_dict']
    source = {k[len('student.'):]: v for k, v in raw.items() if k.startswith('student.')}
    if not source:
        raise ValueError('Missing T3.30 student backbone')
    current = model.state_dict()
    copied, skipped = {}, []
    normalization = ('curve_mean', 'curve_scale', 'stats_mean', 'stats_scale', 'normalization_fitted')
    for key, value in source.items():
        if key.startswith(('cnn.', 'cnn2.', 'reg_feat.', 'cnn_fusion.', 'token_expansion.',
                           'transformer_encoder_layer.', 'relation_heads.')) or key.endswith(normalization):
            skipped.append(key)
        elif key in current and value.shape == current[key].shape:
            copied[key] = value
        elif key in current:
            raise ValueError('Warm-start shape mismatch: '+key)
        else:
            skipped.append(key)
    required = [k for k in current if k.startswith(('transformer_encoder.', 'classification_',
                'concentration_projection.', 'ordinal_heads.', 'peak_guided_attention.'))]
    missing = [k for k in required if k not in copied]
    if missing:
        raise ValueError('Incomplete T3.30 warm start: '+str(missing))
    model.load_state_dict(copied, strict=False)
    return dict(source_checkpoint_version=version, copied_keys=sorted(copied), skipped_keys=sorted(skipped),
                copied_parameters=sum(v.numel() for v in copied.values()),
                note='Image frontend and relation heads are new; normalization refitted on real training only; old frozen teacher routing is not used')
