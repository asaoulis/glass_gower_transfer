from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    """Generalized mean pooling with learnable p (shared or per-channel)."""

    def __init__(self, p_init: float = 3.0, eps: float = 1e-6, per_channel: bool = False, clamp=(0.1, 10.0)):
        super().__init__()
        if per_channel:
            self.p = nn.Parameter(torch.ones(1, 1) * float(p_init))
        else:
            self.p = nn.Parameter(torch.tensor(float(p_init)))
        self.eps = eps
        self.per_channel = per_channel
        self.clamp_min, self.clamp_max = clamp

    def forward(self, x):
        x = torch.clamp(x, min=self.eps)
        p = self.p
        if self.per_channel and p.dim() == 2:
            p = p.expand(1, x.size(1))
        p = torch.clamp(p, min=self.clamp_min, max=self.clamp_max)
        return x.pow(p.unsqueeze(-1).unsqueeze(-1)).mean(dim=(-2, -1)).pow(
            1.0 / p.unsqueeze(-1).unsqueeze(-1)
        ).squeeze(-1).squeeze(-1)


class SpatialPyramidPooling(nn.Module):
    """SPP -> concatenates flattened grids for output_sizes (e.g. (1,2,4))."""

    def __init__(self, output_sizes=(1, 2, 4), mode="avg"):
        super().__init__()
        assert mode in ("avg", "max")
        self.output_sizes = tuple(output_sizes)
        self.mode = mode

    def forward(self, x):
        B, C, H, W = x.shape
        parts = []
        for s in self.output_sizes:
            if self.mode == "avg":
                p = F.adaptive_avg_pool2d(x, output_size=(s, s))
            else:
                p = F.adaptive_max_pool2d(x, output_size=(s, s))
            parts.append(p.view(B, C * s * s))
        return torch.cat(parts, dim=1)


class SpatialAttentionPool(nn.Module):
    """Lightweight spatial attention pooling: weighted average with learned logits."""

    def __init__(self, in_channels, hidden=128):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.attn = nn.Conv2d(hidden, 1, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = F.relu(self.proj(x))
        logits = self.attn(h).view(B, -1)
        weights = F.softmax(logits, dim=-1).view(B, 1, H, W)
        out = (x * weights).sum(dim=(-2, -1))
        return out


class TransformerPool(nn.Module):
    """Tiny transformer encoder readout. Use only if H*W is modest."""

    def __init__(self, in_channels, nhead=4, nhid=128, nlayers=1, add_cls_token=True):
        super().__init__()
        d_model = in_channels
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=nhid,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.add_cls_token = add_cls_token
        if add_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.view(B, C, H * W).permute(0, 2, 1)
        if self.add_cls_token:
            cls = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
        out = self.transformer(tokens)
        if self.add_cls_token:
            pooled = out[:, 0]
        else:
            pooled = out.mean(dim=1)
        return self.ln(pooled)


class PoolProj(nn.Module):
    """Pooling factory + learned projection."""

    def __init__(
        self,
        pool_types,
        in_channels,
        proj_dim,
        spp_sizes=(1, 2, 4),
        gem_per_channel=False,
        transformer_args=None,
        attn_hidden=128,
    ):
        super().__init__()
        assert isinstance(pool_types, (list, tuple))
        self.pool_types = list(pool_types)
        self.in_ch = in_channels
        self.proj_dim = proj_dim
        self.spp_sizes = tuple(spp_sizes)
        self.transformer_args = transformer_args or {}
        self.attn_hidden = attn_hidden

        if "gem" in self.pool_types:
            self.gem = GeM(per_channel=gem_per_channel)
        if "spp" in self.pool_types:
            self.spp = SpatialPyramidPooling(output_sizes=self.spp_sizes, mode="avg")
        if "attn" in self.pool_types:
            self.attn = SpatialAttentionPool(in_channels, hidden=self.attn_hidden)
        if "trans" in self.pool_types or "transformer" in self.pool_types:
            self.trans = TransformerPool(in_channels, **self.transformer_args)

        concat_dim = 0
        for t in self.pool_types:
            if t in ("avg", "max", "gem", "attn", "trans", "transformer"):
                concat_dim += in_channels
            elif t == "spp":
                concat_dim += in_channels * sum([s * s for s in self.spp_sizes])
            else:
                raise ValueError(f"Unknown pool type: {t}")

        self.concat_dim = concat_dim
        self.proj = nn.Sequential(
            nn.Linear(self.concat_dim, max(self.proj_dim, 64)),
            nn.ReLU(inplace=True),
            nn.LayerNorm(max(self.proj_dim, 64)),
            nn.Linear(max(self.proj_dim, 64), self.proj_dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        parts = []
        for t in self.pool_types:
            if t == "avg":
                parts.append(F.adaptive_avg_pool2d(x, 1).view(B, C))
            elif t == "max":
                parts.append(F.adaptive_max_pool2d(x, 1).view(B, C))
            elif t == "gem":
                parts.append(self.gem(x).view(B, C))
            elif t == "spp":
                parts.append(self.spp(x))
            elif t == "attn":
                parts.append(self.attn(x))
            elif t in ("trans", "transformer"):
                parts.append(self.trans(x))
        desc = torch.cat(parts, dim=1)
        return self.proj(desc)
