"""ConvNeXt-style map encoder (counts-extended Phase 2b, 2026-07-17).

A standard, paper-defensible modern CNN (Liu et al. 2022, "A ConvNet for the 2020s"): patchify
stem, depthwise 7x7 + LayerNorm + inverted-bottleneck MLP blocks, layer scale, stochastic depth,
symmetric downsampling only (augmentation-symmetry constraint). Interface matches
PreActResNetEncoder: forward(x: [B,C,H,W]) -> [B, num_outputs].
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .unet.poolproj import PoolProj


class DropPath(nn.Module):
    """Stochastic depth per sample (timm-style)."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = torch.rand(x.shape[0], *([1] * (x.ndim - 1)), device=x.device, dtype=x.dtype)
        mask = (mask < keep).to(x.dtype) / keep
        return x * mask


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)          # applied channels-last
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        inp = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)                        # B,H,W,C
        x = self.pwconv2(self.act(self.pwconv1(self.norm(x))))
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return inp + self.drop_path(x)


class _ChannelsFirstLN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvNeXtEncoder(nn.Module):
    """ConvNeXt-T-style trunk -> PoolProj -> FC, weight-shared per hemisphere."""

    def __init__(
        self,
        in_channels: int,
        num_outputs: int,
        dims: Sequence[int] = (64, 128, 256, 512),
        depths: Sequence[int] = (3, 3, 9, 3),
        drop_path_rate: float = 0.1,
        pool_types: Sequence[str] = ("avg", "gem"),
        head_hidden: int | None = None,
        **kwargs,  # tolerate unused map_kwargs for config compat
    ):
        super().__init__()
        dims = tuple(dims)
        depths = tuple(depths)
        # Patchify stem: 4x4 stride-4 conv + LN (symmetric).
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], 4, stride=4),
            _ChannelsFirstLN(dims[0]),
        )
        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        stages, downsamples = [], []
        i = 0
        for si, (dim, depth) in enumerate(zip(dims, depths)):
            if si > 0:
                downsamples.append(nn.Sequential(
                    _ChannelsFirstLN(dims[si - 1]),
                    nn.Conv2d(dims[si - 1], dim, 2, stride=2),
                ))
            blocks = [ConvNeXtBlock(dim, drop_path=dp_rates[i + b]) for b in range(depth)]
            i += depth
            stages.append(nn.Sequential(*blocks))
        self.downsamples = nn.ModuleList(downsamples)
        self.stages = nn.ModuleList(stages)
        self.final_norm = _ChannelsFirstLN(dims[-1])

        self.poolproj = PoolProj(
            pool_types=list(pool_types),
            in_channels=dims[-1],
            proj_dim=dims[-1],
        )
        head_hidden = head_hidden or dims[-1]
        self.fc = nn.Sequential(
            nn.Linear(self.poolproj.proj_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, num_outputs),
        )
        self.num_outputs = num_outputs

    def forward(self, x, **_ignored):
        h = self.stem(x)
        for si, stage in enumerate(self.stages):
            if si > 0:
                h = self.downsamples[si - 1](h)
            h = stage(h)
        h = self.final_norm(h)
        return self.fc(self.poolproj(h))
