"""Pre-activation ResNet map encoder (counts-extended task, 2026-07-16).

A deliberately boring, well-behaved deep CNN for the weak-lensing map patches, replacing the
diffusion-style UNet encoder whose zero-init residual blocks + scale-shift GroupNorm were measured
degenerating into a rank-1, amplitude-exploding representation under VMIM gradient starvation
(task diagnosis.md). Design follows the cosmic-shear CNN literature (Ribli 2019 / Fluri 2019
depth) with modern residual practice (He 2016 identity mappings):

  - pre-activation residual blocks (GN -> GELU -> 3x3 conv, x2; identity skip),
  - standard (He) init everywhere — residual branches are ALIVE at init by construction,
  - GroupNorm (batch-statistics-free: N/S patches in a batch are physically correlated),
  - symmetric stride-2 downsampling only (flip/rotation augmentation safety — user constraint),
  - GeM/avg pooling head via the repo's PoolProj.

Interface matches UNetO3StyleEncoder's plain call path: forward(x: [B,C,H,W]) -> [B, num_outputs].
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .unet.poolproj import PoolProj


def _gn(ch: int) -> nn.GroupNorm:
    groups = max(1, min(32, ch // 4))
    while ch % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, ch)


class BlurPool2d(nn.Module):
    """Fixed anti-aliasing downsample (Zhang 2019, Tri-3): depthwise [1,2,1]^T[1,2,1]/16 blur
    with stride 2. Symmetric kernel -> flip/180-rotation augmentation safe."""

    def __init__(self, channels: int):
        super().__init__()
        k = torch.tensor([1.0, 2.0, 1.0])
        k = torch.outer(k, k)
        k = k / k.sum()
        self.register_buffer("kernel", k.expand(channels, 1, 3, 3).clone())
        self.channels = channels

    def forward(self, x):
        return nn.functional.conv2d(
            x, self.kernel.to(x.dtype), stride=2, padding=1, groups=self.channels
        )


class PreActBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1, blur: bool = False):
        super().__init__()
        self.norm1 = _gn(cin)
        # anti-aliased variant (counts-ext phase 4, R10): stride-1 conv + fixed Tri-3 blur
        # downsample on BOTH the residual and shortcut paths, replacing the stride-2 conv
        self.blur = bool(blur) and stride == 2
        conv_stride = 1 if self.blur else stride
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=conv_stride, padding=1, bias=False)
        self.norm2 = _gn(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.act = nn.GELU()
        if self.blur:
            self.down = BlurPool2d(cout)
        if stride != 1 or cin != cout:
            self.shortcut = nn.Conv2d(cin, cout, 1, stride=conv_stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.act(self.norm1(x))
        sc = self.shortcut(h if not isinstance(self.shortcut, nn.Identity) else x)
        h = self.conv1(h)
        h = self.conv2(self.act(self.norm2(h)))
        if self.blur:
            h = self.down(h)
            sc = self.down(sc)
        return h + sc


class PreActResNetEncoder(nn.Module):
    """Weight-shared (used per-hemisphere) pre-activation ResNet -> pooled vector."""

    def __init__(
        self,
        in_channels: int,
        num_outputs: int,
        stage_channels: Sequence[int] = (32, 64, 128, 256, 256),
        blocks_per_stage: int = 3,
        pool_types: Sequence[str] = ("avg", "gem"),
        head_hidden: int | None = None,
        dropout: float = 0.0,
        blur_stages: Sequence[int] = (),
        tap_penultimate: bool = False,
        **kwargs,  # tolerate unused map_kwargs (e.g. conditioning flags) for config compat
    ):
        super().__init__()
        stage_channels = tuple(stage_channels)
        blur_stages = tuple(blur_stages)
        self.stem = nn.Conv2d(in_channels, stage_channels[0], 3, padding=1, bias=False)
        stages = []
        cin = stage_channels[0]
        for si, cout in enumerate(stage_channels):
            for bi in range(blocks_per_stage):
                # first block of every stage downsamples symmetrically (stride 2)
                stride = 2 if bi == 0 else 1
                stages.append(
                    PreActBlock(cin, cout, stride=stride, blur=(si in blur_stages))
                )
                cin = cout
        self.stages = nn.Sequential(*stages)
        self.final_norm = _gn(cin)
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.poolproj = PoolProj(
            pool_types=list(pool_types),
            in_channels=cin,
            proj_dim=cin,
        )
        # penultimate-stage tap (counts-ext phase 4, R2/ECAPA-MFA): pool the last-but-one
        # stage's (higher-resolution) features with the same pool set and concat before fc.
        # self.stages stays an nn.Sequential so existing checkpoint keys are unchanged.
        self.tap_penultimate = bool(tap_penultimate) and len(stage_channels) >= 2
        fc_in = self.poolproj.proj_dim
        if self.tap_penultimate:
            self._penult_end = (len(stage_channels) - 1) * blocks_per_stage
            penult_ch = stage_channels[-2]
            self.tap_norm = _gn(penult_ch)
            self.tap_pool = PoolProj(
                pool_types=list(pool_types),
                in_channels=penult_ch,
                proj_dim=penult_ch,
            )
            fc_in += self.tap_pool.proj_dim
        head_hidden = head_hidden or cin
        self.fc = nn.Sequential(
            nn.Linear(fc_in, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, num_outputs),
        )
        self.num_outputs = num_outputs

    def forward(self, x, **_ignored):
        h = self.stem(x)
        if self.tap_penultimate:
            h_pen = self.stages[: self._penult_end](h)
            h = self.stages[self._penult_end :](h_pen)
            pooled_pen = self.tap_pool(self.act(self.tap_norm(h_pen)))
        else:
            h = self.stages(h)
        h = self.dropout(self.act(self.final_norm(h)))
        pooled = self.poolproj(h)
        if self.tap_penultimate:
            pooled = torch.cat([pooled, pooled_pen], dim=-1)
        return self.fc(pooled)
