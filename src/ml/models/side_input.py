"""Side-input modules appended to an encoder's summary (config key ``side_input``; see
``src/ml/data/side_inputs.py`` for the semantics). Attached via
``KidsHybridBandpowersMaps.attach_side_input``; the flow then conditions on ``[z, side]``.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from ..data.side_inputs import SIDE_INPUT_KEY


class DataKeySideInput(nn.Module):
    """Pass through a vector the dataset put in the batch (``data['side_input']``)."""

    def __init__(self, dim: int, key: str = SIDE_INPUT_KEY):
        super().__init__()
        self.dim = int(dim)
        self.key = key

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.key not in data:
            raise KeyError(f"side input '{self.key}' missing from the batch: the dataset was built "
                           "without config.side_input (true_theta)")
        side = data[self.key]
        if side.shape[-1] != self.dim:
            raise ValueError(f"side input dim {side.shape[-1]} != expected {self.dim}")
        return side


class FrozenEncoderSideInput(nn.Module):
    """The summary of a separately trained encoder, frozen and kept in eval mode (BatchNorm/dropout
    stats never move, even while the parent trains)."""

    def __init__(self, encoder: nn.Module, dim: int):
        super().__init__()
        self.encoder = encoder
        self.dim = int(dim)
        self.freeze()

    def freeze(self):
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        self.encoder.only_return_mu = True
        return self.encoder(data)
