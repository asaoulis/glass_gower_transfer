import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from neuralop.models import FNO


MAX_INPUT_HW: Tuple[int, int] = (100, 1000)


def _pad_to_size(x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    """Zero-pad x on bottom/right to target spatial size (H, W)."""
    _, _, h, w = x.shape
    th, tw = target_hw
    pad_h = max(th - h, 0)
    pad_w = max(tw - w, 0)
    if pad_h == 0 and pad_w == 0:
        return x
    # Pad format: (left, right, top, bottom)
    return F.pad(x, (0, pad_w, 0, pad_h))


class FNOBackbone2d(nn.Module):
    """Configurable stack of FNO2d blocks with residuals, norm, and activations.

    Produces a feature map of shape [B, hidden_channels, H, W].
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        n_layers: int = 4,
        n_modes: Tuple[int, int] = (8, 8),
        dropout: float = 0.0,
        use_norm: bool = True,
        activation: str = "gelu",
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        # optional input projection to hidden_channels
        self.in_proj = (
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
            if in_channels != hidden_channels
            else nn.Identity()
        )

        if activation.lower() == "gelu":
            act = nn.GELU()
        elif activation.lower() in ("silu", "swish"):
            act = nn.SiLU()
        else:
            act = nn.LeakyReLU(0.2, inplace=True)

        blocks = []
        for _ in range(n_layers):
            norm = nn.InstanceNorm2d(hidden_channels, affine=True) if use_norm else nn.Identity()
            fno = FNO(
                n_modes=(n_modes[0], n_modes[1]),
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                hidden_channels=hidden_channels,
                n_layers=1,
            )
            blocks.append(nn.ModuleDict({"norm": norm, "fno": fno, "act": act, "dropout": nn.Dropout(dropout)}))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        for blk in self.blocks:
            residual = x
            y = blk["norm"](x)
            y = blk["fno"](y)
            y = blk["act"](y)
            y = blk["dropout"](y)
            x = y + residual
        return x


class GlobalPool2d(nn.Module):
    """Configurable global pooling over spatial dims.

    Supports:
      - "avg": global average pooling
      - "avgmax": concat of avg and max pooling
      - "attn": lightweight attention pooling over HW
      - "spp": simple 1x1 + 2x2 spatial pyramid pooling (avg)
    """

    def __init__(self, in_channels: int, pool_type: str = "avg"):
        super().__init__()
        pool_type = pool_type.lower()
        self.pool_type = pool_type
        if pool_type == "avg":
            self.out_dim = in_channels
            self.attn_query = None
        elif pool_type == "avgmax":
            self.out_dim = 2 * in_channels
            self.attn_query = None
        elif pool_type == "attn":
            self.out_dim = in_channels
            # learnable query vector for attention over spatial positions
            self.attn_query = nn.Parameter(torch.randn(in_channels) * 0.02)
        elif pool_type == "spp":
            # 1x1 + 2x2 pyramid -> (1 + 4) * C
            self.out_dim = 5 * in_channels
            self.attn_query = None
        else:
            raise ValueError(f"Unknown pool_type '{pool_type}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if self.pool_type == "avg":
            return x.mean(dim=(2, 3))
        if self.pool_type == "avgmax":
            avg = x.mean(dim=(2, 3))
            mx = x.amax(dim=(2, 3))
            return torch.cat([avg, mx], dim=1)
        if self.pool_type == "attn":
            # [B, C, H, W] -> [B, HW, C]
            feats = x.view(B, C, H * W).transpose(1, 2)
            q = self.attn_query.view(1, 1, C).expand(B, 1, C)
            # dot-product attention: [B, 1, HW]
            attn_logits = torch.bmm(q, feats.transpose(1, 2))
            attn = torch.softmax(attn_logits, dim=-1)
            pooled = torch.bmm(attn, feats).squeeze(1)
            return pooled
        # spp
        # 1x1
        p1 = F.adaptive_avg_pool2d(x, (1, 1)).view(B, C)
        # 2x2
        p2 = F.adaptive_avg_pool2d(x, (2, 2)).view(B, C * 4)
        return torch.cat([p1, p2], dim=1)


def _build_sinusoidal_2d_embedding(h: int, w: int, c: int, device: torch.device) -> torch.Tensor:
    """2D sinusoidal positional embedding [1, C, H, W]."""
    assert c % 4 == 0, "positional embedding channels must be divisible by 4"
    pe = torch.zeros(1, c, h, w, device=device)
    c_half = c // 2
    c_quarter = c // 4
    # y embeddings
    pos_y = torch.linspace(-1.0, 1.0, steps=h, device=device).view(1, 1, h, 1)
    div_term_y = torch.exp(torch.arange(0, c_half, 2, device=device) * (-torch.log(torch.tensor(10000.0, device=device)) / c_half))
    pe[:, 0:c_half:2, :, :] = torch.sin(pos_y * div_term_y.view(-1, 1, 1))
    pe[:, 1:c_half:2, :, :] = torch.cos(pos_y * div_term_y.view(-1, 1, 1))
    # x embeddings
    pos_x = torch.linspace(-1.0, 1.0, steps=w, device=device).view(1, 1, 1, w)
    div_term_x = torch.exp(torch.arange(0, c_quarter * 2, 2, device=device) * (-torch.log(torch.tensor(10000.0, device=device)) / (c_quarter * 2)))
    pe[:, c_half::2, :, :] = torch.sin(pos_x * div_term_x.view(-1, 1, 1))
    pe[:, c_half + 1::2, :, :] = torch.cos(pos_x * div_term_x.view(-1, 1, 1))
    return pe


class KidsFNONorthSouthEmbedding(nn.Module):
    """Twin-tower FNO encoder for north/south shear maps.

    - Expects dict with keys: 'E_south', 'B_south', 'E_north', 'B_north'
      where each tensor is [B, C=6, H, W].
    - Stacks E/B per hemisphere -> [B, 12, H, W].
    - Applies a shared FNO backbone to north and south.
    - Pools backbone outputs over (H, W) -> feature vectors.
    - Concatenates [south, north] features and maps to latent_dim via MLP.
    """

    def __init__(
        self,
        latent_dim: int,
        channels_per_map: int = 6,
        hidden_channels: int = 32,
        n_modes: Tuple[int, int] = (8, 8),
        n_layers: int = 4,
        dropout: float = 0.1,
        pool_type: str = "avg",           # "avg", "avgmax", "attn", "spp"
        use_coord_channels: bool = False,
        pos_embedding: str = "none",      # "none", "sinusoidal"
        head_hidden_multiple: float = 2.0,
        head_num_layers: int = 2,
        head_dropout: float = 0.1,
        head_activation: str = "gelu",
        use_norm: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.channels_per_map = channels_per_map
        self.use_coord_channels = use_coord_channels
        self.pos_embedding = pos_embedding.lower()

        in_channels = 2 * channels_per_map  # E + B
        if use_coord_channels:
            in_channels += 2  # x, y

        # shared FNO backbone
        self.shared_backbone = FNOBackbone2d(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            n_modes=n_modes,
            dropout=dropout,
            use_norm=use_norm,
            activation=head_activation,
        )
        self._backbone_out_channels = hidden_channels

        # pooling head
        self.pool = GlobalPool2d(in_channels=self._backbone_out_channels, pool_type=pool_type)

        # build flexible MLP head
        in_dim = 2 * self.pool.out_dim
        if head_activation.lower() == "gelu":
            act = nn.GELU()
        elif head_activation.lower() in ("silu", "swish"):
            act = nn.SiLU()
        else:
            act = nn.LeakyReLU(0.2, inplace=True)

        hidden_dim = max(latent_dim, int(head_hidden_multiple * in_dim))
        layers = []
        d_in = in_dim
        for _ in range(max(1, head_num_layers - 1)):
            layers.append(nn.Linear(d_in, hidden_dim))
            layers.append(act)
            if head_dropout > 0:
                layers.append(nn.Dropout(head_dropout))
            d_in = hidden_dim
        layers.append(nn.Linear(d_in, latent_dim))
        self.head = nn.Sequential(*layers)

    def _add_coord_channels(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_coord_channels:
            return x
        B, _, H, W = x.shape
        device = x.device
        ys = torch.linspace(-1.0, 1.0, steps=H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        xs = torch.linspace(-1.0, 1.0, steps=W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
        return torch.cat([x, ys, xs], dim=1)

    def _add_pos_embedding(self, x: torch.Tensor) -> torch.Tensor:
        if self.pos_embedding != "sinusoidal":
            return x
        B, C, H, W = x.shape
        device = x.device
        # ensure C is suitable; if not, just skip to avoid runtime errors
        if C % 4 != 0:
            return x
        pe = _build_sinusoidal_2d_embedding(H, W, C, device=device)
        return x + pe

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        e_south = data["E_south"]
        b_south = data["B_south"]
        e_north = data["E_north"]
        b_north = data["B_north"]

        south_stack = torch.cat([e_south, b_south], dim=1)
        north_stack = torch.cat([e_north, b_north], dim=1)

        south_stack = _pad_to_size(south_stack, MAX_INPUT_HW)
        north_stack = _pad_to_size(north_stack, MAX_INPUT_HW)

        south_stack = self._add_coord_channels(south_stack)
        north_stack = self._add_coord_channels(north_stack)

        south_stack = self._add_pos_embedding(south_stack)
        north_stack = self._add_pos_embedding(north_stack)

        south_feat = self.shared_backbone(south_stack)
        north_feat = self.shared_backbone(north_stack)

        south_vec = self.pool(south_feat)
        north_vec = self.pool(north_feat)

        feats = torch.cat([south_vec, north_vec], dim=1)
        z = self.head(feats)
        return z
