import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

# Maximum input spatial size (H, W) expected by the CNN to size its head.
# The FlexibleO3 backbone will run a dummy pass with this size to infer head dims.
MAX_INPUT_HW: Tuple[int, int] = (100, 1000)


class KidsO3NorthSouthEmbedding(nn.Module):
    """
    Embedding network that:
      - expects a data dict with keys: 'E_south', 'B_south', 'E_north', 'B_north'
      - each entry is a tensor of shape [B, C, H, W] with C=6
      - stacks E and B channelwise per hemisphere to form 12-channel inputs
      - applies the SAME flexible o3 CNN (shared weights) to north and south stacks separately
      - zero-pads the south stack to match the north spatial size (if smaller)
      - concatenates the two feature vectors and passes through a shallow MLP
        to produce a final latent vector of size `latent_dim`.
    """

    def __init__(self, latent_dim: int, cnn_out_dim: int = 256, hidden: int = 12, channels_per_map: int = 6):
        super().__init__()
        self.latent_dim = latent_dim
        self.cnn_out_dim = cnn_out_dim
        self.channels_per_map = channels_per_map

        from .compressors import flexible_o3_model

        in_channels = 2 * channels_per_map  # E + B stacked per hemisphere (6 + 6 = 12)
        # Shared CNN used for both north and south
        self.shared_cnn = flexible_o3_model(
            num_outputs=cnn_out_dim,
            hidden=hidden,
            channels=in_channels,
            max_hw=MAX_INPUT_HW,
            predict_sigmas=False,
        )

        # Shallow head after concatenation of north/south embeddings
        hidden_head = max(latent_dim, 2 * cnn_out_dim // 2)
        self.head = nn.Sequential(
            nn.Linear(2 * cnn_out_dim, hidden_head),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_head, latent_dim),
        )

    @staticmethod
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

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Expect tensors of shape [B, C=6, H, W]
        e_south = data["E_south"]
        b_south = data["B_south"]
        e_north = data["E_north"]
        b_north = data["B_north"]

        # Stack channelwise to get 12-channel inputs per hemisphere
        south_stack = torch.cat([e_south, b_south], dim=1)
        north_stack = torch.cat([e_north, b_north], dim=1)

        # Ensure south matches north spatial size by zero-padding (bottom/right)
        south_stack = self._pad_to_size(south_stack, (MAX_INPUT_HW[0], MAX_INPUT_HW[1]))
        south_stack = self._pad_to_size(south_stack, (MAX_INPUT_HW[0], MAX_INPUT_HW[1]))

        # Shared CNN processes both stacks
        south_feat = self.shared_cnn(south_stack)
        north_feat = self.shared_cnn(north_stack)

        # Concatenate features and map to latent representation
        feats = torch.cat([south_feat, north_feat], dim=1)
        z = self.head(feats)
        return z


class KidsCombinedCNNTransformer(nn.Module):
    """
    Combined architecture that preserves spatial structure from the flexible O3 CNN,
    applies hemisphere-specific sinusoidal positional embeddings and a categorical
    class embedding (0=south, 1=north) to tokens before concatenation, and then
    runs Transformer self-attention with per-layer learnable queries.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden: int = 12,
        channels_per_map: int = 6,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        n_queries: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.latent_dim = latent_dim
        self.channels_per_map = channels_per_map
        self.hidden = hidden
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_queries = n_queries

        from .compressors import flexible_o3_model

        in_channels = 2 * channels_per_map
        # Shared CNN backbone returning spatial features [B, Cout, H', W']
        self.shared_cnn = flexible_o3_model(
            num_outputs=0,  # ignored when return_features=True
            hidden=hidden,
            channels=in_channels,
            max_hw=MAX_INPUT_HW,
            predict_sigmas=False,
            return_features=True,
        )
        # CNN channel dim is 128*hidden after tail_conv
        self.cnn_out_channels = 128 * hidden
        # Project per-width tokens to d_model
        self.proj = nn.Linear(self.cnn_out_channels, d_model)
        self.token_ln = nn.LayerNorm(d_model)

        # Hemisphere class embedding: 0=south, 1=north
        self.class_embed = nn.Embedding(2, d_model)

        # Transformer encoder layers for data tokens (batch_first)
        def make_encoder_layer():
            return nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=int(mlp_ratio * d_model),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        self.encoder = nn.ModuleList([make_encoder_layer() for _ in range(n_layers)])

        # Per-layer learnable queries and cross-attn modules
        self.query_tokens = nn.ParameterList([
            nn.Parameter(torch.randn(n_queries, d_model) * 0.02) for _ in range(n_layers)
        ])
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True) for _ in range(n_layers)
        ])
        self.query_ln = nn.LayerNorm(d_model)

        # Final head from aggregated query to latent_dim
        self.head = nn.Sequential(
            nn.Linear(d_model, max(latent_dim, d_model // 2)),
            nn.GELU(),
            nn.Linear(max(latent_dim, d_model // 2), latent_dim),
        )

    @staticmethod
    def _pad_to_match(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad tensors a and b on bottom/right to the same (H, W)."""
        _, _, ha, wa = a.shape
        _, _, hb, wb = b.shape
        th, tw = max(ha, hb), max(wa, wb)
        pad = lambda x, h, w: F.pad(x, (0, tw - w, 0, th - h)) if (th - h) or (tw - w) else x
        return pad(a, ha, wa), pad(b, hb, wb)

    @staticmethod
    def _sinusoidal_positional_encoding(length: int, d_model: int, device: torch.device) -> torch.Tensor:
        """Return [length, d_model] sinusoidal embeddings for width positions."""
        pe = torch.zeros(length, d_model, device=device)
        position = torch.arange(0, length, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-torch.log(torch.tensor(10000.0, device=device)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _to_width_tokens(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Convert CNN feature map [B, C, H, W] to token sequence along width.
        We average over H to keep width structure, then project to d_model.
        Returns [B, W, d_model].
        """
        B, C, H, W = feat.shape
        print(feat.shape)
        x = feat.mean(dim=2)  # [B, C, W]
        x = x.transpose(1, 2)  # [B, W, C]
        x = self.proj(x)       # [B, W, d_model]
        x = self.token_ln(x)
        return x

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        e_south = data["E_south"]; b_south = data["B_south"]
        e_north = data["E_north"]; b_north = data["B_north"]

        south_stack = torch.cat([e_south, b_south], dim=1)
        north_stack = torch.cat([e_north, b_north], dim=1)
        south_stack, north_stack = self._pad_to_match(south_stack, north_stack)

        # CNN features preserve spatial structure
        south_feat = self.shared_cnn(south_stack)  # [B, Cc, H', W']
        north_feat = self.shared_cnn(north_stack)

        # Build width tokens per hemisphere
        south_tokens = self._to_width_tokens(south_feat)  # [B, W_s, d]
        north_tokens = self._to_width_tokens(north_feat)  # [B, W_n, d]

        # Apply sinusoidal PE separately
        B, W_s, D = south_tokens.shape
        _, W_n, _ = north_tokens.shape
        pe_s = self._sinusoidal_positional_encoding(W_s, D, south_tokens.device).unsqueeze(0)
        pe_n = self._sinusoidal_positional_encoding(W_n, D, north_tokens.device).unsqueeze(0)
        south_tokens = south_tokens + pe_s
        north_tokens = north_tokens + pe_n

        # Add class embedding (0=south, 1=north)
        cls_s = self.class_embed(torch.tensor(0, device=south_tokens.device)).view(1, 1, -1)
        cls_n = self.class_embed(torch.tensor(1, device=north_tokens.device)).view(1, 1, -1)
        south_tokens = south_tokens + cls_s
        north_tokens = north_tokens + cls_n

        # Concatenate sequences only after PE + class embedding
        tokens = torch.cat([south_tokens, north_tokens], dim=1)  # [B, W_s+W_n, d]

        # Self-attention over data tokens with per-layer query extraction
        collected_queries = []
        x = tokens
        for l in range(self.n_layers):
            x = self.encoder[l](x)
            q = self.query_tokens[l].unsqueeze(0).expand(B, -1, -1)  # [B, Q, D]
            q_out, _ = self.cross_attn[l](q, x, x)  # queries attend over data tokens
            q_out = self.query_ln(q_out)
            collected_queries.append(q_out)

        # Aggregate queries across layers and tokens
        Q_all = torch.stack(collected_queries, dim=1)  # [B, Lc, Q, D]
        q_mean = Q_all.mean(dim=(1, 2))  # [B, D]
        z = self.head(q_mean)  # [B, latent_dim]
        return z


# Simple registry to integrate with the existing model selection flow
KIDS_MODEL_BUILDERS = {
    "kids_o3_dual": lambda num_outputs, **kwargs: KidsO3NorthSouthEmbedding(latent_dim=num_outputs, **kwargs),
    "kids_combined_cnn_transformer": lambda num_outputs, **kwargs: KidsCombinedCNNTransformer(latent_dim=num_outputs, **kwargs),
}
