
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Sequence

class QueryCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float,
        dropout: float,
    ):
        super().__init__()

        # --- X path ---
        self.x_self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.x_ln1 = nn.LayerNorm(d_model)
        self.x_ln2 = nn.LayerNorm(d_model)

        self.x_mlp = nn.Sequential(
            nn.Linear(d_model, int(mlp_ratio * d_model)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(mlp_ratio * d_model), d_model),
            nn.Dropout(dropout),
        )

        # --- Q path ---
        self.q_xattn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        self.q_ln1 = nn.LayerNorm(d_model)
        self.q_ln2 = nn.LayerNorm(d_model)
        self.q_ln3 = nn.LayerNorm(d_model)

        self.q_mlp = nn.Sequential(
            nn.Linear(d_model, int(mlp_ratio * d_model)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(mlp_ratio * d_model), d_model),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, x: torch.Tensor):
        """
        q: (B, Q, D)
        x: (B, T, D)
        """

        # ---- Update x (self-attention) ----
        x2, _ = self.x_self_attn(self.x_ln1(x), self.x_ln1(x), self.x_ln1(x))
        x = x + x2
        x = x + self.x_mlp(self.x_ln2(x))

        # ---- Update q (cross-attention) ----
        q2, _ = self.q_xattn(self.q_ln1(q), x, x)
        q = q + q2

        # ---- FFN on q ----
        q = q + self.q_mlp(self.q_ln3(q))

        return q, x
