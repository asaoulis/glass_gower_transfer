from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kids_inference_architectures import KidsInferenceEncoder, MAX_INPUT_HW
from .compressors import flexible_o3_model
from .transformers import QueryCrossAttentionBlock


class KidsCombinedCNNTransformer(KidsInferenceEncoder):
    def __init__(
        self,
        latent_dim: int,
        hidden: int = 12,
        input_channels: int = 12,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        n_queries: int = 8,
        mlp_ratio: float = 2.0,
        dropout: float = 0.15,
        pool_queries: str = "mean",
        encoder_type: str = "flex_o3",
        transformer_type: str = "detr",
        separate_hemispheres=True,
        **kwargs,
    ):
        super().__init__(latent_dim=latent_dim, **kwargs)
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_queries = n_queries
        self.pool_queries = pool_queries
        self.transformer_type = transformer_type
        self.separate_hemispheres = separate_hemispheres

        in_channels = input_channels
        if encoder_type == "flex_o3":
            self.shared_cnn = flexible_o3_model(
                num_outputs=None,
                hidden=hidden,
                channels=in_channels,
                max_hw=MAX_INPUT_HW,
                predict_sigmas=False,
                return_features=True,
            )
            self.cnn_out_channels = 32 * hidden
        elif encoder_type == "unet_o3":
            from ..unet.unet import UNetStyleEncoder

            print("Using UNetStyleEncoder as shared CNN", flush=True)
            model_channels = 32
            channel_mult = (1, 1, 2, 2, 4, 8)
            self.shared_cnn = UNetStyleEncoder(
                image_size=MAX_INPUT_HW[0],
                in_channels=in_channels,
                model_channels=model_channels,
                num_res_blocks=2,
                attention_resolutions=(3,),
                channel_mult=channel_mult,
                cascade_conditioning=False,
            )
            self.cnn_out_channels = self.shared_cnn.out_channels
        else:
            raise ValueError(
                f"Unknown encoder_type '{encoder_type}', expected 'flex_o3' or 'unet_o3'"
            )

        self.proj = nn.Linear(self.cnn_out_channels, d_model)
        self.token_ln = nn.LayerNorm(d_model)
        self.class_embed = nn.Embedding(2, d_model)
        self.query_tokens = nn.Parameter(torch.randn(1, n_queries, d_model) * 0.2)

        if self.transformer_type == "detr":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=int(mlp_ratio * d_model),
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=int(mlp_ratio * d_model),
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
            self.blocks = None
        elif self.transformer_type == "perceiver":
            self.encoder = None
            self.decoder = None
            self.blocks = nn.ModuleList(
                [
                    QueryCrossAttentionBlock(
                        d_model=d_model,
                        n_heads=n_heads,
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                    )
                    for _ in range(n_layers)
                ]
            )
        else:
            raise ValueError(
                f"Unknown transformer_type '{self.transformer_type}', expected 'detr' or 'perceiver'"
            )

        self.final_ln = nn.LayerNorm(d_model)
        self.head = self.build_head(d_model)

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
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device)
            * (-torch.log(torch.tensor(10000.0, device=device)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _to_width_tokens(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Convert CNN feature map [B, C, H, W] to token sequence along width.
        Uses mean over height, projection to d_model, sinusoidal position encoding,
        then LayerNorm.
        Returns [B, W, d_model].
        """
        B, C, H, W = feat.shape
        x = feat.mean(dim=2)
        x = x.transpose(1, 2)
        x = self.proj(x)
        pe = self._sinusoidal_positional_encoding(
            length=W,
            d_model=self.d_model,
            device=x.device,
        ).unsqueeze(0)
        x = self.token_ln(x)
        return x

    def get_representation(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        e_south = data["E_south"]; b_south = data["B_south"]
        e_north = data["E_north"]; b_north = data["B_north"]

        south_stack = torch.cat([e_south, b_south], dim=1)
        north_stack = torch.cat([e_north, b_north], dim=1)
        south_stack, north_stack = self._pad_to_match(south_stack, north_stack)

        south_feat = self.shared_cnn(south_stack)
        north_feat = self.shared_cnn(north_stack)

        south_tokens = self._to_width_tokens(south_feat)
        north_tokens = self._to_width_tokens(north_feat)

        B_s, W_s, D = south_tokens.shape
        B_n, W_n, _ = north_tokens.shape
        assert B_s == B_n, "Batch size mismatch between hemispheres"
        B = B_s

        south_cls = self.class_embed.weight[0].view(1, 1, D)
        north_cls = self.class_embed.weight[1].view(1, 1, D)
        south_tokens = south_tokens + south_cls
        north_tokens = north_tokens + north_cls

        if self.transformer_type == "detr":
            if self.separate_hemispheres:
                south_memory = self.encoder(south_tokens)
                north_memory = self.encoder(north_tokens)
                memory = torch.cat([south_memory, north_memory], dim=1)
            else:
                x = torch.cat([south_tokens, north_tokens], dim=1)
                memory = self.encoder(x)

            q = self.query_tokens.expand(B, -1, -1)
            q = self.decoder(tgt=q, memory=memory)
        else:
            x = torch.cat([south_tokens, north_tokens], dim=1)
            q = self.query_tokens.expand(B, -1, -1)
            for blk in self.blocks:
                q, x = blk(q, x)

        q = self.final_ln(q)
        if self.pool_queries == "first":
            pooled = q[:, 0]
        else:
            pooled = q.mean(dim=1)
        return pooled
