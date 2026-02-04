import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Sequence

from .neural_operators import KidsFNONorthSouthEmbedding
from .transformers import QueryCrossAttentionBlock

# Small common interface for encoders used by NDELightningModule
class KidsInferenceEncoder(nn.Module):
    """Common interface for kids encoders.

    - compress(data) -> latent representation z

    If ``use_kl`` is True the encoder is interpreted as predicting a
    diagonal Gaussian in latent space. In that case ``latent_dim`` is
    the dimensionality of ``mu`` and the underlying network is assumed
    to return a tensor of shape ``[B, 2 * latent_dim]`` which is split
    into ``(mu, logvar)``.

    For backward compatibility, if the underlying network only returns
    ``[B, latent_dim]`` we assume this to be ``mu`` and set
    ``logvar = zeros_like(mu)``.

    The public interface is:
      * forward/compress() ->
          - train mode: logvar (so downstream modules can do reparam).
          - eval mode:  mu     (deterministic latents for inference).

    This keeps ``latent_dim`` equal to the dimension of ``mu`` so that
    downstream code that only cares about latent size does not break.
    """

    def __init__(self, *args, latent_dim: int | None = None, use_kl: bool = False, **kwargs):
        super().__init__()
        # latent_dim is the dimension of mu (the latent code used downstream)
        self.latent_dim = latent_dim
        self.use_kl = use_kl
        # model_output_dim is what the concrete network heads actually emit
        if latent_dim is None:
            self.model_output_dim = None
        else:
            self.model_output_dim = 2 * latent_dim if use_kl else latent_dim
        # When True, forward()/compress() only return mu, even if KL is used
        self.only_return_mu = False

    def _forward_base(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Subclasses should override this instead of forward().

        Must return a tensor of shape:
          * [B, latent_dim]          if ``use_kl=False`` or if only ``mu``
            is predicted.
          * [B, 2 * latent_dim]      if ``use_kl=True`` and the network
            explicitly predicts both (mu, logvar).
        """
        raise NotImplementedError

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.use_kl:
            return self._forward_base(data)

        raw = self._forward_base(data)
        # if self.latent_dim is None:
        #     # Infer latent_dim on first call from the contract that
        #     # latent_dim is the dimension of mu.
        #     if raw.shape[-1] % 2 == 0:
        #         self.latent_dim = raw.shape[-1] // 2
        #     else:
        #         self.latent_dim = raw.shape[-1]
        #     self.model_output_dim = raw.shape[-1]
        # else:
        #     self.model_output_dim = 2 * self.latent_dim

        # if raw.shape[-1] == self.latent_dim:
        #     mu = raw
        #     logvar = torch.zeros_like(mu)
        # else:
        if raw.shape[-1] != 2 * self.latent_dim:
            raise ValueError(
                f"Expected encoder output dim to be latent_dim ({self.latent_dim}) "
                f"or 2*latent_dim ({2 * self.latent_dim}), got {raw.shape[-1]}"
            )
        mu, logvar = torch.chunk(raw, 2, dim=-1)

        if self.only_return_mu:
            return mu
        return mu, logvar

    def compress(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Default implementation: call ``forward``.
        return self.forward(data)

# Maximum input spatial size (H, W) expected by the CNN to size its head.
# The FlexibleO3 backbone will run a dummy pass with this size to infer head dims.
MAX_INPUT_HW: Tuple[int, int] = (100, 1000)


class KidsO3NorthSouthEmbedding(KidsInferenceEncoder):
    """
    Embedding network that:
      - expects a data dict with keys: 'E_south', 'B_south', 'E_north', 'B_north'
      - each entry is a tensor of shape [B, C, H, W] with C=6
      - stacks E and B channelwise per hemisphere to form 12-channel inputs
      - applies the SAME CNN encoder (shared weights) to north and south stacks separately
      - zero-pads the south stack to match the north spatial size (if smaller)
      - concatenates the two feature vectors and passes through a shallow MLP
        to produce a final latent vector of size `latent_dim`.

    The shared encoder can be:
      - 'flex_o3' (default): FlexibleO3 head from ``compressors.py``.
      - 'unet_o3': UNetO3StyleEncoder from ``unet.unet`` using UNet blocks
        with an O3-like head.
    """

    def __init__(
        self,
        latent_dim: int,
        cnn_out_dim: int = 256,
        hidden: int = 12,
        channels_per_map: int = 6,
        encoder_type: str = "flex_o3",
        **kwargs,
    ):
        super().__init__(latent_dim=latent_dim, **kwargs)
        self.cnn_out_dim = cnn_out_dim
        self.channels_per_map = channels_per_map
        self.encoder_type = encoder_type

        in_channels = 2 * channels_per_map  # E + B stacked per hemisphere (6 + 6 = 12)

        if encoder_type == "flex_o3":
            from .compressors import flexible_o3_model
            print("Using FlexibleO3 model as shared CNN", flush=True)
            # Shared CNN used for both north and south
            self.shared_cnn = flexible_o3_model(
                num_outputs=cnn_out_dim,
                hidden=hidden,
                channels=in_channels,
                max_hw=MAX_INPUT_HW,
                predict_sigmas=False,
                **kwargs
            )
        elif encoder_type == "unet_o3":
            from .unet.unet import UNetO3StyleEncoder
            # Use UNet-style encoder with O3-inspired head
            print("Using UNetO3StyleEncoder as shared CNN", flush=True)
            channel_mult = (1, 1, 2, 2,4,8)
            self.shared_cnn = UNetO3StyleEncoder(
                image_size=MAX_INPUT_HW[0],  # assume roughly square-ish height scale
                in_channels=in_channels,
                num_outputs=cnn_out_dim,
                model_channels=32,
                num_res_blocks=2,
                attention_resolutions=(3,),
                channel_mult=channel_mult,
                cascade_conditioning=False,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown encoder_type '{encoder_type}', expected 'flex_o3' or 'unet_o3'")

        # Shallow head after concatenation of north/south embeddings
        hidden_head = max(self.model_output_dim, 2 * cnn_out_dim // 2)
        self.head = nn.Sequential(
            nn.Linear(2 * cnn_out_dim, hidden_head),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_head, self.model_output_dim),
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

    def _forward_base(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Expect tensors of shape [B, C=6, H, W]
        e_south = data["E_south"]
        b_south = data["B_south"]
        e_north = data["E_north"]
        b_north = data["B_north"]

        # Stack channelwise to get 12-channel inputs per hemisphere
        south_stack = torch.cat([e_south, b_south], dim=1)
        north_stack = torch.cat([e_north, b_north], dim=1)

        # Ensure south matches target spatial size by zero-padding (bottom/right)
        south_stack = self._pad_to_size(south_stack, (MAX_INPUT_HW[0], MAX_INPUT_HW[1]))
        north_stack = self._pad_to_size(north_stack, (MAX_INPUT_HW[0], MAX_INPUT_HW[1]))

        # Shared encoder processes both stacks
        # UNetO3StyleEncoder ignores "cond" and only needs x
        south_feat = self.shared_cnn(south_stack)
        north_feat = self.shared_cnn(north_stack)

        # Concatenate features and map to latent representation
        feats = torch.cat([south_feat, north_feat], dim=1)
        z = self.head(feats)
        return z


class KidsCombinedCNNTransformer(KidsInferenceEncoder):
    def __init__(
        self,
        latent_dim: int,
        hidden: int = 12,
        channels_per_map: int = 6,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        n_queries: int = 8,
        mlp_ratio: float = 2.0,
        dropout: float = 0.15,
        pool_queries: str = "mean",
        encoder_type: str = "flex_o3",
        transformer_type: str = "detr",
        **kwargs
    ):
        super().__init__(latent_dim=latent_dim, **kwargs)
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_queries = n_queries
        self.pool_queries = pool_queries
        self.transformer_type = transformer_type

        from .compressors import flexible_o3_model

        in_channels = 2 * channels_per_map
        if encoder_type == "flex_o3":
            self.shared_cnn = flexible_o3_model(
                num_outputs=None, # we are using return features = True so this is ignored
                hidden=hidden,
                channels=in_channels,
                max_hw=MAX_INPUT_HW,
                predict_sigmas=False,
                return_features=True,
            )
            self.cnn_out_channels = 32 * hidden

        elif encoder_type == "unet_o3":
            from .unet.unet import UNetStyleEncoder
            print("Using UNetStyleEncoder as shared CNN", flush=True)
            model_channels = 32
            channel_mult = (1, 1, 2, 2,4,8)
            cnn_out_dim = model_channels * channel_mult[-1]
            self.shared_cnn = UNetStyleEncoder(
                image_size=MAX_INPUT_HW[0],  # assume roughly square-ish height scale
                in_channels=in_channels,
                model_channels=model_channels,
                num_res_blocks=2,
                attention_resolutions=(3,),
                channel_mult=channel_mult,
                cascade_conditioning=False,
            )
            self.cnn_out_channels = self.shared_cnn.out_channels
        else:
            raise ValueError(f"Unknown encoder_type '{encoder_type}', expected 'flex_o3' or 'unet_o3'")

        # Project per-width tokens to d_model
        self.proj = nn.Linear(self.cnn_out_channels, d_model)
        self.token_ln = nn.LayerNorm(d_model)

        # Two classes: south (0) and north (1)
        self.class_embed = nn.Embedding(2, d_model)

        # Learnable query tokens (persistent)
        self.query_tokens = nn.Parameter(
            torch.randn(1, n_queries, d_model) * 0.2
        )

        # -------------------------------
        # Transformer backbone
        # -------------------------------
        if self.transformer_type == "detr":
            # DETR-style encoder/decoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=int(mlp_ratio * d_model),
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=n_layers,
            )

            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=int(mlp_ratio * d_model),
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=n_layers,
            )
            self.blocks = None

        elif self.transformer_type == "perceiver":
            # Perceiver-style stack of cross-attention blocks
            self.encoder = None
            self.decoder = None
            self.blocks = nn.ModuleList([
                QueryCrossAttentionBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ])
        else:
            raise ValueError(
                f"Unknown transformer_type '{self.transformer_type}', expected 'detr' or 'perceiver'"
            )

        self.final_ln = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, max(self.model_output_dim, d_model // 2)),
            nn.GELU(),
            nn.Linear(max(self.model_output_dim, d_model // 2), self.model_output_dim),
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
        Uses mean over height, projection to d_model, sinusoidal position encoding,
        then LayerNorm.
        Returns [B, W, d_model].
        """
        B, C, H, W = feat.shape

        # Collapse height
        x = feat.mean(dim=2)          # [B, C, W]
        x = x.transpose(1, 2)         # [B, W, C]

        # Project to model dim
        x = self.proj(x)              # [B, W, d_model]

        # --- positional encoding (sinusoidal) ---
        pe = self._sinusoidal_positional_encoding(
            length=W,
            d_model=self.d_model,
            device=x.device,
        ).unsqueeze(0)                # [1, W, D]

        # x = x + pe                    # ADD BEFORE LN

        # Normalize
        x = self.token_ln(x)

        return x

    def _forward_base(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        e_south = data["E_south"]; b_south = data["B_south"]
        e_north = data["E_north"]; b_north = data["B_north"]

        # CNN feature extraction
        south_stack = torch.cat([e_south, b_south], dim=1)
        north_stack = torch.cat([e_north, b_north], dim=1)
        south_stack, north_stack = self._pad_to_match(south_stack, north_stack)

        south_feat = self.shared_cnn(south_stack)
        north_feat = self.shared_cnn(north_stack)

        south_tokens = self._to_width_tokens(south_feat)  # [B, W_s, D]
        north_tokens = self._to_width_tokens(north_feat)  # [B, W_n, D]

        B_s, W_s, D = south_tokens.shape
        B_n, W_n, _ = north_tokens.shape
        assert B_s == B_n, "Batch size mismatch between hemispheres"
        B = B_s

        # Add learnt class embeddings to each token sequence
        south_cls = self.class_embed.weight[0].view(1, 1, D)  # [1, 1, D]
        north_cls = self.class_embed.weight[1].view(1, 1, D)  # [1, 1, D]
        south_tokens = south_tokens + south_cls
        north_tokens = north_tokens + north_cls

        if self.transformer_type == "detr":
            # DETR-style: encode each hemisphere separately, then decode with queries
            south_memory = self.encoder(south_tokens)   # [B, W_s, D]
            north_memory = self.encoder(north_tokens)   # [B, W_n, D]
            memory = torch.cat([south_memory, north_memory], dim=1)  # [B, T, D]

            q = self.query_tokens.expand(B, -1, -1)  # [B, Q, D]
            q = self.decoder(tgt=q, memory=memory)   # [B, Q, D]
        else:
            # Perceiver-style: concatenate tokens as inputs and update queries
            x = torch.cat([south_tokens, north_tokens], dim=1)  # [B, T, D]
            q = self.query_tokens.expand(B, -1, -1)            # [B, Q, D]
            for blk in self.blocks:
                q, x = blk(q, x)

        q = self.final_ln(q)

        if self.pool_queries == "first":
            pooled = q[:, 0]
        else:
            pooled = q.mean(dim=1)

        z = self.head(pooled)
        return z



# New simple models for bandpowers inputs
class KidsBandpowersMLP(KidsInferenceEncoder):
    """
    Simple MLP that flattens bandpowers of shape [B, 21, 20] and maps to latent_dim.
    Hidden layer width is latent_dim * hidden_multiple (default 2), with an
    adaptive number of hidden layers controlled by num_layers.
    """
    def __init__(
        self,
        latent_dim: int,
        input_shape: Tuple[int, int] = (21, 8),
        hidden_multiple: int = 2,
        num_layers: int = 5,
        dropout: float = 0.1,
        redundancy_dim: int = 0,
        **kwargs,
    ):
        super().__init__(latent_dim=latent_dim, **kwargs)
        in_features = int(input_shape[0] * input_shape[1])
        width = int(self.latent_dim * hidden_multiple)
        layers = [nn.Linear(in_features, width), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(max(0, num_layers - 1)):
            layers += [nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout)]
        layers += [nn.Linear(width, self.model_output_dim)]
        self.net = nn.Sequential(*layers)
        self.redundancy_dim = redundancy_dim

    def _forward_base(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = data["mixed_bandpowers"]  # [B, 21, 20]
        x = x.view(x.shape[0], -1)
        if self.redundancy_dim != 0:
            zeros = torch.zeros(x.shape[0], self.redundancy_dim, device=x.device, dtype=x.dtype)
            x = torch.cat([x, zeros], dim=1)
        return self.net(x)


class KidsBandpowersCNN1D(KidsInferenceEncoder):
    """
    1D CNN over length-20 series with 21 channels (bandpowers stacked on channel dim).
    The number of Conv1d blocks is controlled by the length of `channels`.
    Produces a latent vector of size latent_dim.
    """
    def __init__(
        self,
        latent_dim: int,
        in_channels: int = 21,
        seq_len: int = 8,
        channels: Sequence[int] = (64, 128),
        kernel_size: int = 3,
        dropout: float = 0.15,
        redundancy_dim: int = 0,
        **kwargs,
    ):
        super().__init__(latent_dim=latent_dim, **kwargs)
        padding = kernel_size // 2
        conv_layers: list[nn.Module] = []
        curr_in = in_channels
        for c in channels:
            conv_layers += [
                nn.Conv1d(curr_in, c, kernel_size=kernel_size, padding=padding, bias=True),
                nn.BatchNorm1d(c),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            curr_in = c
        self.conv = nn.Sequential(*conv_layers)
        # Keep length dimension; project flattened [B, C_out * L] -> latent_dim
        in_features = curr_in * seq_len
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_features, self.model_output_dim),
        )
        self.redundancy_dim = redundancy_dim

    def _forward_base(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = data["mixed_bandpowers"]  # [B, 21, 20]
        if x.dim() != 3:
            raise ValueError(f"Expected bandpowers tensor with 3 dims [B, C, L], got shape {tuple(x.shape)}")
        x = self.conv(x)
        z = self.head(x)
        if self.redundancy_dim != 0:
            zeros = torch.zeros(z.shape[0], self.redundancy_dim, device=z.device, dtype=z.dtype)
            z = torch.cat([z, zeros], dim=1)
        return z


class KidsHybridBandpowersMaps(KidsInferenceEncoder):
    """
    Hybrid model that processes both bandpowers and map patches:
      - A bandpowers encoder (choose 'mlp' or 'cnn') maps bandpowers -> latent_b
      - A patch encoder (choose 'transformer' or 'o3_dual') maps patches -> latent_p
      - Concatenate [latent_b, latent_p] to form final latent of size latent_dim
    Each branch produces roughly half the requested latent_dim (handles odd by
    allocating the remainder to the patch branch).
    """
    def __init__(
        self,
        latent_dim: int,
        bandpower_type: str = 'mlp',
        bandpower_kwargs: Dict = None,
        map_encoder_type: str = 'transformer',
        map_kwargs: Dict = None,
        transformer_kwargs: Dict = None,
        bandpower_latent_dim: int = None,
        **kwargs,
    ):
        super().__init__(latent_dim=latent_dim, **kwargs)
        bandpower_kwargs = bandpower_kwargs or {}
        # Backward-compat for existing configs using transformer_kwargs
        if map_kwargs is None and transformer_kwargs is not None:
            map_kwargs = transformer_kwargs
        map_kwargs = map_kwargs or {}
        print("Map kwargs", map_kwargs, flush=True)

        if bandpower_latent_dim is None:
            dim_band = self.latent_dim // 2
        else:
            dim_band = bandpower_latent_dim
        dim_patch = self.latent_dim - dim_band

        # Bandpowers encoder
        bp_builders = {
            'mlp': KidsBandpowersMLP,
            'cnn': KidsBandpowersCNN1D,
        }
        if bandpower_type not in bp_builders:
            raise ValueError(f"Unknown bandpower_type '{bandpower_type}', expected one of {list(bp_builders.keys())}")
        self.band_encoder = bp_builders[bandpower_type](latent_dim=dim_band, **bandpower_kwargs)

        # Patch encoder
        patch_builders = {
            'transformer': KidsCombinedCNNTransformer,
            'o3_dual': KidsO3NorthSouthEmbedding,
            'kids_o3_dual': KidsO3NorthSouthEmbedding,
            'dual': KidsO3NorthSouthEmbedding,
        }
        if map_encoder_type not in patch_builders:
            raise ValueError(f"Unknown map_encoder_type '{map_encoder_type}', expected one of {list(patch_builders.keys())}")
        self.patch_encoder = patch_builders[map_encoder_type](latent_dim=dim_patch, **map_kwargs)

    def _forward_base(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        z_band = self.band_encoder(data)
        z_patch = self.patch_encoder(data)
        return torch.cat([z_band, z_patch], dim=1)


# Simple registry to integrate with the existing model selection flow
KIDS_MODEL_BUILDERS = {
    "kids_o3_dual": lambda num_outputs, **kwargs: KidsO3NorthSouthEmbedding(latent_dim=num_outputs, **kwargs),
    "kids_combined_cnn_transformer": lambda num_outputs, **kwargs: KidsCombinedCNNTransformer(latent_dim=num_outputs, **kwargs),
    "kids_bandpowers_mlp": lambda num_outputs, **kwargs: KidsBandpowersMLP(latent_dim=num_outputs, **kwargs),
    "kids_bandpowers_cnn1d": lambda num_outputs, **kwargs: KidsBandpowersCNN1D(latent_dim=num_outputs, **kwargs),
    "kids_hybrid_bandpowers_maps": (
        lambda num_outputs,
               bandpower_type='mlp',
               bandpower_kwargs=None,
               map_encoder_type='o3_dual',
               map_kwargs=None,
               transformer_kwargs=None,
               **kwargs: KidsHybridBandpowersMaps(
                   latent_dim=num_outputs,
                   bandpower_type=bandpower_type,
                   bandpower_kwargs=bandpower_kwargs,
                   map_encoder_type=map_encoder_type,
                   map_kwargs=(map_kwargs if map_kwargs is not None else transformer_kwargs),
                   transformer_kwargs=transformer_kwargs,
                   **kwargs,
               )
    ),
    "kids_fno_dual": lambda num_outputs, **kwargs: KidsFNONorthSouthEmbedding(latent_dim=num_outputs, **kwargs),
}
