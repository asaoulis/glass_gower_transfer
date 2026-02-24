import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Sequence

from .neural_operators import KidsFNONorthSouthEmbedding
from .transformers import QueryCrossAttentionBlock
from .compressors import PoolProj

def stack_hemi(data, hemi):
    keys = [k for k in ["E_" + hemi, "B_" + hemi] if k in data]
    return torch.cat([data[k] for k in keys], dim=1)
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
        # Generic head placeholder; subclasses should either use
        # ``build_head`` or override it completely.
        self.head: nn.Module | None = None

    # ------------------------------------------------------------------
    # Head utility
    # ------------------------------------------------------------------
    def build_head(self, in_dim: int, hidden_head: int | None = None) -> nn.Sequential:
        """Utility to build a standard 2-layer MLP head.

        All subclasses should prefer this helper so we keep the same
        pattern everywhere.
        """
        if self.model_output_dim is None:
            raise ValueError("model_output_dim is None; KidsInferenceEncoder must have latent_dim set before building head.")
        if hidden_head is None:
            hidden_head = max(self.model_output_dim, in_dim // 2)
        return nn.Sequential(
            nn.Linear(in_dim, hidden_head),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_head, self.model_output_dim),
        )

    # ------------------------------------------------------------------
    # Representation interface
    # ------------------------------------------------------------------
    def get_representation(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return a feature representation before the final head.

        Subclasses are expected to override this method and use
        ``self.head`` to map the representation to the final latent
        space in ``_forward_base``.
        """
        raise NotImplementedError

    def _forward_base(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Default implementation using get_representation + head.

        Subclasses that override ``get_representation`` and use a
        standard head do not need to override this.
        """
        if self.head is None:
            raise RuntimeError("head is not defined; subclasses must create self.head using build_head or override _forward_base.")
        feats = self.get_representation(data)
        return self.head(feats)

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.use_kl:
            return self._forward_base(data)

        raw = self._forward_base(data)
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

    Optionally, instead of concatenating north and south features, we can
    stack them as (B, C, 2) and aggregate across the hemisphere dimension
    (the last spatial dim) using a PoolProj module. This is controlled by the
    ``aggregate_north_south`` flag for backward compatibility.
    """

    def __init__(
        self,
        latent_dim: int,
        cnn_out_dim: int = 256,
        hidden: int = 12,
        input_channels: int = 12,
        encoder_type: str = "flex_o3",
        aggregate_north_south: bool = False,
        hemi_pool_types: Sequence[str] = ("avg", "max", "gem"),
        **kwargs,
    ):
        super().__init__(latent_dim=latent_dim, **kwargs)
        self.cnn_out_dim = cnn_out_dim
        self.encoder_type = encoder_type
        self.aggregate_north_south = aggregate_north_south

        in_channels = input_channels  # E + B stacked per hemisphere (6 + 6 = 12)
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
        if self.aggregate_north_south:
            # Pool (B, C, 2) -> (B, C) with PoolProj, preserving channel dim
            # Last two dims are treated as spatial by PoolProj, so we use H=1, W=2.
            self.hemi_pool = PoolProj(
                pool_types=hemi_pool_types,
                in_channels=cnn_out_dim,
                proj_dim=2*cnn_out_dim,
            )
            head_in_dim = 2*cnn_out_dim
        else:
            head_in_dim = 2 * cnn_out_dim

        hidden_head = max(self.model_output_dim, head_in_dim // 2)
        self.head = self.build_head(head_in_dim, hidden_head=hidden_head)

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

    def _forward_stack(self, data):
        south_stack = stack_hemi(data, "south")
        north_stack = stack_hemi(data, "north")

        # Ensure south matches target spatial size by zero-padding (bottom/right)
        south_stack = self._pad_to_size(south_stack, (MAX_INPUT_HW[0], MAX_INPUT_HW[1]))
        north_stack = self._pad_to_size(north_stack, (MAX_INPUT_HW[0], MAX_INPUT_HW[1]))

        # Shared encoder processes both stacks
        south_feat = self.shared_cnn(south_stack)
        north_feat = self.shared_cnn(north_stack)

        if self.aggregate_north_south:
            # stack along a pseudo-width dim so PoolProj pools over hemispheres
            x = torch.stack([south_feat, north_feat], dim=-1)  # (B, C, 2)
            x = x.unsqueeze(-2)  # (B, C, 1, 2): H=1, W=2 -> hemispheres as spatial dim
            pooled = self.hemi_pool(x)  # (B, C)
            feats = pooled
        else:
            # Backward-compatible behaviour: concatenate features
            feats = torch.cat([south_feat, north_feat], dim=1)
        return feats
    
    def get_representation(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._forward_stack(data)

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
        separate_hemispheres = True,
        **kwargs
    ):
        super().__init__(latent_dim=latent_dim, **kwargs)
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_queries = n_queries
        self.pool_queries = pool_queries
        self.transformer_type = transformer_type
        self.separate_hemispheres = separate_hemispheres

        from .compressors import flexible_o3_model

        in_channels = input_channels
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

    def get_representation(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
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
            if self.separate_hemispheres:
                # DETR-style: encode each hemisphere separately, then decode with queries
                south_memory = self.encoder(south_tokens)   # [B, W_s, D]
                north_memory = self.encoder(north_tokens)   # [B, W_n, D]
                memory = torch.cat([south_memory, north_memory], dim=1)  # [B, T, D]
            else:
                # Encode concatenated tokens from both hemispheres
                x = torch.cat([south_tokens, north_tokens], dim=1)  # [B, T, D]
                memory = self.encoder(x)                            # [B, T, D]

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

        return pooled


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
        self.net = nn.Sequential(*layers)
        self.head = self.build_head(width)
        self.redundancy_dim = redundancy_dim

    def get_representation(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
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
        conv_layers += [nn.Flatten()]
        self.conv = nn.Sequential(*conv_layers)
        # Keep length dimension; project flattened [B, C_out * L] -> latent_dim
        in_features = curr_in * seq_len
        self.head = self.build_head(in_features)
        self.redundancy_dim = redundancy_dim

    def _forward_base(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Preserve the original behaviour that also handles redundancy_dim
        x = data["mixed_bandpowers"]  # [B, 21, 20]
        if x.dim() != 3:
            raise ValueError(f"Expected bandpowers tensor with 3 dims [B, C, L], got shape {tuple(x.shape)}")
        x = self.conv(x)
        z = self.head(x)
        if self.redundancy_dim != 0:
            zeros = torch.zeros(z.shape[0], self.redundancy_dim, device=z.device, dtype=z.dtype)
            z = torch.cat([z, zeros], dim=1)
        return z
    
    def get_representation(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = data["mixed_bandpowers"]  # [B, 21, 20]
        if x.dim() != 3:
            raise ValueError(f"Expected bandpowers tensor with 3 dims [B, C, L], got shape {tuple(x.shape)}")
        x = self.conv(x)
        return x


class KidsHybridBandpowersMaps(KidsInferenceEncoder):
    """Hybrid model that processes both bandpowers and map patches.

    This subclass owns the KL behaviour for the *concatenated* latent.
    The child encoders (band_encoder, patch_encoder) may themselves be
    KL or non‑KL. We therefore do not assume anything about their
    ``use_kl`` flags and instead inspect their outputs at runtime.
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
        use_kl: bool = False,
        **kwargs,
    ):
        # For the *hybrid* encoder, latent_dim is the dim of mu after
        # concatenation; model_output_dim is 2*latent_dim when KL is used.
        super().__init__(latent_dim=latent_dim, use_kl=use_kl, **kwargs)

        bandpower_kwargs = bandpower_kwargs or {}
        if map_kwargs is None and transformer_kwargs is not None:
            map_kwargs = transformer_kwargs
        
        map_kwargs = map_kwargs or {}
        map_kwargs["input_channels"] = kwargs.get("input_channels", 12)  # default to 12 if not specified
        print("Map kwargs", map_kwargs, flush=True)

        # Split requested mu-dimension between band and patch branches.
        if bandpower_latent_dim is None:
            dim_band = self.latent_dim // 2
        else:
            dim_band = bandpower_latent_dim
        dim_patch = 128 # HACK FOR NOW

        bp_builders = {
            'mlp': KidsBandpowersMLP,
            'cnn': KidsBandpowersCNN1D,
        }
        if bandpower_type not in bp_builders:
            raise ValueError(f"Unknown bandpower_type '{bandpower_type}', expected one of {list(bp_builders.keys())}")
        self.band_encoder = bp_builders[bandpower_type](latent_dim=dim_band, **bandpower_kwargs)

        patch_builders = {
            'transformer': KidsCombinedCNNTransformer,
            'o3_dual': KidsO3NorthSouthEmbedding,
            'kids_o3_dual': KidsO3NorthSouthEmbedding,
            'dual': KidsO3NorthSouthEmbedding,
        }
        if map_encoder_type not in patch_builders:
            raise ValueError(f"Unknown map_encoder_type '{map_encoder_type}', expected one of {list(patch_builders.keys())}")
        self.patch_encoder = patch_builders[map_encoder_type](latent_dim=dim_patch, **map_kwargs)

        # Head maps concatenated mu (and optionally logvar) to final output.
        # When self.use_kl is True, self.model_output_dim == 2 * latent_dim.
        in_features = self.latent_dim if not self.use_kl else 2 * self.latent_dim
        in_features = 192
        self.hybrid_head = nn.Linear(in_features, self.model_output_dim)
        self.freeze_band = False

    def _normalise_child_output(self, out: torch.Tensor | tuple[torch.Tensor, torch.Tensor], expected_mu_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, logvar) for a child encoder output.

        - If the child returns a tuple, interpret as (mu, logvar).
        - If it returns a single tensor, treat it as mu and create
          a zeros logvar of matching shape.
        Ensures mu has the requested dimension.
        """
        if isinstance(out, tuple) and len(out) == 2:
            mu, logvar = out
        else:
            mu = out
            logvar = torch.zeros_like(mu)
        if mu.shape[-1] != expected_mu_dim:
            raise ValueError(
                f"Expected child mu dim {expected_mu_dim}, got {mu.shape[-1]}"
            )
        if logvar.shape != mu.shape:
            raise ValueError(
                f"Child logvar shape {tuple(logvar.shape)} does not match mu shape {tuple(mu.shape)}"
            )
        return mu, logvar

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """KL‑aware forward for the hybrid encoder.

        Child encoders may or may not use KL internally; we always
        expose (mu, logvar) at the hybrid level when self.use_kl=True.
        """
        raw = self._forward_base(data)

        if not self.use_kl:
            # Hybrid KL disabled: raw is [B, latent_dim]. Only mu exists.
            return raw

        if raw.shape[-1] != 2 * self.latent_dim:
            raise ValueError(
                f"Expected hybrid output dim to be 2*latent_dim ({2 * self.latent_dim}), got {raw.shape[-1]}"
            )
        mu, logvar = torch.chunk(raw, 2, dim=-1)
        if self.only_return_mu:
            return mu
        return mu, logvar
    
    def get_representation(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
            """Computes the concatenated features before the hybrid head.
            
            This implements the forward pass logic up to the hybrid head input,
            handling band freezing, patch encoding, and optional KL concatenation.
            """
            # --- band branch ---
            # if not self.freeze_band:
            #     band_out = self.band_encoder.compress(data)
            #     dim_band = self.band_encoder.latent_dim
            #     band_mu, band_logvar = self._normalise_child_output(band_out, dim_band)
            # else:
            #     # purely a representation when frozen
            band_repr = self.band_encoder.get_representation(data)
            # no mu/logvar here
            band_mu = band_repr
            band_logvar = None  # unused

            # --- patch branch (still true latent) ---
            patch_out = self.patch_encoder.compress(data)
            dim_patch = self.patch_encoder.latent_dim
            patch_mu, patch_logvar = self._normalise_child_output(patch_out, dim_patch)

            # concatenate along feature dim
            mu_concat = torch.cat([band_mu, patch_mu], dim=-1)

            if not self.use_kl:
                # If no KL, the head expects just the concatenated mus
                return mu_concat

            # If band is frozen we don't want it in KL, just use patch logvar
            if band_logvar is None:
                logvar_concat = patch_logvar
            else:
                logvar_concat = torch.cat([band_logvar, patch_logvar], dim=-1)

            z_cat = torch.cat([mu_concat, logvar_concat], dim=-1)

            # Optionally: keep latent_dim/model_output_dim consistent
            # (Preserving side effects from original _forward_base logic)
            self.latent_dim = mu_concat.shape[-1]
            self.model_output_dim = z_cat.shape[-1]

            return z_cat

    def _forward_base(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Reuses the logic from get_representation to avoid duplication
        head_input = self.get_representation(data)
        return self.hybrid_head(head_input)


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
