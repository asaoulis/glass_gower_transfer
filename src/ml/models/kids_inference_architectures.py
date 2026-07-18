import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Sequence

from .deprecated.transformers import QueryCrossAttentionBlock
from .unet.poolproj import PoolProj

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

    Optional `patch_conditioning` allows the shared CNN to know which hemisphere
    it is encoding via either an extra input channel ("channel") and/or a
    side-information embedding passed through the UNet FiLM pathway ("side_info").
    Supported values:
        None (default): no conditioning, fully backward compatible.
        ("channel",)
        ("side_info",)
        ("channel", "side_info")
    For `flex_o3` encoders only "channel" is used; "side_info" is ignored.
    For `unet_o3` encoders both mechanisms are supported.
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
        patch_conditioning: None | Tuple[str, ...] = None,
        **kwargs,
    ):
        super().__init__(latent_dim=latent_dim, **kwargs)
        self.cnn_out_dim = cnn_out_dim
        self.encoder_type = encoder_type
        self.aggregate_north_south = aggregate_north_south
        self.patch_conditioning = patch_conditioning
        # Normalize patch_conditioning to a tuple or None
        if self.patch_conditioning is not None:
            if isinstance(self.patch_conditioning, str):
                self.patch_conditioning = (self.patch_conditioning,)
            self.patch_conditioning = tuple(self.patch_conditioning)

        # Determine input channels (add +1 if channel conditioning is used)
        in_channels = input_channels
        if self.patch_conditioning is not None and "channel" in self.patch_conditioning:
            in_channels += 1
        print("Using per-patch conditioning:", self.patch_conditioning, flush=True)
        # E + B stacked per hemisphere (6 + 6 = 12 normally)
        if encoder_type == "flex_o3":
            from .deprecated.compressors import flexible_o3_model
            print("Using FlexibleO3 model as shared CNN", flush=True)
            self.shared_cnn = flexible_o3_model(
                num_outputs=cnn_out_dim,
                hidden=hidden,
                channels=in_channels,
                max_hw=MAX_INPUT_HW,
                predict_sigmas=False,
                **kwargs,
            )
        elif encoder_type == "unet_o3":
            from .unet.unet import UNetO3StyleEncoder
            print("Using UNetO3StyleEncoder as shared CNN", flush=True)
            channel_mult = (1, 1, 2, 2, 4, 8)
            # Enable patch conditioning inside UNet-style backbone when requested
            enable_patch = (
                self.patch_conditioning is not None
                and "side_info" in self.patch_conditioning
            )
            self.shared_cnn = UNetO3StyleEncoder(
                image_size=MAX_INPUT_HW[0],  # assume roughly square-ish height scale
                in_channels=in_channels,
                num_outputs=cnn_out_dim,
                # Backbone depth/width are overridable via map_kwargs (counts-extended task);
                # defaults preserve the historical architecture exactly.
                model_channels=kwargs.pop("model_channels", 32),
                num_res_blocks=kwargs.pop("num_res_blocks", 2),
                attention_resolutions=kwargs.pop("attention_resolutions", (3,)),
                channel_mult=kwargs.pop("channel_mult", channel_mult),
                enable_patch_conditioning=enable_patch,
                side_conditioning=False,
                **kwargs,
            )
        elif encoder_type == "preact_resnet":
            from .resnet_encoder import PreActResNetEncoder
            print("Using PreActResNetEncoder as shared CNN", flush=True)
            # Plain call path (no side_info/FiLM); 'channel' conditioning still works upstream.
            self.shared_cnn = PreActResNetEncoder(
                in_channels=in_channels,
                num_outputs=cnn_out_dim,
                **kwargs,
            )
        elif encoder_type == "convnext":
            from .convnext_encoder import ConvNeXtEncoder
            print("Using ConvNeXtEncoder as shared CNN", flush=True)
            self.shared_cnn = ConvNeXtEncoder(
                in_channels=in_channels,
                num_outputs=cnn_out_dim,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unknown encoder_type '{encoder_type}', expected 'flex_o3', 'unet_o3' "
                "or 'preact_resnet'"
            )

        # Shallow head after concatenation of north/south embeddings
        if self.aggregate_north_south:
            self.hemi_pool = PoolProj(
                pool_types=hemi_pool_types,
                in_channels=cnn_out_dim,
                proj_dim=2 * cnn_out_dim,
            )
            head_in_dim = 2 * cnn_out_dim
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
        return F.pad(x, (0, pad_w, 0, pad_h))

    def _maybe_append_hemi_channel(self, x: torch.Tensor, value: float) -> torch.Tensor:
        """Append a constant-valued hemisphere indicator channel if enabled."""
        if self.patch_conditioning is None or "channel" not in self.patch_conditioning:
            return x
        B, _, H, W = x.shape
        hemi_chan = x.new_full((B, 1, H, W), float(value))
        return torch.cat([x, hemi_chan], dim=1)

    def _forward_stack(self, data):
        south_stack = stack_hemi(data, "south")
        north_stack = stack_hemi(data, "north")

        # Input-channel conditioning: append hemisphere channel before padding
        south_stack = self._maybe_append_hemi_channel(south_stack, value=-1.0)
        north_stack = self._maybe_append_hemi_channel(north_stack, value=+1.0)

        # Ensure stacks match target spatial size by zero-padding (bottom/right)
        south_stack = self._pad_to_size(
            south_stack, (MAX_INPUT_HW[0], MAX_INPUT_HW[1])
        )
        north_stack = self._pad_to_size(
            north_stack, (MAX_INPUT_HW[0], MAX_INPUT_HW[1])
        )

        # Side-information conditioning via patch_id for UNet-based encoders
        use_side_info = (
            self.patch_conditioning is not None
            and "side_info" in self.patch_conditioning
            and self.encoder_type == "unet_o3"
        )
        B = south_stack.shape[0]
        south_ids = north_ids = None
        if use_side_info:
            device = south_stack.device
            south_ids = torch.zeros(B, dtype=torch.long, device=device)
            north_ids = torch.ones(B, dtype=torch.long, device=device)

        # Shared encoder processes both stacks
        if use_side_info:
            south_feat = self.shared_cnn(
                south_stack,
                cond=None,
                patch_id=south_ids,
            )
            north_feat = self.shared_cnn(
                north_stack,
                cond=None,
                patch_id=north_ids,
            )
        else:
            south_feat = self.shared_cnn(south_stack)
            north_feat = self.shared_cnn(north_stack)

        if self.aggregate_north_south:
            x = torch.stack([south_feat, north_feat], dim=-1)  # (B, C, 2)
            x = x.unsqueeze(-2)  # (B, C, 1, 2): H=1, W=2 -> hemispheres as spatial dim
            pooled = self.hemi_pool(x)  # (B, C)
            feats = pooled
        else:
            feats = torch.cat([south_feat, north_feat], dim=1)
        return feats
    
    def get_representation(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._forward_stack(data)

from .deprecated.kids_inference_architectures import KidsCombinedCNNTransformer


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
        hybrid_output_dim: int = None,
        use_kl: bool = False,
        # Fusion controls
        fusion_type: str = "concat",  # {'concat','film','gated'}
        fusion_hidden: int | None = None,
        patch_scale_init: float = 0.1,
        patch_scale_learnable: bool = True,
        # Anti-collapse controls (counts-training-performance task, 2026-07-14). All default OFF.
        # band_dropout_p: per-SAMPLE modality dropout of the (frozen) band branch during training
        #   — with prob p, band_mu is zeroed so the flow must explain theta from the maps alone,
        #   keeping gradient flowing into the map CNN even when the band shortcut saturates the
        #   VMIM loss. Eval/val always sees the full band.
        # patch_head_init_gain: init-only rescale of the patch encoder head's final Linear. At
        #   default init patch_mu enters the concat ~18x smaller than the pretrained band_mu
        #   (measured std 0.012 vs 0.225), leaving the map branch near-invisible to the flow at
        #   the start of training; a gain ~O(10) restores scale parity. Training remains free to
        #   shrink it (weights only scaled at construction).
        # patch_var_reg_coeff: coefficient for a VICReg-style hinge penalty
        #   mean(relu(1 - std(patch_mu, dim=0))^2) added by the Lightning training step —
        #   directly forbids the collapsed constant-patch_mu state observed in stuck runs.
        band_dropout_p: float = 0.0,
        patch_head_init_gain: float = 1.0,
        patch_var_reg_coeff: float = 0.0,
        patch_cov_reg_coeff: float = 0.0,
        sd_band_coeff: float = 0.0,
        # patch_norm (counts-extended task, 2026-07-16): normalisation barrier on patch_mu
        # before fusion. 'layernorm' = non-affine LayerNorm per sample — structurally removes
        # the measured DC-blowup/amplitude-explosion failure mode (patch_mu means grow to O(10)
        # while stds collapse) and fixes the ~280x band/patch scale imbalance at ALL times, not
        # just at init (cf. patch_head_init_gain). Applied in train AND eval (deterministic).
        patch_norm: str | None = None,
        **kwargs,
    ):
        # For the *hybrid* encoder, latent_dim is the dim of mu after concatenation of the
        # band + patch branches: it drives the dim_band/dim_patch split AND the hybrid_head
        # INPUT. hybrid_output_dim, when set, decouples the FINAL summary width (the hybrid_head
        # OUTPUT == model_output_dim == what the flow conditions on) from that concat dim -- e.g.
        # project a band(8)+patch(8)=16-D concat down to a 6-D summary. When None the output
        # equals latent_dim (fully backward compatible). NB: build_model mirrors this on the flow
        # side (src/ml/utils.py: latent_dim <- hybrid_output_dim for conditioning_dim), so the two
        # stay consistent. (Only exercised with use_kl=False; the hybrid_output_dim + KL combo is
        # untested -- forward()'s KL shape check assumes model_output_dim == 2*latent_dim.)
        super().__init__(latent_dim=latent_dim, use_kl=use_kl, **kwargs)
        if hybrid_output_dim is not None:
            # Override ONLY the head output width; keep self.latent_dim == concat dim so the
            # band/patch split and the hybrid_head input stay correct.
            self.model_output_dim = 2 * hybrid_output_dim if use_kl else hybrid_output_dim

        bandpower_kwargs = bandpower_kwargs or {}
        print("Bandpower kwargs", bandpower_kwargs, flush=True)
        if map_kwargs is None and transformer_kwargs is not None:
            map_kwargs = transformer_kwargs
        
        map_kwargs = map_kwargs or {}
        map_kwargs["input_channels"] = kwargs.get("input_channels", 12)  # default to 12 if not specified
        print("Map kwargs", map_kwargs, flush=True)

        self.fusion_type = fusion_type

        # Split requested mu-dimension between band and patch branches.
        if bandpower_latent_dim is None:
            dim_band = self.latent_dim // 2
        else:
            dim_band = bandpower_latent_dim
        if fusion_type == "concat":
            dim_patch = self.latent_dim - dim_band
        else:
            # For film/gated fusion we need dim_band == dim_patch to keep the math simple.
            dim_patch = dim_band
        self.dim_band = dim_band
        self.dim_patch = dim_patch

        bp_builders = {
            'mlp': KidsBandpowersMLP,
            'cnn': KidsBandpowersCNN1D,
        }
        if bandpower_type not in bp_builders:
            raise ValueError(f"Unknown bandpower_type '{bandpower_type}', expected one of {list(bp_builders.keys())}")
        # IMPORTANT: propagate KL-flag to the band encoder so pretrained KL checkpoints
        # (trained with use_KL_loss=True) have matching head parameter shapes.
        self.band_encoder = bp_builders[bandpower_type](
            latent_dim=dim_band,
            use_kl=self.use_kl,
            **bandpower_kwargs,
        )

        patch_builders = {
            'transformer': KidsCombinedCNNTransformer,
            'o3_dual': KidsO3NorthSouthEmbedding,
            'kids_o3_dual': KidsO3NorthSouthEmbedding,
            'dual': KidsO3NorthSouthEmbedding,
        }
        if map_encoder_type not in patch_builders:
            raise ValueError(f"Unknown map_encoder_type '{map_encoder_type}', expected one of {list(patch_builders.keys())}")
        self.patch_encoder = patch_builders[map_encoder_type](latent_dim=dim_patch, **map_kwargs)

        self.band_dropout_p = float(band_dropout_p)
        self.patch_var_reg_coeff = float(patch_var_reg_coeff)
        # counts-ext phase 4: patch_cov_reg_coeff = off-diagonal covariance (decorrelation)
        # penalty on patch_mu (anti-rank-1 pressure); sd_band_coeff = asymmetric spectral-
        # decoupling L2 on the band-only component of the fused latent (Pezeshki-style).
        # Both are read by the Lightning training step; the encoder just stores/caches.
        self.patch_cov_reg_coeff = float(patch_cov_reg_coeff)
        self.sd_band_coeff = float(sd_band_coeff)
        self._last_patch_mu = None
        self._last_band_mu = None
        self.cache_patch_mu = False  # set True by the Lightning module when the aux head is on
        if patch_norm is None:
            self.patch_norm_layer = None
        elif patch_norm == "layernorm":
            self.patch_norm_layer = nn.LayerNorm(self.dim_patch, elementwise_affine=False)
        elif patch_norm == "batchnorm":
            # Non-affine BatchNorm1d: forces per-dim batch std ~1 in training, making the
            # cross-sample constant-output collapse structurally impossible (LayerNorm cannot:
            # it normalises per sample). Eval uses running stats. Single-GPU runs only (no
            # SyncBN wiring) — our counts runs train on one GPU.
            self.patch_norm_layer = nn.BatchNorm1d(self.dim_patch, affine=False)
        else:
            raise ValueError(
                f"Unknown patch_norm '{patch_norm}', expected None, 'layernorm' or 'batchnorm'")
        if patch_head_init_gain != 1.0:
            # Init-only rescale of the patch head's final Linear (build_head ends in a Linear).
            last_linear = None
            head = getattr(self.patch_encoder, "head", None)
            if isinstance(head, nn.Sequential):
                for layer in head:
                    if isinstance(layer, nn.Linear):
                        last_linear = layer
            if last_linear is None:
                raise ValueError(
                    "patch_head_init_gain set but the patch encoder head has no final nn.Linear"
                )
            with torch.no_grad():
                last_linear.weight.mul_(float(patch_head_init_gain))
                if last_linear.bias is not None:
                    last_linear.bias.mul_(float(patch_head_init_gain))
            print(f"Applied patch_head_init_gain={patch_head_init_gain} to the patch head final Linear", flush=True)

        # -----------------
        # Fusion modules
        # -----------------
        # In all cases we produce mu_concat with dimension == self.latent_dim.
        if self.fusion_type not in {"concat", "film", "gated"}:
            raise ValueError("fusion_type must be one of {'concat','film','gated'}")

        # Scale controlling how much the PATCH branch can influence the fused mu early on.
        # When learnable, we optimize log_scale for positivity.
        if self.fusion_type != "concat":
            if patch_scale_learnable:
                self._patch_log_scale = nn.Parameter(torch.log(torch.tensor(float(patch_scale_init))))
            else:
                self.register_buffer("_patch_log_scale", torch.log(torch.tensor(float(patch_scale_init))), persistent=False)

        def _default_hidden() -> int:
            return max(32, 2 * self.dim_band)

        if fusion_hidden is None:
            fusion_hidden = _default_hidden()

        if self.fusion_type == "film":
            if self.dim_patch <= 0:
                raise ValueError("FiLM fusion requires dim_patch > 0")
            if self.dim_band <= 0:
                raise ValueError("FiLM fusion requires dim_band > 0")
            # patch_mu -> (gamma, beta) each of size dim_band
            self.film_mlp = nn.Sequential(
                nn.Linear(self.dim_patch, fusion_hidden),
                nn.GELU(),
                nn.Linear(fusion_hidden, 2 * self.dim_band),
            )
        elif self.fusion_type == "gated":
            if self.dim_band != self.dim_patch:
                raise ValueError(
                    "Gated fusion currently requires dim_band == dim_patch so the convex combination is well-defined. "
                    "Set bandpower_latent_dim to match the patch branch, or use fusion_type='concat'/'film'."
                )
            # gate and candidate are conditioned on patch_mu (keeps z1 dominant).
            self.gate_mlp = nn.Sequential(
                nn.Linear(self.dim_patch, fusion_hidden),
                nn.GELU(),
                nn.Linear(fusion_hidden, self.dim_band),
            )
            self.cand_mlp = nn.Sequential(
                nn.Linear(self.dim_patch, fusion_hidden),
                nn.GELU(),
                nn.Linear(fusion_hidden, self.dim_band),
            )

        # Head maps concatenated mu (and optionally logvar) to final output.
        # When self.use_kl is True, self.model_output_dim == 2 * latent_dim.
        # hybrid_head_hidden (Phase 2b, default None = historical single Linear): when set, use a
        # small MLP so the fusion can model band x map interactions (a single Linear cannot).
        in_features = self.latent_dim if not self.use_kl else 2 * self.latent_dim
        hybrid_head_hidden = kwargs.get("hybrid_head_hidden", None)
        if hybrid_head_hidden:
            self.hybrid_head = nn.Sequential(
                nn.Linear(in_features, int(hybrid_head_hidden)),
                nn.GELU(),
                nn.Linear(int(hybrid_head_hidden), self.model_output_dim),
            )
        else:
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

    def _patch_scale(self) -> torch.Tensor:
        # positive scalar
        return torch.exp(self._patch_log_scale)

    def _fuse_mu(self, band_mu: torch.Tensor, patch_mu: torch.Tensor) -> torch.Tensor:
        """Fuse band and patch mus into a single vector of size latent_dim.

        - concat:    [band_mu, patch_mu]
        - film:      band_mu + s * (gamma(band,patch)*band_mu + beta)
        - gated:     band_mu + s * (g*band_mu + (1-g)*cand(patch))  (requires equal dims)

        The scale s is initialized small (patch_scale_init) to keep patch contribution small at start.
        """
        if self.fusion_type == "concat":
            return torch.cat([band_mu, patch_mu], dim=-1)

        s = self._patch_scale()

        if self.fusion_type == "film":
            gb = self.film_mlp(patch_mu)  # [B, 2*dim_band]
            gamma, beta = torch.split(gb, [self.dim_band, self.dim_band], dim=-1)
            # Stable FiLM-like residual correction anchored at band_mu
            return band_mu + s * (gamma * band_mu + beta)

        # gated
        gate_logits = self.gate_mlp(patch_mu)
        g = torch.sigmoid(gate_logits)
        cand = self.cand_mlp(patch_mu)
        correction = g * band_mu + (1.0 - g) * cand
        return band_mu + s * correction

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
        """Compute the hybrid latent representation.

        - Non-KL: returns a mu-like vector of shape [B, latent_dim].
        - KL: returns a single vector of shape [B, 2*latent_dim] containing
          [mu, logvar] so that the shared ``hybrid_head`` can operate on the
          full latent statistics.
        """

        # --- band branch ---
        band_out = self.band_encoder.compress(data)
        if self.use_kl:
            band_mu, band_logvar = self._normalise_child_output(band_out, self.dim_band)
        else:
            band_mu = band_out

        # --- patch branch ---
        patch_out = self.patch_encoder.compress(data)
        if self.use_kl:
            patch_mu, patch_logvar = self._normalise_child_output(patch_out, self.dim_patch)
        else:
            patch_mu, _ = self._normalise_child_output(patch_out, self.dim_patch)

        if not self.use_kl:
            if self.patch_norm_layer is not None:
                patch_mu = self.patch_norm_layer(patch_mu)
            if self.training and self.band_dropout_p > 0:
                keep = (
                    torch.rand(band_mu.shape[0], 1, device=band_mu.device)
                    >= self.band_dropout_p
                ).to(band_mu.dtype)
                band_mu = band_mu * keep
            if (
                self.patch_var_reg_coeff > 0
                or self.patch_cov_reg_coeff > 0
                or self.sd_band_coeff > 0
                or self.cache_patch_mu
            ):
                self._last_patch_mu = patch_mu
                self._last_band_mu = band_mu
            mu = self._fuse_mu(band_mu, patch_mu)
            if mu.shape[-1] != self.latent_dim:
                raise ValueError(
                    f"Hybrid mu dim mismatch: expected latent_dim={self.latent_dim}, got {mu.shape[-1]}."
                )
            return mu

        mu = torch.cat([band_mu, patch_mu], dim=-1)
        logvar = torch.cat([band_logvar, patch_logvar], dim=-1)

        if mu.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Hybrid KL mu dim mismatch: expected latent_dim={self.latent_dim}, got {mu.shape[-1]}. "
                "If using fusion_type != 'concat', ensure bandpower_latent_dim is set consistently so the fused mu has size latent_dim."
            )
        if logvar.shape != mu.shape:
            raise ValueError(
                f"Hybrid KL logvar shape {tuple(logvar.shape)} does not match mu shape {tuple(mu.shape)}"
            )

        return torch.cat([mu, logvar], dim=-1)

    @torch.no_grad()
    def get_frozen_features(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Pre-head cut for frozen-trunk embedding caching (Phase 2b): band_mu ++ the patch
        encoder's pre-head N/S features (KidsO3 get_representation, i.e. BEFORE its head MLP).
        Eval-mode semantics (no band dropout); used via embedding_cut='hybrid_pre_head'."""
        band_out = self.band_encoder.compress(data)
        band_mu = band_out[0] if isinstance(band_out, tuple) else band_out
        feats = self.patch_encoder.get_representation(data)
        return torch.cat([band_mu, feats], dim=-1)

    def _forward_base(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
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
}
