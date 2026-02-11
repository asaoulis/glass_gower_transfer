from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)

from .unet_helpers import (
    GaussianFourierProjection,
    LocationFourierProjection,
    TimestepEmbedSequential,
    ResBlock,
    AttentionBlock,
    Downsample,
    Upsample,
    SplitConvDownsample,
)

# Import PoolProj so UNetO3StyleEncoder can share the same pooling head design
from ..compressors import PoolProj


class UNetStyleEncoder(nn.Module):
    """Encoder built from the same UNet building blocks.

    This module mirrors the downsampling depth of FlexibleO3 (6 stages with
    stride-2 operations by default) while reusing the same ResBlock,
    AttentionBlock, and Downsample components as UNetModel.

    Differences from UNetModel:
      * No skip connections or decoder path.
      * Returns a feature map with spatial dimensions preserved at the
        final downsampled resolution (no flattening or linear heads).
      * Optional cascade conditioning via SplitConvDownsample at each
        downsampling stage, matching the UNetModel convention.

    Args:
        image_size (int): Input spatial size.
        in_channels (int): Input channels.
        model_channels (int): Base channel width.
        num_res_blocks (int): Residual blocks per resolution level.
        attention_resolutions (sequence[int]): Downsample factors at which
            attention is applied (same semantics as UNetModel).
        dropout (float): Dropout rate for ResBlocks.
        channel_mult (tuple[int]): Per-level channel multipliers.
        conv_resample (bool): Whether to use learned convs for downsampling.
        dims (int): Convolution dimensionality (1, 2, or 3).
        use_checkpoint (bool): Enable gradient checkpointing.
        use_scale_shift_norm (bool): Use FiLM-style conditioning in ResBlocks.
        cascade_conditioning (bool): If True, expect a `cond` tensor that is
            progressively downsampled and concatenated at each downsampling
            stage, similar to UNetModel.cascade_downscalers.
    """

    def __init__(
        self,
        image_size: int,
        in_channels: int,
        model_channels: int = 64,
        num_res_blocks: int = 2,
        attention_resolutions=(4, 2, 1),
        dropout: float = 0.0,
        channel_mult=(1, 4, 8, 8),
        conv_resample: bool = True,
        dims: int = 2,
        use_checkpoint: bool = False,
        use_scale_shift_norm: bool = True,
        cascade_conditioning: bool = False,
        cond_channels: int | None = None,
        **kwargs,
    ):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = set(attention_resolutions)
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.dims = dims
        self.cascade_conditioning = cascade_conditioning

        ch = int(channel_mult[0] * model_channels)
        self.input_conv = conv_nd(dims, in_channels, ch, 3, padding=1)

        # Optional cascade downscalers: one per resolution level
        if self.cascade_conditioning:
            if cond_channels is None:
                raise ValueError("cond_channels must be provided when cascade_conditioning=True")
            self.cascade_downscalers = nn.ModuleList(
                [
                    SplitConvDownsample(cond_channels, dims=dims, out_channels=32)
                    for _ in channel_mult
                ]
            )
            self.cond_out_channels = 32
        else:
            self.cascade_downscalers = None
            self.cond_out_channels = 0

        blocks = []
        ds = 1
        train_size = image_size
        level_idx = 0
        for level, mult in enumerate(channel_mult):
            for res_idx in range(num_res_blocks):
                in_ch = ch
                out_ch = int(mult * model_channels)
                blocks.append(
                    ResBlock(
                        in_ch + (self.cond_out_channels if (self.cascade_conditioning and res_idx == 0 and level > 0) else 0),
                        emb_channels=0,
                        dropout=dropout,
                        num_embeddings=0,
                        out_channels=out_ch,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                )
                ch = out_ch
                # if ds in self.attention_resolutions:
                #     blocks.append(
                #         AttentionBlock(
                #             ch,
                #             attention_type="legacy",
                #             use_checkpoint=use_checkpoint,
                #             num_heads=1,
                #             num_head_channels=-1,
                #             use_new_attention_order=False,
                #             image_size=train_size,
                #         )
                #     )
            if level != len(channel_mult) - 1:
                blocks.append(
                    Downsample(
                        ch,
                        use_conv=conv_resample,
                        dims=dims,
                        out_channels=ch,
                    )
                )
                ds *= 2
                train_size = train_size // 2
            level_idx += 1

        self.blocks = nn.ModuleList(blocks)
        self.out_channels = ch

    def forward(self, x: th.Tensor, cond: th.Tensor | None = None) -> th.Tensor:
        """Encode input tensor into a lower-resolution feature map.

        Shape:
            x: (N, C_in, H, W) -> (N, C_out, H_out, W_out)
        """
        h = self.input_conv(x)
        if not self.cascade_conditioning:
            for m in self.blocks:
                if isinstance(m, ResBlock):
                    h = m(h, None)
                else:
                    h = m(h)
            return h

        # With cascade conditioning: downsample cond in lockstep with resolution levels
        if cond is None:
            raise ValueError("cond must be provided when cascade_conditioning=True")
        h_cond = cond
        level_idx = 0
        res_idx = 0
        for m in self.blocks:
            if isinstance(m, ResBlock):
                # Decide whether to concat conditional feature maps (first res block of each level after level 0)
                if res_idx == 0 and level_idx > 0:
                    h_cond, conv_cond = self.cascade_downscalers[level_idx - 1](h_cond)
                    h = th.cat([h, conv_cond], dim=1)
                h = m(h, None)
                res_idx += 1
            else:
                # Downsample op marks end of a level
                h = m(h)
                level_idx += 1
                res_idx = 0
        return h


class UNetO3StyleEncoder(nn.Module):
    """UNet-style encoder with an O3-inspired PoolProj head.

    Backbone:
      - Uses UNet-style ResBlocks, AttentionBlocks, Downsample, optional
        cascade conditioning, mirroring ``UNetStyleEncoder``.

    Head:
      - Uses the same PoolProj + MLP structure as FlexibleO3:
          tail_conv -> PoolProj(pool_types=['avg','max','gem']) -> FC1 -> FC2.
      - This keeps similar pooling / compression behaviour while reusing
        the UNet backbone.

    The encoder returns a vector of size ``num_outputs``.
    """

    def __init__(
        self,
        image_size: int,
        in_channels: int,
        num_outputs: int,
        model_channels: int = 64,
        num_res_blocks: int = 2,
        attention_resolutions=(4, 2, 1),
        dropout: float = 0.1,
        channel_mult=(1, 2, 4, 8, 16, 32),
        conv_resample: bool = True,
        dims: int = 2,
        use_checkpoint: bool = False,
        use_scale_shift_norm: bool = True,
        cascade_conditioning: bool = False,
        cond_channels: int | None = None,
        head_hidden_mult: float = 1.0,
        pool_types = ("avg", "max", "gem"),
        **kwargs
    ):
        super().__init__()
        self.backbone = UNetStyleEncoder(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=model_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            conv_resample=conv_resample,
            dims=dims,
            use_checkpoint=use_checkpoint,
            use_scale_shift_norm=use_scale_shift_norm,
            cascade_conditioning=cascade_conditioning,
            cond_channels=cond_channels,
        )
        self.num_outputs = num_outputs
        ch = self.backbone.out_channels

        # Tail conv analogous to FlexibleO3.tail_conv; keep channels moderate.
        tail_out = 32 * (model_channels // 64 if model_channels >= 64 else 1)
        self.tail_conv = conv_nd(dims, ch, tail_out, 1)
        self.tail_bn = normalization(tail_out)
        self.act = nn.SiLU()

        # PoolProj head, mirroring FlexibleO3 defaults
        self.poolproj = PoolProj(
            pool_types=list(pool_types),
            in_channels=tail_out,
            proj_dim=tail_out,
            spp_sizes=(1, 2, 4),
            gem_per_channel=False,
            transformer_args={"nhead": 4, "nlayers": 1, "nhid": 128},
            attn_hidden=128,
        )
        self.feature_dim = self.poolproj.proj_dim

        hidden_dim = max(num_outputs, int(head_hidden_mult * tail_out))
        self.FC1 = linear(self.feature_dim, hidden_dim)
        self.FC2 = linear(hidden_dim, num_outputs)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: th.Tensor, cond: th.Tensor | None = None) -> th.Tensor:
        h = self.backbone(x, cond)              # (B, C, H, W)
        h = self.act(self.tail_bn(self.tail_conv(h)))  # (B, tail_out, H, W)
        h = self.poolproj(h)                    # (B, feature_dim)
        h = self.dropout(h)
        h = self.dropout(self.act(self.FC1(h)))
        return self.FC2(h)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        embedding_type="fourier",
        cascade_conditioning=False,
        side_conditioning=False,
        attention_type="legacy",
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=True,
        num_heads=4,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_new_attention_order=False,
        diffusion=True,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.cascade_conditioning = cascade_conditioning
        self.side_conditioning = side_conditioning
        if self.side_conditioning:
            num_embeddings = 2
        else:
            num_embeddings = 1
        if diffusion:
            self.in_channels += out_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        # self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.embedding_type = embedding_type
        time_embed_dim = model_channels * 4

        if embedding_type == "fourier":
            self.timestep_embedding = GaussianFourierProjection(
                embedding_size=model_channels, scale=1
            )
            temb_input_dim = 2 * model_channels
        elif embedding_type == "positional":
            self.timestep_embedding = partial(timestep_embedding, dim=model_channels)
            temb_input_dim = model_channels
        elif embedding_type == "identity":
            self.timestep_embedding = nn.Identity()
            temb_input_dim = model_channels
        else:
            raise ValueError(f"embedding type {embedding_type} unknown.")

        self.time_embed = nn.Sequential(
            linear(temb_input_dim, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.side_conditioning:
            self.location_embedding = LocationFourierProjection(
                embedding_size=model_channels, scale=1
            )
            temb_input_dim = 2 * model_channels
            self.loc_embed = nn.Sequential(
                linear(temb_input_dim, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )

        conditional_channels = self.in_channels - (
            self.out_channels if diffusion else 0
        )
        if cascade_conditioning:
            self.cascade_downscalers = nn.ModuleList(
                [
                    SplitConvDownsample(
                        conditional_channels, dims=dims, out_channels=32
                    )
                    for i, _ in enumerate(channel_mult)
                ]
            )

        ch = input_ch = int(channel_mult[0] * model_channels)

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, self.in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        train_size = self.image_size
        for level, mult in enumerate(channel_mult):
            for res_idx in range(num_res_blocks):
                if self.cascade_conditioning and res_idx == 0 and level > 0:
                    ch = ch + 32
                else:
                    ch = ch
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        num_embeddings=num_embeddings,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            attention_type=attention_type,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                            image_size=train_size,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            num_embeddings=num_embeddings,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                train_size = train_size // 2
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                num_embeddings=num_embeddings,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                attention_type=attention_type,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
                image_size=train_size,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                num_embeddings=num_embeddings,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        num_embeddings=num_embeddings,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            attention_type=attention_type,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                            image_size=train_size,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            num_embeddings=num_embeddings,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                    train_size = train_size * 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    @th.compile()
    def forward(self, x, cond=None, timesteps=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.side_conditioning:
            cond, coords = cond
        elif isinstance(cond, list):
            cond, coords = cond
        if cond is not None:
            x = th.cat([x, cond], dim=1)
        hs = []
        if self.embedding_type == "fourier":
            timesteps = th.log(timesteps)

        emb = self.time_embed(self.timestep_embedding(timesteps))

        if self.side_conditioning:
            emb = th.stack(
                [emb, self.loc_embed(self.location_embedding(coords))], dim=1
            )
        else:
            emb = emb.unsqueeze(dim=1)
        h = x.type(x.dtype)
        downscaling_idx = 0
        for idx, module in enumerate(self.input_blocks):
            if self.cascade_conditioning and (idx - 1) % 5 == 0 and idx > 2:
                cond, conv_cond = self.cascade_downscalers[downscaling_idx](cond)
                downscaling_idx += 1
                h = th.cat([h, conv_cond], dim=1)
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)