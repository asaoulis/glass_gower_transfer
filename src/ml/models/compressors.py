import torch
import torch.nn as nn
from typing import Optional

# optional torchvision import for auxiliary builders
try:
    from torchvision import models
except Exception:  # pragma: no cover - allow environments without torchvision
    models = None


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, first_block=False):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='circular'),
            nn.LeakyReLU(negative_slope=0.2)
        ]
        if not first_block:
            layers.insert(1, nn.BatchNorm2d(out_channels))
        layers += [
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='circular'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=True, padding_mode='circular'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
# drop-in pooling + projection module
import torch
import torch.nn as nn
import torch.nn.functional as F

class GeM(nn.Module):
    """Generalized mean pooling with learnable p (shared or per-channel)."""
    def __init__(self, p_init: float = 3.0, eps: float = 1e-6, per_channel: bool = False, clamp=(0.1, 10.0)):
        super().__init__()
        if per_channel:
            self.p = nn.Parameter(torch.ones(1, 1) * float(p_init))  # will expand for channels if needed
        else:
            self.p = nn.Parameter(torch.tensor(float(p_init)))
        self.eps = eps
        self.per_channel = per_channel
        self.clamp_min, self.clamp_max = clamp

    def forward(self, x):
        # x: (B, C, H, W)
        x = torch.clamp(x, min=self.eps)
        p = self.p
        if self.per_channel and p.dim() == 2:
            # expand to (1, C) at runtime if needed (works with channels known from input)
            p = p.expand(1, x.size(1))
        p = torch.clamp(p, min=self.clamp_min, max=self.clamp_max)
        # mean over H,W of x**p then raise to 1/p
        return x.pow(p.unsqueeze(-1).unsqueeze(-1)).mean(dim=(-2, -1)).pow(1.0 / p.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)

class SpatialPyramidPooling(nn.Module):
    """SPP -> concatenates flattened grids for output_sizes (e.g. (1,2,4))."""
    def __init__(self, output_sizes=(1,2,4), mode='avg'):
        super().__init__()
        assert mode in ('avg', 'max')
        self.output_sizes = tuple(output_sizes)
        self.mode = mode

    def forward(self, x):
        # x: (B, C, H, W) -> returns (B, C * sum(s*s))
        B, C, H, W = x.shape
        parts = []
        for s in self.output_sizes:
            if self.mode == 'avg':
                p = F.adaptive_avg_pool2d(x, output_size=(s, s))
            else:
                p = F.adaptive_max_pool2d(x, output_size=(s, s))
            parts.append(p.view(B, C * s * s))
        return torch.cat(parts, dim=1)

class SpatialAttentionPool(nn.Module):
    """Lightweight spatial attention pooling: weighted average with learned logits."""
    def __init__(self, in_channels, hidden=128):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.attn = nn.Conv2d(hidden, 1, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = F.relu(self.proj(x))                # (B, hidden, H, W)
        logits = self.attn(h).view(B, -1)       # (B, H*W)
        weights = F.softmax(logits, dim=-1).view(B, 1, H, W)
        out = (x * weights).sum(dim=(-2, -1))   # (B, C)
        return out

class TransformerPool(nn.Module):
    """Tiny transformer encoder readout. Use only if H*W is modest."""
    def __init__(self, in_channels, nhead=4, nhid=128, nlayers=1, add_cls_token=True):
        super().__init__()
        d_model = in_channels
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=nhid, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.add_cls_token = add_cls_token
        if add_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        tokens = x.view(B, C, H*W).permute(0, 2, 1)   # (B, H*W, C)
        if self.add_cls_token:
            cls = self.cls_token.expand(B, -1, -1)    # (B,1,C)
            tokens = torch.cat([cls, tokens], dim=1)  # (B, 1+H*W, C)
        out = self.transformer(tokens)                # (B, seq, C)
        if self.add_cls_token:
            pooled = out[:, 0]
        else:
            pooled = out.mean(dim=1)
        return self.ln(pooled)                        # (B, C)

class PoolProj(nn.Module):
    """
    Pooling factory + learned projection.
    - pool_types: list of strings from {'avg','max','gem','spp','attn','trans'}
    - in_channels: number of channels spatial map (tail_conv.out_channels)
    - proj_dim: output dimension of projection (e.g. 32*hidden)
    - spp_sizes: grid sizes for SPP (default (1,2,4)); SPP output expands quickly
    - gem_per_channel: whether GeM has per-channel p
    - transformer_args: dict forwarded to TransformerPool
    """
    def __init__(self, pool_types, in_channels, proj_dim, spp_sizes=(1,2,4), gem_per_channel=False, transformer_args=None, attn_hidden=128):
        super().__init__()
        assert isinstance(pool_types, (list, tuple))
        self.pool_types = list(pool_types)
        self.in_ch = in_channels
        self.proj_dim = proj_dim
        self.spp_sizes = tuple(spp_sizes)
        self.transformer_args = transformer_args or {}
        self.attn_hidden = attn_hidden

        # instantiate subpool modules if needed
        if 'gem' in self.pool_types:
            self.gem = GeM(per_channel=gem_per_channel)
        if 'spp' in self.pool_types:
            self.spp = SpatialPyramidPooling(output_sizes=self.spp_sizes, mode='avg')
        if 'attn' in self.pool_types:
            self.attn = SpatialAttentionPool(in_channels, hidden=self.attn_hidden)
        if 'trans' in self.pool_types or 'transformer' in self.pool_types:
            # alias 'trans' or 'transformer'
            self.trans = TransformerPool(in_channels, **self.transformer_args)

        # compute concatenated descriptor dimension (needed to build projection)
        concat_dim = 0
        for t in self.pool_types:
            if t == 'avg' or t == 'max':
                concat_dim += in_channels
            elif t == 'gem':
                concat_dim += in_channels
            elif t == 'spp':
                # C * sum(s*s)
                concat_dim += in_channels * sum([s*s for s in self.spp_sizes])
            elif t == 'attn':
                concat_dim += in_channels
            elif t in ('trans', 'transformer'):
                concat_dim += in_channels
            else:
                raise ValueError(f"Unknown pool type: {t}")

        self.concat_dim = concat_dim

        # learned projection from concat_dim -> proj_dim
        # include a small bottleneck with activation + layernorm for stability
        self.proj = nn.Sequential(
            nn.Linear(self.concat_dim, max(self.proj_dim, 64)),
            nn.ReLU(inplace=True),
            nn.LayerNorm(max(self.proj_dim, 64)),
            nn.Linear(max(self.proj_dim, 64), self.proj_dim)
        )

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: (B, proj_dim)
        """
        B, C, H, W = x.shape
        parts = []
        for t in self.pool_types:
            if t == 'avg':
                parts.append(F.adaptive_avg_pool2d(x, 1).view(B, C))
            elif t == 'max':
                parts.append(F.adaptive_max_pool2d(x, 1).view(B, C))
            elif t == 'gem':
                parts.append(self.gem(x).view(B, C))
            elif t == 'spp':
                parts.append(self.spp(x))  # already (B, C * sum(s*s))
            elif t == 'attn':
                parts.append(self.attn(x))  # (B, C)
            elif t in ('trans', 'transformer'):
                parts.append(self.trans(x)) # (B, C)
        desc = torch.cat(parts, dim=1)  # (B, concat_dim)
        return self.proj(desc)          # (B, proj_dim)


#####################################################################################
class model_o3_err(nn.Module):
    def __init__(self, num_outputs, hidden, dr=0.35, channels=1, predict_sigmas=False):
        super(model_o3_err, self).__init__()
        self.predict_sigmas = predict_sigmas
        self.num_outputs = num_outputs
        if predict_sigmas:
            num_outputs = 2 * self.num_outputs

        # input: 1x256x256 ---------------> output: 2*hiddenx128x128
        self.C01 = nn.Conv2d(channels, 2 * hidden, kernel_size=3, stride=1, padding=1,
                             padding_mode='circular', bias=True)
        self.C02 = nn.Conv2d(2 * hidden, 2 * hidden, kernel_size=3, stride=1, padding=1,
                             padding_mode='circular', bias=True)
        self.C03 = nn.Conv2d(2 * hidden, 2 * hidden, kernel_size=2, stride=2, padding=0,
                             padding_mode='circular', bias=True)
        self.B01 = nn.BatchNorm2d(2 * hidden)
        self.B02 = nn.BatchNorm2d(2 * hidden)
        self.B03 = nn.BatchNorm2d(2 * hidden)

        # input: 2*hiddenx128x128 ----------> output: 4*hiddenx64x64
        self.C11 = nn.Conv2d(2 * hidden, 4 * hidden, kernel_size=3, stride=1, padding=1,
                             padding_mode='circular', bias=True)
        self.C12 = nn.Conv2d(4 * hidden, 4 * hidden, kernel_size=3, stride=1, padding=1,
                             padding_mode='circular', bias=True)
        self.C13 = nn.Conv2d(4 * hidden, 4 * hidden, kernel_size=2, stride=2, padding=0,
                             padding_mode='circular', bias=True)
        self.B11 = nn.BatchNorm2d(4 * hidden)
        self.B12 = nn.BatchNorm2d(4 * hidden)
        self.B13 = nn.BatchNorm2d(4 * hidden)

        # input: 4*hiddenx64x64 --------> output: 8*hiddenx32x32
        self.C21 = nn.Conv2d(4 * hidden, 8 * hidden, kernel_size=3, stride=1, padding=1,
                             padding_mode='circular', bias=True)
        self.C22 = nn.Conv2d(8 * hidden, 8 * hidden, kernel_size=3, stride=1, padding=1,
                             padding_mode='circular', bias=True)
        self.C23 = nn.Conv2d(8 * hidden, 8 * hidden, kernel_size=2, stride=2, padding=0,
                             padding_mode='circular', bias=True)
        self.B21 = nn.BatchNorm2d(8 * hidden)
        self.B22 = nn.BatchNorm2d(8 * hidden)
        self.B23 = nn.BatchNorm2d(8 * hidden)

        # input: 8*hiddenx32x32 ----------> output: 16*hiddenx16x16
        self.C31 = nn.Conv2d(8 * hidden, 16 * hidden, kernel_size=3, stride=1, padding=1,
                             padding_mode='circular', bias=True)
        self.C32 = nn.Conv2d(16 * hidden, 16 * hidden, kernel_size=3, stride=1, padding=1,
                             padding_mode='circular', bias=True)
        self.C33 = nn.Conv2d(16 * hidden, 16 * hidden, kernel_size=2, stride=2, padding=0,
                             padding_mode='circular', bias=True)
        self.B31 = nn.BatchNorm2d(16 * hidden)
        self.B32 = nn.BatchNorm2d(16 * hidden)
        self.B33 = nn.BatchNorm2d(16 * hidden)

        # input: 16*hiddenx16x16 ----------> output: 32*hiddenx8x8
        self.C41 = nn.Conv2d(16 * hidden, 32 * hidden, kernel_size=3, stride=1, padding=1,
                             padding_mode='circular', bias=True)
        self.C42 = nn.Conv2d(32 * hidden, 32 * hidden, kernel_size=3, stride=1, padding=1,
                             padding_mode='circular', bias=True)
        self.C43 = nn.Conv2d(32 * hidden, 32 * hidden, kernel_size=2, stride=2, padding=0,
                             padding_mode='circular', bias=True)
        self.B41 = nn.BatchNorm2d(32 * hidden)
        self.B42 = nn.BatchNorm2d(32 * hidden)
        self.B43 = nn.BatchNorm2d(32 * hidden)

        # input: 32*hiddenx8x8 ----------> output:64*hiddenx4x4
        self.C51 = nn.Conv2d(32 * hidden, 64 * hidden, kernel_size=3, stride=1, padding=1,
                             padding_mode='circular', bias=True)
        self.C52 = nn.Conv2d(64 * hidden, 64 * hidden, kernel_size=3, stride=1, padding=1,
                             padding_mode='circular', bias=True)
        self.C53 = nn.Conv2d(64 * hidden, 64 * hidden, kernel_size=2, stride=2, padding=0,
                             padding_mode='circular', bias=True)
        self.B51 = nn.BatchNorm2d(64 * hidden)
        self.B52 = nn.BatchNorm2d(64 * hidden)
        self.B53 = nn.BatchNorm2d(64 * hidden)

        # input: 64*hiddenx4x4 ----------> output: 128*hiddenx1x1
        self.C61 = nn.Conv2d(64 * hidden, 128 * hidden, kernel_size=4, stride=1, padding=0,
                             padding_mode='circular', bias=True)
        # self.B61 = nn.BatchNorm2d(128*hidden)
        self.B61 = nn.Identity()  # torch doesn't like BN on 1dim data
        self.P0 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.FC1 = nn.Linear(128 * hidden, 64 * hidden)
        self.FC2 = nn.Linear(64 * hidden, num_outputs)
        self.dropout = nn.Dropout(p=dr)
        self.ReLU = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, image):
        x = self.LeakyReLU(self.C01(image))
        x = self.LeakyReLU(self.B02(self.C02(x)))
        x = self.LeakyReLU(self.B03(self.C03(x)))
        x = self.LeakyReLU(self.B11(self.C11(x)))
        x = self.LeakyReLU(self.B12(self.C12(x)))
        x = self.LeakyReLU(self.B13(self.C13(x)))
        x = self.LeakyReLU(self.B21(self.C21(x)))
        x = self.LeakyReLU(self.B22(self.C22(x)))
        x = self.LeakyReLU(self.B23(self.C23(x)))
        x = self.LeakyReLU(self.B31(self.C31(x)))
        x = self.LeakyReLU(self.B32(self.C32(x)))
        x = self.LeakyReLU(self.B33(self.C33(x)))
        x = self.LeakyReLU(self.B41(self.C41(x)))
        x = self.LeakyReLU(self.B42(self.C42(x)))
        x = self.LeakyReLU(self.B43(self.C43(x)))
        x = self.LeakyReLU(self.B51(self.C51(x)))
        x = self.LeakyReLU(self.B52(self.C52(x)))
        x = self.LeakyReLU(self.B53(self.C53(x)))
        x = self.LeakyReLU(self.B61(self.C61(x)))
        x = x.view(image.shape[0], -1)
        x = self.dropout(x)
        x = self.dropout(self.LeakyReLU(self.FC1(x)))
        x = self.FC2(x)
        if self.predict_sigmas:
            # enforce the errors to be positive
            y = torch.clone(x)
            y[:, self.num_outputs:2*self.num_outputs] = torch.square(x[:, self.num_outputs:2*self.num_outputs])
            return y
        return x


####################################################################################
####################################################################################
class FlexibleO3(nn.Module):
    """
    O3-like CNN that works with arbitrary input sizes.
    - Uses repeated stages with stride-2 downsampling.
    - Applies adaptive pooling to 1x1 then a 1x1 tail conv to fix channel count.
    - Infers the linear head input feature size via a dummy forward using max_hw.
    - Optionally returns spatial feature maps instead of flatten+FFN.
    - Supports configurable normalization per stage: GroupNorm (default) or BatchNorm2d.
    """
    def __init__(self, num_outputs: int, hidden: int = 12, channels: int = 1, dr: float = 0.15, max_hw=(256, 256), predict_sigmas: bool = False, return_features: bool = False, ch_mults = [8, 8, 16, 16, 32, 32], pool_types = ('avg', 'max', 'gem',), norm_type: str = 'group', gn_groups: Optional[int] = None):
        super().__init__()
        self.predict_sigmas = predict_sigmas
        self.num_outputs = num_outputs
        self.return_features = return_features
        # normalization config
        self.norm_type = (norm_type or 'group').lower()
        self.gn_groups = gn_groups
        if predict_sigmas and not return_features:
            num_outputs = 2 * num_outputs
        self.hidden = hidden
        # Build stages programmatically
        self._ch_mults = ch_mults
        stages = []
        in_ch = channels
        for i, m in enumerate(ch_mults):
            out_ch = m * hidden
            stages.append(self._make_stage(in_ch, out_ch, first=(i == 0)))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)
        # Expose a backbone attribute for compatibility with UNetO3StyleEncoder
        # so that loaders can uniformly access the convolutional feature extractor.
        self.backbone = self.stages
        # Tail conv; for features path we skip pooling
        self.adapt_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.tail_conv = nn.Conv2d(ch_mults[-1] * hidden, 32 * hidden, kernel_size=1, bias=True)
        self.tail_bn = nn.Identity()
        self.act = nn.LeakyReLU(0.2)
        # If returning features, we do not build FFN head
        if not self.return_features:
            self.poolproj = PoolProj(pool_types=pool_types, in_channels=32*hidden, proj_dim=32*hidden,
                                    spp_sizes=(1,2,4), gem_per_channel=False,
                                    transformer_args={'nhead':4, 'nlayers':1, 'nhid':128}, attn_hidden=128)

            # keep rest of head but set FC1 input dim to proj_dim
            self.feature_dim = self.poolproj.proj_dim
            self.FC1 = nn.Linear(self.feature_dim, 32 * hidden)
            self.FC2 = nn.Linear(32 * hidden, num_outputs)
            self.dropout = nn.Dropout(p=dr)
        # Init
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)

    def _make_norm(self, num_channels: int):
        if self.norm_type in ('group', 'gn', 'groupnorm'):
            g = self.gn_groups
            if g is None or num_channels % g != 0:
                # choose largest suitable group count from a safe set
                for candidate in [32, 16, 8, 4, 2, 1]:
                    if num_channels % candidate == 0:
                        g = candidate
                        break
            return nn.GroupNorm(g, num_channels)
        # default to BatchNorm2d
        return nn.BatchNorm2d(num_channels)

    def _make_stage(self, in_ch, out_ch, first=False):
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, padding_mode='zeros', bias=True),
            nn.LeakyReLU(0.2),
        ]
        if not first:
            layers.insert(1, self._make_norm(out_ch))
        layers += [
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, padding_mode='zeros', bias=True),
            self._make_norm(out_ch),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_ch, out_ch, kernel_size=2, stride=2, padding=0, padding_mode='zeros', bias=True),
            self._make_norm(out_ch),
            nn.LeakyReLU(0.2),
        ]
        return nn.Sequential(*layers)

    def _infer_feature_dim(self, max_hw, in_channels):
        with torch.no_grad():
            h, w = max_hw
            device = next(self.stages.parameters()).device if any(p.requires_grad for p in self.stages.parameters()) else torch.device('cpu')
            x = torch.zeros(1, in_channels, h, w, device=device)
            x = self.stages(x)
            x = self.act(self.tail_bn(self.tail_conv(self.adapt_pool(x))))
            x = x.view(1, -1)
            return x.shape[1]

    def forward(self, x):
        # Use backbone (stages) as the convolutional feature extractor
        x = self.backbone(x)
        if self.return_features:
            # Keep spatial dims and return feature map after tail conv + activation
            x = self.act(self.tail_bn(self.tail_conv(x)))
            return x
        # Original head path
        x = self.act(self.tail_bn(self.tail_conv(x)))   # (B, tail_ch, H, W)
        x = self.poolproj(x)                             # (B, 32*hidden) == self.feature_dim
        x = self.dropout(x)
        x = self.dropout(self.act(self.FC1(x)))
        x = self.FC2(x)
        if self.predict_sigmas:
            y = torch.clone(x)
            y[:, self.num_outputs:2*self.num_outputs] = torch.square(x[:, self.num_outputs:2*self.num_outputs])
            return y
        return x


def flexible_o3_model(num_outputs, hidden=12, channels=1, max_hw=(256, 256), predict_sigmas=False, return_features=False, **kwargs):
    return FlexibleO3(num_outputs=num_outputs, hidden=hidden, channels=channels, max_hw=max_hw, predict_sigmas=predict_sigmas, return_features=return_features, **kwargs)


# Example usage

def build_resnet(num_outputs, pretrained=True):
    if models is None:
        raise ImportError("torchvision is required for build_resnet but is not installed.")
    resnet = models.resnet18(pretrained=pretrained)
    # Copy weights from the original layer
    original_weights = resnet.conv1.weight.data
    # Average the weights across the RGB channels
    new_weights = original_weights.mean(dim=1, keepdim=True)
    # Replace the conv1 layer and assign the new weights
    resnet.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=resnet.conv1.out_channels,
        kernel_size=resnet.conv1.kernel_size,
        stride=resnet.conv1.stride,
        padding=resnet.conv1.padding,
        bias=resnet.conv1.bias is not None
    )
    resnet.conv1.weight.data = new_weights
    # add two fc layers
    resnet.fc = nn.Sequential(
        # nn.Linear(512, 256),
        # nn.ReLU(),
        nn.Linear(512, num_outputs),
    )
    return resnet


def build_convnext(num_outputs, pretrained=True):
    if models is None:
        raise ImportError("torchvision is required for build_convnext but is not installed.")
    convnext = models.convnext_tiny(pretrained=pretrained)

    # Get the original first convolution layer
    original_conv = convnext.features[0][0]  # First conv layer in ConvNeXt
    original_weights = original_conv.weight.data

    # Average the weights across RGB channels
    new_weights = original_weights.mean(dim=1, keepdim=True)

    # Replace first conv layer with single-channel input
    convnext.features[0][0] = nn.Conv2d(
        in_channels=1,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=original_conv.bias is not None
    )
    convnext.features[0][0].weight.data = new_weights

    # Modify classifier head
    in_features = convnext.classifier[2].in_features
    convnext.classifier = nn.Sequential(
        nn.Flatten(),
        # nn.Linear(in_features, 256),
        # nn.ReLU(),

        nn.Linear(in_features, num_outputs),
    )

    return convnext


_MODEL_BUILDERS = {
    "o3": lambda num_outputs, **kwargs: model_o3_err(num_outputs, hidden=12).to(device='cuda'),
    "flex_o3": lambda num_outputs, max_hw=(256, 256), channels=1, hidden=12, **kwargs: flexible_o3_model(num_outputs, hidden=hidden, channels=channels, max_hw=max_hw, **kwargs),
    "resnet": lambda num_outputs, pretrained=True, **kwargs: build_resnet(num_outputs, pretrained=pretrained),
    "convnext": lambda num_outputs, pretrained=True, **kwargs: build_convnext(num_outputs, pretrained=pretrained)
}
