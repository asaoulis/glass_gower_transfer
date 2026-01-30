"""k-diffusion transformer diffusion models, version 2."""

from functools import reduce
import math

from einops import rearrange
import torch
from torch import nn
import torch._dynamo

from .axial_rope import make_axial_pos


try:
    import natten
except ImportError:
    natten = None

def zero_init(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def checkpoint(function, *args, **kwargs):
    if flags.get_checkpointing():
        kwargs.setdefault("use_reentrant", True)
        return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)
    else:
        return function(*args, **kwargs)


def downscale_pos(pos):
    pos = rearrange(pos, "... (h nh) (w nw) e -> ... h w (nh nw) e", nh=2, nw=2)
    return torch.mean(pos, dim=-2)


def scale_for_cosine_sim(q, k, scale, eps):
    dtype = reduce(torch.promote_types, (q.dtype, k.dtype, scale.dtype, torch.float32))
    sum_sq_q = torch.sum(q.to(dtype) ** 2, dim=-1, keepdim=True)
    sum_sq_k = torch.sum(k.to(dtype) ** 2, dim=-1, keepdim=True)
    sqrt_scale = torch.sqrt(scale.to(dtype))
    scale_q = sqrt_scale * torch.rsqrt(sum_sq_q + eps)
    scale_k = sqrt_scale * torch.rsqrt(sum_sq_k + eps)
    return q * scale_q.to(q.dtype), k * scale_k.to(k.dtype)


# Rotary position embeddings


def apply_rotary_emb(x, theta, conj=False):
    out_dtype = x.dtype
    dtype = reduce(torch.promote_types, (x.dtype, theta.dtype, torch.float32))
    d = theta.shape[-1]
    assert d * 2 <= x.shape[-1]
    x1, x2, x3 = x[..., :d], x[..., d : d * 2], x[..., d * 2 :]
    x1, x2, theta = x1.to(dtype), x2.to(dtype), theta.to(dtype)
    cos, sin = torch.cos(theta), torch.sin(theta)
    sin = -sin if conj else sin
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    y1, y2 = y1.to(out_dtype), y2.to(out_dtype)
    return torch.cat((y1, y2, x3), dim=-1)


def _apply_rotary_emb_inplace(x, theta, conj):
    dtype = reduce(torch.promote_types, (x.dtype, theta.dtype, torch.float32))
    d = theta.shape[-1]
    assert d * 2 <= x.shape[-1]
    x1, x2 = x[..., :d], x[..., d : d * 2]
    x1_, x2_, theta = x1.to(dtype), x2.to(dtype), theta.to(dtype)
    cos, sin = torch.cos(theta), torch.sin(theta)
    sin = -sin if conj else sin
    y1 = x1_ * cos - x2_ * sin
    y2 = x2_ * cos + x1_ * sin
    x1.copy_(y1)
    x2.copy_(y2)


class ApplyRotaryEmbeddingInplace(torch.autograd.Function):
    @staticmethod
    def forward(x, theta, conj):
        _apply_rotary_emb_inplace(x, theta, conj=conj)
        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, theta, conj = inputs
        ctx.save_for_backward(theta)
        ctx.conj = conj

    @staticmethod
    def backward(ctx, grad_output):
        (theta,) = ctx.saved_tensors
        _apply_rotary_emb_inplace(grad_output, theta, conj=not ctx.conj)
        return grad_output, None, None


def apply_rotary_emb_(x, theta):
    return ApplyRotaryEmbeddingInplace.apply(x, theta, False)


class AxialRoPE(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        log_min = math.log(math.pi)
        log_max = math.log(10.0 * math.pi)
        freqs = torch.linspace(log_min, log_max, n_heads * dim // 4 + 1)[:-1].exp()
        self.register_buffer("freqs", freqs.view(dim // 4, n_heads).T.contiguous())

    def extra_repr(self):
        return f"dim={self.freqs.shape[1] * 4}, n_heads={self.freqs.shape[0]}"

    def forward(self, pos):
        theta_h = pos[..., None, 0:1] * self.freqs.to(pos.dtype)
        theta_w = pos[..., None, 1:2] * self.freqs.to(pos.dtype)
        return torch.cat((theta_h, theta_w), dim=-1)


class NeighborhoodSelfAttentionBlock(nn.Module):
    def __init__(
        self, d_model, n_heads, kernel_size, dropout=0.0, rope_embeddings=False
    ):
        super().__init__()
        self.dim = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.rope_embeddings = rope_embeddings
        self.kernel_size = kernel_size
        if self.rope_embeddings:
            self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
            self.pos_emb = AxialRoPE(self.d_head // 2, self.n_heads)

    def extra_repr(self):
        return f"d_head={self.d_head}, kernel_size={self.kernel_size}"

    def forward(self, qkv, spatial=None):
        bs, width, length = qkv.shape
        ch = width // (3 * self.n_heads)
        scale = 1 / math.sqrt(math.sqrt(ch))
        q, k, v = rearrange(
            qkv, "n (nh t e) (h w) -> t n nh h w e", t=3, e=self.d_head, h=spatial[0]
        )
        if self.rope_embeddings:
            q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None, None], 1e-6)
            h, w = spatial
            pos = make_axial_pos(h, w, 1.0, dtype=q.dtype, device=q.device).view(
                h, w, 2
            )
            # pos = rearrange(pos, "... h w e -> ... (h w) e").to(qkv.dtype)
            theta = self.pos_emb(pos).movedim(-2, -4)
            q = apply_rotary_emb_(q, theta)
            k = apply_rotary_emb_(k, theta)

        else:
            q, k = q * scale, k * scale
        qk = natten.functional.na2d_qk(q, k, self.kernel_size)

        # flops.op(flops.op_natten, q.shape, k.shape, v.shape, self.kernel_size)
        a = torch.softmax(qk, dim=-1).to(v.dtype)
        x = natten.functional.na2d_av(a, v, self.kernel_size)
        x = rearrange(x, "n nh h w e -> n (nh e) (h w)")
        return x