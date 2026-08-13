"""Vectorised ensemble likelihood for NLE sampling.

The serial ensemble (`_EnsembleLikelihoodModel` in `ensemble_nle.py`) evaluates the M members'
log-likelihoods one at a time inside the MCMC potential, so ensemble sampling costs ~M× a single
model. The cost is overhead/launch-bound at the small batch sizes MCMC uses (a single flow on M×
the rows costs only a little more than on 1× the rows), so folding the M members into ONE forward
over an M-major-tiled batch recovers most of that factor.

Mechanism (see the task plan for the full derivation):
  - The production ``nsf`` flow is 5 × [RQS-coupling (conditioner = ResidualNet of nn.Linear) +
    LULinear]. The ONLY parameterised leaves are ``nn.Linear`` and ``LULinear``; every other op
    (spline math, base distribution, the deterministic coupling mask) is param-free and identical
    across independently-seeded members.
  - ``LULinear.forward_no_cache`` is two ``F.linear`` calls with a per-row-constant logabsdet, i.e.
    an affine map. So EVERY parameterised op reduces to a (grouped) matmul once weights are stacked.
  - We deep-copy member-0's raw nflows flow as a structural template and replace each ``nn.Linear``
    / ``LULinear`` with a grouped variant holding the stacked params of the corresponding-named
    module across all M members. The forward tiles inputs to M-major blocks ``(M*B, ·)``; each
    grouped module reshapes ``(M*B,·)->(M,B,·)`` and applies member-specific weights via ``einsum``;
    param-free ops run per-row unchanged (row order = member-major blocks is preserved throughout).

``torch.func.vmap`` is NOT used: the RQS spline does a boolean-mask ``index_put`` and a data-dependent
``if torch.any(...)`` branch, both of which vmap rejects. Eager grouped matmuls avoid that entirely
and are numerically equivalent to the serial path (verified by an allclose gate in the bench).

If the members' flow contains any parameterised leaf that is not ``nn.Linear``/``LULinear`` (e.g.
``maf``'s RandomPermutation/MADE, or zuko flows), :func:`build_grouped_ensemble_likelihood` returns
``None`` and the caller falls back to the serial loop, so we never produce wrong numbers silently.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

try:  # LULinear lives in nflows; import defensively so the module still loads if absent.
    from nflows.transforms.lu import LULinear
except Exception:  # pragma: no cover
    LULinear = None


class GroupedLinear(nn.Module):
    """Member-grouped replacement for ``nn.Linear`` over an M-major-tiled batch.

    Input rows are arranged as M contiguous blocks of B (member-major): rows ``[0:B]`` belong to
    member 0, ``[B:2B]`` to member 1, etc. Each block is multiplied by its own member's weight.
    Equivalent to looping ``F.linear(x_m, W_m, b_m)`` over members, but in one einsum.
    """

    def __init__(self, num_members: int, weight: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        self.num_members = int(num_members)
        # weight: (M, out, in); bias: (M, out) or None. Registered as buffers (eval-only, no grad).
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias if bias is not None else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        M = self.num_members
        if x.dim() != 2 or x.shape[0] % M != 0:
            raise ValueError(
                f"GroupedLinear expects a 2D (M*B, in) tensor with leading dim divisible by "
                f"M={M}; got shape {tuple(x.shape)}"
            )
        B = x.shape[0] // M
        xm = x.view(M, B, x.shape[1])
        # out[m,b,o] = sum_i xm[m,b,i] * W[m,o,i]   (mirrors F.linear(x, W) = x @ W.T)
        out = torch.einsum("mbi,moi->mbo", xm, self.weight)
        if self.bias is not None:
            out = out + self.bias.unsqueeze(1)
        return out.reshape(M * B, out.shape[-1])


class GroupedLULinear(nn.Module):
    """Member-grouped replacement for ``LULinear`` (a Transform: returns ``(outputs, logabsdet)``).

    Mirrors ``LULinear.forward_no_cache`` exactly per member:
    ``F.linear(F.linear(x, upper), lower, bias)`` with a per-row-constant logabsdet.
    """

    def __init__(
        self,
        num_members: int,
        lower: torch.Tensor,
        upper: torch.Tensor,
        bias: torch.Tensor | None,
        logabsdet: torch.Tensor,
    ):
        super().__init__()
        self.num_members = int(num_members)
        self.register_buffer("lower", lower)  # (M, D, D)
        self.register_buffer("upper", upper)  # (M, D, D)
        self.register_buffer("bias", bias if bias is not None else None)  # (M, D) or None
        self.register_buffer("logabsdet", logabsdet)  # (M,)

    def forward(self, inputs: torch.Tensor, context=None):
        M = self.num_members
        if inputs.dim() != 2 or inputs.shape[0] % M != 0:
            raise ValueError(
                f"GroupedLULinear expects a 2D (M*B, D) tensor with leading dim divisible by "
                f"M={M}; got shape {tuple(inputs.shape)}"
            )
        B = inputs.shape[0] // M
        D = inputs.shape[1]
        xm = inputs.view(M, B, D)
        # Two-step to match LULinear.forward_no_cache's op order (closest to bit-identical).
        tmp = torch.einsum("mbi,moi->mbo", xm, self.upper)
        out = torch.einsum("mbi,moi->mbo", tmp, self.lower)
        if self.bias is not None:
            out = out + self.bias.unsqueeze(1)
        outputs = out.reshape(M * B, D)
        logabsdet = self.logabsdet.view(M, 1).expand(M, B).reshape(M * B)
        return outputs, logabsdet


def _stack_param(modules, attr: str):
    """Stack ``getattr(m, attr)`` across modules into a leading-M tensor (or None if any is None)."""
    vals = [getattr(m, attr, None) for m in modules]
    if any(v is None for v in vals):
        return None
    return torch.stack([v.detach() for v in vals], dim=0)


def _set_submodule(root: nn.Module, dotted_name: str, new_module: nn.Module):
    """Replace ``root.<dotted_name>`` with ``new_module`` (walks parents, handles list indices)."""
    parts = dotted_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


def _is_groupable_leaf(mod) -> bool:
    """A parameterised leaf is groupable iff it is a PLAIN nn.Linear or an LULinear.

    The exact-type check on nn.Linear is deliberate: ``MaskedLinear`` (MADE's autoregressive layers in
    maf/rqs flows) SUBCLASSES nn.Linear but carries an autoregressive mask buffer that a plain grouped
    matmul would silently ignore — so ``isinstance`` would wrongly accept it. ``type(mod) is nn.Linear``
    rejects such subclasses, forcing the serial fallback for those flows.
    """
    if type(mod) is nn.Linear:
        return True
    if LULinear is not None and isinstance(mod, LULinear):
        return True
    return False


def _is_groupable_flow(raw_flows) -> bool:
    """True iff every parameterised leaf across all member flows is a groupable Linear/LULinear.

    NOTE: this is a necessary-but-not-sufficient screen. Param-free modules with member-varying buffers
    (e.g. ``RandomPermutation``'s random permutation) would slip past it, so the real guarantee is the
    build-time self-validation in :func:`build_grouped_ensemble_likelihood`.
    """
    if LULinear is None:
        return False
    for raw in raw_flows:
        for _, mod in raw.named_modules():
            has_params = any(True for _ in mod.parameters(recurse=False))
            if not has_params:
                continue
            if not _is_groupable_leaf(mod):
                return False
    return True


class GroupedEnsembleFlow(nn.Module):
    """Drop-in vectorised replacement for ``_EnsembleLikelihoodModel`` (same ``log_prob`` API/shape).

    Builds ONE grouped flow from the M members' raw nflows flows and evaluates all members in a
    single forward over an M-major-tiled batch, then reduces (``logmeanexp`` or ``mean_log_prob``).
    """

    def __init__(self, members, reduction: str = "logmeanexp"):
        super().__init__()
        if reduction not in {"mean_log_prob", "logmeanexp"}:
            raise ValueError("reduction must be one of {'mean_log_prob', 'logmeanexp'}")
        if not members:
            raise ValueError("GroupedEnsembleFlow requires at least one member.")
        self.members = nn.ModuleList(members)
        self.reduction = reduction
        self.num_members = len(members)

        # Keep each member's embedding_net (may be Identity for embeddings-NLE, or a CNN).
        self._embedding_nets = [m.model.embedding_net for m in members]
        raw_flows = [m.model.flow.net for m in members]  # the raw nflows Flow of each member

        if not _is_groupable_flow(raw_flows):
            raise ValueError("Member flows contain non-Linear/LULinear params; cannot group.")

        # Structural template = deep copy of member-0's flow (exact transforms, masks, base dist).
        template = copy.deepcopy(raw_flows[0])
        member_module_dicts = [dict(r.named_modules()) for r in raw_flows]

        for name, mod in list(template.named_modules()):
            corresponding = [d[name] for d in member_module_dicts]
            if isinstance(mod, nn.Linear):
                weight = _stack_param(corresponding, "weight")  # (M, out, in)
                bias = _stack_param(corresponding, "bias")  # (M, out) or None
                _set_submodule(template, name, GroupedLinear(self.num_members, weight, bias))
            elif LULinear is not None and isinstance(mod, LULinear):
                lowers, uppers, biases, logabsdets = [], [], [], []
                for lu in corresponding:
                    lower, upper = lu._create_lower_upper()
                    lowers.append(lower.detach())
                    uppers.append(upper.detach())
                    biases.append(lu.bias.detach() if lu.bias is not None else None)
                    logabsdets.append(lu.logabsdet().detach().reshape(()))
                lower = torch.stack(lowers, dim=0)
                upper = torch.stack(uppers, dim=0)
                bias = None if any(b is None for b in biases) else torch.stack(biases, dim=0)
                logabsdet = torch.stack(logabsdets, dim=0)
                _set_submodule(
                    template,
                    name,
                    GroupedLULinear(self.num_members, lower, upper, bias, logabsdet),
                )

        self.grouped_flow = template
        self.eval()

    def _embed_and_tile(self, x):
        """Per-member embed of the fixed data, stacked M-major: returns (M*B, latent)."""
        embs = [emb(x) for emb in self._embedding_nets]
        return torch.cat(embs, dim=0)

    @torch.no_grad()
    def log_prob(self, x, theta):
        M = self.num_members
        x_tiled = self._embed_and_tile(x)  # (M*B, latent)
        B = theta.shape[0]
        theta_tiled = theta.repeat(M, 1)  # (M*B, n_params), member-major (matches x tiling)

        flat = self.grouped_flow.log_prob(x_tiled, theta_tiled)  # (M*B,)
        per_member = flat.view(M, B)  # (M, B)

        if self.reduction == "logmeanexp":
            reduced = torch.logsumexp(per_member, dim=0) - np.log(M)  # (B,)
        else:
            reduced = per_member.mean(dim=0)  # (B,)

        # Match the serial _EnsembleLikelihoodModel output shape: (1, B).
        return reduced.unsqueeze(0)


@torch.no_grad()
def _matches_serial(grouped: "GroupedEnsembleFlow", members, reduction: str) -> bool:
    """Bulletproof, flow-agnostic guard: the grouped log_prob MUST equal the serial member loop.

    Computes the serial reference inline (no import of ``_EnsembleLikelihoodModel`` → no circular
    import) on a random probe batch and checks allclose. Catches ANY structural mismatch the static
    type screen misses (e.g. member-varying ``RandomPermutation`` buffers), guaranteeing we never
    silently return a grouped model that produces wrong numbers.
    """
    first = members[0]
    device = next(first.parameters()).device
    latent = first.inference_dim      # flow INPUT dim (data embedding) for nle
    nparams = first.conditioning_dim  # flow CONDITION dim (theta) for nle
    torch.manual_seed(0)
    x = torch.randn(4, latent, device=device)
    theta = torch.randn(4, nparams, device=device)

    lps = [m.forward(x, cond=theta) for m in members]  # each (1, B)
    stacked = torch.stack(lps, dim=0)  # (M, 1, B)
    if reduction == "logmeanexp":
        ref = torch.logsumexp(stacked, dim=0) - np.log(len(members))
    else:
        ref = stacked.mean(dim=0)

    out = grouped.log_prob(x, theta)
    return out.shape == ref.shape and torch.allclose(ref, out, atol=1e-4, rtol=1e-4)


def build_grouped_ensemble_likelihood(members, reduction: str = "logmeanexp"):
    """Return a :class:`GroupedEnsembleFlow`, or ``None`` if the members can't be safely grouped.

    ``None`` signals the caller to fall back to the serial ``_EnsembleLikelihoodModel`` (e.g. for
    ``maf``/``zuko`` flows, or anything that fails the equivalence self-check).
    """
    if not members:
        return None
    try:
        raw_flows = [m.model.flow.net for m in members]
    except AttributeError:
        return None
    if not _is_groupable_flow(raw_flows):
        return None
    try:
        grouped = GroupedEnsembleFlow(members, reduction=reduction)
    except (ValueError, KeyError):
        return None
    # Final guarantee: never hand back a grouped model that disagrees with the serial loop.
    try:
        if not _matches_serial(grouped, members, reduction):
            return None
    except Exception:
        return None
    return grouped
