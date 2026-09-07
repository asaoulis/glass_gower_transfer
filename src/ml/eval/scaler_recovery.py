"""Recover the training-time data-key scalers of an ALREADY-TRAINED run.

Why this exists
---------------
`_fit_data_key_scalers_from_paths` historically drew its 1000-file subsample from the global,
unseeded RNG, so no run before `config.scaler_fit_seed` recorded -- or could re-derive -- the input
frame its encoder was actually trained in. Refitting lands ~1.1e-2 (median |dz|/sd) away from it,
measured two independent ways: by the reproduction gate, and by the spread across the nine
ensemble-member embedding caches, which also shows every member used a DIFFERENT frame.

That is fine for mock numbers already published with that uncertainty baked in. It is NOT fine for
the real-data analysis, where one observation must be embedded in the frame its model was trained
in. Retraining would fix it; recovering the frame is far cheaper.

How
---
The cached `emb_test.pt` holds `z`, the exact embeddings the trained flow consumed, for ~2000
events. The encoder is frozen and differentiable, and the scalers are affine in log-space with just
TWO scalar parameters per key (`StandardScaler` and `LogNormalScaler` both store a single mean/std
for the whole array). So recovering the frame is a 6-parameter fit against ~16k residuals --
hugely overdetermined -- solved by gradient descent through the frozen encoder.

Note what the objective is: we do not need the "true" scalers, we need scalers that REPRODUCE z.
Those are the same thing wherever the encoder is sensitive, and where it is insensitive the
difference cannot affect any downstream embedding. So the objective is exactly the goal, which is
what makes this robust to the fit being only weakly identified in some direction.

Two things this must get right, or it will silently fit noise:
  * fp32, never the bf16 autocast the encoders are built with -- the target deviation is ~1%, which
    is at bf16's precision floor;
  * initialise at a plain refit, which is already within ~1% -- this is a local refinement, not a
    search.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..data.scaling import LogNormalScaler, StandardScaler


@dataclass
class RecoveryResult:
    key_scalers: Dict[str, object]
    n_events: int
    steps_run: int
    loss_initial: float
    loss_final: float
    z_dev_median_before: float
    z_dev_median_after: float
    z_dev_max_after: float
    params_initial: Dict[str, Dict[str, float]] = field(default_factory=dict)
    params_final: Dict[str, Dict[str, float]] = field(default_factory=dict)
    transform_parity_max: float = float("nan")

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d.pop("key_scalers", None)
        return d


class _LearnableScaler(torch.nn.Module):
    """Differentiable stand-in for one fitted key scaler.

    Mirrors `LogNormalScaler.transform` / `StandardScaler.transform` EXACTLY at the initial
    parameter values -- `check_parity` asserts that rather than trusting it, because a transform
    that merely resembles the real one would fit a frame the encoder never saw.
    """

    def __init__(self, scaler):
        super().__init__()
        self.kind = type(scaler).__name__
        if self.kind not in ("LogNormalScaler", "StandardScaler"):
            raise TypeError(f"cannot recover a {self.kind}: only scalar mean/std scalers are affine")
        self.eps = float(np.asarray(getattr(scaler, "eps", 1e-8)).ravel()[0])
        mean0 = float(np.asarray(scaler.mean).ravel()[0])
        std0 = float(np.asarray(scaler.std).ravel()[0])
        self.mean = torch.nn.Parameter(torch.tensor(mean0, dtype=torch.float64))
        # log-parameterised so std stays strictly positive under an unconstrained optimiser
        self.log_std = torch.nn.Parameter(torch.tensor(math.log(abs(std0)), dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if self.kind == "LogNormalScaler":
            x = torch.log10(torch.clamp(x, min=self.eps))
        return (x - self.mean.to(torch.float32)) / torch.exp(self.log_std).to(torch.float32)

    def to_scaler(self):
        out = LogNormalScaler(eps=self.eps) if self.kind == "LogNormalScaler" else StandardScaler()
        out.mean = float(self.mean.detach())
        out.std = float(torch.exp(self.log_std).detach())
        return out

    def params(self) -> Dict[str, float]:
        return {"mean": float(self.mean.detach()), "std": float(torch.exp(self.log_std).detach())}


def _encoder_forward(encoders, data: Dict[str, torch.Tensor]) -> torch.Tensor:
    """The same cut point and feature-wise concatenation `compute_embeddings` uses, with grad
    enabled and AMP off. `encoders` is a list so a multi-source row recovers the same z layout."""
    zs = []
    for encoder in encoders:
        if getattr(encoder, "embedding_cut", None) == "hybrid_pre_head":
            zs.append(encoder.get_frozen_features(data))
        else:
            encoder.only_return_mu = True
            zs.append(encoder(data))
    return zs[0] if len(zs) == 1 else torch.cat(zs, dim=-1)


def _load_raw_batches(raw_loader, max_events: Optional[int]) -> List[Dict[str, torch.Tensor]]:
    """Materialise the UNSCALED batches once so the optimiser can re-evaluate without re-reading
    HDF5 (LBFGS calls the closure many times per step)."""
    batches, n = [], 0
    for data, _theta in raw_loader:
        batches.append({k: v.clone() for k, v in data.items()})
        n += next(iter(data.values())).shape[0]
        if max_events is not None and n >= max_events:
            break
    return batches


def recover_key_scalers(
    encoders,
    raw_loader,
    z_target: torch.Tensor,
    init_key_scalers: Dict[str, object],
    *,
    max_events: Optional[int] = None,
    steps: int = 40,
    device: Optional[str] = None,
    verbose: bool = True,
) -> RecoveryResult:
    """Fit the data-key scalers that reproduce `z_target` through the frozen `encoder`.

    `raw_loader` must yield UNSCALED data (build it with empty `key_scalers`), in the SAME order as
    `z_target`. `encoders` is the list of frozen source encoders (as `compute_embeddings` takes).
    Returns fitted scaler objects ready for `save_scalers`.
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if not isinstance(encoders, (list, tuple)):
        encoders = [encoders]
    encoders = [e.to(dev).eval().to(torch.float32) for e in encoders]
    for e in encoders:
        for p in e.parameters():
            p.requires_grad_(False)

    batches = _load_raw_batches(raw_loader, max_events)
    n_events = sum(next(iter(b.values())).shape[0] for b in batches)
    z_ref = z_target[:n_events].to(dev, torch.float32)
    if verbose:
        print(f"[recover] {n_events} events, {len(batches)} batches, target z {tuple(z_ref.shape)}",
              flush=True)

    learn = torch.nn.ModuleDict({k: _LearnableScaler(s) for k, s in init_key_scalers.items()
                                 if k in batches[0]}).to(dev)
    params_initial = {k: m.params() for k, m in learn.items()}

    # Parity: the differentiable transform must equal the real fitted one at the init values.
    # Without this the optimiser could be minimising against a subtly different frame.
    parity = 0.0
    with torch.no_grad():
        for k, m in learn.items():
            x = batches[0][k][:2].to(dev)
            got = m(x)
            want = init_key_scalers[k].transform(x.to(torch.float32))
            parity = max(parity, float((got - want).abs().max()))
    if verbose:
        print(f"[recover] transform parity vs fitted scalers: max |d| = {parity:.3e}", flush=True)
    if not math.isfinite(parity) or parity > 1e-4:
        raise RuntimeError(
            f"differentiable transform does not reproduce the fitted scaler (max |d| = {parity:.3e}); "
            "recovery would fit the wrong frame"
        )

    def _z_now() -> torch.Tensor:
        outs = []
        for b in batches:
            scaled = {k: (learn[k](v.to(dev)) if k in learn else v.to(dev)) for k, v in b.items()}
            outs.append(_encoder_forward(encoders, scaled))
        return torch.cat(outs, dim=0)

    def _dev_median(z: torch.Tensor) -> float:
        sd = z_ref.std(dim=0).clamp_min(1e-12)
        return float(((z - z_ref).abs() / sd).median())

    with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=False):
        z0 = _z_now()
        loss0 = float(torch.nn.functional.mse_loss(z0, z_ref))
        dev0 = _dev_median(z0)
    if verbose:
        print(f"[recover] initial: mse = {loss0:.6e}, median |dz|/sd = {dev0:.3e}", flush=True)

    opt = torch.optim.LBFGS(list(learn.parameters()), max_iter=steps, history_size=20,
                            line_search_fn="strong_wolfe", tolerance_grad=1e-12,
                            tolerance_change=1e-14)
    calls = {"n": 0}

    def closure():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev.type, enabled=False):
            loss = torch.nn.functional.mse_loss(_z_now(), z_ref)
        loss.backward()
        calls["n"] += 1
        return loss

    opt.step(closure)

    with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=False):
        z1 = _z_now()
        loss1 = float(torch.nn.functional.mse_loss(z1, z_ref))
        sd = z_ref.std(dim=0).clamp_min(1e-12)
        d1 = ((z1 - z_ref).abs() / sd)
        dev1, dev1max = float(d1.median()), float(d1.max())
    if verbose:
        print(f"[recover] final:   mse = {loss1:.6e}, median |dz|/sd = {dev1:.3e} "
              f"(max {dev1max:.3e}) after {calls['n']} closure evals", flush=True)
        for k, m in learn.items():
            p0, p1 = params_initial[k], m.params()
            print(f"[recover]   {k}: mean {p0['mean']:.6f} -> {p1['mean']:.6f}, "
                  f"std {p0['std']:.6f} -> {p1['std']:.6f}", flush=True)

    return RecoveryResult(
        key_scalers={k: m.to_scaler() for k, m in learn.items()},
        n_events=n_events, steps_run=calls["n"],
        loss_initial=loss0, loss_final=loss1,
        z_dev_median_before=dev0, z_dev_median_after=dev1, z_dev_max_after=dev1max,
        params_initial=params_initial,
        params_final={k: m.params() for k, m in learn.items()},
        transform_parity_max=parity,
    )
