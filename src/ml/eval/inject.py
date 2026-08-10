"""Synthetic noise-variance injection: does the NETWORK read the b_g channel?

Context (`.claude/runs/training-runs/improved-shear-tests/artifacts/`):
`BG_MECHANISM_AND_DES.md` §9 established *what the maps do* under a wrong `b_g` — the excess
power is white, its per-bin amplitude tracks `σ_δ,pix²`, and it is convex in `b_g`. What it did
NOT establish is that the trained compressor's multi-σ shift is *caused* by that channel. Every
piece of evidence so far is a consistency argument.

This module closes that link without any new simulation and without retraining, and it is the
experiment that discriminates the two surviving hypotheses:

* **P1 — the smooth variance modulation.** `Var(m_p) ∝ 1/(1 + b_g δ_p)` (the counts-family
  invariance theorem, `NSIDE512_AND_DES_CNN.md` §2). DES Y3 have this channel too: Jeffrey 2025
  Eq. 14 carries `(Σn̄ / Σn̄(1+b_gδ))^{1/2}` and it is in their fiducial `b_g = 1` training sims.
* **P2 — the discrete/occupancy sector.** Poisson `N_p` drawn from a clustered `λ`: the convex
  `⟨1/N⟩` term, the low-`N` heavy tail, the δ-correlated empty-pixel population. DES have
  **none** of it (their mock galaxies sit at the real catalogue's positions and weights).

**A post-hoc multiplicative modulation of a stored map can synthesise P1 exactly. It cannot
synthesise P2 at all.** So: inject a calibrated P1 modulation onto `b_g = 1.0` events and
re-measure the paired posterior shift.

  * shift ≈ the real `b_g = 1.5` shift (+4.5σ in Ω_m) ⇒ **P1 is sufficient**, and the tension with
    DES's < 0.3σ is about their *compression*, not their forward model;
  * shift ≪ that ⇒ **P2 dominates**, and the DES reconciliation is structural — they are immune
    because they never simulate the discreteness, exactly as the user argued.

Calibration is empirical, not modelled. `AMP` below is set from the ebdiff ensemble-mean power
ratios ⟨P(1.5)/P(1.0)⟩ measured on these very stores (job 1342435), so the injected maps
reproduce the measured per-bin b_g response **by construction** — the harness verifies this and
prints the achieved ratio next to the target.

Why a *multiplicative* modulation is the right synthesis. Writing the map as signal + noise with
`Var(noise) ∝ 1/(1+b_gδ)`, moving `b_g: 1 → b'` scales the noise by `[(1+δ)/(1+b'δ)]^{1/2}`. We
do not know this event's `δ`, so we instead inject an independent modulation field `g` whose
amplitude is chosen to reproduce the *measured* variance of the log-noise-variance field at
`b_g = b'`. That matches the statistic the compressor can actually read (the fluctuation
amplitude and spectrum of the local noise level), which is the whole channel. Two sources bracket
the readout hypothesis:

  * ``grf``   — an independent red Gaussian field. Correct amplitude/spectrum, **uncorrelated**
    with this event's true structure. Moves the posterior only if the network reads the
    *fluctuation amplitude* of the noise level.
  * ``kappa`` — the event's own low-passed E map, rescaled. **Correlated** with the true
    structure, so it also moves the posterior if the network reads the noise-level field's
    *coherence with the shear*. Caveat: it is derived from the map it modulates, and κ_i traces
    all `z < z_i` while the modulation traces `δ` at the bin's own redshift — a partial proxy, so
    treat it as the optimistic end of the bracket.
  * ``null``  — amplitude 0. Control: must return a paired shift consistent with zero, and it
    exercises the identical code path (dtype round-trips included).

The bandpower branch is deliberately **untouched**: it is noise-debiased and measured immune
(Δθ(σ_8) = +0.0009 ± 0.0028), so a map-only perturbation is the clean probe. This also makes the
injection *cleaner* than the real b_g variate, which perturbs both branches.
"""
from __future__ import annotations

import hashlib
import re
from typing import Dict, Optional, Union

import numpy as np

# Ensemble-mean radial power ratio <P(b_g=1.5)/P(b_g=1.0)> per tomographic bin, measured on the
# dual-norm gb stores by `--mode ebdiff` (job 1342435, committed 357c1b9) and reported in
# BG_MECHANISM_AND_DES.md §9. These ARE the calibration; do not replace them with theory.
MEASURED_POWER_RATIO_B1P5: Dict[str, np.ndarray] = {
    "north": np.array([1.063, 1.054, 1.039, 1.028, 1.017, 1.010]),
    "south": np.array([1.061, 1.057, 1.039, 1.029, 1.015, 1.008]),
}

# ---------------------------------------------------------------------------------------------
# The DES-Y3 amplitude profile, for the "could a DES-sized channel pass at <0.3 sigma?" test.
#
# Derivation, stated so it can be checked. Gatti et al. 2024 (arXiv:2307.13860), on the noise
# double-counting the F(phi) factor corrects: "The variance of the noise is increased at most by 5
# (resp. 1) per cent at small scales for the first (resp. fourth) DES tomographic bin." That
# enhancement is b^2 sigma_delta^2 at b = 1, so sigma_delta^2 = 0.05 (bin 1) ... 0.01 (bin 4) at
# their NSIDE-512 pixel scale. The b_g: 1 -> 1.5 power ratio is then 1 + (b^2-1) sigma_delta^2 =
# 1 + 1.25 sigma_delta^2. Their 4 bins are mapped onto our 6 by geometric interpolation in
# sigma_delta^2 across the same fractional bin position (low-z first in both surveys).
#
# Unlike ours these are PREDICTED rather than realised, but the prediction should be much closer to
# truth for DES: their modulation is a smooth analytic factor with no Poisson lambda-clipping, which
# is what suppresses our own realised ratios to ~0.4-0.6 of their prediction in the low bins.
#
# NOTE the headline: 1.0625 (their worst bin) against our measured 1.063. The channel amplitudes
# are MATCHED, not orders apart -- so this arm is expected to land close to the `measured` arm, and
# that is the point of running it.
_DES_SIGMA_D2 = np.array([0.05, 0.0362, 0.0262, 0.0190, 0.0138, 0.01])
DES_POWER_RATIO_B1P5: Dict[str, np.ndarray] = {
    "north": 1.0 + 1.25 * _DES_SIGMA_D2,
    "south": 1.0 + 1.25 * _DES_SIGMA_D2,
}

_PROFILES = {"measured": MEASURED_POWER_RATIO_B1P5, "des": DES_POWER_RATIO_B1P5}

_REF_B = 1.5                      # the b_g the ratios above were measured at
_SIDE_RE = re.compile(r"_(north|south)$")
_EB_RE = re.compile(r"^[EB]_(north|south)$")


_Z = np.linspace(-8.0, 8.0, 20001)
_ZW = np.exp(-0.5 * _Z ** 2)
_ZW = _ZW / _ZW.sum()


def _mean_f2(a: float, floor: float) -> float:
    """E[1 / max(1 + a z, floor)] for z ~ N(0,1) — the power ratio a modulation of size `a` gives."""
    return float((_ZW / np.maximum(1.0 + a * _Z, floor)).sum())


def _solve_amplitude(target_ratio: float, floor: float) -> float:
    """Invert `_mean_f2` for `a`.

    The leading term is `1 + a^2`, but the `3a^4` term is a ~30 % overshoot on the *excess* at the
    bin-0 amplitude (verified against a direct simulation), so solve numerically rather than using
    `a = sqrt(R-1)`. Monotone in `a`, so plain bisection is enough.
    """
    if target_ratio <= 1.0:
        return 0.0
    lo, hi = 0.0, 1.0
    while _mean_f2(hi, floor) < target_ratio and hi < 8.0:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _mean_f2(mid, floor) < target_ratio:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _amplitude(side: str, nbins: int, target_b: float, floor: float = 0.05,
               profile: str = "measured") -> np.ndarray:
    """Per-bin modulation amplitude `a` such that <f^2> reproduces the measured power ratio.

    With `f = (1 + a g)^{-1/2}` and `g` zero-mean unit-variance, `<f^2> = <1/(1+ag)>`; `a` is
    solved numerically (`_solve_amplitude`) so that equals the measured ratio.
    Extrapolating in b_g uses the `(b^2 - 1)` scaling the count sector must have (and which the
    measured 0.7/1.5 asymmetry confirms): the *excess* `R - 1` scales as `(b^2-1)/(b_ref^2-1)`.
    """
    table = _PROFILES.get(profile)
    if table is None:
        raise ValueError(f"profile must be one of {sorted(_PROFILES)}, got {profile!r}")
    ratio = table.get(side)
    if ratio is None:
        raise KeyError(f"no power ratio for side {side!r} in profile {profile!r}")
    if nbins > ratio.size:
        raise ValueError(f"{side}: {nbins} bins but only {ratio.size} calibrated ratios")
    scale = (float(target_b) ** 2 - 1.0) / (_REF_B ** 2 - 1.0)
    if scale < 0:
        raise ValueError(
            f"target_b={target_b} < 1: a multiplicative injection can only ADD noise-variance "
            "modulation. The b_g<1 direction needs the true delta and is not synthesisable here."
        )
    targets = 1.0 + np.maximum(ratio[:nbins] - 1.0, 0.0) * scale
    return np.array([_solve_amplitude(float(t), floor) for t in targets])


def _seed_from(arr: np.ndarray) -> int:
    """Deterministic per-(event, bin, side) seed, content-addressed.

    The loader transform never sees the file path, and `num_workers > 1` rules out a shared
    counter — so derive the seed from the array bytes. Same event, same modulation, every run.
    """
    h = hashlib.blake2b(np.ascontiguousarray(arr).tobytes()[:8192], digest_size=8).digest()
    return int(np.frombuffer(h, dtype="<u8")[0] & 0x7FFFFFFF)


def _red_grf(shape, slope: float, rng: np.random.Generator,
             kband: Optional[tuple] = None) -> np.ndarray:
    """Zero-mean Gaussian field on an (H, W) grid, unit-ish variance.

    Two spectra, so that "how much bias per unit modulation variance" can be measured as a
    function of the SCALE of the modulation rather than only its total power:

    * ``kband=None``  -> a power law `P(k) ∝ k^-slope`. `slope=0` is white. Note that in 2-D the
      variance per log-k is `k^2 P(k)`, so slope < 2 still puts most of the variance at small
      scales; only slope > 2 is large-scale dominated. This is why a slope sweep is informative
      and not just a relabelling.
    * ``kband=(lo, hi)`` -> a top-hat: power ONLY for `lo <= k < hi`, k in cycles/pixel. With the
      field renormalised to unit variance downstream, this puts the ENTIRE modulation variance in
      one octave, so the resulting posterior shift is the network's response to that scale alone.
      Patch pixels are 6.87' (NSIDE 512), so k = 0.5 is 13.7' and k = 0.0125 is ~9 degrees.
    """
    h, w = shape
    white = rng.standard_normal((h, w))
    ky = np.fft.fftfreq(h)[:, None]
    kx = np.fft.fftfreq(w)[None, :]
    k = np.sqrt(ky ** 2 + kx ** 2)
    k[0, 0] = np.inf                      # kill the mean; no DC power
    if kband is None:
        amp = k ** (-0.5 * slope)
    else:
        lo, hi = float(kband[0]), float(kband[1])
        amp = ((k >= lo) & (k < hi)).astype(float)
        if not amp.any():
            raise ValueError(f"kband {kband} selects no modes on a {h}x{w} grid "
                             f"(k resolution is 1/{max(h, w)})")
    g = np.fft.ifft2(np.fft.fft2(white) * amp).real
    return g


def _lowpass(img: np.ndarray, sigma_pix: float) -> np.ndarray:
    """Gaussian low-pass of an (H, W) image, implemented in Fourier space (no scipy dep)."""
    h, w = img.shape
    ky = np.fft.fftfreq(h)[:, None]
    kx = np.fft.fftfreq(w)[None, :]
    k2 = ky ** 2 + kx ** 2
    kern = np.exp(-2.0 * (np.pi ** 2) * (sigma_pix ** 2) * k2)
    return np.fft.ifft2(np.fft.fft2(img) * kern).real


def _unit_variance(g: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance over the footprint; identically 0 if the footprint is degenerate."""
    if mask.sum() < 2:
        return np.zeros_like(g)
    m = g[mask].mean()
    s = g[mask].std()
    if not np.isfinite(s) or s <= 0:
        return np.zeros_like(g)
    return (g - m) / s


class NoiseVarianceInjectTransform:
    """Multiply E/B maps by `(1 + a·g)^{-1/2}`, synthesising a b_g shift in the noise sector.

    Applied to the RAW data dict, i.e. **before** any `EBNoiseNormTransform` / scaler, so the
    downstream pipeline is byte-identical to a real variate read. Zero (off-footprint) pixels stay
    zero because the operation is multiplicative.

    Parameters
    ----------
    source : {'grf', 'kappa', 'null'}
    target_b : float
        The b_g being synthesised (>= 1). Amplitude is calibrated from `MEASURED_POWER_RATIO_B1P5`.
    slope : float
        `P(k) ∝ k^-slope` for the ``grf`` source. Default 1.0 — the projected galaxy overdensity
        is red at these scales. `slope=0` (white) is the useful bracket: a white modulation is one
        the CNN cannot localise, so comparing the two isolates *where* the channel lives.
    kappa_smooth_pix : float
        Gaussian sigma (in patch pixels, 6.87' each at NSIDE 512) for the ``kappa`` low-pass.
    floor : float
        `1 + a·g` is clipped below at this value so the square root stays real for the rare deep
        void. At the calibrated amplitudes (a <= 0.25) clipping is a <1e-4 event.
    """

    def __init__(self, source: str = "grf", target_b: float = 1.5, slope: float = 1.0,
                 kappa_smooth_pix: float = 6.0, floor: float = 0.05,
                 profile: str = "measured", kband: Optional[tuple] = None):
        if source not in ("grf", "kappa", "null"):
            raise ValueError(f"source must be grf|kappa|null, got {source!r}")
        if profile not in _PROFILES:
            raise ValueError(f"profile must be one of {sorted(_PROFILES)}, got {profile!r}")
        self.source = source
        self.profile = profile
        self.kband = None if kband is None else (float(kband[0]), float(kband[1]))
        self.target_b = float(target_b)
        self.slope = float(slope)
        self.kappa_smooth_pix = float(kappa_smooth_pix)
        self.floor = float(floor)
        # Diagnostics accumulated across the epoch and printed by `summary()`: the achieved
        # per-bin power ratio must match MEASURED_POWER_RATIO_B1P5, else the calibration is wrong.
        self._pow_in: Dict[str, np.ndarray] = {}
        self._pow_out: Dict[str, np.ndarray] = {}
        self._n: Dict[str, int] = {}

    # ---------------------------------------------------------------- internals

    def _modulator(self, plane: np.ndarray, mask: np.ndarray, rng) -> np.ndarray:
        if self.source == "grf":
            return _unit_variance(_red_grf(plane.shape, self.slope, rng, self.kband), mask)
        # 'kappa': the event's own low-passed E map as a structure proxy (overdense -> larger g
        # -> smaller f -> LOWER noise variance, the correct sign for source clustering).
        return _unit_variance(_lowpass(plane, self.kappa_smooth_pix), mask)

    def _apply_side(self, arr: np.ndarray, side: str) -> np.ndarray:
        nbins = arr.shape[0]
        amp = _amplitude(side, nbins, self.target_b, self.floor, self.profile)
        out = np.array(arr, dtype=np.float64, copy=True)
        pin = np.zeros(nbins)
        pout = np.zeros(nbins)
        for b in range(nbins):
            plane = out[b]
            mask = plane != 0
            pin[b] = float((plane[mask] ** 2).mean()) if mask.any() else 0.0
            if self.source != "null" and amp[b] > 0:
                rng = np.random.default_rng(_seed_from(arr[b]) + b)
                g = self._modulator(plane, mask, rng)
                s = np.clip(1.0 + amp[b] * g, self.floor, None)
                out[b] = plane / np.sqrt(s)
            pout[b] = float((out[b][mask] ** 2).mean()) if mask.any() else 0.0
        self._pow_in[side] = self._pow_in.get(side, np.zeros(nbins)) + pin
        self._pow_out[side] = self._pow_out.get(side, np.zeros(nbins)) + pout
        self._n[side] = self._n.get(side, 0) + 1
        return out

    # ---------------------------------------------------------------- API

    def __call__(self, data: Dict[str, Union[np.ndarray, "object"]]):
        out = dict(data)
        for key, val in data.items():
            m = _EB_RE.match(key)
            if not m:
                continue
            side = m.group(1)
            is_torch = hasattr(val, "detach")
            arr = val.detach().cpu().numpy() if is_torch else np.asarray(val)
            if arr.ndim != 3:
                raise ValueError(f"{key}: expected (nbins, H, W), got {arr.shape}")
            mod = self._apply_side(arr, side).astype(arr.dtype, copy=False)
            if is_torch:
                import torch
                out[key] = torch.as_tensor(mod)
            else:
                out[key] = mod
        return out

    def summary(self) -> Dict[str, Dict[str, list]]:
        """Achieved vs target per-bin power ratio — the calibration self-check."""
        rep: Dict[str, Dict[str, list]] = {}
        for side, n in self._n.items():
            if not n:
                continue
            pin = self._pow_in[side] / n
            pout = self._pow_out[side] / n
            with np.errstate(invalid="ignore", divide="ignore"):
                achieved = np.where(pin > 0, pout / pin, np.nan)
            tgt = _PROFILES[self.profile][side][: achieved.size]
            if self.source == "null":
                tgt = np.ones_like(tgt)
            rep[side] = {
                "n_events": int(n),
                "achieved_power_ratio": [round(float(x), 5) for x in achieved],
                "target_power_ratio": [round(float(x), 5) for x in tgt],
            }
        return rep

    def __repr__(self):
        return (f"NoiseVarianceInjectTransform(source={self.source!r}, "
                f"target_b={self.target_b}, slope={self.slope}, profile={self.profile!r}, "
                f"kband={self.kband})")


def build_inject_transform(spec: Optional[Dict]) -> Optional[NoiseVarianceInjectTransform]:
    """Build from a variate dict's ``inject`` entry (None -> no injection)."""
    if not spec:
        return None
    return NoiseVarianceInjectTransform(**dict(spec))
