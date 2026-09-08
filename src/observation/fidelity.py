"""Rel-RMS comparison of a built observation against a stored mock ``output_*.h5``.

Modelled on ``scripts/shear_replay/replay.py:fidelity_check`` (rel_rms = ||a-b|| / ||b||), extended
to every shared dataset. Bit-parity is NOT expected everywhere:

  * ``mixed_bandpowers``, ``cls`` (EE/BB) and the E/B/E_sc8 patches derive from the SAME galaxies
    through the SAME code -> parity to float rounding, except through the shape-noise debias term
    (``denoise_shear_cls`` subtracts a high-ell average of the RANDOM-rotation spectrum, which is a
    different realisation here) -> a small relative difference in ``cls``/bandpowers at the noise
    level. The patches do not involve the random map -> parity (up to the float16 cast).
  * ``noise_std_*`` are stds of a random-rotation realisation -> agree to ~1/sqrt(N_pix) only.
"""
from __future__ import annotations

from typing import Dict, Optional

import h5py
import numpy as np

# Default tolerances (rel_rms) per dataset family; the gate reports every number, these only
# decide PASS/FAIL. MEASURED 2026-09-08 on the smoke fixture with the mock's exact block RNG:
# bandpowers 5.5e-3, cls 3.6e-3, E/B patches 2-4e-3 -- and a half-ulp float32 jitter of the
# catalogue reproduces exactly those numbers (6.6e-3 / 4.5e-3 / 3-6e-3), i.e. the residual is the
# FIXTURE's float32 storage, not the code. Prefer the self-calibrating ``jitter_floor`` gate
# (pass if rel_rms <= JITTER_FACTOR x the measured floor); these absolute numbers are the fallback.
# (A DENSE fixture -- 30M galaxies at nside 256, ~40/pixel -- has a larger floor: bandpowers 3.1e-2,
# cls 2.2e-2, counts-E/B patches 5-7e-2, E_sc8 5-7e-3, noise_std_sc8 1e-4. The absolute numbers
# below are therefore loose; the jitter-floor gate is the one that means something.)
DEFAULT_TOL = {
    "bandpower_ls": 1e-9,
    "mixed_bandpowers": 5e-2, "bb_bandpowers": 1e-1, "cls": 5e-2,
    "E_": 1e-1, "B_": 1e-1, "E_sc8_": 2e-2, "noise_std_": 5e-2,
}
JITTER_FACTOR = 3.0


def rel_rms(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    den = np.sqrt(np.mean(b ** 2))
    return float(np.sqrt(np.mean((a - b) ** 2)) / den) if den > 0 else float("nan")


def _family(name: str) -> Optional[str]:
    for fam in ("E_sc8_", "noise_std_", "E_", "B_"):
        if name.startswith(fam):
            return fam
    return name if name in DEFAULT_TOL else None


def compare_observation_to_mock(obs_path: str, mock_path: str, tol: Optional[Dict] = None,
                                floor: Optional[Dict[str, float]] = None,
                                floor_factor: float = JITTER_FACTOR) -> Dict:
    """Return {'per_dataset': {name: rel_rms}, 'pass': bool, 'failures': [...], 'missing': [...]}.

    ``floor``: per-dataset rel_rms of a half-ulp-jittered rebuild vs the un-jittered one (see
    ``jitter_floor``). When given, a dataset passes if rel_rms <= max(floor_factor * floor, 1e-6)
    (noise_std_* keep the absolute tolerance: they are realisation-level quantities)."""
    tol = {**DEFAULT_TOL, **(tol or {})}
    res: Dict[str, float] = {}
    missing = []
    with h5py.File(obs_path, "r") as fo, h5py.File(mock_path, "r") as fm:
        cf_o, cf_m = fo["cls_results"]["full"], fm["cls_results"]["full"]
        # band edges first: a band-definition mismatch must show up as its own line, not as an
        # unexplained bandpower residual
        if "bandpower_ls" in cf_o and "bandpower_ls" in cf_m:
            res["bandpower_ls"] = rel_rms(cf_o["bandpower_ls"][()], cf_m["bandpower_ls"][()])
        for k in ("mixed_bandpowers", "cls", "bb_bandpowers"):
            if k in cf_o and k in cf_m:
                res[k] = rel_rms(cf_o[k][()], cf_m[k][()])
            elif k in cf_o:
                missing.append(f"mock lacks cls_results/full/{k}")
        po, pm = fo["pixelised_results"], fm["pixelised_results"]
        for name in po.keys():
            if name.startswith("_"):
                continue
            if name not in pm:
                missing.append(f"mock lacks pixelised_results/{name}")
                continue
            for sub in po[name].keys():
                if sub in pm[name]:
                    res[f"{name}/{sub}"] = rel_rms(po[name][sub][()], pm[name][sub][()])
    # exact (bit-level) equality per dataset, for the identity gate record
    exact: Dict[str, bool] = {}
    with h5py.File(obs_path, "r") as fo, h5py.File(mock_path, "r") as fm:
        cf_o, cf_m = fo["cls_results"]["full"], fm["cls_results"]["full"]
        for k in ("bandpower_ls", "mixed_bandpowers", "cls"):
            if k in cf_o and k in cf_m:
                exact[k] = bool(np.array_equal(cf_o[k][()], cf_m[k][()]))
        po, pm = fo["pixelised_results"], fm["pixelised_results"]
        for name in po.keys():
            if name.startswith("_") or name not in pm:
                continue
            for sub in po[name].keys():
                if sub in pm[name]:
                    exact[f"{name}/{sub}"] = bool(np.array_equal(po[name][sub][()], pm[name][sub][()]))
    failures = []
    for name, v in res.items():
        fam = _family(name.split("/")[0])
        t = tol.get(fam) if fam else None
        if floor is not None and name in floor and fam != "noise_std_":
            t = max(floor_factor * float(floor[name]), 1e-6)
        if t is not None and not (np.isfinite(v) and v <= t):
            failures.append((name, v, t))
    return {"per_dataset": res, "pass": not failures, "failures": failures, "missing": missing,
            "floor_used": floor is not None, "exact": exact, "all_exact": bool(exact) and all(exact.values())}


def jitter_floor(cat, m_bias, *, geometry, variants, rng_factory, workdir: str, seed: int = 3) -> Dict[str, float]:
    """Measure the float32-storage floor: rebuild the observation from the catalogue jittered by
    +-0.5 ulp(float32) in RA/DEC/E1/E2 and return rel_rms(jittered, unjittered) per dataset.
    ``rng_factory()`` must return a FRESH copy of the same random-rotation generator each call."""
    import copy
    import os
    from .build import build_observation
    os.makedirs(workdir, exist_ok=True)
    base = build_observation(cat, m_bias, out_path=os.path.join(workdir, "floor_base.h5"), label="floor_base",
                             geometry=geometry, variants=variants, rng=rng_factory(), verbose=False)
    jit = copy.deepcopy(cat)
    rng = np.random.default_rng(seed)
    for col in ("RA", "DEC", "E1", "E2"):
        v = jit.data[col]
        ulp = np.spacing(v.astype(np.float32)).astype(float)
        jit.data[col] = v + rng.uniform(-0.5, 0.5, size=v.shape) * ulp
    jp = build_observation(jit, m_bias, out_path=os.path.join(workdir, "floor_jit.h5"), label="floor_jit",
                           geometry=geometry, variants=variants, rng=rng_factory(), verbose=False)
    return compare_observation_to_mock(jp, base)["per_dataset"]


def format_report(rep: Dict) -> str:
    lines = ["dataset".ljust(42) + "rel_rms".rjust(10)]
    for k, v in rep["per_dataset"].items():
        lines.append(k.ljust(42) + f"{v:10.3e}")
    for m in rep["missing"]:
        lines.append(f"MISSING: {m}")
    for name, v, t in rep["failures"]:
        lines.append(f"FAIL: {name} rel_rms={v:.3e} > tol={t:.1e}")
    lines.append(("PASS" if rep["pass"] else "FAIL") + (" (jitter-floor gate)" if rep.get("floor_used") else " (absolute tolerances)")
                 + ("; BIT-IDENTICAL on every compared dataset" if rep.get("all_exact") else ""))
    return "\n".join(lines)
