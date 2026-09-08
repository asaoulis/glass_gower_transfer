"""The SOLE reader of raw observation posteriors. Returns standardised samples only.

Frames (all per parameter, independent rescaling; S_8 derived per sample before anything else):

  self      z = (x - mean_self)/std_self             every posterior to N(0,1): shapes/degeneracies
                                                    only (Plot A, maximally blind)
  real      z = (x - mean_REAL)/std_REAL             the frame of ONE reference posterior (the real
                                                    data's, per arm), applied to that posterior
                                                    (-> N(0,1)) and to OTHER posteriors:
                                                      - a matched MOCK: its OWN mean subtracted,
                                                        divided by the REAL std (Plot B: widths and
                                                        degeneracies comparable, location hidden)
                                                      - another ARM/label of the real data: mean of
                                                        the flagship subtracted (Plot B-prime: inter-arm
                                                        offsets visible in sigma units, absolute
                                                        location hidden -- user-approved 2026-09-08)

No function here prints, logs, returns or persists a frame (mean/std). The standardised outputs are
written as npz with ``z``, ``names``, ``frame``, ``source`` only.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import BLIND_ROOT, STANDARDISED_ROOT


@dataclass(frozen=True)
class Standardised:
    z: np.ndarray            # [S, D] standardised samples
    names: Tuple[str, ...]   # parameter names (S8 appended)
    frame: str               # 'self' | 'real:<label>/<exp>' | 'real:...+ownmean'
    source: str              # basename of the raw file (never its location-revealing content)
    n_samples: int


# --------------------------------------------------------------------------------------------
# raw access (private)
# --------------------------------------------------------------------------------------------
def prior_of_dump(path: str) -> str:
    """The prior tag encoded in a sample-dump FILENAME, '' when it carries none.

    Both dump families put the prior in the name because an NLE posterior is only defined together
    with its prior — and, more sharply, because the prior decides the dump's COLUMN SET (see
    `_scaler_box`). The two shapes are
    ``external_posterior_samples_<prior>_ncosmo<...>_<repeat>.npz`` (per repeat, also the matched
    mock dumps) and ``pooled_posterior_samples_<prior>.npz`` (pooled).
    """
    base = os.path.basename(path)
    if not base.endswith(".npz"):
        return ""
    for stem, has_match in (("external_" + "posterior_samples_", True),
                            ("pooled_" + "posterior_samples_", False)):
        if base.startswith(stem):
            body = base[len(stem):-4]
            return body.split("_ncosmo", 1)[0] if has_match else body
    return ""


def _scaler_box(experiment: str, prior_mode: Optional[str] = None):
    """-> (names, min, max) of the columns a dump of `experiment` under `prior_mode` actually has.

    ⚠️ `prior_mode` is load-bearing, not decoration. A mode that PINS a parameter (`LCDM_fixed_w0`
    pins w0) is sampled through sbi's `conditional_potential`, which returns only the free
    dimensions — so the dump has one column FEWER than `cfg.cosmo_param_names` and unscaling it
    with the full box is a broadcast error (or, for an unlucky arm, a silent column shift). The
    pinned name is DROPPED rather than re-inserted as a constant: a zero-variance column makes
    every standardised frame NaN. Everything downstream aligns by name, so this composes.
    ``theta0s`` keeps the full vector — use ``prior_mode=None`` for truth.
    """
    import eval as _eval
    cfg, _ = _eval.load_config(experiment)
    from src.ml.utils import _build_cosmo_preset_scaler
    from src.ml.data.constants import COSMO_PARAM_PRESET_MINMAX
    names = list(cfg.cosmo_param_names)
    if prior_mode:
        from src.ml.eval.nle_external import free_param_names
        names = free_param_names(prior_mode, names)
    preset = dict(getattr(cfg, "scaler_options", {}).get("cosmo", {}).get("preset_overrides", {}) or {})
    sc = _build_cosmo_preset_scaler({**COSMO_PARAM_PRESET_MINMAX, **preset}, names)
    return names, np.asarray(sc.min, dtype=np.float64), np.asarray(sc.max, dtype=np.float64)


def _load_physical(path: str, experiment: str, event: int = 0) -> Tuple[np.ndarray, List[str]]:
    """[S, D+1] physical samples (S8 appended) of ONE event of a sample dump. PRIVATE."""
    names, lo, hi = _scaler_box(experiment, prior_of_dump(path))
    with np.load(path, allow_pickle=False) as d:
        S = np.asarray(d["samples"])            # [S, N, D] scaled
        if S.ndim == 2:
            S = S[:, None, :]
        if S.shape[-1] != len(names):
            raise ValueError(
                "%s has %d sampled columns but the '%s' prior box for %s has %d (%s); the dump's "
                "prior tag and the experiment's parameter vector disagree"
                % (os.path.basename(path), S.shape[-1], prior_of_dump(path) or "gower",
                   experiment, len(names), ", ".join(names)))
        x = S[:, event, :].astype(np.float64) * (hi - lo) + lo
    io_, is_ = names.index("omega_m"), names.index("sigma_8")
    s8 = x[:, is_] * np.sqrt(x[:, io_] / 0.3)
    return np.concatenate([x, s8[:, None]], axis=1), names + ["S8"]


def _frame(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return x.mean(axis=0), x.std(axis=0, ddof=1)


# --------------------------------------------------------------------------------------------
# public API: standardised outputs only
# --------------------------------------------------------------------------------------------
def standardise_self(path: str, experiment: str, event: int = 0) -> Standardised:
    x, names = _load_physical(path, experiment, event)
    mu, sd = _frame(x)
    z = (x - mu) / sd
    return Standardised(z=z, names=tuple(names), frame="self", source=os.path.basename(path), n_samples=len(z))


def standardise_in_real_frame(target_path: str, target_experiment: str, real_path: str, real_experiment: str,
                              *, subtract: str = "own", event: int = 0, real_event: int = 0) -> Standardised:
    """Standardise ``target`` with the REAL posterior's per-parameter std.
    subtract='own'  : target's own mean subtracted (Plot B; matched mocks) -> location hidden
    subtract='real' : the real posterior's mean subtracted (Plot B-prime; other arms/labels of the
                      real data) -> inter-arm offsets in sigma units, absolute location hidden"""
    x, names = _load_physical(target_path, target_experiment, event)
    xr, names_r = _load_physical(real_path, real_experiment, real_event)
    # align by NAME on the shared parameter set (arms differ in their IA nuisance, e.g. b_ia vs b_z)
    shared = [n for n in names_r if n in names]
    it = [names.index(n) for n in shared]
    ir = [names_r.index(n) for n in shared]
    x, xr = x[:, it], xr[:, ir]
    mu_r, sd_r = _frame(xr)
    mu_t, _ = _frame(x)
    centre = mu_t if subtract == "own" else mu_r
    z = (x - centre) / sd_r
    tag = "ownmean" if subtract == "own" else "realmean"
    return Standardised(z=z, names=tuple(shared), frame=f"real:{os.path.basename(real_path)}+{tag}",
                        source=os.path.basename(target_path), n_samples=len(z))


def write_standardised(st: Standardised, out_dir: str, stem: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"{stem}.npz")
    np.savez(p, z=st.z.astype(np.float32), names=np.array(st.names), frame=np.array(st.frame),
             source=np.array(st.source), n_samples=np.array(st.n_samples))
    return p


def load_standardised(path: str) -> Standardised:
    with np.load(path, allow_pickle=False) as d:
        return Standardised(z=d["z"], names=tuple(str(n) for n in d["names"]), frame=str(d["frame"]),
                            source=str(d["source"]), n_samples=int(d["n_samples"]))


# --------------------------------------------------------------------------------------------
# registry of raw runs (FILENAMES only)
# --------------------------------------------------------------------------------------------
def _arm_of(experiment: str) -> str:
    """Arm name of a per-repeat experiment (from the production arm table; '' if unknown)."""
    try:
        from src.ml.eval.arms import ARMS
    except Exception:  # pragma: no cover
        return ""
    for arm, (tmpl, _bake, reps) in ARMS.items():
        if any(tmpl.format(r=r) == experiment for r in reps):
            return arm
    return ""


def list_raw_runs(root: str = BLIND_ROOT) -> List[Dict[str, str]]:
    """Walk the blind store and return {label, arm, experiment, prior, match, pooled, path} from
    FILENAMES only.

    Two families live under the store (`scripts/sample_observation.py fetch`):
      <root>/<label>/<experiment>/.../external_posterior_samples_<prior>_<match>.npz   per-repeat
      <root>/<label>/pooled_<arm>/.../pooled_posterior_samples_<prior>.npz            POOLED (final)
    A pooled entry carries ``match='pooled'`` and ``experiment`` = the arm's repeat-0 member, whose
    scaler box is the shared frame of every member (`eval.py --mode pool` refuses mixed frames).
    """
    out = []
    for dirpath, _, files in os.walk(root):
        rel = os.path.relpath(dirpath, root).split(os.sep)
        for f in files:
            if f.startswith("external_posterior_samples_") and f.endswith(".npz"):
                label, exp = (rel[0], rel[1]) if len(rel) >= 2 else (rel[0], "")
                body = f[len("external_posterior_samples_"):-4]
                prior, match = prior_of_dump(f), body.split("_ncosmo", 1)[1]
                out.append({"label": label, "arm": _arm_of(exp), "experiment": exp, "prior": prior,
                            "match": "ncosmo" + match, "pooled": False, "path": os.path.join(dirpath, f)})
            elif f.startswith("pooled_posterior_samples_") and f.endswith(".npz"):
                label = rel[0]
                arm_dir = next((d for d in rel[1:] if d.startswith("pooled_")), "")
                arm = arm_dir[len("pooled_"):]
                prior = prior_of_dump(f)
                try:
                    from src.ml.eval.arms import ARMS
                    exp = ARMS[arm][0].format(r=ARMS[arm][2][0])
                except Exception:
                    exp = ""
                out.append({"label": label, "arm": arm, "experiment": exp, "prior": prior,
                            "match": "pooled", "pooled": True, "path": os.path.join(dirpath, f)})
    return sorted(out, key=lambda r: (r["label"], r["arm"], r["experiment"], r["prior"], r["match"]))


def _selftest():
    """The returned object must not allow location recovery: mean(z) == 0 and std(z) == 1 in the
    self frame, and no attribute of Standardised carries the frame."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5000, 3)) * [0.1, 2.0, 5.0] + [0.3, 0.8, -1.0]
    mu, sd = _frame(x)
    z = (x - mu) / sd
    assert np.allclose(z.mean(0), 0, atol=1e-12) and np.allclose(z.std(0, ddof=1), 1, atol=1e-12)
    assert set(Standardised.__dataclass_fields__) == {"z", "names", "frame", "source", "n_samples"}

    # --- the fixed-parameter contract (regression: LCDM_fixed_w0 dumps are D-1 wide) ------------
    # sbi samples only the free dimensions under a pinning prior, so a reader that unscales with
    # the full parameter box either crashes or shifts every column. These two asserts are the only
    # gate on that; `_load_physical` is exercised end-to-end by the plot battery, which until
    # 2026-09-08 had only ever been run on the `gower` prior, where the bug is invisible.
    assert prior_of_dump("external_" + "posterior_samples_LCDM_fixed_w0_ncosmoNone_0.npz") == "LCDM_fixed_w0"
    assert prior_of_dump("external_" + "posterior_samples_kids_s8_analytic_ncosmo300_2.npz") == "kids_s8_analytic"
    assert prior_of_dump("pooled_" + "posterior_samples_gower.npz") == "gower"
    assert prior_of_dump("notadump.txt") == ""
    from src.ml.eval.nle_external import free_param_names
    p9 = ["omega_m", "sigma_8", "w0", "mnu", "h", "ns", "ombh2", "a_ia", "b_ia"]
    assert free_param_names("gower", p9) == p9
    assert free_param_names("kids_s8_analytic", p9) == p9
    assert free_param_names("LCDM_fixed_w0", p9) == [n for n in p9 if n != "w0"]
    print("blind.standardise selftest OK")


if __name__ == "__main__":
    _selftest()
