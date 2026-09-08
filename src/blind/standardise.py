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
def _scaler_box(experiment: str):
    import eval as _eval
    cfg, _ = _eval.load_config(experiment)
    from src.ml.utils import _build_cosmo_preset_scaler
    from src.ml.data.constants import COSMO_PARAM_PRESET_MINMAX
    names = list(cfg.cosmo_param_names)
    preset = dict(getattr(cfg, "scaler_options", {}).get("cosmo", {}).get("preset_overrides", {}) or {})
    sc = _build_cosmo_preset_scaler({**COSMO_PARAM_PRESET_MINMAX, **preset}, names)
    return names, np.asarray(sc.min, dtype=np.float64), np.asarray(sc.max, dtype=np.float64)


def _load_physical(path: str, experiment: str, event: int = 0) -> Tuple[np.ndarray, List[str]]:
    """[S, D+1] physical samples (S8 appended) of ONE event of a sample dump. PRIVATE."""
    names, lo, hi = _scaler_box(experiment)
    with np.load(path, allow_pickle=False) as d:
        S = np.asarray(d["samples"])            # [S, N, D] scaled
        if S.ndim == 2:
            S = S[:, None, :]
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
def list_raw_runs(root: str = BLIND_ROOT) -> List[Dict[str, str]]:
    """Walk the blind store and return {label, experiment, prior, match, path} from FILENAMES."""
    out = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.startswith("external_posterior_samples_") and f.endswith(".npz"):
                rel = os.path.relpath(dirpath, root).split(os.sep)
                label, exp = (rel[0], rel[1]) if len(rel) >= 2 else (rel[0], "")
                body = f[len("external_posterior_samples_"):-4]
                prior, match = body.split("_ncosmo", 1)
                out.append({"label": label, "experiment": exp, "prior": prior, "match": "ncosmo" + match,
                            "path": os.path.join(dirpath, f)})
    return sorted(out, key=lambda r: (r["label"], r["experiment"], r["prior"], r["match"]))


def _selftest():
    """The returned object must not allow location recovery: mean(z) == 0 and std(z) == 1 in the
    self frame, and no attribute of Standardised carries the frame."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5000, 3)) * [0.1, 2.0, 5.0] + [0.3, 0.8, -1.0]
    mu, sd = _frame(x)
    z = (x - mu) / sd
    assert np.allclose(z.mean(0), 0, atol=1e-12) and np.allclose(z.std(0, ddof=1), 1, atol=1e-12)
    assert set(Standardised.__dataclass_fields__) == {"z", "names", "frame", "source", "n_samples"}
    print("blind.standardise selftest OK")


if __name__ == "__main__":
    _selftest()
