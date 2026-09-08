"""Reference-cloud extraction: the per-mock Tier-1 vectors over a whole store, and the same
vectors for an observation file.

Per file we extract, into one npz:
  files, sim_ids, aug_ids                      identity (cosmology-aware resampling needs sim_ids)
  bandpowers   [N, 21, nbands]                 EE ``cls_results/full/mixed_bandpowers``
  bb           [N, 21, nbands]                 BB bandpowers: ``cls_results/full/bb_bandpowers`` if
                                               present, else re-binned from ``cls[:, :, 1, :]``
                                               with the IDENTICAL Brown binning (raw stores only)
  emap         [N, D]  + emap_names            hand-crafted E-map summaries (checks/summaries.py)
                                               of the noise-normed E patches. For a BAKED store the
                                               bare ``E`` group is read as is; for a RAW store the
                                               bake is replicated on the fly (E_<variant> divided by
                                               noise_std_<variant>/all) so both agree.

Nothing here reads ``cosmo_dict``. The mapping sim_id -> cosmology is never joined in this package.
"""
from __future__ import annotations

import glob
import os
import re
from multiprocessing import get_context
from typing import Dict, List, Optional, Sequence

import numpy as np

from .checks.summaries import emap_summary_vector, summary_names

_AUG_RE = re.compile(r"output_(\d+)_out(\d+)_rot(\d+)_(\d+)\.h5$")


def parse_ids(path: str):
    m = _AUG_RE.search(os.path.basename(path))
    if not m:
        m2 = re.search(r"output_(\d+)_", os.path.basename(path))
        return (int(m2.group(1)) if m2 else -1), -1
    sim, outer, rot, cat = (int(g) for g in m.groups())
    return sim, outer * 10000 + rot * 100 + cat


def _bands_for(f, geometry_name: Optional[str]):
    from .geometry import Geometry
    if geometry_name:
        g = Geometry.named(geometry_name)
    else:
        n_ell = f["cls_results"]["full"]["cls"].shape[-1] if "cls" in f["cls_results"]["full"] else 2049
        g = Geometry.production() if n_ell >= 2049 else Geometry.smoke()
    return g


def extract_one(path: str, *, eb_variant: Optional[str] = None, noise_norm: Optional[str] = "rand",
                want: Sequence[str] = ("bandpowers", "bb", "emap"), geometry_name: Optional[str] = None,
                patch_names: Sequence[str] = ("north", "south")) -> Optional[Dict]:
    """One file -> dict of vectors, or None if unreadable. ``eb_variant`` None = baked bare ``E``."""
    import h5py
    try:
        out: Dict = {"file": os.path.basename(path)}
        out["sim_id"], out["aug_id"] = parse_ids(path)
        with h5py.File(path, "r") as f:
            cf = f["cls_results"]["full"]
            if "bandpowers" in want:
                out["bandpowers"] = cf["mixed_bandpowers"][()].astype(np.float64)
            if "bb" in want:
                if "bb_bandpowers" in cf:
                    out["bb"] = cf["bb_bandpowers"][()].astype(np.float64)
                elif "cls" in cf:
                    from .build import bb_bandpowers_from_cls
                    g = _bands_for(f, geometry_name)
                    cls = cf["cls"][()]
                    cut = cls[:, :, :, g.lower_lscale:g.upper_lscale + 1]
                    out["bb"] = bb_bandpowers_from_cls(cut, g.nbins, g.lower_lscale, g.upper_lscale, g.nbands)
            if "emap" in want:
                pix = f["pixelised_results"]
                e_group = f"E_{eb_variant}" if eb_variant else "E"
                if e_group not in pix:
                    return None
                patches = {}
                sd = None
                if eb_variant and noise_norm and noise_norm != "none":
                    ns = f"noise_std_{eb_variant}"
                    if ns not in pix:
                        return None
                    sd = np.asarray(pix[ns]["all"][()], dtype=np.float64)
                for p in patch_names:
                    m = pix[e_group][p][()].astype(np.float64)
                    if sd is not None:
                        m = m / sd[:, None, None]
                    patches[p] = m
                out["emap"] = emap_summary_vector(patches, patch_names)
        return out
    except Exception as ex:  # corrupt / truncated / missing group
        return {"file": os.path.basename(path), "error": f"{type(ex).__name__}: {ex}"}


def _worker(args):
    path, kw = args
    return extract_one(path, **kw)


def extract_reference(paths: Sequence[str], out_npz: str, *, workers: int = 8, max_files: Optional[int] = None,
                      seed: int = 0, **kw) -> Dict:
    """Run ``extract_one`` over ``paths`` (a random subset of ``max_files`` if given, chosen by
    WHOLE cosmologies so the sim_id structure survives) and write one npz. Returns a summary."""
    paths = sorted(paths)
    if max_files is not None and len(paths) > max_files:
        rng = np.random.default_rng(seed)
        by_sim: Dict[int, List[str]] = {}
        for p in paths:
            by_sim.setdefault(parse_ids(p)[0], []).append(p)
        sims = list(by_sim)
        rng.shuffle(sims)
        chosen: List[str] = []
        for s in sims:
            if len(chosen) >= max_files:
                break
            chosen += by_sim[s]
        paths = sorted(chosen)
    jobs = [(p, kw) for p in paths]
    if workers > 1:
        with get_context("fork").Pool(workers) as pool:
            results = pool.map(_worker, jobs, chunksize=8)
    else:
        results = [_worker(j) for j in jobs]
    good = [r for r in results if r is not None and "error" not in r]
    bad = [r for r in results if r is None or "error" in r]
    if not good:
        raise RuntimeError(f"no readable files among {len(paths)} (first errors: {bad[:3]})")
    npz = {
        "files": np.array([r["file"] for r in good]),
        "sim_ids": np.array([r["sim_id"] for r in good], dtype=np.int64),
        "aug_ids": np.array([r["aug_id"] for r in good], dtype=np.int64),
    }
    for key in ("bandpowers", "bb", "emap"):
        if all(key in r for r in good):
            npz[key] = np.stack([r[key] for r in good]).astype(np.float32 if key == "emap" else np.float64)
    if "emap" in npz:
        nb = npz["emap"].shape[1]
        npz["emap_names"] = np.array(summary_names(nbins=6, patch_names=kw.get("patch_names", ("north", "south"))))
        assert len(npz["emap_names"]) == nb, (len(npz["emap_names"]), nb)
    npz["extract_kwargs"] = np.array(str({k: v for k, v in kw.items()}))
    os.makedirs(os.path.dirname(os.path.abspath(out_npz)), exist_ok=True)
    np.savez(out_npz, **npz)
    return {"n_files": len(paths), "n_good": len(good), "n_bad": len(bad), "keys": sorted(npz),
            "n_cosmologies": int(len(set(npz["sim_ids"].tolist()))), "out": out_npz,
            "bad_examples": [str(b)[:200] for b in bad[:5]]}


def extract_observation(paths: Sequence[str], out_npz: str, **kw) -> Dict:
    """The observation counterpart (N labels, one file each; no subsetting, no workers)."""
    return extract_reference(list(paths), out_npz, workers=1, max_files=None, **kw)


def load_reference(npz_path: str) -> Dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
