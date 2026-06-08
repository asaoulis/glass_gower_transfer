"""Load saved mocks and compute empirical/theory bandpower ratios over an ensemble.

Each mock ``.h5`` written by the master simulator stores ``mixed_bandpowers`` (21, 8)
and ``bandpower_ls`` (8,) under ``cls_results/full``.  For every file we read those,
compute the matching theory bandpowers for that file's cosmology, and form the ratio
``empirical / theory`` per (spectrum, band).

Production files contain ONLY ``mixed_bandpowers`` (no raw ``bandpowers`` key), so the
default empirical source is ``mixed_bandpowers`` and the theory side is mixed with the
KiDS mixing matrix to match.  A legacy ``empirical_key='bandpowers'`` + ``mixing=None``
path is kept for old datasets that pre-date the mixed-only writer.
"""

from __future__ import annotations

import contextlib
import glob
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.ml.data.data_loading import unpack_data

from . import config as cfg
from .theory import compute_bandpower_theory_from_cosmo_vec

# HDF5 path map for unpack_data (see src/ml/data/data_augmentations.py contract).
DEFAULT_NESTED_KEYS = {
    "mixed_bandpowers": ("cls_results", "full", "mixed_bandpowers"),
    "bandpower_cls": ("cls_results", "full", "bandpower_ls"),
}


def glob_sim_files(
    folder: str,
    pattern: str = "*.h5",
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
) -> List[str]:
    """Glob ``folder`` for mock files, with optional substring include/exclude filters."""
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if include is not None:
        files = [f for f in files if any(s in os.path.basename(f) for s in include)]
    if exclude is not None:
        files = [f for f in files if not any(s in os.path.basename(f) for s in exclude)]
    return files


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Patch joblib so a Parallel run reports into the given tqdm progress bar."""
    import joblib

    def tqdm_print_progress(self):
        if self.n_completed_tasks > tqdm_object.n:
            tqdm_object.update(n=self.n_completed_tasks - tqdm_object.n)

    original = joblib.parallel.Parallel.print_progress
    joblib.parallel.Parallel.print_progress = tqdm_print_progress
    try:
        yield tqdm_object
    finally:
        joblib.parallel.Parallel.print_progress = original
        tqdm_object.close()


def load_and_compute_ratios(
    path: str,
    nested_keys: Dict[str, Tuple[str, ...]],
    cosmo_params,
    *,
    nside: int = cfg.NSIDE,
    nbins: int = cfg.NBINS,
    lmin_cut: int = cfg.LMIN_CUT,
    lmax_cut: int = cfg.LMAX_CUT,
    nbands: int = cfg.NBANDS,
    mixing_matrix: Optional[np.ndarray] = None,
    nonlinear: str = cfg.NONLINEAR,
    theory_mode: str = cfg.THEORY_MODE,
    data_dir: str = cfg.DATA_DIR,
    empirical_key: str = "mixed_bandpowers",
):
    """Load one mock, compute its theory bandpowers, return ratios (or None on failure).

    Returns ``{path, ratios (n_spectra, nbands), cls (nbands,), labels}``.  A bad/missing
    file logs a clear message and returns None so one failure does not kill a joblib
    batch; the caller counts the failures.
    """
    try:
        data, cosmo_vec = unpack_data(
            path, nested_keys, cosmo_params, as_torch=False, return_names=True
        )

        if empirical_key not in data:
            raise KeyError(
                f"empirical bandpower field '{empirical_key}' missing from {path}; "
                f"available: {sorted(data)}"
            )
        cl_bandpowers = np.asarray(data[empirical_key])      # (n_spectra, nbands)
        cls = np.asarray(data["bandpower_cls"])              # (nbands,)

        theory_bandpowers = compute_bandpower_theory_from_cosmo_vec(
            cosmo_vec,
            nside=nside, nbins=nbins, lmin_cut=lmin_cut, lmax_cut=lmax_cut,
            nbands=nbands, mixing_matrix=mixing_matrix, nonlinear=nonlinear,
            theory_mode=theory_mode, data_dir=data_dir,
        )

        ratios, labels = [], []
        for i in range(nbins):
            for j in range(nbins):
                if i < j:
                    continue
                lbl = f"S{i + 1}-S{j + 1}"
                idx = int(i * (i + 1) / 2 + j)
                ratios.append(cl_bandpowers[idx] / theory_bandpowers[lbl])
                labels.append(lbl)

        return {
            "path": path,
            "ratios": np.array(ratios),   # (n_spectra, nbands)
            "cls": cls,
            "labels": labels,
        }
    except Exception as e:  # noqa: BLE001 - keep the batch alive, report the failure
        print(f"[validation] ERROR processing {path}: {e}")
        return None


def compute_ensemble_ratios(
    folder: str,
    nested_keys: Optional[Dict[str, Tuple[str, ...]]] = None,
    cosmo_params=None,
    *,
    nside: int = cfg.NSIDE,
    nbins: int = cfg.NBINS,
    lmin_cut: int = cfg.LMIN_CUT,
    lmax_cut: int = cfg.LMAX_CUT,
    nbands: int = cfg.NBANDS,
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
    mixing_matrix: Optional[np.ndarray] = None,
    nonlinear: str = cfg.NONLINEAR,
    theory_mode: str = cfg.THEORY_MODE,
    data_dir: str = cfg.DATA_DIR,
    empirical_key: str = "mixed_bandpowers",
    max_sims: Optional[int] = None,
    n_jobs: int = 8,
    shuffle_seed: int = 0,
) -> dict:
    """Compute the (n_files, n_spectra, nbands) ratio cube over all matching mocks."""
    import joblib
    from tqdm import tqdm

    if nested_keys is None:
        nested_keys = DEFAULT_NESTED_KEYS

    files = glob_sim_files(folder, include=include, exclude=exclude)
    if max_sims is not None and len(files) > max_sims:
        rng = np.random.default_rng(shuffle_seed)
        files = list(np.array(files)[rng.permutation(len(files))[:max_sims]])
        files = sorted(files)

    if not files:
        raise FileNotFoundError(
            f"no mock files matched in {folder} (include={include}, exclude={exclude})"
        )
    print(f"[validation] found {len(files)} simulation files in {folder}")

    # Shell-projection theory uses a slow non-Limber CAMB matter_cls per cosmology; warm the
    # disk cache SERIALLY first so the parallel ratio loop only hits cache (avoids many heavy
    # concurrent CAMB jobs / OOM, and computes each unique cosmology exactly once).
    if theory_mode == "shell_projection":
        from .theory import warm_matter_cache
        cosmo_vecs = []
        for f in files:
            try:
                _, cv = unpack_data(f, nested_keys, cosmo_params, as_torch=False,
                                    return_names=True)
                cosmo_vecs.append(cv)
            except Exception as e:  # noqa: BLE001 - skip unreadable file; loop will report
                print(f"[validation] (cache warm: skipping {f}: {e})")
        # Run several CAMB calls concurrently, each pinned to ~8 threads (n_jobs*omp<=cores).
        omp = 8
        warm_jobs = max(1, n_jobs // omp)
        n_cos = warm_matter_cache(cosmo_vecs, nside=nside, n_jobs=warm_jobs, omp_threads=omp)
        print(f"[validation] matter_cls cache warm for {n_cos} unique cosmologies")

    with tqdm_joblib(tqdm(total=len(files), desc="ratios")):
        results = joblib.Parallel(n_jobs=n_jobs, backend="loky", verbose=1)(
            joblib.delayed(load_and_compute_ratios)(
                f, nested_keys, cosmo_params,
                nside=nside, nbins=nbins, lmin_cut=lmin_cut, lmax_cut=lmax_cut,
                nbands=nbands, mixing_matrix=mixing_matrix, nonlinear=nonlinear,
                theory_mode=theory_mode, data_dir=data_dir, empirical_key=empirical_key,
            )
            for f in files
        )

    n_failed = sum(1 for r in results if r is None)
    results = [r for r in results if r is not None]
    if not results:
        raise RuntimeError(
            f"all {len(files)} files failed to load/compute ratios (see errors above)"
        )

    ratios = np.stack([r["ratios"] for r in results], axis=0)  # (n_loaded, 21, nbands)
    return {
        "ratios": ratios,
        "cls": results[0]["cls"],
        "labels": results[0]["labels"],
        "files": [r["path"] for r in results],
        "n_loaded": len(results),
        "n_failed": n_failed,
    }
