"""Load saved mocks and compute empirical/theory bandpower ratios over an ensemble.

Each mock's cls_results/full/mixed_bandpowers (21, 8) is divided by its cosmology's theory
bandpowers, mixed with the KiDS mixing matrix, per (spectrum, band). empirical_key='bandpowers'
with mixing=None serves datasets older than the mixed-only writer.
"""

from __future__ import annotations

import contextlib
import glob
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
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


_COSMO_ID_RE = re.compile(r"output_(\d+)")


def cosmo_id_from_path(path: str) -> str:
    """Cosmology identifier for a mock = the first integer after ``output_`` in the basename.

    Mocks are named output_{sim}_out{outer}_rot{rot}_{inner}.h5, and all augmentations of one
    {sim} share its cosmology. Falls back to the full basename when the pattern does not match.
    """
    base = os.path.basename(path)
    m = _COSMO_ID_RE.match(base)
    return m.group(1) if m else base


def load_numeric_cosmo_vec(path: str) -> Tuple[np.ndarray, List[str]]:
    """Read cosmo_dict keeping only float-convertible scalar entries -> (values, names).

    NLA-era mocks store a string ia_model in cosmo_dict, which load_cosmo_params would float().
    """
    names: List[str] = []
    values: List[float] = []
    with h5py.File(path, "r") as f:
        grp = f["cosmo_dict"]
        for k in grp.keys():
            v = np.asarray(grp[k][()])
            if v.shape != ():
                continue
            try:
                fv = float(v)
            except (ValueError, TypeError):
                continue
            names.append(k)
            values.append(fv)
    return np.asarray(values), names


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


def _ratios_from_theory(cl_bandpowers, theory_bandpowers: Dict[str, np.ndarray],
                        nbins: int) -> Tuple[List[np.ndarray], List[str]]:
    """Divide empirical bandpowers by theory, in the canonical lower-triangle order.

    Row ``idx = i*(i+1)/2 + j`` of ``cl_bandpowers`` is spectrum ``S{i+1}-S{j+1}`` (``i >= j``);
    ``theory_bandpowers`` is keyed by that same label.
    """
    ratios, labels = [], []
    for i in range(nbins):
        for j in range(nbins):
            if i < j:
                continue
            lbl = f"S{i + 1}-S{j + 1}"
            idx = int(i * (i + 1) / 2 + j)
            ratios.append(cl_bandpowers[idx] / theory_bandpowers[lbl])
            labels.append(lbl)
    return ratios, labels


def _load_empirical(path, nested_keys, empirical_key):
    """Read one mock's empirical bandpowers + ℓ-bandpowers (NO cosmo).

    cosmo_params=[]: NLA-era mocks store a string ia_model in cosmo_dict.
    Returns (cl_bandpowers (n_spectra, nbands), cls (nbands,)). Raises KeyError if the requested
    empirical field is missing.
    """
    data, _ = unpack_data(path, nested_keys, [], as_torch=False, return_names=True)
    if empirical_key not in data:
        raise KeyError(
            f"empirical bandpower field '{empirical_key}' missing from {path}; "
            f"available: {sorted(data)}"
        )
    cl_bandpowers = np.asarray(data[empirical_key])      # (n_spectra, nbands)
    cls = np.asarray(data["bandpower_cls"])              # (nbands,)
    return cl_bandpowers, cls


def load_ratios_with_theory(
    path: str,
    theory_bandpowers: Optional[Dict[str, np.ndarray]],
    nested_keys: Dict[str, Tuple[str, ...]],
    cosmo_params,
    *,
    nbins: int = cfg.NBINS,
    empirical_key: str = "mixed_bandpowers",
):
    """Load one mock's empirical bandpowers and divide by a PRECOMPUTED theory dict.

    theory_bandpowers comes from compute_theory_by_cosmology, once per cosmology.
    Returns {path, ratios (n_spectra, nbands), cls (nbands,), labels}, or None on failure so one
    bad file does not kill the joblib batch.
    """
    try:
        if theory_bandpowers is None:
            raise RuntimeError(
                "no theory available for this cosmology (representative file unreadable?)"
            )
        cl_bandpowers, cls = _load_empirical(path, nested_keys, empirical_key)
        ratios, labels = _ratios_from_theory(cl_bandpowers, theory_bandpowers, nbins)
        return {
            "path": path,
            "ratios": np.array(ratios),   # (n_spectra, nbands)
            "cls": cls,
            "labels": labels,
        }
    except Exception as e:  # noqa: BLE001 - keep the batch alive, report the failure
        print(f"[validation] ERROR processing {path}: {e}")
        return None


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
    """Load one mock, compute its OWN theory bandpowers, return ratios (or None on failure).

    Single-file convenience: computes theory from the file's own cosmology.  The ensemble
    driver instead computes theory once per cosmology and calls
    :func:`load_ratios_with_theory` — avoiding the per-augmentation recompute.
    Returns ``{path, ratios (n_spectra, nbands), cls (nbands,), labels}``.
    """
    try:
        cl_bandpowers, cls = _load_empirical(path, nested_keys, empirical_key)
        cosmo_vec = load_numeric_cosmo_vec(path)
        theory_bandpowers = compute_bandpower_theory_from_cosmo_vec(
            cosmo_vec,
            nside=nside, nbins=nbins, lmin_cut=lmin_cut, lmax_cut=lmax_cut,
            nbands=nbands, mixing_matrix=mixing_matrix, nonlinear=nonlinear,
            theory_mode=theory_mode, data_dir=data_dir,
        )
        ratios, labels = _ratios_from_theory(cl_bandpowers, theory_bandpowers, nbins)
        return {
            "path": path,
            "ratios": np.array(ratios),   # (n_spectra, nbands)
            "cls": cls,
            "labels": labels,
        }
    except Exception as e:  # noqa: BLE001 - keep the batch alive, report the failure
        print(f"[validation] ERROR processing {path}: {e}")
        return None


def compute_theory_by_cosmology(
    files: Sequence[str],
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
    n_jobs: int = 8,
    omp_threads: int = 8,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Compute the theory bandpowers once per unique cosmology.

    Files are grouped by cosmo_id_from_path, one representative per cosmology. Returns
    {cosmo_id: {"S{i}-S{j}": (nbands,)}}, omitting a cosmology whose theory fails. For
    shell_projection the non-Limber matter_cls is warmed into the disk cache first.
    """
    import joblib

    groups: Dict[str, str] = {}
    for f in files:
        groups.setdefault(cosmo_id_from_path(f), f)
    n_files, n_cos = len(files), len(groups)
    print(f"[validation] {n_files} files -> {n_cos} unique cosmologies "
          f"(~{n_files / max(1, n_cos):.0f} augmentations/cosmology); "
          f"computing theory once per cosmology")

    ids: List[str] = []
    vecs: List = []
    for cid, rep in groups.items():
        try:
            cv = load_numeric_cosmo_vec(rep)
            ids.append(cid)
            vecs.append(cv)
        except Exception as e:  # noqa: BLE001 - skip; its files become counted failures
            print(f"[validation] (theory: skipping cosmology {cid}, rep {rep} unreadable: {e})")

    if not vecs:
        return {}

    # Warm the matter_cls cache serially first, so the parallel step avoids concurrent CAMB jobs (OOM).
    if theory_mode == "shell_projection":
        from .theory import warm_matter_cache
        warm_jobs = max(1, n_jobs // omp_threads)
        warm_matter_cache(vecs, nside=nside, n_jobs=warm_jobs, omp_threads=omp_threads)

    def _theory(cosmo_vec):
        try:
            return compute_bandpower_theory_from_cosmo_vec(
                cosmo_vec,
                nside=nside, nbins=nbins, lmin_cut=lmin_cut, lmax_cut=lmax_cut,
                nbands=nbands, mixing_matrix=mixing_matrix, nonlinear=nonlinear,
                theory_mode=theory_mode, data_dir=data_dir,
            )
        except Exception as e:  # noqa: BLE001 - drop this cosmology; its files fail downstream
            print(f"[validation] (theory: cosmology computation failed: {e})")
            return None

    theories = joblib.Parallel(n_jobs=min(n_jobs, len(vecs)), backend="loky", verbose=1)(
        joblib.delayed(_theory)(cv) for cv in vecs
    )
    return {cid: th for cid, th in zip(ids, theories) if th is not None}


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

    theory_by_cosmo = compute_theory_by_cosmology(
        files, nested_keys, cosmo_params,
        nside=nside, nbins=nbins, lmin_cut=lmin_cut, lmax_cut=lmax_cut,
        nbands=nbands, mixing_matrix=mixing_matrix, nonlinear=nonlinear,
        theory_mode=theory_mode, data_dir=data_dir, n_jobs=n_jobs,
    )

    with tqdm_joblib(tqdm(total=len(files), desc="ratios")):
        results = joblib.Parallel(n_jobs=n_jobs, backend="loky", verbose=1)(
            joblib.delayed(load_ratios_with_theory)(
                f, theory_by_cosmo.get(cosmo_id_from_path(f)),
                nested_keys, cosmo_params, nbins=nbins, empirical_key=empirical_key,
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
