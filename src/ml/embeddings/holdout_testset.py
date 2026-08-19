"""Swap a cache-only run's test set for a larger, explicitly-specified held-out set.

Why this exists
---------------
The whitened NLE chains run `embeddings_cache_only`: the cached `emb_{train,val,test}.pt` ARE the
dataset, because the raw `gower_mocks` store no longer exists. That pins the test set to whatever
the run's own cache holds (59 cosmologies), which is not the 190-cosmology set the paper's coverage
figures are computed on. `gen_samples.py`'s `test_shape_noise_idx` / `N_extra_test_cosmologies`
overrides cannot help: they act on the file-loading path that cache-only bypasses.

Every cosmology needed does exist in cached form -- in ANOTHER run's cache (the single chain's
N=530 repeat-0 cache holds 589 cosmologies and every published test row). Two things make
borrowing it safe:

1. **Leakage.** The spec is built to exclude both whitened models' train/val cosmologies, and this
   module re-asserts that at load time against the train/val tensors actually in memory. The lock
   file could be regenerated wrongly; the invariant is enforced where it is used.

2. **Encoder frame.** Caches are NOT interchangeable: each run embedded its rows with a different
   realisation of the frozen source encoder, and a raw swap moves the 1-D profile-likelihood peak
   by 0.15-0.6 x the constraint width -- a systematic aimed straight at the statistic a calibration
   figure measures. But every cache contains the SAME 1117 test rows with IDENTICAL theta: the same
   mocks embedded twice. That is paired data, so the offset is fittable as a plain affine map
   (`z_native ~ z_foreign @ W + b`, 16->16 least squares). Measured out-of-sample (fit on 558 rows,
   tested on 559) it takes the peak bias from (+0.15, +0.17, -0.15) w to (0.000, 0.000, 0.000) w
   for the N=530 cache, and from (+0.61, +0.56, -0.46) w to (+0.03, +0.06, 0.000) w for a worse
   one. The map is fit only on paired embeddings of identical mocks, so it carries no information
   about the parameters being inferred.

The swap happens on RAW z, before whitening, so the run's own whitener and scalers then apply
unchanged.
"""
from __future__ import annotations

import json
import os
from typing import Optional, Tuple

import numpy as np
import torch

SPLITS = ("train", "val", "test")


def _cache_dir(base_path: str, experiment: str, run: str) -> str:
    return os.path.join(base_path, "checkpoints", experiment, run, "datasets")


def _load_split(cache_dir: str, split: str):
    path = os.path.join(cache_dir, f"emb_{split}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"[holdout] source cache missing: {path}")
    st = torch.load(path, map_location="cpu")
    return st["z"], st["theta"]


def _concat_source(cache_dir: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Concatenate the source run's splits in the order the spec's row_index assumes."""
    zs, ths = [], []
    for sp in SPLITS:
        z, th = _load_split(cache_dir, sp)
        zs.append(z)
        ths.append(th)
    return torch.cat(zs, dim=0), torch.cat(ths, dim=0)


def fit_affine_frame_map(z_from: torch.Tensor, z_to: torch.Tensor, holdout_frac: float = 0.5,
                         seed: int = 0):
    """Least-squares affine map taking `z_from` onto `z_to`, plus its out-of-sample rms.

    Both arguments must be the SAME rows (same mocks, identical theta) embedded by the two encoder
    realisations. Fitting on a subset and reporting the error on the rest is what makes the map
    auditable rather than a free parameter.
    """
    a = z_from.double().numpy()
    b = z_to.double().numpy()
    n = a.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_fit = max(1, int(round(n * (1.0 - holdout_frac))))
    fit, held = idx[:n_fit], idx[n_fit:]

    design = np.hstack([a[fit], np.ones((len(fit), 1))])
    W, *_ = np.linalg.lstsq(design, b[fit], rcond=None)

    def apply(x: torch.Tensor) -> torch.Tensor:
        xd = x.double().numpy()
        out = np.hstack([xd, np.ones((xd.shape[0], 1))]) @ W
        return torch.from_numpy(out).to(x.dtype)

    if len(held):
        rms_before = float(np.sqrt(((a[held] - b[held]) ** 2).mean()))
        pred = np.hstack([a[held], np.ones((len(held), 1))]) @ W
        rms_after = float(np.sqrt(((pred - b[held]) ** 2).mean()))
    else:
        rms_before = rms_after = float("nan")
    return apply, rms_before, rms_after


def load_holdout_test_set(
    spec_cfg,
    base_path: str,
    native_test_z: torch.Tensor,
    native_test_theta: torch.Tensor,
    train_theta: Optional[torch.Tensor] = None,
    val_theta: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Assemble the spec's rows from the source cache, aligned into the native encoder frame.

    `native_test_*` are this run's own cached test tensors: the paired anchor for the frame map,
    and nothing else. Returns RAW (unwhitened) z and theta.
    """
    # `spec_cfg` arrives as a plain dict OR an ml_collections.ConfigDict (config build converts
    # nested dicts); both support item access, so key off "is it a bare path string".
    if isinstance(spec_cfg, (str, bytes, os.PathLike)):
        spec_path, align = str(spec_cfg), True
    else:
        spec_path = str(spec_cfg["path"])
        align = bool(spec_cfg["align_to_native"]) if "align_to_native" in spec_cfg else True
    if not os.path.isabs(spec_path):
        spec_path = os.path.join(os.getcwd(), spec_path)
    with open(spec_path) as f:
        spec = json.load(f)

    cache_dir = _cache_dir(base_path, spec["source_experiment"], spec["source_run"])
    src_z, src_theta = _concat_source(cache_dir)
    print(f"[holdout] spec {os.path.basename(spec_path)}: {spec['n_rows']} rows / "
          f"{spec['n_cosmologies']} cosmologies from {spec['source_experiment']}/{spec['source_run']} "
          f"({src_theta.shape[0]} cached rows)")

    idx = torch.tensor([int(r["row_index"]) for r in spec["rows"]], dtype=torch.long)
    if int(idx.max()) >= src_theta.shape[0]:
        raise RuntimeError(
            f"[holdout] spec row_index {int(idx.max())} exceeds the source cache "
            f"({src_theta.shape[0]} rows) -- the spec was built against a different cache.")
    sel_z, sel_theta = src_z[idx], src_theta[idx]

    # theta checksum: the row indices must still point at the rows the spec was built from
    want = torch.tensor(np.array([r["theta"] for r in spec["rows"]], dtype=np.float64))
    if not torch.allclose(sel_theta.double(), want, atol=1e-6, rtol=0):
        bad = int((~torch.isclose(sel_theta.double(), want, atol=1e-6, rtol=0)).any(dim=1).sum())
        raise RuntimeError(f"[holdout] theta mismatch on {bad} rows -- source cache has changed "
                           "since the spec was built. Rebuild the spec; do not proceed.")

    # HARD no-leakage gate, against the tensors actually loaded for this run
    sel_c = {tuple(np.round(r[:7].astype(np.float64), 9)) for r in sel_theta.numpy()}
    for name, th in (("train", train_theta), ("val", val_theta)):
        if th is None:
            continue
        seen = {tuple(np.round(r[:7].astype(np.float64), 9)) for r in th.numpy()}
        leak = sel_c & seen
        if leak:
            raise RuntimeError(
                f"[holdout] LEAKAGE: {len(leak)} held-out cosmologies are in this run's {name} "
                "split. The spec's exclusion list does not match the model being sampled.")
    print(f"[holdout] no-leakage gate passed: {len(sel_c)} cosmologies, disjoint from train+val")

    if align:
        src_test_z, src_test_theta = _load_split(cache_dir, "test")
        if src_test_theta.shape != native_test_theta.shape or not torch.equal(
                src_test_theta, native_test_theta):
            raise RuntimeError(
                "[holdout] cannot fit the frame map: the source run's cached test rows are not the "
                "same rows as this run's (theta differs). Alignment needs paired embeddings.")
        apply, rms_before, rms_after = fit_affine_frame_map(src_test_z, native_test_z)
        sel_z = apply(sel_z)
        print(f"[holdout] affine frame alignment fit on {src_test_z.shape[0]} paired test rows: "
              f"held-out rms |z_src - z_native| {rms_before:.4f} -> {rms_after:.4f}")
    else:
        print("[holdout] align_to_native=False -- using source-frame embeddings VERBATIM")

    return sel_z, sel_theta
