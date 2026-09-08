"""Bake an observation into the compact per-arm store the trained models read.

The flagship BGP arms train on ``prebake_maps.py --eb-variant sc8_fwhm4_lmin56_lcut1400 --noise-norm
rand --dtype float16`` (no ``--keep-variant-tag`` => bare ``E`` group). The a1/nobgp arms on
``--eb-variant fwhm4_lmin56_lcut1400 --noise-norm rand`` (verify per arm: see ``ARM_BAKES``).
This module calls the SAME ``_bake_one`` worker, so an observation goes through the identical bake.

File naming contract (``src/ml/data/data_selection.py:extract_cosmo_index`` requires
``output_(\\d+)_``): an observation with opaque label ``L`` is written as
``<store>/output_<obs_id>_out0_rot0_0.h5`` with ``obs_id`` from ``OBS_IDS`` (9001.. by label
order) -- the label itself lives in the STORE directory name and the provenance group only.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional

REPO = Path(__file__).resolve().parents[2]

# arm-family -> prebake flags (eb_variant, noise_norm, dtype, keep_tag)
ARM_BAKES: Dict[str, dict] = {
    # flagship / all BGP arms (nla_m, nla_hf, nla_z_hf, vd_hf, k2_hf) + NPE ens1 pack
    "sc8a1": {"eb_variant": "sc8_fwhm4_lmin56_lcut1400", "noise_norm": "rand", "dtype": "float16", "keep_tag": False},
    # plain counts A1 (noise-normed) at the primary cut
    "a1": {"eb_variant": "fwhm4_lmin56_lcut1400", "noise_norm": "rand", "dtype": "float16", "keep_tag": False},
    # plain counts A0 (no noise norm) -- the pre-dual-norm gower_mocks_nla_m_f16_fwhm4_* product
    "a0": {"eb_variant": "fwhm4_lmin56_lcut1400", "noise_norm": "none", "dtype": "float16", "keep_tag": False},
    # keep the variant tag (configs with eb_map_variant=<tag>)
    "a0_tagged": {"eb_variant": "fwhm4_lmin56_lcut1400", "noise_norm": "none", "dtype": "float16", "keep_tag": True},
}

# opaque label -> numeric obs id used in the filename (so every loader's sim-id regex works)
def obs_id_for(label: str) -> int:
    import hashlib
    labels = {"A": 9001, "B": 9002, "C": 9003}
    if label in labels:
        return labels[label]
    # any other label: stable 4-digit id from a hash, in 9100-9999
    return 9100 + int(hashlib.sha1(label.encode()).hexdigest(), 16) % 900


def baked_filename(label: str) -> str:
    return f"output_{obs_id_for(label)}_out0_rot0_0.h5"


def bake_observation(obs_path: str, store_dir: str, *, arm: str = "sc8a1", label: Optional[str] = None,
                     overwrite: bool = False) -> str:
    """Bake ``obs_path`` (an observation_<label>.h5) into ``store_dir/output_<id>_out0_rot0_0.h5``.
    Returns the baked path. The bake worker drops nothing here (one file); it raises on a missing
    E/noise group, which is the contract check."""
    sys.path.insert(0, str(REPO / "scripts"))
    import prebake_maps  # noqa: WPS433
    cfg = ARM_BAKES[arm]
    if label is None:
        base = os.path.basename(obs_path)
        label = base[len("observation_"):-3] if base.startswith("observation_") else os.path.splitext(base)[0]
    store = Path(store_dir)
    store.mkdir(parents=True, exist_ok=True)
    dst = store / baked_filename(label)
    if dst.exists():
        if not overwrite:
            return str(dst)
        dst.unlink()
    # _bake_one derives dst from the SOURCE basename, so bake via a temp link with the target name
    tmpdir = store / "_bake_tmp"
    tmpdir.mkdir(exist_ok=True)
    src_link = tmpdir / dst.name
    if src_link.exists() or src_link.is_symlink():
        src_link.unlink()
    os.symlink(os.path.abspath(obs_path), src_link)
    try:
        status, info, secs, nbytes = prebake_maps._bake_one(
            (str(src_link), str(store), cfg["eb_variant"], cfg["dtype"], "h5", None, 4,
             cfg["keep_tag"], cfg["noise_norm"]))
    finally:
        src_link.unlink()
        shutil.rmtree(tmpdir, ignore_errors=True)
    if status != "ok":
        raise RuntimeError(f"bake failed: {status} {info}")
    # carry the provenance group across (the bake worker copies only the ML-relevant groups)
    import h5py
    with h5py.File(obs_path, "r") as fo, h5py.File(dst, "a") as fd:
        if "observation" in fo and "observation" not in fd:
            fo.copy("observation", fd)
        if "bb_bandpowers" in fo["cls_results"]["full"] and "bb_bandpowers" not in fd["cls_results"]["full"]:
            fd["cls_results"]["full"].create_dataset("bb_bandpowers", data=fo["cls_results"]["full"]["bb_bandpowers"][()])
        if "cls" in fo["cls_results"]["full"] and "cls" not in fd["cls_results"]["full"]:
            fd["cls_results"]["full"].create_dataset("cls", data=fo["cls_results"]["full"]["cls"][()])
        fd.attrs["bake_arm"] = arm
        fd.attrs["bake_flags"] = str(cfg)
        fd.attrs["label"] = label
    return str(dst)
