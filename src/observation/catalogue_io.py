"""External catalogue -> the simulator's structured-array contract, plus the deferred treatment hooks.

The simulator's catalogue contract (``src/cosmology/simulators.py``, ``master_kids_legacy_simulator
.CATALOGUE_DTYPES``) is a structured array with fields ``RA, DEC, Z_TRUE, ZBIN, E1, E2`` — no weight
column. ``make_alm_shear_convergence`` (protected) then subtracts the GLOBAL per-tomo-bin mean
ellipticity and divides by ``1/(1+m_i)``.

Three corrections that real data needs and the forward model handles differently are DEFERRED by
user decision (2026-09-08) and exposed as ONE explicit call each — identity by default, always
recorded in the provenance so a later change is a one-line, auditable edit:

    apply_weights(cat, mode=...)   lensfit weights vs the unweighted forward model
    apply_m_bias(cat, m_bias=...)  the per-bin multiplicative bias vector passed to the map maker
    apply_c_terms(cat, mode=...)   additive c-terms beyond the global per-bin mean subtraction

NOTHING here reads a ``cosmo_dict`` group, even when the file has one (a saved mock catalogue does).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

# The simulator's on-disk catalogue dtypes (mirror of master's CATALOGUE_DTYPES; parity-checked in
# geometry.check_master_parity's sibling below).
CATALOGUE_DTYPES = {
    "RA": np.float32, "DEC": np.float32, "Z_TRUE": np.float32,
    "ZBIN": np.int8, "E1": np.float32, "E2": np.float32,
}
_ROW_DTYPE = np.dtype([("RA", float), ("DEC", float), ("Z_TRUE", float), ("ZBIN", int),
                       ("E1", float), ("E2", float)])

# Column-map spec for an external (real) catalogue. Values are dataset paths (HDF5) or column
# names (FITS). Only ra/dec/e1/e2 + (zbin | z_b+zbin_edges) are required.
DEFAULT_COLUMN_MAP = {
    "ra": "RA", "dec": "DEC", "e1": "e1", "e2": "e2",
    "zbin": None,            # integer tomographic bin (see zbin_offset)
    "z_b": None,             # photometric redshift, digitised with zbin_edges when zbin is None
    "weight": None,          # lensfit weight (kept aside; see apply_weights)
    "z_true": None,
    "zbin_offset": 0,        # subtract this so bins are 0-based (KiDS tables are often 1-based)
    "zbin_edges": None,      # e.g. [0.1,0.3,0.5,0.7,0.9,1.2,2.0]
    "flip_e1": False, "flip_e2": False,   # sign conventions (record, don't guess)
}


@dataclass
class Catalogue:
    data: np.ndarray                      # structured, fields of _ROW_DTYPE
    weight: Optional[np.ndarray] = None   # lensfit weight, same length as data, or None
    provenance: Dict = field(default_factory=dict)

    @property
    def n_gal(self) -> int:
        return int(self.data.shape[0])

    def counts_per_bin(self, nbins: int) -> list:
        return [int(np.sum(self.data["ZBIN"] == i)) for i in range(nbins)]


def _file_fingerprint(path: str, nbytes: int = 1 << 20) -> dict:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    return {"path": os.path.abspath(path), "size": os.path.getsize(path),
            "sha256_first_MiB": h.hexdigest()}


def _validate(data: np.ndarray, nbins: int) -> dict:
    """Range checks + a NaN/out-of-range drop with counts (never silent)."""
    n0 = data.shape[0]
    ok = np.isfinite(data["RA"]) & np.isfinite(data["DEC"]) & np.isfinite(data["E1"]) & np.isfinite(data["E2"])
    ok &= (np.abs(data["DEC"]) <= 90.0)
    ok &= (np.hypot(data["E1"], data["E2"]) < 1.5)
    ok &= (data["ZBIN"] >= 0) & (data["ZBIN"] < nbins)
    dropped = int(n0 - ok.sum())
    if dropped:
        data = data[ok]
    # RA -> [0, 360)
    data["RA"] = np.mod(data["RA"], 360.0)
    if data.shape[0] == 0:
        raise ValueError("catalogue is empty after validation")
    return {"n_in": int(n0), "n_dropped": dropped, "n_out": int(data.shape[0])}, data


def load_catalogue(path: str, *, column_map: Optional[dict] = None, nbins: int = 6,
                   kind: str = "auto") -> Catalogue:
    """Load a catalogue into the simulator contract.

    kind='sim'   a ``catalogue_*.h5`` written by ``master --save-catalogues`` (group ``catalogue/``
                 + file attrs). Its ``cosmo_dict`` group is NEVER read.
    kind='h5'    a generic HDF5 file addressed by ``column_map``.
    kind='fits'  a FITS table addressed by ``column_map`` (astropy).
    kind='auto'  sim if the file has a ``catalogue`` group, else by extension.
    """
    import h5py
    prov = {"source": _file_fingerprint(path), "kind": kind, "nbins": int(nbins)}
    if kind == "auto":
        if path.endswith((".fits", ".fit", ".fits.gz")):
            kind = "fits"
        else:
            with h5py.File(path, "r") as f:
                kind = "sim" if "catalogue" in f else "h5"
        prov["kind"] = kind

    weight = None
    if kind == "sim":
        with h5py.File(path, "r") as f:
            g = f["catalogue"]
            n = g["RA"].shape[0]
            data = np.empty(n, dtype=_ROW_DTYPE)
            for col in ("RA", "DEC", "ZBIN", "E1", "E2"):
                data[col] = g[col][()]
            data["Z_TRUE"] = g["Z_TRUE"][()] if "Z_TRUE" in g else np.nan
            attrs = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in f.attrs.items()}
        # keep the non-cosmological attrs the replay needs (m_bias, nside, seeds, ids)
        prov["sim_attrs"] = attrs
    else:
        cm = dict(DEFAULT_COLUMN_MAP)
        cm.update(column_map or {})
        prov["column_map"] = cm
        cols = _read_columns(path, kind, cm)
        n = cols["ra"].shape[0]
        data = np.empty(n, dtype=_ROW_DTYPE)
        data["RA"] = cols["ra"]
        data["DEC"] = cols["dec"]
        data["E1"] = (-1.0 if cm["flip_e1"] else 1.0) * cols["e1"]
        data["E2"] = (-1.0 if cm["flip_e2"] else 1.0) * cols["e2"]
        data["Z_TRUE"] = cols["z_true"] if cols.get("z_true") is not None else np.nan
        if cols.get("zbin") is not None:
            data["ZBIN"] = np.asarray(cols["zbin"]).astype(int) - int(cm["zbin_offset"])
        elif cols.get("z_b") is not None and cm["zbin_edges"] is not None:
            edges = np.asarray(cm["zbin_edges"], dtype=float)
            # bin i <- edges[i] <= z_B < edges[i+1]; outside -> -1 (dropped by _validate)
            zb = np.digitize(cols["z_b"], edges) - 1
            zb[(cols["z_b"] < edges[0]) | (cols["z_b"] >= edges[-1])] = -1
            data["ZBIN"] = zb
        else:
            raise ValueError("column_map needs 'zbin' or ('z_b' + 'zbin_edges')")
        weight = cols.get("weight")
        if weight is not None:
            weight = np.asarray(weight, dtype=float)

    stats, data = _validate(data, nbins)
    if weight is not None and stats["n_dropped"]:
        # re-apply the same row filter to the weights (recompute mask on the validated array
        # is not possible, so filter alongside): rebuild from a boolean index computed above
        raise NotImplementedError("row drops with a weight column present: filter weights first")
    prov["validation"] = stats
    prov["counts_per_bin"] = [int(np.sum(data["ZBIN"] == i)) for i in range(nbins)]
    prov["treatment"] = {}
    return Catalogue(data=data, weight=weight, provenance=prov)


def _read_columns(path: str, kind: str, cm: dict) -> dict:
    out = {}
    keys = ("ra", "dec", "e1", "e2", "zbin", "z_b", "weight", "z_true")
    if kind == "h5":
        import h5py
        with h5py.File(path, "r") as f:
            for k in keys:
                src = cm.get(k)
                out[k] = None if src is None else np.asarray(f[src][()])
    elif kind == "fits":
        from astropy.io import fits
        with fits.open(path, memmap=True) as hdul:
            tab = next(h for h in hdul if getattr(h, "columns", None) is not None and h.data is not None)
            names = {c.name.lower(): c.name for c in tab.columns}
            for k in keys:
                src = cm.get(k)
                if src is None:
                    out[k] = None
                else:
                    name = names.get(str(src).lower(), src)
                    out[k] = np.asarray(tab.data[name])
    else:
        raise ValueError(kind)
    for k in ("ra", "dec", "e1", "e2"):
        if out[k] is None:
            raise ValueError(f"column_map has no source for required column {k!r}")
    return out


# ----------------------------------------------------------------------------------------------
# DEFERRED treatment hooks — one call each; identity by default; always logged.
# ----------------------------------------------------------------------------------------------
def apply_weights(cat: Catalogue, mode: str = "ignore", **kw) -> Catalogue:
    """Lensfit-weight treatment. DEFERRED (user, 2026-09-08).

    mode='ignore'  (default) the forward model is unweighted (``map_shears(..., gal_wht=None)`` in
                   the protected ``make_alm_shear_convergence``), so weights are set aside.
    Future modes (e.g. 'resample_to_neff', 'weighted_maps') plug in HERE and nowhere else.
    """
    if mode != "ignore":
        raise NotImplementedError(f"apply_weights mode {mode!r} is not implemented yet (deferred)")
    cat.provenance["treatment"]["weights"] = {"mode": mode, "had_weight_column": cat.weight is not None}
    return cat


def apply_m_bias(cat: Catalogue, m_bias=None, source: str = "auto") -> np.ndarray:
    """Return the per-bin multiplicative-bias vector handed to ``make_alm_shear_convergence``,
    whose estimator divides the shears by ``(1 + m)`` (``map_shears.py``).

    USER DECISION (2026-09-08, afternoon): the real shear catalogue arrives ALREADY m- and
    c-corrected -- that correction step is what the forward model simulates -- so the estimator
    must not de-bias it again: a real catalogue gets ``m = 0``. A SIM catalogue (mock-as-real) is
    the raw product of the simulator and is processed exactly as the master does it, with its
    realised ``m_bias_for_shear`` (this is what the bit-identity gate reproduces).

    source='auto'   sim catalogue -> its ``m_bias_for_shear``; anything else -> zeros;
    source='given'  use ``m_bias`` verbatim; source='zero' -> zeros;
    source='fiducial' -> ``src.KiDS.systematics.m_bias`` (only for a catalogue that is NOT corrected).
    """
    nbins = int(cat.provenance.get("nbins", 6))
    if source == "given":
        m = np.asarray(m_bias, dtype=float)
    elif source == "zero":
        m = np.zeros(nbins)
    elif source == "auto":
        attrs = cat.provenance.get("sim_attrs", {})
        if "m_bias_for_shear" in attrs:
            m = np.asarray(attrs["m_bias_for_shear"], dtype=float)
        else:
            m = np.zeros(nbins)            # real catalogue: already m-corrected (user, 2026-09-08)
    elif source == "fiducial":
        from src.KiDS.systematics import m_bias as _fid
        m = np.asarray(_fid, dtype=float)
    else:
        raise ValueError(source)
    if m.ndim == 0:
        m = np.repeat(m, nbins)
    assert m.shape == (nbins,), m.shape
    cat.provenance["treatment"]["m_bias"] = {"source": source, "values": m.tolist()}
    return m


def apply_c_terms(cat: Catalogue, mode: str = "global_mean_only", **kw) -> Catalogue:
    """Additive c-term treatment.

    USER DECISION (2026-09-08, afternoon): the real catalogue arrives c-corrected; nothing is
    applied here. mode='global_mean_only' (default) only RECORDS the per-bin means that the
    production estimator subtracts anyway inside ``make_alm_shear_convergence`` (identically for
    the mocks, which inject per-N/S c-terms). Future modes (per-patch, per-bin c-map) plug in HERE.
    """
    if mode != "global_mean_only":
        raise NotImplementedError(f"apply_c_terms mode {mode!r} is not implemented yet (deferred)")
    nb = int(cat.provenance.get("nbins", 6))
    means = [[float(np.mean(cat.data["E1"][cat.data["ZBIN"] == i])) if np.any(cat.data["ZBIN"] == i) else 0.0,
              float(np.mean(cat.data["E2"][cat.data["ZBIN"] == i])) if np.any(cat.data["ZBIN"] == i) else 0.0]
             for i in range(nb)]
    cat.provenance["treatment"]["c_terms"] = {"mode": mode, "per_bin_mean_e1_e2_removed_by_estimator": means}
    return cat


def check_master_parity() -> None:
    import master_kids_legacy_simulator as m
    assert {k: v for k, v in m.CATALOGUE_DTYPES.items()} == CATALOGUE_DTYPES, m.CATALOGUE_DTYPES
