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

    @property
    def estimator_weights(self) -> Optional[np.ndarray]:
        """The per-galaxy weights the estimator must use: ``weight`` only when ``apply_weights``
        recorded mode='lensfit'; None otherwise (the unweighted forward model)."""
        mode = self.provenance.get("treatment", {}).get("weights", {}).get("mode", "ignore")
        return self.weight if mode == "lensfit" else None


def _file_fingerprint(path: str, nbytes: int = 1 << 20) -> dict:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    return {"path": os.path.abspath(path), "size": os.path.getsize(path),
            "sha256_first_MiB": h.hexdigest()}


def _validate(data: np.ndarray, nbins: int, weight: Optional[np.ndarray] = None):
    """Range checks + a NaN/out-of-range drop with counts (never silent).

    Returns ``(stats, data, keep)`` where ``keep`` is the boolean row mask applied, so a weight
    column is filtered with the IDENTICAL mask (spec §5.1). A non-finite or non-positive weight
    drops its row too. The drop order and the RA wrap are unchanged from the unweighted path, so a
    sim catalogue (no weight column) is processed bit-identically."""
    n0 = data.shape[0]
    ok = np.isfinite(data["RA"]) & np.isfinite(data["DEC"]) & np.isfinite(data["E1"]) & np.isfinite(data["E2"])
    ok &= (np.abs(data["DEC"]) <= 90.0)
    ok &= (np.hypot(data["E1"], data["E2"]) < 1.5)
    ok &= (data["ZBIN"] >= 0) & (data["ZBIN"] < nbins)
    n_bad_weight = 0
    if weight is not None:
        if weight.shape[0] != n0:
            raise ValueError(f"weight column has {weight.shape[0]} rows, catalogue has {n0}")
        wok = np.isfinite(weight) & (weight > 0)
        n_bad_weight = int(np.sum(ok & ~wok))
        ok &= wok
    dropped = int(n0 - ok.sum())
    if dropped:
        data = data[ok]
    # RA -> [0, 360)
    data["RA"] = np.mod(data["RA"], 360.0)
    if data.shape[0] == 0:
        raise ValueError("catalogue is empty after validation")
    stats = {"n_in": int(n0), "n_dropped": dropped, "n_out": int(data.shape[0]),
             "n_dropped_bad_weight": n_bad_weight}
    return stats, data, ok


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
            zraw = np.asarray(cols["zbin"]).astype(int)
            if int(cm["zbin_offset"]) == 0 and zraw.min() == 1:
                # KiDS TOMOBIN is 1-based: with offset 0 every galaxy moves up one bin, bin 0 is
                # empty and bin `nbins` is silently dropped by _validate. Refuse rather than guess.
                raise ValueError(f"zbin column runs {zraw.min()}..{zraw.max()} with zbin_offset=0 -- looks 1-based; "
                                 "set 'zbin_offset': 1 in the column map (KiDS TOMOBIN)")
            data["ZBIN"] = zraw - int(cm["zbin_offset"])
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

    stats, data, keep = _validate(data, nbins, weight)
    if weight is not None and stats["n_dropped"]:
        weight = weight[keep]                      # the SAME row mask, so rows and weights stay aligned
    if weight is not None:
        assert weight.shape[0] == data.shape[0], (weight.shape, data.shape)
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
# Survey constants for the REPORTED weight summary only (spec §2.2; never enter an estimator).
# frac-weighted area of the repo mask (N 495.79 + S 471.59) and np.sum(mask) at nside 1024.
KIDS_AREA_DEG2 = 967.39
KIDS_NPIX_NORM = 295070.25


def weight_summary(cat: "Catalogue", nbins: int, area_deg2: float = KIDS_AREA_DEG2,
                   npix_norm: float = KIDS_NPIX_NORM) -> dict:
    """Per-bin weight bookkeeping (spec §2.2 / gate 5): N, sum w, sum w^2, N_eff = (sum w)^2/sum w^2,
    n_eff per arcmin^2 over ``area_deg2`` (compare to ``src/KiDS/tomo.py`` n_arcmin2) and the
    Eq.-11 scalar W_i under normalisation (b) (sum w~ = N_eff): W_i = N_eff / N_pix, which is what
    the mocks' mean count per pixel is. Reported only -- the estimators are scale-free (§2.1)."""
    w = cat.weight
    out = {"per_bin": [], "area_deg2": float(area_deg2), "npix_norm": float(npix_norm)}
    for i in range(nbins):
        sel = cat.data["ZBIN"] == i
        n = int(sel.sum())
        if w is None or n == 0:
            out["per_bin"].append({"bin": i, "n": n})
            continue
        wi = w[sel]
        sw, sw2 = float(wi.sum()), float((wi * wi).sum())
        neff = sw * sw / sw2 if sw2 > 0 else 0.0
        out["per_bin"].append({"bin": i, "n": n, "sum_w": sw, "sum_w2": sw2, "n_eff": neff,
                               "mean_w": sw / n, "n_eff_per_arcmin2": neff / (area_deg2 * 3600.0),
                               "W_i_neff_norm": neff / npix_norm, "W_i_raw": sw / npix_norm})
    return out


def apply_weights(cat: Catalogue, mode: str = "ignore", nbins: Optional[int] = None, **kw) -> Catalogue:
    """Lensfit-weight treatment (spec .claude/plans/shear_normalisation_spec.md §5.1).

    mode='ignore'   (default) identity: the estimator runs unweighted (``gal_wht=None``), the
                    weight column is kept aside. This is the mock path.
    mode='lensfit'  carry ``cat.weight`` (= shear_weight_only * gold_weight_only) through to the
                    estimator as ``weights=`` (spec §5.2). Requires a weight column. NO rescaling
                    is applied: every normalisation mode is invariant under w -> alpha*w (§2.1),
                    so a "safety" normalisation would be a no-op with a bin-indexing hazard.
    Either way the per-bin weight summary (N, sum w, N_eff, n_eff/arcmin^2, W_i) is recorded.
    """
    if mode not in ("ignore", "lensfit"):
        raise NotImplementedError(f"apply_weights mode {mode!r} is not implemented (ignore | lensfit)")
    if mode == "lensfit" and cat.weight is None:
        raise ValueError("apply_weights(mode='lensfit') needs a weight column (column_map['weight'])")
    nb = int(nbins or cat.provenance.get("nbins", 6))
    cat.provenance["treatment"]["weights"] = {"mode": mode, "had_weight_column": cat.weight is not None,
                                              "rescaled": False, "summary": weight_summary(cat, nb)}
    return cat


def fold_weights_for_mean_norm(cat: Catalogue, nbins: Optional[int] = None) -> Catalogue:
    """Spec §4.1: the 'mean'-normalisation ingestion transform, for CROSS-CHECKS only.

    Replaces e by  e~ = (N_i / sum_i w) * w * (e - <e>_w,i)  per bin, so the UNWEIGHTED estimator with
    normalization='mean' (and the mask passed!) returns exactly S_p / W_i. The weighted per-bin mean is
    subtracted here, so the estimator's own mean subtraction becomes a true no-op. Does NOT generalise
    to 'counts' / 'smoothed_counts' (their denominator is per pixel). Production never calls this."""
    if cat.weight is None:
        raise ValueError("fold_weights_for_mean_norm needs a weight column")
    nb = int(nbins or cat.provenance.get("nbins", 6))
    data = cat.data.copy()
    w = cat.weight
    for i in range(nb):
        sel = data["ZBIN"] == i
        if not sel.any():
            continue
        ww = w[sel]
        k = sel.sum() / ww.sum()
        for comp in ("E1", "E2"):
            v = data[comp][sel]
            data[comp][sel] = k * ww * (v - (ww * v).sum() / ww.sum())
    prov = dict(cat.provenance)
    prov["treatment"] = dict(prov.get("treatment", {}))
    prov["treatment"]["weights"] = {"mode": "folded_mean_norm", "had_weight_column": True,
                                    "note": "spec 4.1 ingestion transform; run the estimator with "
                                            "normalization='mean' and the mask; weights NOT passed"}
    return Catalogue(data=data, weight=None, provenance=prov)


def apply_m_bias(cat: Catalogue, m_bias=None, source: str = "auto") -> np.ndarray:
    """Return the per-bin multiplicative-bias vector handed to ``make_alm_shear_convergence``,
    whose estimator divides the shears by ``(1 + m)`` (``map_shears.py``).

    SETTLED (audit .claude/background/real_catalogue_audit.md finding 1; spec §3, 2026-09-08 evening):
    the real KiDS-Legacy catalogue is NOT m-corrected -- sigma_e,w/(1+m) reproduces
    ``systematics.py:sigma_e`` to 0.02 % in all six bins -- so it is de-biased with the FIDUCIAL
    vector ``src/KiDS/systematics.py:m_bias``, exactly as the simulator de-biases its mocks (it
    injects ``m_bias_realised`` and divides by the fiducial vector, leaving the m-uncertainty as a
    residual in the mocks). A SIM catalogue (mock-as-real) uses its realised ``m_bias_for_shear``,
    which is what the bit-identity gate reproduces. Apply 1/(1+m) ONCE: here (via the estimator),
    never also at ingestion.

    source='auto'      sim catalogue -> its ``m_bias_for_shear``; real catalogue -> fiducial;
    source='fiducial'  ``src.KiDS.systematics.m_bias``;
    source='given'     use ``m_bias`` verbatim;  source='zero' -> zeros (an ALREADY m-corrected input only).
    """
    nbins = int(cat.provenance.get("nbins", 6))
    attrs = cat.provenance.get("sim_attrs", {})
    if source == "given":
        m = np.asarray(m_bias, dtype=float)
    elif source == "zero":
        m = np.zeros(nbins)
    elif source == "auto":
        if "m_bias_for_shear" in attrs:
            m = np.asarray(attrs["m_bias_for_shear"], dtype=float)
            source = "auto->sim"
        else:
            from src.KiDS.systematics import m_bias as _fid
            m = np.asarray(_fid, dtype=float)
            source = "auto->fiducial"
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


def _patch_of(dec: np.ndarray) -> np.ndarray:
    """KiDS-North (DEC ~ -5..+5) vs KiDS-South (DEC ~ -36..-26): a diagnostic split only."""
    return np.where(dec > -15.0, "N", "S")


def apply_c_terms(cat: Catalogue, mode: str = "global_mean_only", **kw) -> Catalogue:
    """Additive c-term treatment.

    SETTLED (spec §3, 2026-09-08 evening): the per-bin mean subtraction stays GLOBAL over N+S and is
    done by the estimator (weighted when weights are passed, unweighted otherwise), exactly as for
    the mocks, which inject per-patch c-terms and remove only a global mean. Nothing is subtracted
    here. mode='global_mean_only' RECORDS: the per-bin means the estimator will remove (weighted
    when mode='lensfit', else unweighted) and, as a leakage diagnostic, the per-patch (N/S)
    residual means that stay in the data after the global subtraction. Do not pre-clean per patch:
    the mocks keep a ~3-10e-4 per-patch offset, and a cleaner data vector is a one-sided bias.
    """
    if mode != "global_mean_only":
        raise NotImplementedError(f"apply_c_terms mode {mode!r} is not implemented yet (deferred)")
    nb = int(cat.provenance.get("nbins", 6))
    w = cat.estimator_weights
    patch = _patch_of(cat.data["DEC"])
    means, per_patch = [], []
    for i in range(nb):
        sel = cat.data["ZBIN"] == i
        if not sel.any():
            means.append([0.0, 0.0]); per_patch.append({}); continue
        ww = None if w is None else w[sel]
        def _mean(v, m=None):
            if ww is None:
                return float(np.mean(v if m is None else v[m]))
            wm = ww if m is None else ww[m]
            return float((wm * (v if m is None else v[m])).sum() / wm.sum()) if wm.sum() > 0 else 0.0
        e1, e2 = cat.data["E1"][sel], cat.data["E2"][sel]
        g = [_mean(e1), _mean(e2)]
        means.append(g)
        pp = {}
        for name in ("N", "S"):
            m = patch[sel] == name
            if m.any():
                pp[name] = {"n": int(m.sum()), "residual_mean_e1": _mean(e1, m) - g[0],
                            "residual_mean_e2": _mean(e2, m) - g[1]}
        per_patch.append(pp)
    cat.provenance["treatment"]["c_terms"] = {
        "mode": mode, "weighted": w is not None,
        "per_bin_mean_e1_e2_removed_by_estimator": means,
        "per_patch_residual_after_global_subtraction": per_patch,
    }
    return cat


def check_master_parity() -> None:
    import master_kids_legacy_simulator as m
    assert {k: v for k, v in m.CATALOGUE_DTYPES.items()} == CATALOGUE_DTYPES, m.CATALOGUE_DTYPES
