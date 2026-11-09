"""External catalogue -> the simulator's structured-array contract, plus the deferred treatment hooks.

The contract is a structured array RA, DEC, Z_TRUE, ZBIN, E1, E2 (CATALOGUE_DTYPES). apply_weights,
apply_m_bias and apply_c_terms are the three treatments, identity by default and recorded in the
provenance. Nothing here reads a cosmo_dict group.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

# The simulator's on-disk catalogue dtypes, a mirror of master's CATALOGUE_DTYPES.
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
        """The per-galaxy estimator weights: `weight` when apply_weights recorded mode='lensfit',
        else None (the unweighted forward model)."""
        mode = self.provenance.get("treatment", {}).get("weights", {}).get("mode", "ignore")
        return self.weight if mode == "lensfit" else None


def _file_fingerprint(path: str, nbytes: int = 1 << 20) -> dict:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    return {"path": os.path.abspath(path), "size": os.path.getsize(path),
            "sha256_first_MiB": h.hexdigest()}


def _validate(data: np.ndarray, nbins: int, weight: Optional[np.ndarray] = None):
    """Range checks and a counted drop of NaN or out-of-range rows.

    Returns (stats, data, keep), keep being the row mask, so a weight column is filtered with the
    same mask. A non-finite or non-positive weight drops its row too."""
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


# --- deferred treatment hooks ---
# Survey constants for the reported weight summary only: the frac-weighted area of the repo mask
# (N 495.79 + S 471.59 deg^2) and np.sum(mask) at nside 1024.
KIDS_AREA_DEG2 = 967.39
KIDS_NPIX_NORM = 295070.25


def weight_summary(cat: "Catalogue", nbins: int, area_deg2: float = KIDS_AREA_DEG2,
                   npix_norm: float = KIDS_NPIX_NORM) -> dict:
    """Per-bin weight bookkeeping: N, sum w, sum w^2, N_eff = (sum w)^2/sum w^2, n_eff per arcmin^2
    over area_deg2, and W_i = N_eff / N_pix, the mocks' mean count per pixel. Reported only."""
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


def apply_weights(cat: Catalogue, mode: str = "ignore", nbins: Optional[int] = None,
                  rescale: str = "none", **kw) -> Catalogue:
    """Lensfit-weight treatment, recorded with the per-bin weight summary.

    mode='ignore' (default) runs the estimator unweighted, the mock path. mode='lensfit' passes
    cat.weight to the estimator as weights= and needs a weight column. rescale='neff' sets, per
    0-based tomographic bin, w~ = w * sum(w)/sum(w^2) so sum(w~) = N_eff. The maps are unchanged
    by it, but W_i and any exported count field take the mocks' scale.
    """
    if mode not in ("ignore", "lensfit"):
        raise NotImplementedError(f"apply_weights mode {mode!r} is not implemented (ignore | lensfit)")
    if rescale not in ("none", "neff"):
        raise NotImplementedError(f"apply_weights rescale {rescale!r} is not implemented (none | neff)")
    if mode == "lensfit" and cat.weight is None:
        raise ValueError("apply_weights(mode='lensfit') needs a weight column (column_map['weight'])")
    if rescale != "none" and cat.weight is None:
        raise ValueError(f"apply_weights(rescale={rescale!r}) needs a weight column")
    nb = int(nbins or cat.provenance.get("nbins", 6))

    record = {"mode": mode, "had_weight_column": cat.weight is not None, "rescaled": False}
    if rescale == "neff":
        raw_summary = weight_summary(cat, nb)
        w = np.asarray(cat.weight, dtype=float).copy()
        divisors = []
        for i in range(nb):
            sel = cat.data["ZBIN"] == i          # 0-based: load_catalogue removed zbin_offset
            if not sel.any():
                divisors.append(None)
                continue
            wi = w[sel]
            sw, sw2 = float(wi.sum()), float((wi * wi).sum())
            if not (sw > 0 and sw2 > 0):
                raise ValueError(f"bin {i}: non-positive weight sums (sum w={sw}, sum w^2={sw2})")
            divisor = sw2 / sw                   # w~ = w / divisor  =>  sum w~ = (sum w)^2/sum w^2
            w[sel] = wi / divisor
            divisors.append(divisor)
        cat = Catalogue(data=cat.data, weight=w, provenance=cat.provenance)
        record.update({
            "rescaled": True, "rescale": "neff",
            "rescale_note": "per bin w~ = w * sum(w)/sum(w^2), so sum(w~) = N_eff; the estimators "
                            "are invariant under this per-bin rescale (maps/alms unchanged), it "
                            "fixes the SCALE of W_i and of any exported count field",
            "divisor_sum_w2_over_sum_w": divisors,
            "summary_raw": raw_summary,
        })
    record["summary"] = weight_summary(cat, nb)
    cat.provenance["treatment"]["weights"] = record
    return cat


def fold_weights_for_mean_norm(cat: Catalogue, nbins: Optional[int] = None) -> Catalogue:
    """The 'mean'-normalisation ingestion transform, for cross-checks only.

    Replaces e by  e~ = (N_i / sum_i w) * w * (e - <e>_w,i)  per bin, so the unweighted estimator with
    normalization='mean' and the mask returns S_p / W_i. It does not apply to 'counts' or
    'smoothed_counts', whose denominator is per pixel."""
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
    """Return the per-bin m vector for make_alm_shear_convergence, which divides shears by (1 + m).

    The real catalogue is not m-corrected, so it takes the fiducial src.KiDS.systematics.m_bias, as
    the mocks do. Apply 1/(1 + m) once, here, never also at ingestion. source='auto' uses a sim
    catalogue's m_bias_for_shear, else the fiducial vector. 'fiducial', 'given' (m_bias verbatim)
    and 'zero' (an already m-corrected input only) are explicit.
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


RESTORE_C_MODES = {"restore_release_c_sim": "sim", "restore_release_c_catalogue": "catalogue"}


def _restore_release_c(cat: Catalogue, frame: str = "sim", scale: float = 1.0) -> dict:
    from src.KiDS import systematics as S
    nb = int(cat.provenance.get("nbins", 6))
    flip2 = bool(cat.provenance.get("column_map", {}).get("flip_e2", False))
    s2 = -1.0 if (frame == "catalogue" and flip2) else 1.0
    north = np.abs(cat.data["DEC"]) < 15.0          # the simulator's hemisphere split (apply_shear_bias)
    added = {"frame": frame, "scale": scale, "c2_sign_applied": s2, "source": "src.KiDS.systematics prior means (fallback)",
             "north": [], "south": []}
    for i in range(nb):
        sel = cat.data["ZBIN"] == i
        for name, m, c1, c2 in (("north", north, S.c_1_bias_north, S.c_2_bias_north),
                                ("south", ~north, S.c_1_bias_south, S.c_2_bias_south)):
            a1, a2 = scale * float(c1[i]), scale * s2 * float(c2[i])
            cat.data["E1"][sel & m] += a1
            cat.data["E2"][sel & m] += a2
            added[name].append([a1, a2])
    return added


def apply_c_terms(cat: Catalogue, mode: str = "global_mean_only", **kw) -> Catalogue:
    """Additive c-term treatment.

    The estimator removes the global per-bin mean over N+S, as for the mocks. mode='global_mean_only'
    (default) subtracts nothing and records those means and the per-patch residuals. The opt-in
    RESTORE_C_MODES add back the per-patch c-terms the release subtracted (src.KiDS.systematics
    prior means): 'restore_release_c_sim' in the simulator frame, '_catalogue' in the KiDS frame.
    """
    restored = None
    if mode in RESTORE_C_MODES:
        restored = _restore_release_c(cat, frame=RESTORE_C_MODES[mode], scale=float(kw.get("scale", 1.0)))
    elif mode != "global_mean_only":
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
        "mode": mode, "weighted": w is not None, "restored_release_c": restored,
        "per_bin_mean_e1_e2_removed_by_estimator": means,
        "per_patch_residual_after_global_subtraction": per_patch,
    }
    return cat


def check_master_parity() -> None:
    import master_kids_legacy_simulator as m
    assert {k: v for k, v in m.CATALOGUE_DTYPES.items()} == CATALOGUE_DTYPES, m.CATALOGUE_DTYPES
