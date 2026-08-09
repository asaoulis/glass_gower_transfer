"""Paired b_g difference-map forensics: is the surviving channel SIGNAL or NOISE?

Context. Under `counts` normalisation the density modulation of the shear *signal* cancels
exactly (`Σ_s N_p^s/N_p = 1`), yet a trained field-level model still shifts multi-σ under a wrong
b_g while the bandpower model does not. The leading hypothesis is that the residual lives in the
per-pixel shape-noise variance `v_p = σ_e²/2N_p`, whose fluctuation is `δ_v ≈ −b_g δ` — i.e. the
map's local noise amplitude is itself a galaxy-clustering map, which a CNN can read.

This module tests that WITHOUT training anything, from the b_g stores directly.

⚠️ **Per-event differencing does not work here, and the run that showed it is kept as a guard.**
Matching on ``(sim_id, aug_id)`` pairs the *cosmology, shells and footprint rotation* but NOT the
galaxy realisation: changing b_g changes how many galaxies are drawn, which desynchronises the RNG
stream, so the two maps carry independent shape noise. Measured on glass_dn_a0: ``var(Δ)/var(ref)``
= 1.93–1.98 and ``corr(Δ, ref)`` = −0.70, i.e. exactly the 2 and −1/√2 of two independent draws.
The b_g effect (a ~1.5 % change in map RMS) is invisible under that.

So the primary statistic is the **ensemble-mean power spectrum per b_g store**, which needs no
pairing at all: the b_g response is a coherent few-% change in ⟨P(k)⟩, and averaging over ~150
events per bin per patch beats the realisation scatter down far enough to see it.

The discriminator is the SHAPE of the difference map's power spectrum:

  * shape **noise** is white — flat in 2-D power up to the smoothing beam, which then rolls it off;
  * the lensing (+IA) **signal** is red — power falling steeply with k.

So for `Δm = m(b_g) − m(1.0)`:

  * the RATIO ⟨P_bg(k)⟩ / ⟨P_1.0(k)⟩ flat across k ⇒ the extra power is WHITE ⇒ noise-sector;
  * the ratio elevated only at low k, where the red signal lives, and → 1 at high k where shape
    noise dominates ⇒ signal-sector.

Reported per tomographic bin and patch: `var(Δm)/var(m_ref)`, the Pearson correlation
`corr(Δm, m_ref)` (a signal-level multiplicative modulation is coherent with the signal; a noise
modulation is not), and the radially binned 2-D power spectra of both.

Reads only the pre-baked E stores, so no new simulation and no B-mode bake is required. Paths come
from ``VARIATE_SETS`` in ``misspec.py`` (committed code) — never from a caller-supplied path.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

import h5py
import numpy as np

REF_VARIATE = "glass_gb1p0"
_E_GROUP = ("pixelised_results", "E")


def _key(path: str):
    """(sim_id, aug_id) pairing key, using the same parsers as the misspec sample dumps."""
    from src.ml.data.data_selection import extract_cosmo_index
    from .utils import _parse_aug_id
    return int(extract_cosmo_index(path)), int(_parse_aug_id(path))


def _read_e(path: str, eb_variant: Optional[str]) -> Optional[Dict[str, np.ndarray]]:
    grp = f"E_{eb_variant}" if eb_variant else "E"
    try:
        with h5py.File(path, "r") as f:
            pix = f[_E_GROUP[0]]
            if grp not in pix:
                return None
            return {side: np.asarray(pix[grp][side][()], dtype=np.float64)
                    for side in ("north", "south") if side in pix[grp]}
    except Exception:
        return None                      # truncated / corrupt — same policy as the loader


def _radial_power(img: np.ndarray, nbins: int = 12):
    """Radially binned 2-D power spectrum of one (H, W) patch, mean-subtracted."""
    a = img - img.mean()
    p = np.abs(np.fft.rfft2(a)) ** 2 / a.size
    ky = np.fft.fftfreq(a.shape[0])[:, None]
    kx = np.fft.rfftfreq(a.shape[1])[None, :]
    k = np.sqrt(ky ** 2 + kx ** 2)
    kmax = k.max()
    edges = np.linspace(0.0, kmax, nbins + 1)
    idx = np.clip(np.digitize(k.ravel(), edges) - 1, 0, nbins - 1)
    out = np.bincount(idx, weights=p.ravel(), minlength=nbins)
    cnt = np.bincount(idx, minlength=nbins)
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = np.where(cnt > 0, out / np.maximum(cnt, 1), np.nan)
    centres = 0.5 * (edges[1:] + edges[:-1])
    return centres, prof


def run_ebdiff_analysis(variate_set: str, eb_variant: Optional[str] = None,
                        max_files: int = 120, nbins_k: int = 12,
                        out_root: Optional[str] = None) -> Dict:
    """Difference-map forensics for one arm's paired b_g stores. Returns the report dict."""
    from .misspec import VARIATE_SETS

    if variate_set not in VARIATE_SETS:
        raise KeyError(f"unknown variate set {variate_set!r}")
    spec = {v["name"]: v["patterns"] for v in VARIATE_SETS[variate_set]}
    if REF_VARIATE not in spec:
        raise KeyError(f"{variate_set} has no {REF_VARIATE} reference arm")

    from src.ml.data.data_selection import collect_paths

    def index(patterns):
        out = {}
        for p in collect_paths(patterns):
            try:
                out[_key(p)] = p
            except Exception:
                continue
        return out

    ref_idx = index(spec[REF_VARIATE])
    print(f"[ebdiff] {variate_set}: reference {REF_VARIATE} has {len(ref_idx)} events")

    report: Dict = {"variate_set": variate_set, "eb_variant": eb_variant,
                    "reference": REF_VARIATE, "max_files": max_files, "variates": {}}

    for name, patterns in spec.items():
        if name == REF_VARIATE or name.startswith("glass_dn_"):
            continue                                    # skip the in-distribution arm
        ood_idx = index(patterns)
        keys = sorted(set(ref_idx) & set(ood_idx))[:max_files]
        if not keys:
            print(f"[ebdiff] {name}: no (sim_id, aug_id) overlap with the reference — skipped")
            continue

        acc = defaultdict(list)
        kc = None
        spec_ood = defaultdict(list)
        spec_ref = defaultdict(list)
        for k in keys:
            a = _read_e(ref_idx[k], eb_variant)
            b = _read_e(ood_idx[k], eb_variant)
            if not a or not b:
                continue
            for side in sorted(set(a) & set(b)):
                ea, eb_ = a[side], b[side]
                if ea.shape != eb_.shape:
                    continue
                d = eb_ - ea
                for i in range(ea.shape[0]):               # tomographic bin
                    r, dd = ea[i], d[i]
                    vr, vd = float(r.var()), float(dd.var())
                    acc[f"var_ratio/{side}/{i}"].append(vd / vr if vr > 0 else np.nan)
                    rc, dc = r - r.mean(), dd - dd.mean()
                    den = float(np.sqrt((rc ** 2).sum() * (dc ** 2).sum()))
                    acc[f"corr/{side}/{i}"].append(float((rc * dc).sum() / den) if den > 0
                                                   else np.nan)
                    kc, pd_ = _radial_power(dd, nbins_k)
                    _, pr_ = _radial_power(r, nbins_k)
                    _, po_ = _radial_power(eb_[i], nbins_k)     # the OOD map itself, not Δ
                    acc[f"P_delta/{side}/{i}"].append(pd_)
                    acc[f"P_ref/{side}/{i}"].append(pr_)
                    # ⭐ the pairing-free statistic: ensemble-mean spectra per store
                    spec_ood[f"{side}/{i}"].append(po_)
                    spec_ref[f"{side}/{i}"].append(pr_)

        if kc is None:
            print(f"[ebdiff] {name}: every paired read failed — skipped")
            continue

        entry: Dict = {"n_events": len(keys), "k_centres": kc.tolist(), "per_bin": {}}
        for kk, v in sorted(acc.items()):
            stat, side, ib = kk.split("/")
            arr = np.asarray(v, dtype=np.float64)
            slot = entry["per_bin"].setdefault(f"{side}/{ib}", {})
            if stat.startswith("P_"):
                slot[stat] = np.nanmean(arr, axis=0).tolist()
            else:
                slot[stat] = float(np.nanmean(arr))
                slot[stat + "_sem"] = float(np.nanstd(arr, ddof=1) / np.sqrt(len(arr)))
        # ---- the headline: is P_delta flat where P_ref is red? ----
        # "redness" = log-log slope over the first half of the k range, where the beam has not
        # yet taken over. Signal-sector => slope(P_delta) ~ slope(P_ref); noise-sector => ~0.
        for key_, slot in entry["per_bin"].items():
            kk_ = np.asarray(entry["k_centres"])
            half = max(3, len(kk_) // 2)
            for tag in ("P_delta", "P_ref"):
                y = np.asarray(slot[tag][:half]); x = kk_[:half]
                m = np.isfinite(y) & (y > 0) & (x > 0)
                slot[tag + "_slope"] = (float(np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0])
                                        if m.sum() >= 3 else float("nan"))
        entry["mean_slope_delta"] = float(np.nanmean(
            [s["P_delta_slope"] for s in entry["per_bin"].values()]))
        entry["mean_slope_ref"] = float(np.nanmean(
            [s["P_ref_slope"] for s in entry["per_bin"].values()]))
        # ---- ⭐ ensemble-mean spectra and their ratio (the pairing-free discriminator) ----
        for bk in sorted(spec_ref):
            mo = np.nanmean(np.asarray(spec_ood[bk]), axis=0)
            mr = np.nanmean(np.asarray(spec_ref[bk]), axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.where(mr > 0, mo / mr, np.nan)
            slot = entry["per_bin"][bk]
            slot["P_ood_mean"] = mo.tolist()
            slot["P_ref_mean"] = mr.tolist()
            slot["P_ratio"] = ratio.tolist()
            n_ev = len(spec_ref[bk])
            sd = np.nanstd(np.asarray(spec_ood[bk]) / np.maximum(np.asarray(spec_ref[bk]), 1e-300),
                           axis=0, ddof=1)
            slot["P_ratio_sem"] = (sd / np.sqrt(max(n_ev, 1))).tolist()
        entry["mean_corr"] = float(np.nanmean([s["corr"] for s in entry["per_bin"].values()]))
        entry["mean_var_ratio"] = float(np.nanmean(
            [s["var_ratio"] for s in entry["per_bin"].values()]))
        report["variates"][name] = entry
        print(f"[ebdiff] {name}: n={len(keys)}  var(Δ)/var(ref)={entry['mean_var_ratio']:.4f}  "
              f"corr(Δ,ref)={entry['mean_corr']:+.4f}  "
              f"slope P_Δ={entry['mean_slope_delta']:+.2f} vs P_ref={entry['mean_slope_ref']:+.2f}")

    if out_root:
        os.makedirs(out_root, exist_ok=True)
        dst = os.path.join(out_root, f"ebdiff_{variate_set}.json")
        with open(dst, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[ebdiff] wrote {dst}")
    return report
