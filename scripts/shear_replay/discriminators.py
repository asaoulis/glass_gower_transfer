"""Per-mock discriminators (D1-D10) + paired-triplet mechanism probes (M1-M6).

Every function returns flat dicts of scalars so the sweep can serialise rows straight to JSONL.
Shape-stat definitions copy src/ml/eval/misspec.py:_summarise_variate_inputs (z against each
mock's own std => scale-free), extended with void fractions.
"""
from __future__ import annotations

import numpy as np
import healpy as hp

from src.cosmology.map_shears import _build_ell_filter  # noqa: E402 (read-only import)


# --- D1/D2/D9: per-bin, per-patch map statistics ----------------------------------------------
def patch_stats(patches_by_name, prefix):
    out = {}
    for pname, arr in patches_by_name.items():          # arr: (nbins, H, W)
        x = np.asarray(arr, dtype=np.float64).reshape(arr.shape[0], -1)
        mu = x.mean(axis=1)
        sd = x.std(axis=1)
        z = (x - mu[:, None]) / np.clip(sd[:, None], 1e-12, None)
        for i in range(x.shape[0]):
            k = f"{prefix}_{pname}_b{i}"
            out[f"{k}_mean"] = float(mu[i])
            out[f"{k}_std"] = float(sd[i])
            out[f"{k}_skew"] = float((z[i] ** 3).mean())
            out[f"{k}_kurt"] = float((z[i] ** 4).mean() - 3.0)
            out[f"{k}_zero_frac"] = float((x[i] == 0).mean())
            for t in (2, 3, 4):
                out[f"{k}_peak{t}"] = float((z[i] > t).mean())
            for t in (2, 3):
                out[f"{k}_void{t}"] = float((z[i] < -t).mean())
        # aggregate over bins (the legacy single-number summary)
        out[f"{prefix}_{pname}_std_all"] = float(x.std())
    return out


# --- D4/D6: harmonic-space probes from the (unfiltered) alms + the variant filter --------------
def harmonic_stats(replay_out):
    lmax = replay_out["lmax"]
    fwhm, lmin, lcut = replay_out["eb_variant"]
    geom = replay_out["geom"]
    lmax_out = min(lmax, 3 * geom["nside_out"] - 1, 1500)
    fl = _build_ell_filter(lmax_in=lmax, lmax_out=lmax_out, fwhm_arcmin=fwhm,
                           taper_start_frac=0.95, lmin=lmin, lcut=lcut)
    ell = np.arange(lmax + 1)
    w = (2.0 * ell + 1.0) * fl ** 2
    lo_band = (ell >= 56) & (ell <= 300)
    hi_band = (ell >= 800) & (ell <= min(1400, lmax))
    out = {}
    for i, ((almE, _), (almE_r, _)) in enumerate(zip(replay_out["alm"], replay_out["alm_rand"])):
        clE = hp.alm2cl(almE)
        clR = hp.alm2cl(almE_r)
        v_lo = float(np.sum((w * clE)[lo_band]) / (4 * np.pi))
        v_hi = float(np.sum((w * clE)[hi_band]) / (4 * np.pi))
        out[f"D4_varlo_b{i}"] = v_lo
        out[f"D4_varhi_b{i}"] = v_hi
        out[f"D4_hilo_ratio_b{i}"] = v_hi / v_lo if v_lo > 0 else np.nan
        sel = ell >= min(200, lmax // 2)
        out[f"D6_noise_cl_b{i}"] = float(np.average(clR[sel], weights=2 * ell[sel]))
    return out


# --- D3 + noise meter from the filtered maps ---------------------------------------------------
def map_level_stats(replay_out):
    out = {}
    foot = replay_out["footprint"]
    for i in range(len(replay_out["E_maps"])):
        fp = foot[i]
        if not fp.any():
            continue
        vE = float(replay_out["E_maps"][i][fp].var())
        vB = float(replay_out["B_maps"][i][fp].var())
        vR = float(replay_out["randE_maps"][i][fp].var())
        out[f"D3_varE_minus_varB_b{i}"] = vE - vB
        out[f"D2_Bstd_b{i}"] = np.sqrt(vB)
        out[f"D6b_randstd_b{i}"] = np.sqrt(vR)
        out[f"fnoise_b{i}"] = vR / vE if vE > 0 else np.nan
    return out


# --- D7: in-situ variance-vs-counts coupling (single mock, works on real data) -----------------
def coupling_stats(replay_out):
    out = {}
    E, N, foot = replay_out["E_maps"], replay_out["counts_out"], replay_out["footprint"]
    for i in range(len(E)):
        fp = foot[i]
        if fp.sum() < 100:
            continue
        e2 = E[i][fp] ** 2
        n = N[i][fp]
        r = np.corrcoef(e2, n)[0, 1]
        out[f"D7_corr_E2_N_b{i}"] = float(r)
        nbar = n.mean()
        slope = np.polyfit(n / nbar - 1.0, e2 / e2.mean() - 1.0, 1)[0]
        out[f"D7_slope_b{i}"] = float(slope)
    return out


# --- counts profile: the H1 modulator measured directly from the cache (also M6 audit) ---------
def counts_profile(cache, prefix="cnt"):
    out = {}
    for i, cb in enumerate(cache["bins"]):
        if "N" not in cb:
            continue
        n = cb["N"].astype(np.float64)
        nbar = n.mean()
        out[f"{prefix}_nbar_b{i}"] = float(nbar)
        out[f"{prefix}_varN_rel_b{i}"] = float(n.var() / nbar ** 2)     # = 1/nbar + b^2 sigma_d^2
        out[f"{prefix}_invN_mean_b{i}"] = float((1.0 / n).mean())
        out[f"{prefix}_occupied_b{i}"] = int(n.size)
        out[f"{prefix}_fracN1_b{i}"] = float((n == 1).mean())
        out[f"{prefix}_fracN_le2_b{i}"] = float((n <= 2).mean())
        q = cb["Q11"].astype(np.float64).sum() + cb["Q22"].astype(np.float64).sum()
        sig2 = q / (2.0 * max(int(n.sum()), 1))
        out[f"{prefix}_sigma2gal_b{i}"] = float(sig2)
        out[f"{prefix}_noiseterm_b{i}"] = float(sig2 * (1.0 / n).mean())  # <sigma^2/N>, H1's carrier
    return out


def arm_row(cache, replay_out):
    """All single-arm discriminators for one (mock, candidate, variant)."""
    row = {}
    row.update(patch_stats(replay_out["E_patches"], "E"))
    row.update(patch_stats(replay_out["B_patches"], "B"))
    row.update(harmonic_stats(replay_out))
    row.update(map_level_stats(replay_out))
    row.update(coupling_stats(replay_out))
    row.update(counts_profile(cache))
    bp = np.asarray(replay_out["bandpowers"], dtype=np.float64)
    row["D10_bp_mean"] = float(np.nanmean(bp))
    row["D10_bp"] = bp.tolist()
    for i in range(len(replay_out["post_factors"])):
        row[f"post_factor_b{i}"] = float(replay_out["post_factors"][i])
    return row


# --- triplet-level probes (need all three arms in memory) --------------------------------------
def triplet_stats(rep_by_bg, cache_by_bg, ref_bg=1.0):
    """Paired-difference probes: M1 closure, M4 cross-correlation, D8 leakage map."""
    out = {}
    ref = rep_by_bg[ref_bg]
    nbins = len(ref["E_maps"])
    for bg, rep in rep_by_bg.items():
        if bg == ref_bg:
            continue
        tag = f"bg{bg:g}".replace(".", "p")
        cache, cache_ref = cache_by_bg[bg], cache_by_bg[ref_bg]
        for i in range(nbins):
            fp = ref["footprint"][i] & rep["footprint"][i]
            if fp.sum() < 100:
                continue
            dE = rep["E_maps"][i] - ref["E_maps"][i]
            out[f"D8_dEstd_{tag}_b{i}"] = float(dE[fp].std())
            vE_ref = float(ref["E_maps"][i][fp].var())
            vE = float(rep["E_maps"][i][fp].var())
            meas_dvar = vE / vE_ref - 1.0
            out[f"M_dvar_meas_{tag}_b{i}"] = meas_dvar
            # M1 closure: predicted from the H1 noise term <sigma^2/N> and the rand-based f_noise
            cb, cbr = cache["bins"][i], cache_ref["bins"][i]
            nt = counts_profile({"bins": [cb]}, "x")["x_noiseterm_b0"]
            ntr = counts_profile({"bins": [cbr]}, "x")["x_noiseterm_b0"]
            f_noise = float(ref["randE_maps"][i][fp].var() / vE_ref) if vE_ref > 0 else np.nan
            pred_dvar = f_noise * (nt / ntr - 1.0)
            out[f"M1_dvar_pred_{tag}_b{i}"] = pred_dvar
            out[f"M1_closure_frac_{tag}_b{i}"] = (pred_dvar / meas_dvar
                                                  if meas_dvar not in (0.0,) else np.nan)
            # M4: cross-spectrum of the paired difference map with the count-contrast map
            dN = rep["counts_out"][i] - cache_by_bg_counts_scale(rep, ref, i)
            dn_map = np.zeros_like(dE)
            nb = ref["counts_out"][i][fp].mean()
            dn_map[fp] = dN[fp] / max(nb, 1e-9)
            cl_x = hp.anafast(dE * fp, dn_map, lmax=400)
            cl_a = hp.anafast(dE * fp, lmax=400)
            cl_b = hp.anafast(dn_map, lmax=400)
            denom = np.sqrt(np.clip(cl_a[40:] * cl_b[40:], 1e-30, None))
            out[f"M4_xcorr_{tag}_b{i}"] = float(np.mean(cl_x[40:] / denom))
    return out


def cache_by_bg_counts_scale(rep, ref, i):
    """Rescale ref counts to rep's mean so dN isolates the clustering pattern, not total-N drift."""
    a = rep["counts_out"][i]
    b = ref["counts_out"][i]
    s = a.sum() / b.sum() if b.sum() > 0 else 1.0
    return b * s
