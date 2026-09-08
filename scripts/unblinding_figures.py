#!/usr/bin/env python
"""The unblinding figure set: one numbered folder that walks through every tier of the protocol
with the numbers measured on the two mock-as-real controls (T = GLASS b_g=1 catalogue, known-OOD
for the Gower nla_m cloud; S = held-out flagship mock, in-distribution).

    PYTHONPATH=. python scripts/unblinding_figures.py --out-dir /data/alex/unblinding/figures \\
        [--palette tol-muted] [--palette-options] \\
        [--tier1 T=/data/alex/unblinding/tier1_T S=/data/alex/unblinding/tier1_S] \\
        [--obs-score T=<obs_score_T.json> S=<obs_score_S.json>] [--plots-dir /data/alex/unblinding/plots_pilot]

Style: `src.viz.style` -- the paper's scienceplots science+muted context (serif, LaTeX, 18 pt
drawn at ~6 in per panel), colours pinned by role (real = black, mock ensembles = light grey,
labels T/S and the six arms by key). `--palette` swaps the cycle without changing any role;
`--palette-options` additionally renders the Tier-1/2 panels in every option under
`palette_options/<name>/` plus a synthetic swatch sheet `00_style_palette_options.png`.
Captions live in README.md (figures carry panel letters, not explanatory titles).

Blind-safe by construction: every input is a Tier-1/Tier-2 scalar file, the Tier-2 row cache
(mock rows only), or an already-standardised Tier-3 figure. No posterior sample is opened here.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from src.viz import style as S  # noqa: E402

ART = REPO / ".claude/runs/eval-and-viz/unblinding-prep/artifacts"
T = S.tex
PAL = S.DEFAULT_PALETTE          # set from --palette in main()


def R(role):
    return S.role_colour(role, PAL)


def LC(lab):
    return S.label_colour(lab, PAL)


CAPTIONS = {}


def _save(fig, out_dir: Path, name: str, index: list, caption: str):
    S.save(fig, out_dir / name, formats=("png", "pdf"))
    index.append(name)
    CAPTIONS[name] = caption
    print(f"  wrote {name}")


# ---------------------------------------------------------------------------------------------
# 00 palette swatches (synthetic data only)
# ---------------------------------------------------------------------------------------------
def fig_swatches(out_dir, index):
    rng = np.random.default_rng(3)
    names = list(S.PALETTES)
    with S.context(PAL, font_size=13):
        fig, axes = plt.subplots(len(names), 4, figsize=(19, 3.6 * len(names)))
        fig.subplots_adjust(wspace=0.32, hspace=0.55)
        x = np.linspace(0, 1, 60)
        for row, nm in zip(axes, names):
            cyc = S.palette(nm)
            # (a) the cycle
            ax = row[0]
            for i, c in enumerate(cyc):
                ax.add_patch(patches.Rectangle((i, 0), 1, 1, color=c))
                ax.text(i + 0.5, -0.18, S.PALETTES[nm]["names"][i], ha="center", va="top", fontsize=8, rotation=35)
            ax.set_xlim(0, len(cyc)); ax.set_ylim(-0.9, 1.05); ax.axis("off")
            ax.set_title(T(f"{nm}: cycle"), fontsize=12, loc="left")
            # (b) categorical lines = the six arms, the reference and the observation roles
            ax = row[1]
            for j, arm in enumerate(S.ARM_ORDER):
                ax.plot(x, np.exp(-(x - 0.15 * j - 0.1) ** 2 / 0.02) * (1 + 0.1 * j), lw=2, color=S.arm_colour(arm, nm), label=S.arm_name(arm))
            ax.axhline(0.5, ls="--", color=S.role_colour("reference", nm), lw=1.2, label="reference")
            ax.axvspan(0.85, 1.0, color=S.role_colour("stop", nm), alpha=0.12, label="stop rule")
            ax.set_title(T("arms + roles"), fontsize=12, loc="left"); ax.set_xlabel(T("x")); ax.set_ylabel(T("y"))
            ax.legend(fontsize=7.5, ncol=2, loc="upper right")
            # (c) real vs mock ensemble vs flagship arm (the Tier-3 conventions) as 1-D densities
            ax = row[2]
            g = np.linspace(-4, 4, 300)
            for k in range(8):
                m = rng.normal(0, 0.35); sd = rng.uniform(0.8, 1.1)
                ax.plot(g, np.exp(-0.5 * ((g - m) / sd) ** 2) / sd, lw=0.9, color=S.role_colour("mock", nm),
                        label="matched mocks" if k == 0 else None)
            ax.plot(g, np.exp(-0.5 * ((g - 0.4) / 0.9) ** 2) / 0.9, lw=2, color=S.role_colour("flagship", nm), label="flagship arm")
            ax.fill_between(g, 0, np.exp(-0.5 * ((g - 0.4) / 0.9) ** 2) / 0.9, color=S.role_colour("flagship", nm), alpha=0.25, lw=0)
            ax.plot(g, np.exp(-0.5 * g ** 2), lw=2.4, color=S.role_colour("real", nm), label="observation")
            ax.set_yticks([]); ax.set_xlabel(T("standardised parameter ($\\sigma$)")); ax.legend(fontsize=8)
            ax.set_title(T("Tier-3 roles"), fontsize=12, loc="left")
            # (d) histogram + label markers (the Tier-1/2 convention)
            ax = row[3]
            null = rng.normal(0, 1, 4000)
            ax.hist(null, bins=40, color=S.role_colour("mock", nm), alpha=0.7, density=True, label="null mocks")
            ax.axvline(1.1, color=S.role_colour("T", nm), lw=2.2, label="label T")
            ax.axvline(-0.3, color=S.role_colour("S", nm), lw=2.2, label="label S")
            ax.axvline(2.33, color=S.role_colour("stop", nm), ls="--", lw=1.2, label="stop rule")
            ax.axvspan(2.33, 4, color=S.role_colour("stop", nm), alpha=0.10)
            ax.set_xlim(-4, 4); ax.set_yticks([]); ax.set_xlabel(T("statistic")); ax.legend(fontsize=8)
            ax.set_title(T("Tier-1/2 roles"), fontsize=12, loc="left")
        fig.text(0.5, 0.995, T("Colour options (src/viz/style.py): same roles, different cycle. Default = tol-muted (the paper's)"),
                 ha="center", va="top", fontsize=14)
    _save(fig, out_dir, "00_style_palette_options.png", index,
          "Synthetic swatch sheet: each row is one palette option applied to the same four role sets "
          "(the cycle; the six arms + reference/stop roles; the Tier-3 real/mock/flagship convention; the "
          "Tier-1/2 null-histogram convention). Data are synthetic Gaussians.")


# ---------------------------------------------------------------------------------------------
# 01 protocol overview
# ---------------------------------------------------------------------------------------------
def fig_overview(out_dir, index):
    tiers = [
        ("Tier 0", "observation", "catalogue $\\rightarrow$ bandpowers,\nB-modes, E-maps with the\nmaster's own functions\n(bit-identity gate)\n$\\rightarrow$ baked stores per arm", R("reference")),
        ("Tier 1", "data checks", "2-pt in distribution\n(kNN / Mahalanobis)\nB-mode null ($\\chi^2$ vs mock-B)\nE-map summaries (250-d)\npower on variate clouds", S.arm_colour("nla_m", PAL)),
        ("Tier 2", "misspecification", "5-encoder mean-$p$\n(kNN, recalibrated)\ncross-encoder KL\n$\\rightarrow$ bias tables:\n$P(|z| > t)$ per bin", R("pass")),
        ("Tier 3", "blind sampling", "6 arms $\\times$ 5 repeats\n$\\times$ 2 priors\npooled posterior\nper arm (final)\nstandardised plots ($\\sigma$)", S.arm_colour("nla_z", PAL)),
        ("Tier 4", "notebook", "Sections 1--7 blind\nSection 8 gated:\n\\texttt{UNBLIND = False}\n$\\rightarrow$ physical contours,\npooled first", R("stop")),
    ]
    with S.context(PAL, font_size=12):
        fig, ax = plt.subplots(figsize=(15, 5.6))
        ax.set_xlim(0, 11.2); ax.set_ylim(0, 5.4); ax.axis("off")
        x0, w, h, y = 0.15, 2.0, 3.5, 0.95
        for i, (tier, sub, body, col) in enumerate(tiers):
            x = x0 + i * (w + 0.2)
            ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc="white", ec=col, lw=1.8))
            ax.add_patch(patches.FancyBboxPatch((x, y + h - 0.8), w, 0.8, boxstyle="round,pad=0.02,rounding_size=0.08", fc=col, ec=col, lw=1.8))
            ax.text(x + w / 2, y + h - 0.4, T(f"{tier}: {sub}"), ha="center", va="center", color="white", fontsize=12, weight="bold")
            ax.text(x + w / 2, y + (h - 0.8) / 2 + 0.05, body, ha="center", va="center", fontsize=10.5, color="black", linespacing=1.55)
            if i < len(tiers) - 1:
                ax.annotate("", xy=(x + w + 0.2, y + h / 2), xytext=(x + w, y + h / 2), arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2))
        ax.text(0.15, 5.1, T("KiDS-Legacy unblinding protocol (multifidelity SBI)"), fontsize=15, weight="bold")
        ax.text(0.15, 4.72, T("Pre-registered stop rules: Tier 1 p < 0.01 or B-mode PTE < 0.01 → stop;  Tier 2 mean-p < 0.01 or KL in the top bin → do not unblind;  Tier 3 has no stop rule (shapes only)."),
                fontsize=9.5)
        ax.text(0.15, 0.45, T("Blind rule: no posterior location or width of any observation label is computed or read before Tier 4. Raw posteriors live only in the blind store; a PreToolUse guard denies every other access."),
                fontsize=9.5, color=R("reference"), style="italic")
    _save(fig, out_dir, "01_protocol_overview.png", index,
          "The five tiers of the protocol, left to right, with the pre-registered stop rules and the blind rule.")


# ---------------------------------------------------------------------------------------------
# 02 tier 0 fidelity
# ---------------------------------------------------------------------------------------------
def fig_tier0(out_dir, index):
    ident = json.load(open(ART / "GATE_identity_prodgeom_f64_fidelity.json"))
    g0b = json.load(open(ART / "GATE0b_cluster_prodgeom_fidelity.json"))
    per = g0b["per_dataset"]; floor = g0b.get("floor", {}) or {}
    keys = [k for k in per if k in floor and np.isfinite(per[k]) and floor[k] > 0]
    shorten = lambda k: k.replace("_lmin56_lcut1400", "").replace("_lmin56_lcut1024", "-1024")   # noqa: E731
    with S.context(PAL, font_size=14):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.1, 2.4]})
        fig.subplots_adjust(wspace=0.35)
        ax = axes[0]
        ik = list(ident["per_dataset"])
        ax.barh(range(len(ik)), [max(ident["per_dataset"][k], 1e-16) for k in ik], color=R("pass"))
        ax.set_yticks(range(len(ik))); ax.set_yticklabels([T(shorten(k)) for k in ik], fontsize=9)
        ax.set_xscale("log"); ax.set_xlim(1e-17, 1); ax.invert_yaxis()
        ax.set_xlabel(T("relative rms, observation vs master (float64)"))
        S.panel_label(ax, "(a)", x=0.85)
        ax = axes[1]
        x = np.arange(len(keys))
        ax.bar(x - 0.2, [per[k] for k in keys], 0.4, color=S.arm_colour("nla_m", PAL), label=T("observation vs master mock"))
        ax.bar(x + 0.2, [floor[k] for k in keys], 0.4, color=R("mock"), label=T("half-ulp float32 jitter floor"))
        ax.plot(x, [3 * floor[k] for k in keys], "_", color=R("stop"), ms=16, mew=2, label=T("gate: 3 × floor"))
        ax.set_xticks(x); ax.set_xticklabels([T(shorten(k).replace("/", " ")) for k in keys], fontsize=8, rotation=30, ha="right")
        ax.set_yscale("log"); ax.set_ylabel(T("relative rms (float32 store)"))
        ax.legend(loc="upper left", fontsize=11)
        S.panel_label(ax, "(b)")
    _save(fig, out_dir, "02_tier0_processing_fidelity.png", index,
          "Tier 0, the observation builder reproduces the simulator's own processing. (a) Identity gate on the "
          "float64 catalogue at nside 1024: all 13 datasets bit-identical (relative rms 0). (b) Gate 0b on the "
          "cluster catalogue stored as float32: every dataset sits at its half-ulp storage floor, below the "
          "3x-floor gate. PASS.")


# ---------------------------------------------------------------------------------------------
# 03-06 tier 1
# ---------------------------------------------------------------------------------------------
def fig_tier1_bandpowers(res, out_dir, index):
    labels = list(res)
    with S.context(PAL, font_size=14):
        fig, axes = plt.subplots(1, len(labels), figsize=(5.6 * len(labels) + 1.2, 6.2), squeeze=False)
        fig.subplots_adjust(wspace=0.35)
        for k, (ax, lab) in enumerate(zip(axes[0], labels)):
            r = res[lab]["twopoint"]
            z = np.asarray(r["robust_z"])[0]           # (21, 8)
            im = ax.imshow(z, cmap=S.DIVERGING_CMAP, vmin=-3, vmax=3, aspect="auto")
            ax.set_yticks(range(21)); ax.set_yticklabels([T(s) for s in r["spectrum_labels"]], fontsize=8)
            ax.set_xticks(range(z.shape[1])); ax.set_xticklabels([f"{i+1}" for i in range(z.shape[1])], fontsize=10)
            ax.set_xlabel(T("bandpower")); ax.set_ylabel(T("tomographic pair") if k == 0 else "")
            ax.set_title(T(f"label {lab}:  kNN $p$ = {r['knn']['p'][0]:.2f},  Mahalanobis $p$ = {r['mahalanobis']['p'][0]:.2f}"),
                         fontsize=12, color=LC(lab))
            S.panel_label(ax, f"({'ab'[k]})", x=-0.32, y=1.04, color="black")
        cb = fig.colorbar(im, ax=axes[0], fraction=0.025, pad=0.02)
        cb.set_label(T("robust $z$ of the observed EE bandpower vs the nla\\_m mock cloud"), fontsize=11)
    _save(fig, out_dir, "03_tier1a_bandpowers_vs_cloud.png", index,
          "Tier 1a, two-point statistics in distribution? Robust z of each observed EE bandpower against the nla_m "
          "mock cloud, and the whitened 168-d kNN / Mahalanobis p-values against the held-out cosmology half "
          "(stop rule p < 0.01). (a) label T, (b) label S.")


def fig_tier1_bmodes(res, out_dir, index):
    labels = list(res)
    with S.context(PAL, font_size=14):
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})
        fig.subplots_adjust(wspace=0.22)
        ax = axes[0]
        d = res[labels[0]]["bmodes"]["dim"]
        x = np.linspace(100, 300, 400)
        ax.plot(x, stats.chi2.pdf(x, d), color=R("reference"), lw=1.4, label=T(f"analytic $\\chi^2$({d}), guide only"))
        for lab in labels:
            b = res[lab]["bmodes"]
            ax.axvline(b["chi2_vs_mockmean"][0], color=LC(lab), lw=2.4,
                       label=T(f"label {lab}: $\\chi^2$ = {b['chi2_vs_mockmean'][0]:.0f}, PTE = {b['pte_empirical'][0]:.2f}"))
        b = res[labels[0]]["bmodes"]
        ax.axvline(b["null_chi2_median"], color="black", ls=":", lw=1.2, label=T(f"mock null median ({b['null_chi2_median']:.0f}) / 95\\% ({b['null_chi2_p95']:.0f})"))
        ax.axvline(b["null_chi2_p95"], color="black", ls=":", lw=1.2)
        ax.axvspan(b["null_chi2_p95"], 300, color=R("stop"), alpha=0.10)
        ax.set_xlim(100, 300); ax.set_ylim(0, stats.chi2.pdf(d - 2, d) * 1.9); ax.set_yticks([])
        ax.set_xlabel(T("$\\chi^2$ of the 168 BB bandpowers (mock-B mean and covariance)"))
        ax.legend(loc="upper right", fontsize=10.5)
        S.panel_label(ax, "(a)")
        ax = axes[1]
        for lab in labels:
            pa = res[lab]["bmodes"]["per_auto"]
            ax.plot(range(len(pa)), [p["pte_empirical"][0] for p in pa], "o-", lw=2, ms=7, color=LC(lab), label=T(f"label {lab}"))
            ax.set_xticks(range(len(pa))); ax.set_xticklabels([T(p["spectrum"]) for p in pa])
        ax.axhline(0.01, color=R("stop"), ls="--", lw=1.4, label=T("stop rule (PTE < 0.01)"))
        ax.set_ylim(0, 1.05); ax.set_ylabel(T("empirical PTE")); ax.set_xlabel(T("auto-spectrum (bin $i$--$i$)"))
        ax.legend(loc="lower right", fontsize=10.5)
        S.panel_label(ax, "(b)")
    _save(fig, out_dir, "04_tier1b_bmode_null.png", index,
          "Tier 1b, B-mode null test (KiDS-Legacy convention PTE > 0.01). (a) Observed chi^2 of the 168 BB bandpowers "
          "against the held-out mock-B distribution; the empirical PTE is the record (the analytic curve is a "
          "guide only, Hartlap is unreachable at this dimension). (b) Empirical PTE per tomographic auto-spectrum.")


def fig_tier1_emap(res, out_dir, index):
    labels = list(res)
    with S.context(PAL, font_size=14):
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})
        fig.subplots_adjust(wspace=0.22)
        ax = axes[0]
        null = np.asarray(res[labels[0]]["emap"]["full"]["knn"]["null_scores"])
        lo, hi = np.quantile(null, [0.001, 0.995])
        ax.hist(np.clip(null, lo, hi), bins=60, range=(lo, hi), color=R("mock"), alpha=0.75, density=True,
                label=T(f"held-out mocks ($n$ = {len(null):,}; tail beyond 99.5\\% clipped)"))
        for lab in labels:
            e = res[lab]["emap"]["full"]["knn"]
            ax.axvline(e["obs"][0], color=LC(lab), lw=2.4, label=T(f"label {lab}: kNN $p$ = {e['p'][0]:.2f}"))
        q99 = np.quantile(null, 0.99)
        ax.axvline(q99, color=R("stop"), ls="--", lw=1.4, label=T("stop rule ($p$ < 0.01)"))
        ax.axvspan(q99, hi, color=R("stop"), alpha=0.10)
        ax.set_xlim(lo, hi); ax.set_xlabel(T("whitened kNN distance ($k$ = 10), 250-d E-map summary space")); ax.set_yticks([])
        ax.legend(fontsize=10.5)
        S.panel_label(ax, "(a)", x=0.92)
        ax = axes[1]
        groups = list(res[labels[0]]["emap"]["groups"])
        w = 0.8 / len(labels)
        for i, lab in enumerate(labels):
            g = res[lab]["emap"]["groups"]
            ax.bar(np.arange(len(groups)) + (i - (len(labels) - 1) / 2) * w, [g[k]["knn"]["p"][0] for k in groups], w,
                   color=LC(lab), label=T(f"label {lab}"))
        ax.axhline(0.01, color=R("stop"), ls="--", lw=1.4)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([T(f"{k}\n({res[labels[0]]['emap']['groups'][k]['dim']}-d)") for k in groups], fontsize=9.5)
        ax.set_ylim(0, 1.05); ax.set_ylabel(T("kNN $p$ per summary group"))
        ax.legend(fontsize=10.5)
        S.panel_label(ax, "(b)")
    _save(fig, out_dir, "05_tier1c_emap_ood.png", index,
          "Tier 1c, are the E-mode maps in distribution before any CNN sees them? (a) Whitened kNN distance of the "
          "250-d pre-CNN E-map summary vector (moments, PDF quantiles, peaks/voids, radial power, bin correlations) "
          "against the held-out mocks, stop rule p < 0.01. (b) The same test per summary group (diagnostic only).")


def fig_tier1_power(res, out_dir, index):
    lab = next((l for l in res if "power" in res[l]), None)
    if lab is None:
        return
    pw = res[lab]["power"]
    names = list(pw)
    pretty = {"nla": "NLA (no mass dep.)", "nla_z": "NLA-$z$", "nla_m_vd": "variable depth", "gb1p0": "fixed $b_g$ = 1.0", "gb1p3": "fixed $b_g$ = 1.3"}
    cyc = S.palette(PAL)
    series = [("bandpowers", "2-pt kNN", S.arm_colour("nla_m", PAL)), ("bandpowers", "2-pt Mahalanobis", S.arm_colour("nla_m_nobgp", PAL)),
              ("emap", "E-map kNN", R("pass")), ("emap", "E-map Mahalanobis", cyc[2])]
    with S.context(PAL, font_size=14):
        fig, ax = plt.subplots(figsize=(10, 4.8))
        x = np.arange(len(names)); w = 0.2
        for j, (which, lab_s, col) in enumerate(series):
            key = "auroc_knn" if "kNN" in lab_s else "auroc_mahalanobis"
            ax.bar(x + (j - 1.5) * w, [pw[n][which]["full"][key] for n in names], w, color=col, label=T(lab_s))
        ax.axhline(0.5, color=R("reference"), ls=":", lw=1.4); ax.text(len(names) - 0.5, 0.52, T("chance"), color=R("reference"), fontsize=11, ha="right")
        ax.set_ylim(0, 1.05); ax.set_ylabel(T("AUROC (variate cloud vs in-distribution null)"))
        ax.set_xticks(x); ax.set_xticklabels([T(pretty.get(n, n)) for n in names])
        ax.legend(ncol=4, loc="upper right", fontsize=10.5)
    _save(fig, out_dir, "06_tier1_power_on_variates.png", index,
          "Tier-1 power, measured: AUROC of each statistic at separating a known-OOD mock cloud from the nla_m cloud "
          "(same held-out cosmologies on both sides). NLA and NLA-z are seen (0.90-0.95); variable depth and the "
          "fixed-b_g clouds sit at chance, as the Tier-2 study found for the compressor.")


# ---------------------------------------------------------------------------------------------
# 07-08 tier 2
# ---------------------------------------------------------------------------------------------
def _load_rows(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def fig_tier2_detectors(rows, scores, out_dir, index):
    var = rows["variate"].astype(str)
    idm = np.array([("nla_m" == v) or v.endswith("nla_m") or v == "in_dist" for v in var])
    if not idm.any():
        idm = np.array([("vd" not in v and "gb" not in v and "nla_z" not in v and v.endswith("nla_m")) for v in var])
    edges_mp = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    with S.context(PAL, font_size=14):
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8)); fig.subplots_adjust(wspace=0.16)
        ax = axes[0]
        ax.hist(rows["meanp"][idm][np.isfinite(rows["meanp"][idm])], bins=40, range=(0, 1), color=R("mock"), alpha=0.75, density=True, label=T("in-distribution mocks"))
        for e in edges_mp[1:-1]:
            ax.axvline(e, color="black", ls=":", lw=0.9)
        for lab, sc in scores.items():
            v = sc.get("meanp_recalibrated", sc.get("meanp_raw"))
            ax.axvline(v, color=LC(lab), lw=2.4, label=T(f"label {lab}: mean-$p$ = {v:.2f}"))
        ax.axvspan(0, 0.01, color=R("stop"), alpha=0.18); ax.axvspan(0.01, 0.05, color=R("stop"), alpha=0.08)
        ax.set_xlim(0, 1); ax.set_yticks([]); ax.set_xlabel(T("5-encoder mean kNN $p$ (recalibrated; low = out of support)"))
        ax.legend(loc="upper right", fontsize=10.5)
        S.panel_label(ax, "(a)")
        ax = axes[1]
        kl = rows["kl"][idm]; kl = kl[np.isfinite(kl)]
        ax.hist(kl, bins=50, range=(0, 1.5), color=R("mock"), alpha=0.75, density=True, label=T("in-distribution mocks"))
        for lab, sc in scores.items():
            if "kl" in sc:
                ax.axvline(sc["kl"], color=LC(lab), lw=2.4, label=T(f"label {lab}: KL = {sc['kl']:.2f}"))
        p95 = float(np.quantile(kl, 0.95))
        ax.axvline(p95, color=R("stop"), ls="--", lw=1.4, label=T(f"top bin (≥ null 95\\%: {p95:.2f})"))
        ax.axvspan(p95, 1.5, color=R("stop"), alpha=0.10)
        ax.set_xlim(0, 1.5); ax.set_yticks([]); ax.set_xlabel(T("symmetric KL between the 5 encoders' posteriors (high = OOD)"))
        ax.legend(fontsize=10.5)
        S.panel_label(ax, "(b)", x=0.92)
    _save(fig, out_dir, "07_tier2_detectors.png", index,
          "Tier 2, the two model-misspecification detectors read against their in-distribution null. (a) Summary-space "
          "kNN: 5-encoder mean p, bins stop < 0.01 | flag < 0.05 | mild < 0.2 | in distribution. (b) Cross-encoder "
          "symmetric KL, the only axis that sees galaxy-bias misspecification; the top bin starts at the null 95th "
          "percentile.")


def fig_tier2_bias(rows, scores, out_dir, index):
    tab = json.load(open(ART / "tier2" / "bias_tables.json"))
    edges = {}
    for det in ("meanp", "kl"):          # the SAME bins the readings (obs_reading_<label>.md) use
        b = tab[det]["bins"]
        edges[det] = [b[0]["lo"]] + [x["hi"] for x in b]
    var = rows["variate"].astype(str)
    idm = np.array([v.endswith("nla_m") for v in var])
    with S.context(PAL, font_size=14):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.4)); fig.subplots_adjust(wspace=0.22)
        for k, (ax, det) in enumerate(zip(axes, ("meanp", "kl"))):
            x = rows[det]; e = edges[det]
            cen, p05, p1, pid = [], [], [], []
            for lo, hi in zip(e[:-1], e[1:]):
                m = np.isfinite(x) & (x >= lo) & (x < hi)
                zs = np.abs(rows["z_S8"][m]); zi = np.abs(rows["z_S8"][m & idm])
                cen.append(0.5 * (lo + hi)); p05.append(np.mean(zs > 0.5) if zs.size else np.nan)
                p1.append(np.mean(zs > 1) if zs.size else np.nan); pid.append(np.mean(zi > 0.5) if zi.size else np.nan)
            xx = np.arange(len(cen))
            ax.plot(xx, p05, "o-", lw=2, ms=7, color=S.arm_colour("nla_m", PAL), label=T("$P(|z_{S_8}| > 0.5)$, all variates"))
            ax.plot(xx, p1, "s-", lw=2, ms=7, color=R("pass"), label=T("$P(|z_{S_8}| > 1)$, all variates"))
            ax.plot(xx, pid, "o--", lw=2, ms=7, color=R("reference"), label=T("$P(|z_{S_8}| > 0.5)$, in-distribution rows (calibrated: 0.62)"))
            ax.set_xticks(xx); ax.set_xticklabels([T(f"[{lo:.2g}, {hi:.2g})") for lo, hi in zip(e[:-1], e[1:])], fontsize=9.5, rotation=20)
            for lab, sc in scores.items():
                v = sc.get("meanp_recalibrated", sc.get("meanp_raw")) if det == "meanp" else sc.get("kl")
                if v is None:
                    continue
                b = int(np.clip(np.searchsorted(e, v, side="right") - 1, 0, len(cen) - 1))
                ax.axvline(b, color=LC(lab), lw=2.4, alpha=0.8, label=T(f"label {lab} reads bin {b}"))
            ax.set_ylim(0, 1.0); ax.set_ylabel(T("$P(|z_{S_8}| > t)$"))
            ax.set_xlabel(T("mean-$p$ bin (low = OOD)" if det == "meanp" else "KL bin (high = OOD)"))
            ax.legend(fontsize=9.5, loc="lower left" if det == "meanp" else "upper left", ncol=1)
            S.panel_label(ax, f"({'ab'[k]})", x=0.92 if det == "meanp" else 0.02, y=0.96 if det == "meanp" else 0.5)
    _save(fig, out_dir, "08_tier2_bias_tables.png", index,
          "Tier 2, what a detector reading buys: P(|z| > t) for S8 from 40,499 mock posteriors (5 encoders x 7 "
          "variates), binned by (a) the kNN mean p and (b) the cross-encoder KL, with the bin each label reads. "
          "The in-distribution reference is 0.62 for a calibrated posterior.")


# ---------------------------------------------------------------------------------------------
# 09-12 tier 3 (copies of the standardised figures made by scripts/plot_blind_posteriors.py)
# ---------------------------------------------------------------------------------------------
TIER3 = [("plotB_S", "09_tier3_widths_vs_matched_mocks_S",
          "Tier 3, Plot B: the flagship (pooled) posterior of label S in its own standardised frame against "
          "near-fiducial matched mocks standardised with the flagship's width and their own mean. Sizes and "
          "degeneracy directions are comparable; the location is hidden. Pilot caveat: the S pool here has "
          "4000 draws (2 x 2000), hence the noisy w row; the production pool has 5 x 25k."),
         ("plotD_S_nla_m", "10_tier3_pooled_vs_repeats_S",
          "Tier 3, Plot D: the repeats of the flagship arm in the frame of the POOLED posterior (seed spread "
          "in sigma of the pooled)."),
         ("plotBprime_T", "11_tier3_arm_consistency_T",
          "Tier 3, Plot B': every arm of label T in the flagship frame (mean and width of the flagship), so "
          "inter-arm offsets read in flagship sigma. The no-BGP arm sits ~2 sigma high in S8, the expected imprint "
          "of a fixed galaxy bias."),
         ("plotA_S", "12_tier3_self_standardised_S",
          "Tier 3, Plot A: every arm of label S standardised to N(0,1) per parameter (shapes and degeneracy "
          "directions only; maximally blind).")]


def copy_tier3(plots_dir: Path, out_dir, index):
    for src, dst, cap in TIER3:
        for ext in ("png", "pdf"):
            p = plots_dir / f"{src}.{ext}"
            if p.exists():
                shutil.copyfile(p, out_dir / f"{dst}.{ext}")
                if ext == "png":
                    index.append(f"{dst}.png"); CAPTIONS[f"{dst}.png"] = cap; print(f"  copied {dst}.png")


def _readme(out: Path, index):
    lines = ["# Unblinding figure set", "",
             "Style: `src/viz/style.py` (see `.claude/runs/eval-and-viz/unblinding-prep/artifacts/STYLE_GUIDE.md (copy next to the figures: /data/alex/unblinding/figures/STYLE_GUIDE.md)`), palette option `%s`. " % PAL +
             "Controls: T = GLASS b_g=1 catalogue (known-OOD for the Gower nla_m cloud), S = held-out flagship N-body mock "
             "(in distribution). All Tier-3 panels are in sigma units of a standardised posterior; no physical "
             "posterior location appears anywhere. Every figure is also written as PDF.", ""]
    for n in index:
        lines.append(f"- **{n}** -- {CAPTIONS.get(n, '')}")
    (out / "README.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    global PAL
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="/data/alex/unblinding/figures")
    ap.add_argument("--palette", default=S.DEFAULT_PALETTE, choices=list(S.PALETTES))
    ap.add_argument("--palette-options", action="store_true", help="also render the Tier-1/2 panels in every palette + the swatch sheet")
    ap.add_argument("--tier1", nargs="*", default=["T=/data/alex/unblinding/tier1_T", "S=/data/alex/unblinding/tier1_S"])
    ap.add_argument("--obs-score", nargs="*", default=["T=/data/alex/unblinding/cluster_fetch/obs_gate0b_sc8a1/obs_score_T.json",
                                                       "S=/data/alex/unblinding/cluster_fetch/obs_strip_S_sc8a1/obs_score_S.json"])
    ap.add_argument("--rows", default="/data/alex/unblinding/localroots/tier2_rows.npz")
    ap.add_argument("--plots-dir", default="/data/alex/unblinding/plots_pilot")
    ap.add_argument("--also-copy-to", default=str(ART / "figures"))
    args = ap.parse_args(argv)
    PAL = args.palette
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    res = {}
    for tok in args.tier1:
        lab, d = tok.split("=", 1)
        p = Path(d) / "tier1_results.json"
        if p.exists():
            res[lab] = json.load(open(p))
    scores = {}
    for tok in args.obs_score:
        lab, p = tok.split("=", 1)
        if os.path.exists(p):
            scores[lab] = json.load(open(p))
    rows = _load_rows(args.rows) if os.path.exists(args.rows) else None

    def render(out_dir, index, with_swatch, with_tier3):
        if with_swatch:
            fig_swatches(out_dir, index)
        fig_overview(out_dir, index)
        fig_tier0(out_dir, index)
        if res:
            fig_tier1_bandpowers(res, out_dir, index); fig_tier1_bmodes(res, out_dir, index)
            fig_tier1_emap(res, out_dir, index); fig_tier1_power(res, out_dir, index)
        if rows is not None:
            fig_tier2_detectors(rows, scores, out_dir, index); fig_tier2_bias(rows, scores, out_dir, index)
        if with_tier3:
            copy_tier3(Path(args.plots_dir), out_dir, index)

    index = []
    render(out, index, with_swatch=args.palette_options, with_tier3=True)
    _readme(out, index)
    if args.palette_options:
        chosen = PAL
        for name in S.PALETTES:
            PAL = name
            sub = out / "palette_options" / name; sub.mkdir(parents=True, exist_ok=True)
            print(f"[palette option {name}]")
            render(sub, [], with_swatch=False, with_tier3=False)
        PAL = chosen
    if args.also_copy_to:
        dst = Path(args.also_copy_to); dst.mkdir(parents=True, exist_ok=True)
        for n in index + ["README.md"]:
            shutil.copyfile(out / n, dst / n)
            if (out / n).with_suffix(".pdf").exists():
                shutil.copyfile((out / n).with_suffix(".pdf"), (dst / n).with_suffix(".pdf"))
    print(f"{len(index)} figures under {out} (mirrored to {args.also_copy_to})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
