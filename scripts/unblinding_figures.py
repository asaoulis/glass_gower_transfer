#!/usr/bin/env python
"""The unblinding figure set: one numbered folder that walks through every tier of the protocol
with the numbers measured on the two mock-as-real controls (T = GLASS b_g=1 catalogue, known-OOD
for the Gower nla_m cloud; S = held-out flagship mock, in-distribution).

    PYTHONPATH=. python scripts/unblinding_figures.py --out-dir /data/alex/unblinding/figures \\
        [--tier1 T=/data/alex/unblinding/tier1_T S=/data/alex/unblinding/tier1_S] \\
        [--obs-score T=<obs_score_T.json> S=<obs_score_S.json>] [--plots-dir /data/alex/unblinding/plots_pilot]

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
ART = REPO / ".claude/runs/eval-and-viz/unblinding-prep/artifacts"

# Okabe-Ito, colour-blind safe
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
     "purple": "#CC79A7", "sky": "#56B4E9", "grey": "#8C8C8C", "ink": "#222222"}
LABEL_COL = {"T": C["orange"], "S": C["blue"]}
LABEL_NAME = {"T": "T  (GLASS $b_g$=1 catalogue: known-OOD control)",
              "S": "S  (held-out N-body mock: in-distribution control)"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def _save(fig, out_dir: Path, name: str, index: list):
    p = out_dir / name
    fig.savefig(p)
    plt.close(fig)
    index.append(name)
    print(f"  wrote {p.name}")


# ---------------------------------------------------------------------------------------------
# 01 protocol overview
# ---------------------------------------------------------------------------------------------
def fig_overview(out_dir, index):
    fig, ax = plt.subplots(figsize=(15, 5.4))
    ax.set_xlim(0, 11.2); ax.set_ylim(0, 5.2); ax.axis("off")
    tiers = [
        ("Tier 0\nobservation", "catalogue → bandpowers,\nB-modes, E-maps with the\nmaster's own functions\n(bit-identity gate)\n→ baked stores per arm", C["grey"]),
        ("Tier 1\ndata checks", "2-pt in distribution\n(kNN / Mahalanobis)\nB-mode null (χ² vs mock-B)\nE-map summaries (250-d)\npower on variate clouds", C["blue"]),
        ("Tier 2\nmisspecification", "5-encoder mean-p\n(kNN, recalibrated)\ncross-encoder KL\n→ bias tables:\nP(|z| > t) per bin", C["green"]),
        ("Tier 3\nblind sampling", "6 arms × 5 repeats\n× 2 priors\nPOOLED posterior\nper arm (final)\nstandardised plots (σ)", C["purple"]),
        ("Tier 4\nnotebook", "Sections 1–7 blind\nSection 8 gated:\nUNBLIND = False\n→ physical contours,\npooled first", C["red"]),
    ]
    x0, w, h, y = 0.15, 2.0, 3.4, 0.9
    for i, (title, body, col) in enumerate(tiers):
        x = x0 + i * (w + 0.2)
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                            fc="white", ec=col, lw=1.8))
        ax.add_patch(patches.FancyBboxPatch((x, y + h - 0.75), w, 0.75, boxstyle="round,pad=0.02,rounding_size=0.08",
                                            fc=col, ec=col, lw=1.8))
        ax.text(x + w / 2, y + h - 0.37, title, ha="center", va="center", color="white", fontsize=10.5, weight="bold")
        ax.text(x + w / 2, y + (h - 0.75) / 2 + 0.05, body, ha="center", va="center", fontsize=9, color=C["ink"], linespacing=1.5)
        if i < len(tiers) - 1:
            ax.annotate("", xy=(x + w + 0.2, y + h / 2), xytext=(x + w, y + h / 2),
                        arrowprops=dict(arrowstyle="-|>", color=C["ink"], lw=1.2))
    ax.text(0.15, 4.85, "KiDS-Legacy unblinding protocol (multifidelity SBI)", fontsize=13, weight="bold", color=C["ink"])
    ax.text(0.15, 4.5, "Pre-registered stop rules: Tier 1 p < 0.01 or B-mode PTE < 0.01 → stop;  Tier 2 mean-p < 0.01 or KL in the top bin → do not unblind;  "
                       "Tier 3 has no stop rule (shapes only).", fontsize=8.8, color=C["ink"])
    ax.text(0.15, 0.45, "Blind rule: no posterior location or width of any observation label is computed or read before Tier 4. "
                        "Raw posteriors live only in the blind store; a PreToolUse guard denies every other access.",
            fontsize=8.8, color=C["grey"], style="italic")
    _save(fig, out_dir, "01_protocol_overview.png", index)


# ---------------------------------------------------------------------------------------------
# 02 tier 0 fidelity
# ---------------------------------------------------------------------------------------------
def fig_tier0(out_dir, index):
    ident = json.load(open(ART / "GATE_identity_prodgeom_f64_fidelity.json"))
    g0b = json.load(open(ART / "GATE0b_cluster_prodgeom_fidelity.json"))
    per = g0b["per_dataset"]; floor = g0b.get("floor", {}) or {}
    keys = [k for k in per if k in floor and np.isfinite(per[k]) and floor[k] > 0]
    short = [k.replace("_lmin56_lcut1400", "").replace("_lmin56_lcut1024", "-1024").replace("/", "\n") for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.15, 2.6]})
    ax = axes[0]
    ik = list(ident["per_dataset"])
    ax.barh(range(len(ik)), [max(ident["per_dataset"][k], 1e-16) for k in ik], color=C["green"])
    ax.set_yticks(range(len(ik))); ax.set_yticklabels([k.replace("_lmin56_lcut1400", "").replace("_lmin56_lcut1024", "-1024") for k in ik], fontsize=7)
    ax.set_xscale("log"); ax.set_xlim(1e-17, 1); ax.invert_yaxis()
    ax.set_xlabel("relative rms (observation vs master)")
    ax.set_title("Identity gate (float64 catalogue, nside 1024)\nall 13 datasets: rel. rms = 0, array_equal", fontsize=9.5)
    ax = axes[1]
    x = np.arange(len(keys))
    ax.bar(x - 0.2, [per[k] for k in keys], 0.4, color=C["blue"], label="observation vs master mock")
    ax.bar(x + 0.2, [floor[k] for k in keys], 0.4, color=C["grey"], label="half-ulp float32 jitter floor")
    ax.plot(x, [3 * floor[k] for k in keys], "_", color=C["red"], ms=14, mew=1.6, label="gate: 3 × floor")
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=6.5)
    ax.set_yscale("log"); ax.set_ylabel("relative rms")
    ax.set_title("GATE 0b: cluster catalogue stored as float32 — every dataset at its storage floor  →  PASS", fontsize=9.5)
    ax.legend(loc="upper left", frameon=False)
    fig.suptitle("Tier 0 — the observation builder reproduces the simulator's own processing", fontsize=12, y=1.02)
    _save(fig, out_dir, "02_tier0_processing_fidelity.png", index)


# ---------------------------------------------------------------------------------------------
# 03-06 tier 1
# ---------------------------------------------------------------------------------------------
def fig_tier1_bandpowers(res, out_dir, index):
    labels = list(res)
    fig, axes = plt.subplots(1, len(labels), figsize=(5.2 * len(labels) + 1, 5.6), squeeze=False)
    for ax, lab in zip(axes[0], labels):
        r = res[lab]["twopoint"]
        z = np.asarray(r["robust_z"])[0]           # (21, 8)
        im = ax.imshow(z, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
        ax.set_yticks(range(21)); ax.set_yticklabels(r["spectrum_labels"], fontsize=7)
        ax.set_xticks(range(z.shape[1])); ax.set_xticklabels([f"b{i+1}" for i in range(z.shape[1])], fontsize=8)
        ax.set_xlabel("bandpower"); ax.set_ylabel("tomographic pair" if ax is axes[0][0] else "")
        ax.set_title(f"label {lab}\nkNN p = {r['knn']['p'][0]:.2f}   Mahalanobis p = {r['mahalanobis']['p'][0]:.2f}", fontsize=10,
                     color=LABEL_COL.get(lab, C["ink"]))
        for s in ax.spines.values(): s.set_visible(True)
    cb = fig.colorbar(im, ax=axes[0], fraction=0.025, pad=0.02); cb.set_label("robust z of the observed EE bandpower vs the nla_m mock cloud")
    fig.suptitle("Tier 1a — two-point statistics in distribution?  (168-d whitened kNN / Mahalanobis vs the held-out cosmology half; pass: p > 0.01)", fontsize=11, y=0.98)
    _save(fig, out_dir, "03_tier1a_bandpowers_vs_cloud.png", index)


def fig_tier1_bmodes(res, out_dir, index):
    labels = list(res)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    d = res[labels[0]]["bmodes"]["dim"]
    x = np.linspace(100, 300, 400)
    ax.plot(x, stats.chi2.pdf(x, d), color=C["grey"], lw=1.2, label=f"analytic χ²({d}) (guide only)")
    for lab in labels:
        b = res[lab]["bmodes"]
        ax.axvline(b["chi2_vs_mockmean"][0], color=LABEL_COL[lab], lw=2.2,
                   label=f"label {lab}: χ² = {b['chi2_vs_mockmean'][0]:.0f}, empirical PTE = {b['pte_empirical'][0]:.2f}")
    b = res[labels[0]]["bmodes"]
    ax.axvline(b["null_chi2_median"], color=C["ink"], ls=":", lw=1, label=f"mock null median ({b['null_chi2_median']:.0f}) / 95% ({b['null_chi2_p95']:.0f})")
    ax.axvline(b["null_chi2_p95"], color=C["ink"], ls=":", lw=1)
    ax.axvspan(b["null_chi2_p95"], 300, color=C["red"], alpha=0.06)
    ax.set_xlim(100, 300); ax.set_ylim(0, stats.chi2.pdf(d - 2, d) * 1.9); ax.set_yticks([])
    ax.set_xlabel("χ² of the 168 BB bandpowers vs the mock-B mean and covariance")
    ax.set_title("observed χ² against the held-out mock-B distribution", fontsize=10)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    ax = axes[1]
    for i, lab in enumerate(labels):
        pa = res[lab]["bmodes"]["per_auto"]
        ax.plot(range(len(pa)), [p["pte_empirical"][0] for p in pa], "o-", color=LABEL_COL[lab], label=f"label {lab}")
        ax.set_xticks(range(len(pa))); ax.set_xticklabels([p["spectrum"] for p in pa])
    ax.axhline(0.01, color=C["red"], ls="--", lw=1, label="stop rule (PTE < 0.01)")
    ax.set_ylim(0, 1.05); ax.set_ylabel("empirical PTE"); ax.set_xlabel("auto-spectrum (bin i–i)")
    ax.set_title("per tomographic auto-spectrum", fontsize=10); ax.legend(frameon=False, loc="lower right")
    fig.suptitle("Tier 1b — B-mode null test (KiDS-Legacy convention: PTE > 0.01)", fontsize=11, y=1.02)
    _save(fig, out_dir, "04_tier1b_bmode_null.png", index)


def fig_tier1_emap(res, out_dir, index):
    labels = list(res)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    null = np.asarray(res[labels[0]]["emap"]["full"]["knn"]["null_scores"])
    lo, hi = np.quantile(null, [0.001, 0.995])
    ax.hist(np.clip(null, lo, hi), bins=60, range=(lo, hi), color=C["grey"], alpha=0.55, density=True,
            label=f"held-out mocks (n = {len(null):,}; tail beyond the 99.5th pct. clipped)")
    for lab in labels:
        e = res[lab]["emap"]["full"]["knn"]
        ax.axvline(e["obs"][0], color=LABEL_COL[lab], lw=2.2, label=f"label {lab}: kNN p = {e['p'][0]:.2f}")
    q99 = np.quantile(null, 0.99)
    ax.axvline(q99, color=C["red"], ls="--", lw=1, label="stop rule (p < 0.01)")
    ax.set_xlim(lo, hi); ax.set_xlabel("whitened kNN distance (k = 10) in the 250-d E-map summary space"); ax.set_yticks([])
    ax.set_title("pre-CNN E-map summaries: moments, PDF quantiles, peaks/voids, radial power, bin correlations", fontsize=9.5)
    ax.legend(frameon=False)
    ax = axes[1]
    groups = list(res[labels[0]]["emap"]["groups"])
    w = 0.8 / len(labels)
    for i, lab in enumerate(labels):
        g = res[lab]["emap"]["groups"]
        ax.bar(np.arange(len(groups)) + (i - (len(labels) - 1) / 2) * w, [g[k]["knn"]["p"][0] for k in groups], w,
               color=LABEL_COL[lab], label=f"label {lab}")
    ax.axhline(0.01, color=C["red"], ls="--", lw=1)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([f"{k}\n({res[labels[0]]['emap']['groups'][k]['dim']}-d)" for k in groups], fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("kNN p per summary group"); ax.set_title("per group (diagnostic, no stop rule)", fontsize=10)
    ax.legend(frameon=False)
    fig.suptitle("Tier 1c — are the E-mode maps in distribution before any CNN sees them?", fontsize=11, y=1.02)
    _save(fig, out_dir, "05_tier1c_emap_ood.png", index)


def fig_tier1_power(res, out_dir, index):
    lab = next((l for l in res if "power" in res[l]), None)
    if lab is None:
        return
    pw = res[lab]["power"]
    names = list(pw)
    pretty = {"nla": "NLA (no mass dep.)", "nla_z": "NLA-z", "nla_m_vd": "variable depth", "gb1p0": "fixed $b_g$=1.0", "gb1p3": "fixed $b_g$=1.3"}
    fig, ax = plt.subplots(figsize=(8.5, 4))
    x = np.arange(len(names)); w = 0.2
    series = [("bandpowers", "full", "2-pt kNN", C["blue"]), ("bandpowers", "full", "2-pt Mahalanobis", C["sky"]),
              ("emap", "full", "E-map kNN", C["green"]), ("emap", "full", "E-map Mahalanobis", C["orange"])]
    for j, (which, tag, lab_s, col) in enumerate(series):
        key = "auroc_knn" if "kNN" in lab_s else "auroc_mahalanobis"
        ax.bar(x + (j - 1.5) * w, [pw[n][which][tag][key] for n in names], w, color=col, label=lab_s)
    ax.axhline(0.5, color=C["grey"], ls=":", lw=1); ax.text(len(names) - 0.5, 0.515, "chance", color=C["grey"], fontsize=8, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("AUROC (variate cloud vs in-distribution null)")
    ax.set_xticks(x); ax.set_xticklabels([pretty.get(n, n) for n in names])
    ax.set_title("Tier 1 power, measured: how well each statistic separates a known-OOD mock cloud from the nla_m cloud\n"
                 "(same held-out cosmologies on both sides)", fontsize=10)
    ax.legend(frameon=False, ncol=4, loc="upper right")
    _save(fig, out_dir, "06_tier1_power_on_variates.png", index)


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
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2)); fig.subplots_adjust(wspace=0.18)
    edges_mp = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    ax = axes[0]
    ax.hist(rows["meanp"][idm][np.isfinite(rows["meanp"][idm])], bins=40, range=(0, 1), color=C["grey"], alpha=0.55, density=True, label="in-distribution mocks")
    for e in edges_mp[1:-1]:
        ax.axvline(e, color=C["ink"], ls=":", lw=0.8)
    for lab, sc in scores.items():
        v = sc.get("meanp_recalibrated", sc.get("meanp_raw"))
        ax.axvline(v, color=LABEL_COL[lab], lw=2.2, label=f"label {lab}: mean-p = {v:.2f}")
    ax.axvspan(0, 0.01, color=C["red"], alpha=0.12); ax.axvspan(0.01, 0.05, color=C["orange"], alpha=0.12)
    ax.set_xlim(0, 1); ax.set_yticks([]); ax.set_xlabel("5-encoder mean kNN p (recalibrated; low = out of support)")
    ax.set_title("detector 1: summary-space kNN\nbins: stop < 0.01 | flag < 0.05 | mild < 0.2 | in-distribution", fontsize=9.5)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8) if False else ax.legend(frameon=False, loc="upper center", fontsize=8)
    ax = axes[1]
    kl = rows["kl"][idm]; kl = kl[np.isfinite(kl)]
    ax.hist(kl, bins=50, range=(0, 1.5), color=C["grey"], alpha=0.55, density=True, label="in-distribution mocks")
    for lab, sc in scores.items():
        if "kl" in sc:
            ax.axvline(sc["kl"], color=LABEL_COL[lab], lw=2.2, label=f"label {lab}: KL = {sc['kl']:.2f}")
    p95 = float(np.quantile(kl, 0.95))
    ax.axvline(p95, color=C["red"], ls="--", lw=1, label=f"top bin (≥ null 95%: {p95:.2f})")
    ax.set_xlim(0, 1.5); ax.set_yticks([]); ax.set_xlabel("symmetric KL between the 5 encoders' posteriors (high = OOD)")
    ax.set_title("detector 2: encoder disagreement\n(the only axis that sees galaxy-bias misspecification)", fontsize=9.5)
    ax.legend(frameon=False)
    fig.suptitle("Tier 2 — model-misspecification detectors read against their in-distribution null", fontsize=11, y=1.02)
    _save(fig, out_dir, "07_tier2_detectors.png", index)


def fig_tier2_bias(rows, scores, out_dir, index):
    tab = json.load(open(ART / "tier2" / "bias_tables.json"))
    edges = {}
    for det in ("meanp", "kl"):          # the SAME bins the readings (obs_reading_<label>.md) use
        b = tab[det]["bins"]
        edges[det] = [b[0]["lo"]] + [x["hi"] for x in b]
    var = rows["variate"].astype(str)
    idm = np.array([v.endswith("nla_m") for v in var])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6)); fig.subplots_adjust(wspace=0.22)
    for ax, det in zip(axes, ("meanp", "kl")):
        x = rows[det]; e = edges[det]
        cen, p05, p1, pid = [], [], [], []
        for lo, hi in zip(e[:-1], e[1:]):
            m = np.isfinite(x) & (x >= lo) & (x < hi)
            zs = np.abs(rows["z_S8"][m]); zi = np.abs(rows["z_S8"][m & idm])
            cen.append(0.5 * (lo + hi)); p05.append(np.mean(zs > 0.5) if zs.size else np.nan)
            p1.append(np.mean(zs > 1) if zs.size else np.nan); pid.append(np.mean(zi > 0.5) if zi.size else np.nan)
        xx = np.arange(len(cen))
        ax.plot(xx, p05, "o-", color=C["blue"], label=r"P(|z$_{S_8}$| > 0.5), all variates pooled")
        ax.plot(xx, p1, "s-", color=C["green"], label=r"P(|z$_{S_8}$| > 1), all variates pooled")
        ax.plot(xx, pid, "o--", color=C["grey"], label=r"P(|z$_{S_8}$| > 0.5), in-distribution rows (calibrated: 0.62)")
        ax.set_xticks(xx); ax.set_xticklabels([f"[{lo:.2g}, {hi:.2g})" for lo, hi in zip(e[:-1], e[1:])], fontsize=7.5, rotation=20)
        for lab, sc in scores.items():
            v = sc.get("meanp_recalibrated", sc.get("meanp_raw")) if det == "meanp" else sc.get("kl")
            if v is None:
                continue
            b = int(np.clip(np.searchsorted(e, v, side="right") - 1, 0, len(cen) - 1))
            ax.axvline(b, color=LABEL_COL[lab], lw=2, alpha=0.7, label=f"label {lab} reads bin {b}")
        ax.set_ylim(0, 1.0); ax.set_ylabel("P(|z$_{S_8}$| > t)")
        ax.set_xlabel("mean-p bin (low = OOD)" if det == "meanp" else "KL bin (high = OOD)")
        ax.set_title("bias risk vs the kNN mean-p" if det == "meanp" else "bias risk vs the cross-encoder KL", fontsize=10)
        ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    fig.suptitle("Tier 2 — what a detector reading buys: P(|z| > t) for $S_8$ from 40,499 mock posteriors (5 encoders × 7 variates)", fontsize=11, y=1.02)
    _save(fig, out_dir, "08_tier2_bias_tables.png", index)


# ---------------------------------------------------------------------------------------------
# 09-11 tier 3 (copies of the standardised figures)
# ---------------------------------------------------------------------------------------------
def copy_tier3(plots_dir: Path, out_dir, index):
    wanted = [("plotB_S.png", "09_tier3_widths_vs_matched_mocks_S.png"),
              ("plotD_S_nla_m.png", "10_tier3_pooled_vs_repeats_S.png"),
              ("plotBprime_T.png", "11_tier3_arm_consistency_T.png"),
              ("plotA_S.png", "12_tier3_self_standardised_S.png")]
    for src, dst in wanted:
        p = plots_dir / src
        if p.exists():
            shutil.copyfile(p, out_dir / dst); index.append(dst); print(f"  copied {dst}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="/data/alex/unblinding/figures")
    ap.add_argument("--tier1", nargs="*", default=["T=/data/alex/unblinding/tier1_T", "S=/data/alex/unblinding/tier1_S"])
    ap.add_argument("--obs-score", nargs="*", default=["T=/data/alex/unblinding/cluster_fetch/obs_gate0b_sc8a1/obs_score_T.json",
                                                       "S=/data/alex/unblinding/cluster_fetch/obs_strip_S_sc8a1/obs_score_S.json"])
    ap.add_argument("--rows", default="/data/alex/unblinding/localroots/tier2_rows.npz")
    ap.add_argument("--plots-dir", default="/data/alex/unblinding/plots_pilot")
    ap.add_argument("--also-copy-to", default=str(ART / "figures"))
    args = ap.parse_args(argv)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    index = []
    fig_overview(out, index)
    fig_tier0(out, index)
    res = {}
    for tok in args.tier1:
        lab, d = tok.split("=", 1)
        p = Path(d) / "tier1_results.json"
        if p.exists():
            res[lab] = json.load(open(p))
    if res:
        fig_tier1_bandpowers(res, out, index); fig_tier1_bmodes(res, out, index)
        fig_tier1_emap(res, out, index); fig_tier1_power(res, out, index)
    scores = {}
    for tok in args.obs_score:
        lab, p = tok.split("=", 1)
        if os.path.exists(p):
            scores[lab] = json.load(open(p))
    if os.path.exists(args.rows):
        rows = _load_rows(args.rows)
        fig_tier2_detectors(rows, scores, out, index); fig_tier2_bias(rows, scores, out, index)
    copy_tier3(Path(args.plots_dir), out, index)
    with open(out / "README.md", "w") as fh:
        fh.write("# Unblinding figure set\n\nControls: T = GLASS b_g=1 catalogue (known-OOD for the Gower nla_m cloud), "
                 "S = held-out flagship N-body mock (in distribution). All Tier-3 panels are in σ units of a standardised "
                 "posterior; no physical posterior location appears anywhere.\n\n" + "\n".join(f"- {n}" for n in index) + "\n")
    if args.also_copy_to:
        dst = Path(args.also_copy_to); dst.mkdir(parents=True, exist_ok=True)
        for n in index + ["README.md"]:
            shutil.copyfile(out / n, dst / n)
    print(f"{len(index)} figures under {out} (mirrored to {args.also_copy_to})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
