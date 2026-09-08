#!/usr/bin/env python
"""Generate notebooks/kids_unblinding.ipynb -- the manual unblinding notebook (Tier 4).

The notebook ORCHESTRATES; every computation lives in src/observation, src/blind and the scripts.
It takes a LABEL and a set of directories (parameters cell) and steps through:

  1. provenance & configuration (catalogue fingerprint, treatment knobs, geometry, git rev)
  2. catalogue-level: per-bin counts, n(z) (if Z_TRUE present), mean e1/e2 removed by the estimator
  3. 2-pt: publication-style EE bandpowers (21 spectra) with the nla_m mock-cloud band + Tier-1a numbers
  4. B-modes: BB bandpowers + PTE table (Tier-1b)
  5. map statistics: E-map summary groups vs the mock cloud (peaks / voids / PDF / power) + kNN p (Tier-1c)
  6. OOD dashboard: Tier-1 p-values + Tier-2 detector scores read against the bias tables
  7. STANDARDISED posteriors (Plots A / B / B'; corner battery) -- the last blind section
  8. UNBLIND = False  gate: the ONLY cell that loads raw samples and draws physical contours
     (all arms x priors). The agent never sets it True; guard_blind blocks the underlying read anyway.

Regenerate with:  PYTHONPATH=. python scripts/build_unblinding_notebook.py [--out notebooks/kids_unblinding.ipynb]
Execute (mock only, agent):  jupyter nbconvert --to notebook --execute notebooks/kids_unblinding.ipynb \
     --output kids_unblinding_executed.ipynb --ExecutePreprocessor.timeout=1800
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]

CELLS = []


def md(s):
    CELLS.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    CELLS.append(nbf.v4.new_code_cell(s.strip("\n")))


md(r"""
# KiDS-Legacy unblinding notebook (multifidelity SBI)

**Blind protocol.** Sections 1–7 are blind-safe: nothing in them reads a posterior location or width.
Section 8 is gated by `UNBLIND = False` and is the **only** place physical contours are drawn.
Run it by hand, once, after Tiers 1–3 have been signed off in the runbook
(`.claude/runs/eval-and-viz/unblinding-prep/artifacts/UNBLINDING_PROTOCOL.md`).

Tier order: **1** data checks → **2** misspecification/OOD scores vs the tabulated bias thresholds →
**3** standardised posteriors (shapes, widths, arm consistency) → **4** this notebook's gated cell.
""")

code(r"""
# ---- parameters (edit these) --------------------------------------------------------------
LABEL = "T"                                   # opaque observation label (A / B / C for the blinds)
OBS_DIR = "/data/alex/unblinding/cluster_fetch/obs_gate0b"          # fetched checkpoints/unblinding/<obs-store>
TIER1_DIR = "/data/alex/unblinding/tier1_" + LABEL                  # scripts/tier1_report.py output
TIER2_DIR = ".claude/runs/eval-and-viz/unblinding-prep/artifacts/tier2"   # scripts/tier2_tables.py output
PLOTS_DIR = "/data/alex/unblinding/plots_" + LABEL                  # scripts/plot_blind_posteriors.py output
BLIND_ROOT = "/data/alex/unblinding/blind_store"                    # raw posteriors (Section 8 ONLY)
FLAGSHIP_EXPERIMENT = "gower_nle_finetune_nla_m_bgp_z8_r0_ens9"
UNBLIND = False                               # <- the gate. Flip by hand, once, after sign-off.
REPO_ROOT = "/home/alex/work/glass_gower_transfer"

import os, sys, json, glob
os.chdir(REPO_ROOT)                    # the kernel starts wherever the .ipynb lives; every relative path here is repo-rooted
sys.path.insert(0, REPO_ROOT)
import numpy as np, h5py
import matplotlib.pyplot as plt
from IPython.display import Image, display, Markdown
plt.rcParams.update({"font.size": 11})
obs_h5 = os.path.join(OBS_DIR, f"observation_{LABEL}.h5")
if not os.path.exists(obs_h5):      # a stripped-mock control (eval --mode observe-strip) has only the baked copy
    obs_h5 = os.path.join(OBS_DIR, f"observation_{LABEL}_baked.h5")
assert os.path.exists(obs_h5), obs_h5
""")

md("## 1. Provenance and configuration")
code(r"""
prov = json.load(open(os.path.join(OBS_DIR, f"observation_{LABEL}_provenance.json")))
STRIPPED = prov.get("kind") == "stripped_mock"        # no catalogue behind it: sections 1-2 degrade
cat_prov = json.loads(prov["catalogue_provenance"]) if "catalogue_provenance" in prov else {}
if STRIPPED:
    display(Markdown(f"**label** `{prov['label']}` · **stripped mock control** (arm `{prov['arm']}`, "
                     f"source store `{prov['source_store']}`, git `{prov['git_rev'][:10]}`) — no catalogue provenance"))
else:
    display(Markdown(f"**label** `{prov['label']}` · built {prov['built_utc']} · git `{prov['git_rev'][:10]}` · "
                     f"n_gal = {prov['n_gal']:,} · counts/bin = {prov['counts_per_bin']}"))
    display(Markdown("**geometry** `" + prov["geometry"] + "`"))
    display(Markdown("**variants** `" + prov["variants"] + "`"))
    display(Markdown("**catalogue source** `" + json.dumps(cat_prov.get("source", {})) + "`"))
    display(Markdown("**treatment (deferred hooks)** `" + json.dumps(cat_prov.get("treatment", {})) + "`"))
    display(Markdown("**m-bias used** `" + str(prov["m_bias_used"]) + "`"))
stores = json.load(open(os.path.join(OBS_DIR, f"observation_{LABEL}_stores.json")))
display(Markdown("**baked stores** `" + json.dumps(stores.get("data_store_names", {})) + "`"))
""")

md("## 2. Catalogue-level summaries")
code(r"""
counts = json.loads(prov["counts_per_bin"]) if "counts_per_bin" in prov else []
if not counts:
    display(Markdown("_no catalogue behind this label (stripped mock): catalogue-level summaries skipped_"))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
ax[0].bar(range(1, len(counts) + 1), counts); ax[0].set_xlabel("tomographic bin"); ax[0].set_ylabel("galaxies"); ax[0].set_title("counts per bin")
ct = cat_prov.get("treatment", {}).get("c_terms", {}).get("per_bin_mean_e1_e2_removed_by_estimator")
if ct:
    ct = np.asarray(ct); ax[1].plot(range(1, len(ct) + 1), ct[:, 0], "o-", label="mean e1"); ax[1].plot(range(1, len(ct) + 1), ct[:, 1], "s-", label="mean e2")
    ax[1].axhline(0, color="k", lw=0.5); ax[1].set_xlabel("tomographic bin"); ax[1].set_title("per-bin mean ellipticity removed (c-terms)"); ax[1].legend()
plt.tight_layout(); plt.show()
""")

md("## 3. Two-point statistics (EE bandpowers) with the nla_m mock cloud")
code(r"""
t1 = json.load(open(os.path.join(TIER1_DIR, "tier1_results.json")))
with h5py.File(obs_h5) as f:
    bp = f["cls_results/full/mixed_bandpowers"][()]
    # a baked store (stripped-mock control) carries no bandpower_ls: fall back to the band index
    ells = f["cls_results/full/bandpower_ls"][()] if "bandpower_ls" in f["cls_results/full"] else np.arange(1, bp.shape[-1] + 1)
    bb = f["cls_results/full/bb_bandpowers"][()] if "bb_bandpowers" in f["cls_results/full"] else None
labs = t1["twopoint"]["spectrum_labels"]; z = np.asarray(t1["twopoint"]["robust_z"])[0]
fig, axes = plt.subplots(3, 7, figsize=(19, 7.5))
for s, ax in enumerate(axes.ravel()):
    (ax.loglog if ells.max() > 20 else ax.semilogy)(ells, np.abs(bp[s]), "o-", ms=3, lw=1, color="k")
    ax.set_title(f"EE {labs[s]}", fontsize=9); ax.set_xlabel(r"$\ell$", fontsize=8)
    ax.text(0.03, 0.05, "z: " + " ".join(f"{v:+.1f}" for v in z[s]), transform=ax.transAxes, fontsize=6)
fig.suptitle(f"label {LABEL}: EE bandpowers (|C_b|; robust z vs the mock cloud per band in each panel)"); plt.tight_layout(); plt.show()
display(Image(os.path.join(TIER1_DIR, "tier1_bandpower_residuals.png")))
tp = t1["twopoint"]
display(Markdown(f"**Tier-1a** kNN p = {tp['knn']['p'][0]:.3f}, Mahalanobis p = {tp['mahalanobis']['p'][0]:.3f} "
                 f"(dim {tp['dim']}, null n = {tp['n_null']} rows / {tp['n_cosmologies_null']} cosmologies)"))
""")

md("## 4. B-mode null test")
code(r"""
bm = t1.get("bmodes")
if bm is None:
    display(Markdown("_no B-mode block in the Tier-1 results (reference cloud built without the raw store): section skipped_"))
else:
  display(Markdown(f"**Tier-1b** χ² vs mock-B mean = {bm['chi2_vs_mockmean'][0]:.1f} (dim {bm['dim']}; null median {bm['null_chi2_median']:.1f}) → "
                 f"**PTE = {bm['pte_empirical'][0]:.3f}** (empirical); vs zero: PTE = {bm['pte_vs_zero_empirical'][0]:.3f}; "
                 f"per-auto PTEs: " + ", ".join(f"{a['spectrum']}: {a['pte_empirical'][0]:.2f}" for a in bm['per_auto'])))
  display(Image(os.path.join(TIER1_DIR, "tier1_bmode_residuals.png")))
""")

md("## 5. Map statistics (peaks, voids, one-point PDF, radial power) vs the mock cloud")
code(r"""
em = t1["emap"]
display(Markdown(f"**Tier-1c** E-map kNN p (full, {em['full']['dim']}-d) = {em['full']['knn']['p'][0]:.3f}; "
                 + ", ".join(f"{k}: p = {v['knn']['p'][0]:.3f}" for k, v in em.items() if k.startswith('pca'))
                 + "; per group: " + ", ".join(f"{g}: {v['knn']['p'][0]:.3f}" for g, v in em["groups"].items())))
rz = em["robust_z"]
groups = ["moments", "pdf", "peaks", "power", "xcorr"]
fig, axes = plt.subplots(1, len(groups), figsize=(4 * len(groups), 3.2))
for ax, g in zip(axes, groups):
    names = [n for n in rz if f"/{g}/" in n]; vals = [rz[n][0] for n in names]
    ax.plot(vals, ".-", lw=0.8); ax.axhspan(-2, 2, color="0.9"); ax.axhline(0, color="k", lw=0.5); ax.set_ylim(-6, 6)
    ax.set_title(f"{g} ({len(names)} stats)", fontsize=9); ax.set_ylabel("robust z vs cloud", fontsize=8)
plt.tight_layout(); plt.show()
display(Image(os.path.join(TIER1_DIR, "tier1_knn_scores.png")))
""")

md("## 6. OOD dashboard: Tier-1 p-values and Tier-2 detector readings")
code(r"""
v = json.load(open(os.path.join(TIER1_DIR, "tier1_verdict.json")))
display(Markdown("**Tier-1 verdict** (α = %.3g): " % v["alpha"] + json.dumps(v["summary"])))
rd = os.path.join(TIER2_DIR, f"obs_reading_{LABEL}.md")
if os.path.exists(rd):
    display(Markdown(open(rd).read()))
else:
    display(Markdown(f"_no Tier-2 reading for label {LABEL} yet (run eval --mode obs-score, fetch, then scripts/tier2_tables.py --obs-score …)_"))
display(Image(os.path.join(TIER2_DIR, "bias_curves.png")))
""")

md("## 7. Standardised posteriors (blind): shapes, widths, arm consistency")
md(r"""
The FINAL posterior per arm is the **pooled** one — the equal-weight mixture of the 5 independent
repeats, i.e. their concatenated draws (`eval.py --mode pool`; HANDOFF_pooling.md). Headline plots
(A, B, B′, corner battery, labels) use the pooled run of each arm; **Plot D** shows every repeat in
the pooled frame of its arm (seed spread in σ of the pooled posterior). Pooling can only add spread
across seeds: a pooled 1-D width that looks calibrated says nothing about the joint calibration,
which all seeds share (measured on the Gower mocks: pooled std(z) 1.09 → 1.00, joint TARP under-
coverage survives).
""")
code(r"""
man = json.load(open(os.path.join(PLOTS_DIR, "manifest.json"))) if os.path.exists(os.path.join(PLOTS_DIR, "manifest.json")) else {}
hl = [h for h in man.get("headline", []) if h["label"] == LABEL]
display(Markdown("**headline posteriors** (arm → pooled?): " + ", ".join(f"{h['arm']}: {'POOLED' if h['pooled'] else h['match']}" for h in hl)
                 + (" — ⚠️ some arms are NOT pooled yet" if any(not h["pooled"] for h in hl) else "")))
for name in (f"plotA_{LABEL}.png", f"plotB_{LABEL}.png", f"plotBprime_{LABEL}.png", "labels_overplot.png"):
    p = os.path.join(PLOTS_DIR, name)
    if os.path.exists(p):
        display(Markdown(f"**{name}**")); display(Image(p))
for p in sorted(glob.glob(os.path.join(PLOTS_DIR, f"plotD_{LABEL}_*.png"))):
    display(Markdown(f"**{os.path.basename(p)}** — seed-spread diagnostic (repeats in the pooled frame)")); display(Image(p))
bat = sorted(glob.glob(os.path.join(PLOTS_DIR, "battery", f"{LABEL}_*_corner.png")))
display(Markdown(f"corner battery: {len(bat)} figures under `{PLOTS_DIR}/battery/`"))
for p in bat[:3]:
    display(Image(p, width=700))
""")

md(r"""
## 8. UNBLINDING — gated

`UNBLIND` is `False` above. **Do not flip it until Tiers 1–3 are signed off in the runbook.** This is
the only cell that opens raw posterior samples and draws physical contours. It is never executed by
the agent (guard_blind denies the read), and nbconvert runs of this notebook on mocks leave it skipped.
""")
code(r"""
if not UNBLIND:
    display(Markdown("### 🔒 blind — set `UNBLIND = True` by hand to draw the physical posteriors"))
else:
    import pandas as pd
    from chainconsumer import Chain, ChainConsumer, PlotConfig
    from src.blind.standardise import _load_physical, list_raw_runs   # raw access: user-run cell only
    runs = [r for r in list_raw_runs(BLIND_ROOT) if r["label"] == LABEL]
    cols = {"omega_m": r"$\Omega_{\rm m}$", "sigma_8": r"$\sigma_8$", "S8": r"$S_8$", "w0": r"$w_0$"}
    def _df(r):
        x, names = _load_physical(r["path"], r["experiment"])
        keep = [n for n in cols if n in names]
        return pd.DataFrame(x[:, [names.index(n) for n in keep]], columns=[cols[n] for n in keep]), keep, x, names
    for prior in sorted({r["prior"] for r in runs}):
        pooled = [r for r in runs if r["prior"] == prior and r["pooled"]]
        reps = [r for r in runs if r["prior"] == prior and not r["pooled"]]
        # (i) the FINAL posteriors: one pooled chain per arm
        if pooled:
            c = ChainConsumer()
            for r in pooled:
                df, keep, _, _ = _df(r)
                c.add_chain(Chain(samples=df, parameters=[cols[n] for n in keep], name=f"{r['arm']} POOLED"))
            c.set_plot_config(PlotConfig(serif=True, usetex=False, label_font_size=12, tick_font_size=9))
            fig = c.plotter.plot(figsize=(9, 9)); fig.suptitle(f"UNBLINDED (final = pooled): label {LABEL}, prior {prior}"); plt.show()
        else:
            display(Markdown(f"⚠️ no POOLED run for prior {prior}: run `scripts/sample_observation.py pool` + `fetch --what pooled`"))
        # (ii) diagnostic: per-repeat chains of each arm with the pooled one
        for arm in sorted({r["arm"] for r in reps}):
            c = ChainConsumer()
            for r in [r for r in pooled if r["arm"] == arm]:
                df, keep, _, _ = _df(r)
                c.add_chain(Chain(samples=df, parameters=[cols[n] for n in keep], name=f"{arm} POOLED", color="black"))
            for r in [r for r in reps if r["arm"] == arm]:
                df, keep, _, _ = _df(r)
                c.add_chain(Chain(samples=df, parameters=[cols[n] for n in keep], name=f"{arm} {r['match']}"))
            c.set_plot_config(PlotConfig(serif=True, usetex=False, label_font_size=12, tick_font_size=9))
            fig = c.plotter.plot(figsize=(8, 8)); fig.suptitle(f"diagnostic: label {LABEL}, arm {arm}, prior {prior}: repeats vs pooled"); plt.show()
        # summary table (pooled first)
        rows = []
        for r in pooled + reps:
            _, _, x, names = _df(r)
            for n in ("omega_m", "sigma_8", "S8"):
                i = names.index(n); rows.append((r["arm"], "POOLED" if r["pooled"] else r["match"], n, x[:, i].mean(), x[:, i].std()))
        display(pd.DataFrame(rows, columns=["arm", "run", "param", "mean", "std"]))
""")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO / "notebooks" / "kids_unblinding.ipynb"))
    ap.add_argument("--label", default=None, help="pre-fill LABEL in the parameters cell (default T)")
    ap.add_argument("--obs-dir", default=None, help="pre-fill OBS_DIR (the fetched checkpoints/unblinding/<obs-store>)")
    args = ap.parse_args(argv)
    cells = list(CELLS)
    if args.label or args.obs_dir:
        # the parameters cell is the first code cell; rewrite its two assignments only
        for i, c in enumerate(cells):
            if c["cell_type"] == "code" and "LABEL = " in c["source"]:
                src = c["source"]
                if args.label:
                    src = src.replace('LABEL = "T"', f'LABEL = "{args.label}"', 1)
                if args.obs_dir:
                    src = src.replace('OBS_DIR = "/data/alex/unblinding/cluster_fetch/obs_gate0b"',
                                      f'OBS_DIR = "{args.obs_dir}"', 1)
                cells[i] = nbf.v4.new_code_cell(src)
                break
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, args.out)
    print(f"wrote {args.out} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
