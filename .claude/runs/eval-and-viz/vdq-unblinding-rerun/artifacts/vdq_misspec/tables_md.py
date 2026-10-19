#!/usr/bin/env python
"""P5.8 -- assemble TABLES.md + numbers.{json,tex} from the tables.py jsons, the Tier-2 bias-table
outputs (scripts/tier2_tables.py) and the obs-score jsons. Pure formatting: no statistic is
computed here except the observation's in-dist PTE on the KL axis (fraction of in-dist events at
least as discordant), which needs only the per-event in-dist KL.
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as K  # noqa: E402
import vdq_config as C  # noqa: E402

NUM = {}


def num(key, val):
    NUM[key] = val
    return val


def _t(pk, kind):
    p = os.path.join(C.OUT_MISSPEC, "tables", f"{pk['key']}_{kind}.json")
    return K.load_json(p) if os.path.exists(p) else None


def ci(x, c, d=3, sign=False):
    f = f"{{:{'+' if sign else ''}.{d}f}}"
    return f"{f.format(x)} [{f.format(c[0])}, {f.format(c[1])}]"


def _reading(path):
    return K.load_json(path) if os.path.exists(path) else None


def section_pack(pk):
    L = []
    det, kla, bg = _t(pk, "detector"), _t(pk, "klaxis"), _t(pk, "bgia")
    key = pk["key"]
    L.append(f"## {pk['title']} pack `{pk['exp']}` (repeats {pk['repeats']}, adopted KL `{pk['kl']}`)\n")
    if not (det and kla):
        L.append("_tables.py output missing for this pack._\n")
        return L
    vs = [v for v in C.VARIATES if v in kla["variates"]]
    L.append(f"Variates present: {', '.join('`' + v + '`' for v in vs)}.\n")
    # ---------------- kNN axis
    L.append("### kNN (summary-space OOD) axis — AUROC vs in-distribution, 95% paired cosmology-bootstrap CI\n")
    L.append("Per-encoder detectors: AUROC on −log10 p averaged over the encoders (member 0 per repeat). "
             "`meanp_knn` = the obs-identical statistic (encoder-mean unconditional kNN p, recalibrated on the "
             "`_idtest` null; in-dist rows leave-one-cosmology-out). `meanp_cond` = phase6 conditional combiner "
             "(needs the true θ; mock-side detection power only). flag = fraction with p < 0.05 (in-dist rate in "
             "brackets). PTE = probability that a cluster-matched in-dist draw has a median p at least as low.\n")
    dets = ["knn", "cond_knn", "mahalanobis", "meanp_knn", "meanp_cond"]
    L.append("| variate | n (cos × rows) | " + " | ".join(f"AUROC {d}" for d in dets)
             + " | flag05 meanp_knn [in-dist] | median-p PTE meanp_knn |")
    L.append("|---|---|" + "---|" * len(dets) + "---|---|")
    for v in vs:
        if v == C.IN_DIST:
            continue
        r = det["variates"].get(v)
        if r is None:
            continue
        cells = [f"{C.SHORT[v]}", f"{r['n_rows']} ({r['n_cos']}×{r['rows_per_cos']})"]
        for d in dets:
            x = r["detectors"].get(d)
            cells.append(ci(x["auroc"], x["auroc_ci95"]) if x else "–")
            if x:
                num(f"{key}_{C.SHORT[v]}_auroc_{d}", x["auroc"])
        m = r["detectors"].get("meanp_knn")
        cells.append(f"{m['frac_p_lt_0.05']:.3f} [{m['in_dist_frac_p_lt_0.05']:.3f}]" if m else "–")
        cells.append(f"{m['median_p_pte']:.3g}" if m else "–")
        L.append("| " + " | ".join(cells) + " |")
    L.append("")
    # ---------------- KL axis
    L.append(f"### KL (cross-repeat disagreement) axis + calibration + bias\n")
    kls = kla["kl_reported"]
    L.append("KL columns: median per-event KL [95% cosmology CI]; AUROC vs in-dist [95% paired CI]; power = fraction "
             "above the in-dist 95th percentile. Calibration = TARP error, mean ± sd over the repeats, vs its "
             "floor (big variates: the in-dist repeat spread on the same parameters; gb slices: cluster-matched "
             "40 cos × 2 rows in-dist draws). z = (truth − mean)/sd over all (repeat, event) rows, 95% cosmology CI.\n")
    hdr = ["variate", "n"] + [f"KL[{w}] median" for w in kls] + [f"AUROC KL[{w}]" for w in kls] + \
          [f"power5 KL[{kls[0]}]", "cal_full (z vs floor)", "cal(Ωm,σ8,w) (z vs floor)", "E[z_Ωm]", "E[z_σ8]", "E[z_S8]", "median z_S8",
           "P(\\|z_S8\\|>0.5) [excess]", "P(\\|z_S8\\|>1) [excess]"]
    L.append("| " + " | ".join(hdr) + " |")
    L.append("|" + "---|" * len(hdr))
    for v in vs:
        r = kla["variates"][v]
        s = C.SHORT[v]
        dr = r.get("n_dropped_nonfinite_per_repeat") or []
        cells = [s, str(r[f"kl_{kls[0]}"]["n_events"]) + (f" (drop {min(dr)}-{max(dr)}/rep)" if dr and max(dr) else "")]
        for w in kls:
            e = r[f"kl_{w}"]
            cells.append(ci(e["median"], e["median_ci95"]))
            num(f"{key}_{s}_kl_{w}_median", e["median"])
        for w in kls:
            e = r[f"kl_{w}"]
            cells.append(ci(e["auroc"], e["auroc_ci95"]) if "auroc" in e else "–")
            if "auroc" in e:
                num(f"{key}_{s}_auroc_kl_{w}", e["auroc"])
        e0 = r[f"kl_{kls[0]}"]
        cells.append(ci(e0["power_at_5pct"], e0["power_at_5pct_ci95"]) if "power_at_5pct" in e0 else "–")
        c = r["cal"]
        for cc in ("cal_full", "cal3"):
            zf = c.get(f"{cc}_z_vs_floor")
            # a z built from a tiny floor sd is meaningless far above the floor: quote the ratio there
            tag = "" if zf is None else (f" ({zf:+.1f})" if abs(zf) <= 20 else f" ({c[f'{cc}_ratio_to_floor']:.0f}× floor)")
            cells.append(f"{c[f'{cc}_mean']:.4f}±{c[f'{cc}_sd']:.4f}" + tag)
            num(f"{key}_{s}_{cc}", c[f"{cc}_mean"])
        if len(c["available_params"]) < 9:
            cells[-2] += f" [{len(c['available_params'])}p]"
        for p in ("omega_m", "sigma_8", "S8"):
            z = r["z"][p]
            cells.append(ci(z["mean_z"], z["mean_z_ci95"], 2, True))
            num(f"{key}_{s}_meanz_{p}", z["mean_z"])
        zz = r["z"]["S8"]
        cells.append(ci(zz["median_z"], zz["median_z_ci95"], 2, True) if "median_z" in zz else "–")
        if "median_z" in zz:
            num(f"{key}_{s}_medianz_S8", zz["median_z"])
        for t in ("0.5", "1"):
            z = r["z"]["S8"]
            cells.append(f"{z[f'P_absz_gt_{t}']:.2f} [{z[f'P_absz_gt_{t}_excess']:+.2f}]")
        L.append("| " + " | ".join(cells) + " |")
    L.append("")
    L.append("Floors used: " + "; ".join(
        f"`{k}` cal_full {f['cal_full_mean']:.4f}±{f['cal_full_sd']:.4f}, cal3 {f['cal3_mean']:.4f}±{f['cal3_sd']:.4f} ({f['kind']})"
        for k, f in kla["floors"].items()) + "\n")
    # ---------------- observation
    o = K.obs(pk)
    _f, _s, k_ind = K.kl_event(pk, C.IN_DIST, pk["kl"])
    kv = float(o[f"kl_{pk['kl']}"])
    pte = float((np.sum(k_ind >= kv) + 1) / (k_ind.size + 1))
    num(f"{key}_obs_kl", kv)
    num(f"{key}_obs_kl_pte", pte)
    num(f"{key}_obs_meanp_recal", o["meanp_recalibrated"])
    L.append(f"### Real observation `{o['label']}` on these axes\n")
    L.append(f"- KL[`{pk['kl']}`] = {kv:.4f} (read from `kl_{pk['kl']}`; obs json `kl_params` = `{o.get('kl_params')}`) "
             f"→ in-dist PTE {pte:.3f} (fraction of {k_ind.size} in-dist events with KL ≥ obs).")
    L.append(f"- obs-identical mean-p: raw {o['meanp_raw']:.4f}, recalibrated {o['meanp_recalibrated']:.4f} "
             f"(the recalibrated value IS its in-dist PTE; n_null {o.get('n_idtest_null')}).")
    for sub, lab in (("", "pooled, all variates"), ("pooled_without_vdqk2_nla", "pooled without κ=2+NLA (D2)")):
        rd = _reading(os.path.join(pk["tier2_dir"], sub, f"obs_reading_{o['label']}.json"))
        if not rd:
            continue
        for d in ("meanp", "kl"):
            if d not in rd or rd[d].get("per_param") is None:
                continue
            pp = rd[d]["per_param"]
            s8 = pp.get("S8", {})
            if "mean_z" in s8:
                L.append(f"- Tier-2 reading ({lab}), **{d}** bin [{rd[d]['bin'][0]:.3g}, {rd[d]['bin'][1]:.3g}) "
                         f"n={rd[d]['n_rows']}: E[z_S8] = {s8['mean_z']:+.2f} (CI68 {s8['mean_z_ci68'][0]:+.2f}..{s8['mean_z_ci68'][1]:+.2f}); "
                         + "; ".join(f"P(|z_S8|>{t}) {s8[f'P_absz_gt_{t}']['pooled']:.2f} [excess {s8[f'P_absz_gt_{t}']['excess']:+.2f}]"
                                     for t in ("0.5", "1")))
    L.append("")
    # ---------------- b_g / IA
    if bg:
        L.append("### Galaxy-bias and IA framing\n")
        if "gb_pair" in bg:
            gp = bg["gb_pair"]
            L.append(f"- gb1p3 − gb1p0 paired (same sim/aug ids: {gp['paired_all_repeats']}): " + "; ".join(
                f"{p}: Δz {x['dz_mean_over_repeats']:+.3f}±{x['dz_sd_over_repeats']:.3f}"
                + (f", Δθ {x['dtheta_mean_over_repeats']:+.4f} = {x['dtheta_over_sigma']:+.2f} σ_post" if "dtheta_over_sigma" in x else "")
                for p, x in gp["summary"].items()) + " (mean ± sd over encoders).")
            for p, x in gp["summary"].items():
                num(f"{key}_gbpair_dz_{p}", x["dz_mean_over_repeats"])
        for v, a in bg["a_ia"].items():
            L.append(f"- `{v}`: A_IA^total ∈ [{a['A_total_min']:.2f}, {a['A_total_max']:.2f}] (median {a['A_total_median']:.2f}; {a['convention']})")
        for v, b in bg.get("bg_severity", {}).items():
            if "error" in b:
                L.append(f"- `{v}` per-event b_g: {b['error']}")
                continue
            L.append(f"- `{v}` per-event galaxy-bias severity d = |z_bg| vs the κ=1 prior (χ6 median 2.35 / 95th 3.55): "
                     f"median {b['d_median']:.2f}, 90th {b['d_p90']:.2f}; {b['frac_any_bin_outside_3sigma']:.1%} of events have a bin "
                     f"outside ±3σ. KL power (in-dist 5% false alarm) inside / outside support: "
                     f"{b['kl_power_inside_support']:.3f} / {b['kl_power_outside_support']:.3f}.")
            L.append("")
            L.append("  | d bin | n (cos) | KL power [95%] | mean-p power [95%] | E[z_Ωm] [95%] | E[z_S8] [95%] |")
            L.append("  |---|---|---|---|---|---|")
            for e in b["bins"]:
                mp = ci(e["power_meanp"], e["power_meanp_ci95"], 2) if "power_meanp" in e else "–"
                L.append(f"  | [{e['d_lo']:g}, {e['d_hi']:g}) | {e['n']} ({e['n_cos']}) | {ci(e['power_kl'], e['power_kl_ci95'], 2)} | {mp} | "
                         f"{ci(e['mean_z_omega_m'], e['mean_z_omega_m_ci95'], 2, True)} | {ci(e['mean_z_S8'], e['mean_z_S8_ci95'], 2, True)} |")
            L.append("")
        L.append("")
    return L


def main():
    L = ["# VDQ model-misspecification tables (kNN + KL axes, both NPE packs)\n",
         "Generated by `artifacts/vdq_misspec/tables_md.py` from `tables.py`, `meanp_knn.py`, `scripts/tier2_tables.py` "
         "outputs. All intervals are COSMOLOGY-cluster bootstraps (rows are ~8, gb slices 2, augmentations per "
         "cosmology). Per-variate rows are PRIMARY; pooled Tier-2 tables weight every (variate, repeat, event) row "
         "equally, so the ~1592-event variates dominate the two 80-event gb slices; the pooled table is given with and "
         "without the double misspecification `gower_vdqk2_nla` (κ=2 + NLA, sits on neither single axis; D2). "
         "No ΛCDM-conditioned rows (D4).\n",
         "**Mean-p statistic (D3).** The mean-p column here is the OBS-IDENTICAL statistic: encoder-mean of the "
         "unconditional kNN p, recalibrated against the `_idtest` null — exactly what `obs_score_*.json` "
         "`meanp_recalibrated` is (gate: the obs's own per-encoder p reproduce its recorded value to 6 d.p. on both "
         "packs). **The 2026-09-24 Tier-2 readings binned the obs's unconditional-kNN mean-p into a table built from "
         "the phase6 CONDITIONAL-kNN mean-p** (`cond_knn`, needs the true θ) — a different detector under the same "
         "name. Those readings are superseded by this table; phase6 survives only as the mock-side `meanp_cond` "
         "detection-power column.\n"]
    L += ["## Provenance (data quirks that affect the numbers)\n",
          "- **2-pt nla / nla_z / vdqk2_nla KL rebuilt offline.** Those misspec cells ran as 5 separate per-repeat jobs "
          "(the 5-repeat job timed out), so their `misspec_repeat_disagreement_*` files were rebuilt by the orchestrator "
          "with `rebuild_kl.py` (`misspec._compute_repeat_disagreement` on mean/std² from the moments npz; validated "
          "against the in-job 2-pt gbk2 file: kl_mean 0.063070 == 0.063070, max rel dev 2.7e-6). Every other cell has an "
          "in-job KL file.",
          "- **Dropped events (NLA family).** The misspec eval drops events whose posterior samples are non-finite "
          "(far-OOD sampling degeneracy): 10-27 % per repeat, concentrated at A_IA^total < -3 (37-67 % of those events). "
          "The cross-repeat KL keeps only events retained in ALL repeats (field 1166/1260/1227, 2-pt 1131/1145/1176 of "
          "1592/1590/1592). KL, calibration and z statistics for these variates describe the RETAINED events -- a lower "
          "bound on the misspecification. Oddity recorded, not explained: 2-pt repeat 1 dropped 0 events on all three "
          "NLA-family variates (others 73-349).",
          "- **Field summaries of the 5 new variates were encoded on CPU** (CORES40; a zombie process blocked the GPU), "
          "against a CPU-encoded train cloud, while the local `_idtest` null and the observation score are GPU-encoded. "
          "Measured in `meanp_knn.py`: in-job p vs local recomputation differ by <= 1 rank step of the null on 0.2-0.8 % "
          "of (row, encoder) cells (gbk2: <= 2 steps on 1.1 %); 2-pt summaries (GPU) show 0 deviation except gb1p0 (CPU, "
          "1 step on 1 %). Immaterial at the table's resolution.",
          "- **Per-event b_g** (kappa=2 variates) comes from the FIELD misspec moments `extra_b_g_bin1..6` (commit af9c295) "
          "and is mapped by test file onto the 2-pt pack (same lock + store).",
          "- In-dist `_idtest` md5 unchanged by the fetch; both obs-score gates reproduce `meanp_recalibrated` to 6 d.p.",
          ""]
    for key in ("field", "band"):
        L += section_pack(C.PACKS[key])
    L += ["## Standing caveats\n",
          "- **X-8 resolution floor:** a single observation cannot resolve a sub-σ systematic; a table bin gives the "
          "bias distribution of the equal-weight MOCK MIXTURE that scores like the observation, not the observation's bias.",
          "- **X-14 invisible bias:** a variate whose detectors sit at AUROC ≈ 0.5 can still carry bias (the E[z] "
          "columns); that budget is not bounded by any score. κ=2 contains the κ=1 prior, so its population AUROC ≈ 0.5 "
          "is expected and is NOT 'undetectable' per event.",
          "- NLA-family cal_full is over the available parameters (a_ia excluded, b_ia absent → 7p); its floor is the "
          "in-dist calibration recomputed on the same 7 parameters.", ""]
    with open(os.path.join(C.OUT_MISSPEC, "TABLES.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    with open(os.path.join(C.OUT_MISSPEC, "numbers.json"), "w") as fh:
        json.dump(NUM, fh, indent=1, default=float)
    tex = ["% generated by tables_md.py"]
    for k, v in NUM.items():
        name = "vdq" + "".join(w.capitalize() for w in re.split(r"[^A-Za-z]+", re.sub(r"\d", lambda m: "ABCDEFGHIJ"[int(m.group())], k)) if w)
        tex.append(f"\\newcommand{{\\{name}}}{{{v:.3f}}}" if isinstance(v, float) else f"\\newcommand{{\\{name}}}{{{v}}}")
    with open(os.path.join(C.OUT_MISSPEC, "numbers.tex"), "w") as fh:
        fh.write("\n".join(tex) + "\n")
    print(f"[tables_md] wrote {C.OUT_MISSPEC}/TABLES.md, numbers.json ({len(NUM)} keys), numbers.tex")


if __name__ == "__main__":
    main()
