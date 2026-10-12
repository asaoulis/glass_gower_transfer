#!/usr/bin/env python
"""Tier-4 diagnostic: is the field-level vs 2-pt shift along the Omega_m -- sigma_8 degeneracy seen on
the observation normal for the matched Gower mocks?

For every LCDM matched-mock event (the same mock is scored by both arms of a repeat), take the
posterior medians of the two arms and form the shift field - 2-pt in (Omega_m, sigma_8, S8), raw and
in units of the 2-pt posterior std. Compare the observation's shift (read from the Tier-4 table, the
pooled rows) with that cloud, report its percentile, and draw the mock events whose shift is closest
to the observed one as 2-pt + field corners (same style as the reveal set, MOCK-tagged).

Outputs (in --out-dir):
  shift_cloud.png/.pdf          arrows 2-pt -> field in the Omega_m--sigma_8 plane (mocks grey, observation
                                in the arm colours) + histograms of the standardised shifts
  analogue_<k>_sim<id>_aug<id>_corner.png/.pdf   the k closest mock analogues (field + 2-pt + truth)
  shifts.csv, summary.md        per-event table and the percentiles

Usage:
  python scripts/tier4_shift_diagnostics.py --out-dir /data/alex/unblinding/real/work/tier4_diagnostics \
      --real-table /data/alex/unblinding/real/work/tier4_reveal/tables/02_2pt_field_LCDM.md
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from src.blind.standardise import _scaler_box  # noqa: E402
from src.viz import style as S  # noqa: E402
import tier4_reveal_plots as T4  # noqa: E402

PRIOR = "LCDM_fixed_w0"
PARAMS = ("omega_m", "sigma_8", "S8")


def load_arm(run: dict):
    """-> (medians [N,3], stds [N,3], samples [draws,N,3] physical, truth [N,3], ids [N,2], files [N])."""
    names, lo, hi = _scaler_box(run["experiment"], PRIOR)
    tn, tlo, thi = _scaler_box(run["experiment"], None)
    with np.load(run["path"], allow_pickle=False) as d:
        X = np.asarray(d["samples"], dtype=np.float64) * (hi - lo) + lo
        Tr = np.asarray(d["theta0s"], dtype=np.float64) * (thi - tlo) + tlo
        ids = np.stack([d["sim_ids"], d["aug_ids"]], 1) if "sim_ids" in d.files else None
        files = np.asarray(d["test_files"]) if "test_files" in d.files else None
    io, is8 = names.index("omega_m"), names.index("sigma_8")
    P = np.stack([X[:, :, io], X[:, :, is8], X[:, :, is8] * np.sqrt(X[:, :, io] / 0.3)], -1)
    to, ts = tn.index("omega_m"), tn.index("sigma_8")
    truth = np.stack([Tr[:, to], Tr[:, ts], Tr[:, ts] * np.sqrt(Tr[:, to] / 0.3)], -1)
    return np.median(P, 0), P.std(0), P, truth, ids, files


def read_real_table(path: str) -> dict:
    """Pooled rows of a Tier-4 table -> {arm: {param: (median, std)}}."""
    out = {"band": {}, "nla_m": {}}
    key = {"2-pt (bandpowers, VD) (pooled)": "band", "field level (NLA-M, VD) (pooled)": "nla_m"}
    with open(path) as fh:
        for line in fh:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 6 and cells[0] in key and cells[1] in PARAMS:
                out[key[cells[0]]][cells[1]] = (float(cells[4]), float(cells[3]))
    for a in out:
        if set(out[a]) != set(PARAMS):
            raise ValueError("pooled rows for %s missing in %s" % (a, path))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--real-table", required=True)
    ap.add_argument("--matched-root", default="/data/alex/variate_samples_kids")
    ap.add_argument("--matched-tag", default="gower_match_vdq")
    ap.add_argument("--repeats", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-analogues", type=int, default=3)
    ap.add_argument("--max-samples", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    real = read_real_table(args.real_table)
    r_med = {a: np.array([real[a][p][0] for p in PARAMS]) for a in real}
    r_sd = {a: np.array([real[a][p][1] for p in PARAMS]) for a in real}
    real_shift = r_med["nla_m"] - r_med["band"]
    real_z = real_shift / r_sd["band"]

    runs = T4.mock_runs(args.matched_root, args.matched_tag, ["band", "nla_m"], args.repeats, [PRIOR])
    by = {(r["arm"], r["repeat"]): r for r in runs}
    rows, cache = [], {}
    for r in args.repeats:
        if ("band", r) not in by or ("nla_m", r) not in by:
            print("[skip] repeat %d: one arm missing" % r); continue
        mb, sb, Pb, trb, idb, fb = load_arm(by[("band", r)])
        mf, sf, Pf, trf, idf, ff = load_arm(by[("nla_m", r)])
        if fb is not None and ff is not None and not np.array_equal(fb, ff):
            raise ValueError("repeat %d: the two arms score different events" % r)
        n = min(len(mb), len(mf))
        cache[r] = (Pb, Pf, trb)
        for i in range(n):
            d = mf[i] - mb[i]
            rows.append(dict(repeat=r, event=i, sim=int(idb[i, 0]) if idb is not None else -1,
                             aug=int(idb[i, 1]) if idb is not None else -1,
                             om_b=mb[i, 0], s8_b=mb[i, 1], S8_b=mb[i, 2], om_f=mf[i, 0], s8_f=mf[i, 1], S8_f=mf[i, 2],
                             om_t=trb[i, 0], s8_t=trb[i, 1], S8_t=trb[i, 2],
                             d_om=d[0], d_s8=d[1], d_S8=d[2], z_om=d[0] / sb[i, 0], z_s8=d[1] / sb[i, 1], z_S8=d[2] / sb[i, 2],
                             sd_om_b=sb[i, 0], sd_s8_b=sb[i, 1], sd_S8_b=sb[i, 2]))
    if not rows:
        raise SystemExit("no paired events")
    keys = list(rows[0])
    with open(os.path.join(args.out_dir, "shifts.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    A = {k: np.array([r[k] for r in rows]) for k in keys}
    N = len(rows)

    def pct(x, v):
        return 100.0 * (x <= v).mean()

    # distance to the observed shift in standardised (Omega_m, sigma_8) units
    dist = np.hypot(A["z_om"] - real_z[0], A["z_s8"] - real_z[1])
    order = np.argsort(dist)
    top = order[: args.n_analogues]

    lines = ["# Field-level vs 2-pt shift along the Omega_m--sigma_8 degeneracy: observation vs matched mocks", "",
             "Shift = field median - 2-pt median of the SAME mock (LCDM, %d paired events over repeats %s)." % (N, args.repeats),
             "z = shift / 2-pt posterior std of that event. Observation: pooled rows of `%s`." % os.path.basename(args.real_table), "",
             "| quantity | observation | mock mean | mock std | mock 5 / 50 / 95 % | obs. percentile |", "|---|---|---|---|---|---|"]
    for lab, key, rv in (("d Omega_m", "d_om", real_shift[0]), ("d sigma_8", "d_s8", real_shift[1]), ("d S8", "d_S8", real_shift[2]),
                         ("z Omega_m", "z_om", real_z[0]), ("z sigma_8", "z_s8", real_z[1]), ("z S8", "z_S8", real_z[2])):
        x = A[key]; q = np.percentile(x, [5, 50, 95])
        lines.append("| %s | %+.3f | %+.3f | %.3f | %+.3f / %+.3f / %+.3f | %.1f %% |" % (lab, rv, x.mean(), x.std(), q[0], q[1], q[2], pct(x, rv)))
    both = ((A["z_om"] <= real_z[0]) & (A["z_s8"] >= real_z[1])).mean()
    lines += ["", "Fraction of mock events shifted at least as far as the observation in BOTH Omega_m (down) and sigma_8 (up): %.1f %% (%d of %d)."
              % (100 * both, int(round(both * N)), N),
              "", "Closest analogues (distance in z-space to the observed shift):", "",
              "| k | repeat | sim | aug | truth (Om, s8, S8) | 2-pt median | field median | z Om | z s8 | z S8 | dist |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, i in enumerate(top):
        r = rows[i]
        lines.append("| %d | %d | %d | %d | (%.3f, %.3f, %.3f) | (%.3f, %.3f, %.3f) | (%.3f, %.3f, %.3f) | %+.2f | %+.2f | %+.2f | %.2f |"
                     % (k, r["repeat"], r["sim"], r["aug"], r["om_t"], r["s8_t"], r["S8_t"], r["om_b"], r["s8_b"], r["S8_b"],
                        r["om_f"], r["s8_f"], r["S8_f"], r["z_om"], r["z_s8"], r["z_S8"], dist[i]))
    with open(os.path.join(args.out_dir, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- figure 1: arrow cloud + histograms -------------------------------------------------
    import matplotlib.pyplot as plt
    cb, cf = T4.arm_colour("band"), T4.arm_colour("nla_m")
    with S.context():
        fig = plt.figure(figsize=(16, 6))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1], wspace=0.3)
        ax = fig.add_subplot(gs[0, 0])
        for r in rows:
            ax.annotate("", xy=(r["om_f"], r["s8_f"]), xytext=(r["om_b"], r["s8_b"]),
                        arrowprops=dict(arrowstyle="-|>", color="0.55", lw=0.6, alpha=0.35, mutation_scale=7))
        for i in top:
            r = rows[i]
            ax.annotate("", xy=(r["om_f"], r["s8_f"]), xytext=(r["om_b"], r["s8_b"]),
                        arrowprops=dict(arrowstyle="-|>", color="0.15", lw=1.6, mutation_scale=12))
        ax.annotate("", xy=(r_med["nla_m"][0], r_med["nla_m"][1]), xytext=(r_med["band"][0], r_med["band"][1]),
                    arrowprops=dict(arrowstyle="-|>", color=cf, lw=3.0, mutation_scale=18))
        ax.plot([r_med["band"][0]], [r_med["band"][1]], "o", color=cb, ms=9, label=S.tex("2-pt (observation)"))
        ax.plot([r_med["nla_m"][0]], [r_med["nla_m"][1]], "o", color=cf, ms=9, label=S.tex("field level (observation)"))
        om = np.linspace(0.15, 0.5, 100)
        for s8c in (0.76, 0.80, 0.84, 0.88):
            ax.plot(om, s8c / np.sqrt(om / 0.3), ":", color="0.75", lw=0.9)
        ax.plot([], [], "-", color="0.55", lw=0.8, label=S.tex("mock: 2-pt $\\to$ field (%d events)" % N))
        ax.plot([], [], "-", color="0.15", lw=1.6, label=S.tex("closest %d analogues" % len(top)))
        ax.set_xlabel(S.label("omega_m")); ax.set_ylabel(S.label("sigma_8"))
        ax.set_xlim(0.15, 0.5); ax.set_ylim(0.55, 1.15)
        ax.legend(loc="upper right", fontsize=11, frameon=False)
        for j, (key, lab, rv) in enumerate((("z_om", r"$\Delta\Omega_\mathrm{m}\,/\,\sigma_{\rm 2pt}$", real_z[0]),
                                              ("z_s8", r"$\Delta\sigma_8\,/\,\sigma_{\rm 2pt}$", real_z[1]))):
            axh = fig.add_subplot(gs[0, 1 + j])
            axh.hist(A[key], bins=40, color="0.6", alpha=0.8)
            axh.axvline(rv, color=cf, lw=2.5)
            axh.axvline(0, color="0.3", lw=0.8, ls="--")
            axh.set_xlabel(S.tex(lab)); axh.set_yticks([])
            axh.set_title(S.tex("observation at the %.1f %% percentile" % pct(A[key], rv)), fontsize=13)
        T4._mock_tag(fig)
        fig.text(0.01, 0.955, S.tex("matched Gower mocks, $\\Lambda$CDM; the observation is the coloured arrow"), fontsize=11, color="0.45", ha="left", va="top")
        S.save(fig, os.path.join(args.out_dir, "shift_cloud.png"))
    print("[fig] %s" % os.path.join(args.out_dir, "shift_cloud.png"))

    # ---- analogue corners (same machinery as the reveal set) -------------------------------
    for k, i in enumerate(top):
        r = rows[i]; Pb, Pf, trb = cache[r["repeat"]]; ev = r["event"]
        posts = [T4.Post("2-pt (bandpowers, VD), repeat %d" % r["repeat"], Pb[:, ev, :], list(PARAMS), kind="ours",
                         colour=cb, role="arm", key="band", linewidth=T4.LW["arm"], label="2-pt"),
                 T4.Post("field level (NLA-M, VD), repeat %d" % r["repeat"], Pf[:, ev, :], list(PARAMS), kind="ours",
                         colour=cf, role="flagship", key="nla_m", linewidth=T4.LW["flagship"], label="field level")]
        truth = dict(zip(PARAMS, trb[ev].tolist()))
        out = os.path.join(args.out_dir, "analogue_%d_sim%d_aug%d_corner.png" % (k, r["sim"], r["aug"]))
        T4.corner(posts, PARAMS, out, "mock analogue %d (sim %d, aug %d)" % (k, r["sim"], r["aug"]), truth, rng,
                  args.max_samples, titles=True, mock=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
