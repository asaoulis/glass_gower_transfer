"""Two follow-up questions, computed from the same on-disk products:

(1) Does combining the cross-encoder KL with the summary-space kNN score raise the AUROC?
    Stouffer combination of the two calibrated (empirical-p -> probit) statistics.
(2) How does the kNN detector scale with the size of its reference cloud?  The reference cloud is
    re-fit (whitening + k=10 kNN tree, exactly as src/ml/eval/ood.py) on random subsets of the
    training summaries: by number of training COSMOLOGIES and by number of raw ROWS.

Outputs: numbers_supp.{json,tex}, fig_knn_refsize.pdf.  CIs: paired cluster bootstrap by cosmology.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def probit(p):
    return norm.isf(np.clip(p, 1e-6, 1 - 1e-6))

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_report_figures import (auroc, cosmo_draws, boot_stat, empirical_p, load_stage0, load_kl, load_p6,  # noqa: E402
                                 CK, INDIST, MATCHES, N_BOOT, SEED, DET_COL, COLORS, LABELS)
from src.ml.eval.ood import TrainWhitener, knn_scores  # noqa: E402

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "pdf.fonttype": 42, "font.family": "serif",
    "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif", "axes.linewidth": 0.6})
SUMM = f"{CK}/summaries"
SETS = ["gower_nla", "gower_nla_z", "gower_gb1p0", "gower_gb1p3", "gower_vd"]
N_DRAWS = 8
K = 10


def load_summ(setname, r):
    with np.load(f"{SUMM}/{setname}/summaries_{MATCHES[r]}.npz", allow_pickle=True) as f:
        return {"z": f["z"].astype(np.float64), "sim_ids": f["sim_ids"], "test_files": f["test_files"].astype(str)}


def main():
    rng = np.random.default_rng(SEED + 11)
    sc, ids, meta = load_stage0()
    R = meta["repeats"]
    numbers = {}
    sim_null = ids["_idtest"]["sim_ids"]

    # ------------------------------------------------------------------ (1) KL + kNN combination
    kl_null = load_kl(INDIST)
    order = {f: i for i, f in enumerate(kl_null["test_files"])}
    sel = np.array([order[f] for f in ids["_idtest"]["test_files"]])
    kl_null = kl_null["kl"][sel]
    u_null_kl = probit(empirical_p(kl_null, kl_null))
    u_null_knn = np.stack([probit(empirical_p(sc[r]["_idtest"]["knn"], sc[r]["_idtest"]["knn"])) for r in R])
    numbers["rho_kl_knn_null_probit"] = float(np.mean([np.corrcoef(u_null_kl, u_null_knn[r])[0, 1] for r in R]))
    print(f"null probit corr(KL, kNN) = {numbers['rho_kl_knn_null_probit']:.3f}")
    print("\n=== (1) KL + kNN Stouffer combination: AUROC [95% by-cosmology CI] ===")
    for S in SETS:
        klv = load_kl(S)
        files0 = ids[S]["test_files"]
        common = sorted(set(klv["test_files"]) & set(files0))
        i0 = {f: i for i, f in enumerate(files0)}; ik = {f: i for i, f in enumerate(klv["test_files"])}
        s0 = np.array([i0[f] for f in common]); sk = np.array([ik[f] for f in common])
        sim_q = ids[S]["sim_ids"][s0]
        u_kl = probit(empirical_p(kl_null, klv["kl"][sk]))
        u_knn = np.stack([probit(empirical_p(sc[r]["_idtest"]["knn"], sc[r][S]["knn"][s0])) for r in R])
        draws = cosmo_draws(sim_null, sim_q, N_BOOT, rng)
        res = {}
        for name, fnull, fq in [
            ("kl", lambda r: u_null_kl, lambda r: u_kl),
            ("knn1", lambda r: u_null_knn[r], lambda r: u_knn[r]),
            ("knn5", lambda r: u_null_knn.mean(0), lambda r: u_knn.mean(0)),
            ("kl_plus_knn1", lambda r: (u_null_kl + u_null_knn[r]) / np.sqrt(2), lambda r: (u_kl + u_knn[r]) / np.sqrt(2)),
            ("kl_plus_knn5", lambda r: (u_null_kl + u_null_knn.mean(0)) / np.sqrt(2), lambda r: (u_kl + u_knn.mean(0)) / np.sqrt(2)),
        ]:
            def fn(rn, rq, fnull=fnull, fq=fq):
                return float(np.mean([auroc(fnull(r)[rn], fq(r)[rq]) for r in R]))
            pt = fn(np.arange(sim_null.size), np.arange(sim_q.size))
            lo, hi, sd = boot_stat(fn, draws)
            res[name] = (pt, lo, hi)
            numbers[f"comb_{S}_{name}"] = pt; numbers[f"comb_{S}_{name}_lo"] = lo; numbers[f"comb_{S}_{name}_hi"] = hi
        print(f"  {S:12s} " + "  ".join(f"{k}={v[0]:.3f}[{v[1]:.3f},{v[2]:.3f}]" for k, v in res.items()))

    # ------------------------------------------------------------------ (2) reference-cloud size scan
    print("\n=== (2) kNN AUROC vs reference-cloud size (mean over 5 encoders; sd over random subsets) ===")
    train = {r: load_summ("_train", r) for r in R}
    idt = {r: load_summ("_idtest", r) for r in R}
    for r in R:
        assert (idt[r]["test_files"] == ids["_idtest"]["test_files"]).all()
    query = {S: {r: load_summ(S, r) for r in R} for S in SETS}
    for S in SETS:
        for r in R:
            assert (query[S][r]["test_files"] == ids[S]["test_files"]).all()
    train_cos = np.unique(train[0]["sim_ids"])  # each repeat has its own train/val split; sizes quoted for repeat 0
    n_train_rows = train[0]["z"].shape[0]
    numbers["n_train_cos"] = int(train_cos.size); numbers["n_train_rows"] = int(n_train_rows)
    print(f"  training cloud: {train_cos.size} cosmologies, {n_train_rows} rows")

    def scan(kind, sizes):
        out = {S: {} for S in SETS}
        for n in sizes:
            per_draw = {S: [] for S in SETS}
            n_draws = 1 if (kind == "cos" and n >= train_cos.size) or (kind == "rows" and n >= n_train_rows) else N_DRAWS
            for d in range(n_draws):
                rows_r = {}
                for r in R:  # each repeat has its own training split: subsample within it
                    if kind == "cos":
                        cos_r = np.unique(train[r]["sim_ids"])
                        pick = rng.choice(cos_r, min(n, cos_r.size), replace=False)
                        rows_r[r] = np.flatnonzero(np.isin(train[r]["sim_ids"], pick))
                    else:
                        rows_r[r] = rng.choice(train[r]["z"].shape[0], min(n, train[r]["z"].shape[0]), replace=False)
                rows = rows_r[0]
                for S in SETS:
                    vals = []
                    for r in R:
                        zt = train[r]["z"][rows_r[r]]
                        w = TrainWhitener.fit(zt)
                        zw_t = w(zt)
                        s_null = knn_scores(zw_t, w(idt[r]["z"]), k=K)
                        s_q = knn_scores(zw_t, w(query[S][r]["z"]), k=K)
                        vals.append(auroc(s_null, s_q))
                    per_draw[S].append(float(np.mean(vals)))
            for S in SETS:
                v = np.array(per_draw[S])
                out[S][n] = (float(v.mean()), float(v.std()) if v.size > 1 else 0.0, int(rows.size))
                print(f"  {kind}={n:6d} ({rows.size:5d} rows) {S:12s} AUROC={v.mean():.3f} sd_over_subsets={out[S][n][1]:.3f}")
        return out

    cos_sizes = [5, 10, 20, 40, 80, 120, 240]
    row_sizes = [50, 100, 200, 500, 1000, 3000, 10000, 19167]
    scan_cos = scan("cos", cos_sizes)
    scan_rows = scan("rows", row_sizes)
    for S in SETS:
        for n, (m, sd, nr) in scan_cos[S].items():
            numbers[f"refcos_{S}_n{n}"] = m; numbers[f"refcos_{S}_n{n}_sd"] = sd; numbers[f"refcos_{S}_n{n}_rows"] = nr
        for n, (m, sd, nr) in scan_rows[S].items():
            numbers[f"refrows_{S}_n{n}"] = m; numbers[f"refrows_{S}_n{n}_sd"] = sd

    # ------------------------------------------------------------------ figure
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.15))
    for S in ["gower_nla", "gower_nla_z", "gower_gb1p0", "gower_gb1p3", "gower_vd"]:
        m = np.array([scan_cos[S][n][0] for n in cos_sizes]); sd = np.array([scan_cos[S][n][1] for n in cos_sizes])
        ax.errorbar(cos_sizes, m, yerr=sd, color=COLORS[S], marker="o", ms=3, lw=1.1, capsize=1.5, elinewidth=0.7,
                    label=LABELS[S])
    ax.axhline(0.5, color="0.4", lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_xlabel("training cosmologies in the kNN reference cloud")
    ax.set_ylabel("kNN AUROC (mean of 5 enc.)")
    ax.set_ylim(0.4, 1.0); ax.grid(alpha=0.3, lw=0.5)
    ax.legend(loc="center right", frameon=False, ncol=2, fontsize=6.5, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(f"{HERE}/fig_knn_refsize.pdf")

    with open(f"{HERE}/numbers_supp.json", "w") as f:
        json.dump(numbers, f, indent=1)
    digits = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
    lines = []
    for k, v in numbers.items():
        name = "S" + "".join(digits.get(ch, ch) for ch in k.replace("gower_", "").replace("_", " ").title().replace(" ", "") if ch.isalnum())
        txt = f"{v}" if isinstance(v, int) else (f"{v:.3f}" if "sd" not in k else f"{v:.3f}")
        lines.append(f"\\newcommand{{\\{name}}}{{{txt}}}  % {k}")
    with open(f"{HERE}/numbers_supp.tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} macros")


if __name__ == "__main__":
    main()
