"""Per-event KL vs kNN (encoder 0) on the null and on the NLA suite: are the two detectors redundant?"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, norm
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from make_report_figures import load_stage0, load_kl, empirical_p, INDIST, COLORS
sc, ids, meta = load_stage0()
kn = load_kl(INDIST); o = {f: i for i, f in enumerate(kn["test_files"])}
sel = np.array([o[f] for f in ids["_idtest"]["test_files"]]); kl_null = kn["kl"][sel]
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
for ax, S in zip(axes, ["_idtest", "gower_nla", "gower_gb1p0"]):
    if S == "_idtest":
        kl, knn = kl_null, sc[0]["_idtest"]["knn"]
    else:
        kv = load_kl(S); ok = {f: i for i, f in enumerate(kv["test_files"])}
        files = [f for f in ids[S]["test_files"] if f in ok]
        i0 = {f: i for i, f in enumerate(ids[S]["test_files"])}
        kl = kv["kl"][[ok[f] for f in files]]; knn = sc[0][S]["knn"][[i0[f] for f in files]]
    u_kl = norm.isf(np.clip(empirical_p(kl_null, kl), 1e-6, 1 - 1e-6))
    u_kn = norm.isf(np.clip(empirical_p(sc[0]["_idtest"]["knn"], knn), 1e-6, 1 - 1e-6))
    rho_s = spearmanr(kl, knn).correlation; rho_p = np.corrcoef(u_kl, u_kn)[0, 1]
    ax.scatter(u_kn, u_kl, s=6, alpha=0.4, lw=0, color=COLORS.get(S, "0.3"))
    ax.axhline(norm.isf(0.05), ls=":", color="0.4"); ax.axvline(norm.isf(0.05), ls=":", color="0.4")
    ax.set_title(f"{S.strip('_')}: Spearman {rho_s:+.2f}, probit r {rho_p:+.2f}", fontsize=9)
    ax.set_xlabel("kNN (encoder 0), probit of empirical p"); ax.set_ylabel("cross-encoder KL, probit of empirical p")
    print(f"{S:12s} n={kl.size}: Spearman(KL, kNN) = {rho_s:+.3f}; probit corr = {rho_p:+.3f}; "
          f"frac flagged by both = {np.mean((u_kl > norm.isf(0.05)) & (u_kn > norm.isf(0.05))):.3f}, "
          f"kNN only = {np.mean((u_kl <= norm.isf(0.05)) & (u_kn > norm.isf(0.05))):.3f}, "
          f"KL only = {np.mean((u_kl > norm.isf(0.05)) & (u_kn <= norm.isf(0.05))):.3f}")
fig.tight_layout(); fig.savefig(f"{HERE}/kl_vs_knn_scatter.png", dpi=150); print("wrote kl_vs_knn_scatter.png")
