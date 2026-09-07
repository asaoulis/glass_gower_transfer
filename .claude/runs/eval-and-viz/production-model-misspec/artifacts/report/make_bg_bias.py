"""Per-suite posterior bias of the two fixed-b_g suites relative to the truth, on the SAME 80 mocks
(paired at the mock level with the in-distribution suite), plus the joint prior chi^2 of each pinned
b_g configuration.  Outputs numbers_bg.{json,tex}.  CIs: cluster bootstrap by cosmology."""
import json
import os
import sys

import numpy as np
from scipy.stats import chi2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_report_figures import load_theta, cosmo_draws, MIS, MATCHES, N_BOOT, SEED  # noqa: E402
from src.KiDS.simulation_config import GALAXY_BIAS_PRIOR_MEANS, GALAXY_BIAS_PRIOR_SIGMAS  # noqa: E402

PARAMS = ["omega_m", "sigma_8", "w0", "mnu", "h", "ns", "ombh2", "a_ia", "b_ia"]
KEY = ["omega_m", "sigma_8", "w0"]


def load_samples(S, m):
    with np.load(f"{MIS}/{S}/misspec_posterior_samples_{m}.npz", allow_pickle=True) as f:
        return f["samples"].astype(np.float32), f["theta0s"].astype(np.float64), f["test_files"].astype(str), f["sim_ids"]


def main():
    rng = np.random.default_rng(SEED + 5)
    numbers = {}
    m = np.array(GALAXY_BIAS_PRIOR_MEANS); s = np.array(GALAXY_BIAS_PRIOR_SIGMAS)
    for name, bg in [("gb1p0", 1.0), ("gb1p3", 1.3)]:
        c2 = float((((bg - m) / s) ** 2).sum())
        numbers[f"{name}_prior_chi2"] = c2
        numbers[f"{name}_prior_chi2_p"] = float(chi2.sf(c2, 6))
        print(f"{name}: joint prior chi2 = {c2:.1f} (6 dof), p = {chi2.sf(c2, 6):.3g}")
    # per-suite mean z on the common 80 mocks, per encoder, then averaged
    files80 = None
    per = {S: {k: [] for k in KEY} for S in ["gower_bgp_nla_m", "gower_gb1p0", "gower_gb1p3"]}
    boot = {S: {k: [] for k in KEY} for S in per}
    for mt in MATCHES:
        data = {}
        for S in per:
            smp, th, files, sims = load_samples(S, mt)
            data[S] = (smp, th, files, sims)
        f0 = set(data["gower_gb1p0"][2]) & set(data["gower_gb1p3"][2]) & set(data["gower_bgp_nla_m"][2])
        files80 = sorted(f0)
        for S in per:
            smp, th, files, sims = data[S]
            idx = {f: i for i, f in enumerate(files)}
            sel = np.array([idx[f] for f in files80])
            mu, sd = smp[:, sel].mean(0), smp[:, sel].std(0)
            z = (th[sel] - mu) / np.maximum(sd, 1e-12)
            sim = sims[sel]
            draws = cosmo_draws(sim, sim, N_BOOT // 5, rng)
            for k in KEY:
                j = PARAMS.index(k)
                per[S][k].append(float(z[:, j].mean()))
                boot[S][k].append(np.array([z[rn, j].mean() for rn, _ in draws]))
    numbers["n_paired_mocks"] = len(files80)
    print(f"\nmean z = (truth - posterior mean)/posterior sd on the same {len(files80)} mocks, mean over 5 encoders [95% by-cosmology CI]")
    for S in per:
        for k in KEY:
            v = float(np.mean(per[S][k])); b = np.mean(boot[S][k], axis=0)
            lo, hi = float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
            tag = S.replace("gower_", "")
            numbers[f"{tag}_meanz_{k}"] = v; numbers[f"{tag}_meanz_{k}_lo"] = lo; numbers[f"{tag}_meanz_{k}_hi"] = hi
            numbers[f"{tag}_meanz_{k}_encsd"] = float(np.std(per[S][k]))
            print(f"  {S:16s} {k:8s} {v:+.2f} [{lo:+.2f}, {hi:+.2f}]  (sd over encoders {np.std(per[S][k]):.2f})")
    # linear response: does the two-slice line predict the in-distribution point at the prior-mean b_g?
    bg_mean = float(m.mean()); numbers["bg_prior_mean_avg"] = bg_mean
    for k in KEY:
        z0, z3 = numbers[f"gb1p0_meanz_{k}"], numbers[f"gb1p3_meanz_{k}"]
        pred = z0 + (z3 - z0) / 0.3 * (bg_mean - 1.0)
        numbers[f"linpred_meanz_{k}"] = float(pred)
        print(f"  linear prediction at b_g={bg_mean:.3f} for {k}: {pred:+.2f} (observed in-dist {numbers[f'bgp_nla_m_meanz_{k}']:+.2f})")
    with open(f"{HERE}/numbers_bg.json", "w") as f:
        json.dump(numbers, f, indent=1)
    digits = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
    lines = []
    for k, v in numbers.items():
        name = "B" + "".join(digits.get(ch, ch) for ch in k.replace("_", " ").title().replace(" ", "") if ch.isalnum())
        if isinstance(v, int):
            txt = f"{v}"
        elif "chi2_p" in k:
            if v > 1e-3:
                txt = f"{v:.2g}"
            else:
                e = int(np.floor(np.log10(v))); txt = f"{v / 10 ** e:.1f}\\times10^{{{e}}}"
        elif "chi2" in k:
            txt = f"{v:.1f}"
        elif "prior_mean_avg" in k:
            txt = f"{v:.2f}"
        else:
            txt = f"{v:+.2f}"
        lines.append(f"\\newcommand{{\\{name}}}{{{txt}}}  % {k}")
    with open(f"{HERE}/numbers_bg.tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} macros")


if __name__ == "__main__":
    main()
