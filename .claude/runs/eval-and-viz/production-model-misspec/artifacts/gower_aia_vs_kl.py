"""Misspecification signal vs the EFFECTIVE IA amplitude A_IA^total.

⚠️ THE CONVERSION THIS SCRIPT EXISTS TO GET RIGHT
--------------------------------------------------
`a_ia` does NOT mean the same thing in the three IA models, so the raw parameter must never
be plotted on a shared axis. Measured on the stored `theta0s` of this arm:

  gower_bgp_nla_m   a_ia in [4.48, 7.00], median 5.75   <- RAW NLA-M amplitude parameter
  gower_nla         a_ia in [-6.0, +6.0], median -0.11  <- ALREADY A_IA^total
  gower_nla_z       a_ia in [-6.0, +6.0], median +0.12  <- ALREADY A_IA^total

The NLA and NLA-z variates store the amplitude directly in the KiDS-Legacy A_IA^total
convention (Wright et al. 2025), centred on zero. NLA-M instead stores the raw prefactor of

    A_eff^(i) = a_ia * f_red^(i) * (10^{log10 M^(i)} / 10^{13.5})^{b_ia}

(`src/cosmology/nla.py:kappa_ia_nla_m`, `f_red` from `src/KiDS/systematics.py:19`, per-bin
halo masses from `kids-legacy-sbi/data/priors/massdep_means.txt`). Averaged over the six
tomographic bins at the prior centre (a_ia = 5.74, b_ia = 0.44) this gives A_IA^total = 0.437,
reproducing the 0.44 recorded in the earlier `first-npe-misspecification` analysis.

An earlier version of this figure shaded the RAW NLA-M box [4.48, 7.0] as the "training
support" on an axis carrying NLA's A_IA^total. That put the training band an order of
magnitude away from where the model was actually trained and inverted the reading of the
figure: the NLA/NLA-z events nearest A_IA^total = 0 are the ones the NLA-M model is LEAST
prepared for, not the most. This script converts NLA-M per event and shades the resulting
band, so both fidelities sit in one convention.
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gower_style import COLORS, LABELS  # noqa: E402

PARAMS = ["omega_m", "sigma_8", "w0", "mnu", "h", "ns", "ombh2", "a_ia", "b_ia"]
J_AIA, J_BIA = PARAMS.index("a_ia"), PARAMS.index("b_ia")

# the scaler mapped every variate's a_ia/b_ia through the NLA-M preset box, so unscaling with
# it recovers each variate's own stored physical value (raw for nla_m, A_IA^total for nla*).
AIA_BOX, BIA_BOX = (4.48, 7.0), (0.28, 0.6)

# NLA-M effective-amplitude conversion (see docstring)
F_RED = np.array([0.15, 0.20, 0.17, 0.24, 0.19, 0.03])
LOG10_M = np.array([11.69224599, 12.45768451, 12.76280679, 12.93435839,
                    13.08446641, 13.21675774])
LOG10_M_PIVOT = 13.5


def nla_m_effective(a_ia, b_ia):
    """Raw NLA-M (a_ia, b_ia) -> A_IA^total, averaged over the six tomographic bins."""
    a = np.asarray(a_ia, dtype=np.float64)[:, None]
    b = np.asarray(b_ia, dtype=np.float64)[:, None]
    per_bin = a * F_RED[None, :] * 10.0 ** ((LOG10_M[None, :] - LOG10_M_PIVOT) * b)
    return per_bin.mean(axis=1)


def unscale(x, box):
    return x * (box[1] - box[0]) + box[0]


def load(root, variate, match):
    p = os.path.join(root, variate, f"misspec_posterior_samples_{match}.npz")
    with np.load(p, allow_pickle=True) as f:
        th, s, files = f["theta0s"], f["samples"], f["test_files"]
    mu, sd = s.mean(0), s.std(0)
    z = (th - mu) / np.maximum(sd, 1e-12)
    a_raw = unscale(th[:, J_AIA], AIA_BOX)
    b_raw = unscale(th[:, J_BIA], BIA_BOX)
    return {"a_raw": a_raw, "b_raw": b_raw, "z": z, "files": files}


def kl_map(root, variate, matches):
    """Per-event cross-repeat KL, keyed by test_file."""
    tag = "_".join(matches)
    p = os.path.join(root, variate, f"misspec_repeat_disagreement_{tag}.npz")
    if not os.path.exists(p):
        return None
    with np.load(p, allow_pickle=True) as f:
        d = {k: f[k] for k in f.files}
    key = "kl" if "kl" in d else next((k for k in d if "kl" in k.lower()), None)
    fk = "test_files" if "test_files" in d else None
    if key is None or fk is None:
        return None
    return dict(zip(d[fk].tolist(), np.asarray(d[key], dtype=np.float64)))


def binned(x, y, n=14, minc=6):
    e = np.unique(np.quantile(x, np.linspace(0, 1, n + 1)))
    c, m, s = [], [], []
    for lo, hi in zip(e[:-1], e[1:]):
        k = (x >= lo) & (x <= hi)
        if k.sum() >= minc:
            c.append(0.5 * (lo + hi))
            m.append(np.median(y[k]))
            # 16-84 percentiles, not mean +- sd: the KL is heavy-tailed and plotted on a
            # log axis, where a symmetric sd band underflows into a meaningless funnel.
            s.append(np.percentile(y[k], [16, 84]))
    return np.array(c), np.array(m), np.array(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--matches", nargs="+",
                    default=[f"ncosmo300_{i}" for i in range(5)])
    ap.add_argument("--match", default="ncosmo300_0")
    ap.add_argument("--variates", nargs="+", default=["gower_nla", "gower_nla_z"])
    ap.add_argument("--indist", default="gower_bgp_nla_m")
    ap.add_argument("--out", default="gower_aia_vs_kl.png")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    ind = load(args.root, args.indist, args.match)
    a_eff_id = nla_m_effective(ind["a_raw"], ind["b_raw"])
    lo, hi = np.percentile(a_eff_id, [5, 95])
    print(f"[{args.indist}] NLA-M effective A_IA^total: median {np.median(a_eff_id):.3f}, "
          f"5-95% [{lo:.3f}, {hi:.3f}], full [{a_eff_id.min():.3f}, {a_eff_id.max():.3f}]")
    print("    (earlier first-npe-misspecification analysis adopted the band [0.09, 0.78])")

    fig, axes = plt.subplots(1, len(args.variates),
                             figsize=(5.9 * len(args.variates), 4.5), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, name in zip(axes, args.variates):
        d = load(args.root, name, args.match)
        km = kl_map(args.root, name, args.matches)
        if km is None:
            raise SystemExit(f"no cross-repeat KL npz for {name}")
        y = np.array([km.get(f, np.nan) for f in d["files"].tolist()])
        x = d["a_raw"]                     # already A_IA^total for nla / nla_z
        k = np.isfinite(x) & np.isfinite(y)
        x, y = x[k], y[k]

        ax.axvspan(lo, hi, color="0.86", zorder=0,
                   label=f"NLA-M training support\n"
                         rf"$A_{{\rm IA}}^{{\rm total}} \in [{lo:.2f},\ {hi:.2f}]$ (5–95%)")
        ax.axvline(float(np.median(a_eff_id)), color="0.45", ls="-", lw=1.2, zorder=1,
                   label=rf"NLA-M median {np.median(a_eff_id):.2f}")
        c = COLORS.get(name, "tab:orange")
        ax.scatter(x, y, s=9, alpha=0.30, color=c, lw=0, zorder=2)
        bc, bm, bs = binned(x, y)
        ax.plot(bc, bm, color=c, lw=2.4, zorder=3, label=f"{LABELS.get(name, name)}: binned median")
        ax.fill_between(bc, bs[:, 0], bs[:, 1], color=c, alpha=0.16, lw=0, zorder=2)

        ax.set_yscale("log")
        ax.set_xlabel(r"true $A_{\rm IA}^{\rm total}$")
        ax.set_title(LABELS.get(name, name))
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        ax.grid(alpha=0.25, lw=0.5)

    axes[0].set_ylabel("per-event cross-repeat KL")
    fig.suptitle(r"Misspecification signal vs effective IA amplitude — "
                 r"NLA-M converted to $A_{\rm IA}^{\rm total}$", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, dpi=args.dpi)
    print(f"wrote {args.out}")

    print("\nCAVEATS (kept from the original figure):")
    print("  * a_ia under nla/nla_z is a DIFFERENT parametrisation from NLA-M's; only after the")
    print("    conversion above do the two sit in one convention.")
    print("  * the ~25% intractable events are absent, so the trend is a LOWER bound.")


if __name__ == "__main__":
    main()
