#!/usr/bin/env python
"""Tier-4 REVEAL figure set: PHYSICAL-unit posteriors of the KiDS-Legacy observation, numbered in
the notebook's section-8 order, with the public external chains over-plotted.

    01  2-pt (bandpowers) ΛCDM                       -- Stage 1 (the 2-pt arm alone)
    02  2-pt + field level ΛCDM                      -- Stage 2
    03  2-pt wCDM                                    -- Stage 1
    04  2-pt + field level wCDM                      -- Stage 2
    05  final ΛCDM: field + 2-pt vs Planck 2018      -- Stage 2
    06  S_8 summary whiskers (ours, KiDS-Legacy, companion 2-pt SBI, Planck)   -- Stage 2
  per group: `a` = headline corner (pooled / final), `b` = repeats-vs-pooled diagnostic,
  `c` = headline + external comparisons; `tables/<group>.{csv,md}` = mean / std / median / 68 %.

Two sources, one code path:
  --source mock   a matched Gower mock (truth known, drawn as black lines): the LAYOUT rehearsal.
                  Reads `<matched-root>/<exp>/external/<matched-tag>/` per arm x repeat x prior
                  (`--event K` picks the event). Pooled-equivalent = the repeats' draws concatenated
                  (equal weight, like `eval.py --mode pool`). Every file/title is `MOCK_`-prefixed.
  --source real   the blind store label (`--label A2long`): pooled runs + per-repeat runs, read
                  ONLY through `src.blind.standardise` (the sole raw reader). THIS IS THE REVEAL:
                  it is run by the user (guard_blind denies the agent); nothing is printed except
                  output paths -- every number goes to files.

Stage gate (UNBLINDING_PROTOCOL Tier 4, one-way): `--stage 1` writes only the 2-pt groups (01, 03)
into `stage1_2pt/`; `--stage 2` the field groups (02, 04, 05, 06) into `stage2_field/`; `all` both.

External chains (public):
  KiDS-Legacy (Wright et al. 2025) CosmoSIS/Nautilus text chains, `--kids-dir` = the extracted
  release's `chains_and_config_files/`, `--kids-stats cosebis bandpowers [xipm cosebis_plus_external]`;
  weight = exp(log_weight). ΛCDM only (w = -1 fixed) -> never on the wCDM figures.
  Planck 2018 GetDist chains (`--planck-dir` = the `base/plikHM_TTTEEE_lowl_lowE_lensing/` dir):
  col 0 weight, col 1 -lnL, params per `.paramnames`; first `--planck-burnin` fraction of each chain dropped.
  Companion score-based 2-pt SBI (`--companion-dir` with `novd/` and `vd/`): loader `load_companion`
  (format set once the files are inspected; a missing dir is skipped with a note).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.blind import BLIND_ROOT                                   # noqa: E402
from src.blind.standardise import _load_physical, _scaler_box, list_raw_runs, prior_of_dump  # noqa: E402
from src.ml.eval.arms import ARMS                                  # noqa: E402

PRIOR_NAMES = {"LCDM_fixed_w0": r"$\Lambda$CDM", "kids_s8_analytic": r"$w$CDM"}
PRIOR_PARAMS = {"LCDM_fixed_w0": ("omega_m", "sigma_8", "S8"),
                "kids_s8_analytic": ("omega_m", "sigma_8", "S8", "w0")}
KIDS_STAT_NAMES = {"cosebis": "KiDS-Legacy COSEBIs (Wright+25)", "bandpowers": "KiDS-Legacy band powers (Wright+25)",
                   "xipm": r"KiDS-Legacy $\xi_\pm$ (Wright+25)",
                   "cosebis_plus_external": "KiDS-Legacy COSEBIs + DES Y3 + BAO + SN (Stölzner+25)"}
COMPANION_NAMES = {"novd": "companion 2-pt SBI (score, no VD)", "vd": "companion 2-pt SBI (score, VD)"}
PLANCK_NAME = "Planck 2018 TT,TE,EE+lowE+lensing"
# external chain colours: fixed, distinct from the arm colours (src.viz.style pins those per arm)
EXT_COLOURS = {"cosebis": "#555555", "bandpowers": "#999999", "xipm": "#bbbbbb", "cosebis_plus_external": "#332288",
               "novd": "#882255", "vd": "#117733", "planck": "#cc3311"}


# --------------------------------------------------------------------------------------------
# one posterior = physical draws + names (+ optional weights, truth)
# --------------------------------------------------------------------------------------------
class Post:
    def __init__(self, name, x, names, weights=None, kind="ours", colour=None, role="outline", key=""):
        self.name, self.x, self.names = name, np.asarray(x, dtype=np.float64), list(names)
        self.weights = None if weights is None else np.asarray(weights, dtype=np.float64)
        self.kind, self.colour, self.role, self.key = kind, colour, role, key

    def cols(self, params: Sequence[str]) -> np.ndarray:
        idx = [self.names.index(p) for p in params if p in self.names]
        return self.x[:, idx]

    def has(self, params: Sequence[str]) -> bool:
        return all(p in self.names for p in params)

    def thinned(self, n: int, rng: np.random.Generator):
        """Seeded UNIFORM thinning (never a head slice: pooled dumps are repeats concatenated)."""
        if n <= 0 or len(self.x) <= n:
            return self.x, self.weights
        sel = np.sort(rng.choice(len(self.x), size=n, replace=False))
        return self.x[sel], (None if self.weights is None else self.weights[sel])


def _wq(x: np.ndarray, q: Sequence[float], w: Optional[np.ndarray]) -> np.ndarray:
    if w is None:
        return np.quantile(x, q)
    o = np.argsort(x)
    cw = np.cumsum(w[o]); cw /= cw[-1]
    return np.interp(q, cw, x[o])


def summarise(p: Post, params: Sequence[str]) -> List[dict]:
    rows = []
    for n in params:
        if n not in p.names:
            continue
        v = p.x[:, p.names.index(n)]
        w = p.weights
        mean = np.average(v, weights=w)
        std = np.sqrt(np.average((v - mean) ** 2, weights=w))
        q16, q50, q84 = _wq(v, [0.16, 0.5, 0.84], w)
        rows.append(dict(posterior=p.name, param=n, mean=mean, std=std, median=q50, p16=q16, p84=q84,
                         n_draws=len(v), weighted=w is not None))
    return rows


# --------------------------------------------------------------------------------------------
# our posteriors: mock (matched dumps) or real (blind store)
# --------------------------------------------------------------------------------------------
def _dump_match(fname: str) -> str:
    body = os.path.splitext(fname)[0]
    return "ncosmo" + body.split("_ncosmo", 1)[1] if "_ncosmo" in body else ""


def mock_runs(root: str, tag: str, arms: Sequence[str], repeats: Sequence[int], priors: Sequence[str]) -> List[dict]:
    out = []
    for arm in arms:
        tmpl = ARMS[arm][0]
        for r in repeats:
            exp = tmpl.format(r=r)
            d = os.path.join(root, exp, "external", tag)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                pr = prior_of_dump(f)
                if not pr or pr not in priors or "pooled" in f or not f.startswith("external_"):
                    continue
                m = _dump_match(f)
                if not m.endswith("_%d" % r):
                    continue
                out.append(dict(label="MOCK", arm=arm, experiment=exp, prior=pr, match=m, pooled=False,
                                path=os.path.join(d, f), repeat=r))
    return out


def mock_truth(run: dict, event: int) -> Dict[str, float]:
    """Physical truth of ONE event from the dump's scaled `theta0s` (the experiment's own 9-param
    box, `_scaler_box(exp, None)`); S8 derived."""
    with np.load(run["path"], allow_pickle=False) as d:
        if "theta0s" not in d.files:
            return {}
        t = np.asarray(d["theta0s"])[event]
        sim = int(d["sim_ids"][event]) if "sim_ids" in d.files else -1
        aug = int(d["aug_ids"][event]) if "aug_ids" in d.files else -1
    names, lo, hi = _scaler_box(run["experiment"], None)
    if len(names) != len(t):
        raise ValueError("theta0s has %d columns, the experiment box %d (%s)" % (len(t), len(names), names))
    x = t * (hi - lo) + lo
    tr = dict(zip(names, x.tolist()))
    tr["S8"] = tr["sigma_8"] * np.sqrt(tr["omega_m"] / 0.3)
    tr["_sim_id"], tr["_aug_id"] = sim, aug
    return tr


def load_ours(args) -> tuple[Dict[tuple, Post], Dict[tuple, List[Post]], Dict[str, float], List[dict]]:
    """{(arm, prior): headline Post}, {(arm, prior): [repeat Posts]}, truth (mock only), provenance."""
    from src.viz import style as S
    heads, reps, truth, prov = {}, {}, {}, []
    priors = list(PRIOR_PARAMS)
    if args.source == "mock":
        runs = mock_runs(args.matched_root, args.matched_tag, args.arms, args.repeats, priors)
        event = args.event
    else:
        runs = [r for r in list_raw_runs(BLIND_ROOT) if r["label"] == args.label and r["arm"] in args.arms]
        event = 0
    if not runs:
        raise SystemExit("no runs found for source=%s" % args.source)
    for arm in args.arms:
        for prior in priors:
            rr = sorted([r for r in runs if r["arm"] == arm and r["prior"] == prior and not r["pooled"]],
                        key=lambda r: r["match"])
            pooled = [r for r in runs if r["arm"] == arm and r["prior"] == prior and r["pooled"]]
            if not rr and not pooled:
                continue
            ramp = S.sequential(max(len(rr), 2))
            rl = []
            for i, r in enumerate(rr):
                x, names = _load_physical(r["path"], r["experiment"], event)
                rl.append(Post("%s, repeat %s" % (S.arm_name(arm), r["match"].rsplit("_", 1)[-1]), x, names,
                               kind="ours", colour=ramp[i], role="outline", key="%s_%s" % (arm, r["match"])))
                prov.append(dict(arm=arm, prior=prior, match=r["match"], file=os.path.basename(r["path"]),
                                 experiment=r["experiment"], event=event, n=len(x)))
                if args.source == "mock" and not truth:
                    truth = mock_truth(r, event)
            reps[(arm, prior)] = rl
            if pooled:
                x, names = _load_physical(pooled[0]["path"], pooled[0]["experiment"], 0)
                hname = "%s (pooled)" % S.arm_name(arm)
                prov.append(dict(arm=arm, prior=prior, match="pooled", file=os.path.basename(pooled[0]["path"]),
                                 experiment=pooled[0]["experiment"], event=0, n=len(x)))
            elif len(rl) > 1:
                x = np.concatenate([p.x for p in rl], axis=0); names = rl[0].names
                hname = "%s (pooled %d repeats)" % (S.arm_name(arm), len(rl))
            else:
                x, names = rl[0].x, rl[0].names
                hname = "%s (repeat %s only, no pool)" % (S.arm_name(arm), rr[0]["match"].rsplit("_", 1)[-1])
            heads[(arm, prior)] = Post(hname, x, names, kind="ours", colour=S.arm_colour(arm),
                                       role="flagship" if arm == "nla_m" else "arm", key=arm)
    return heads, reps, truth, prov


# --------------------------------------------------------------------------------------------
# external chains
# --------------------------------------------------------------------------------------------
def load_kids(kids_dir: str, stats: Sequence[str]) -> List[Post]:
    out = []
    for st in stats:
        d = os.path.join(kids_dir, st)
        if not os.path.isdir(d):
            print("[kids] missing %s -> skipped" % d); continue
        fs = [f for f in os.listdir(d) if f.startswith("output_nautilus") and f.endswith(".txt")]
        if not fs:
            print("[kids] no chain in %s -> skipped" % d); continue
        path = os.path.join(d, sorted(fs)[0])
        with open(path) as fh:
            hdr = fh.readline().lstrip("#").strip().split("\t")
        X = np.loadtxt(path)
        col = {h.upper(): i for i, h in enumerate(hdr)}
        # the release's derived `COSMOLOGICAL_PARAMETERS--S_8` column is all-NaN (the sampled
        # parameter is `s_8_input`); derive S8 = sigma_8 sqrt(Omega_m/0.3) like everywhere else.
        # Rows with log_weight = -inf are prior-rejected Nautilus points: dropped.
        idx = [col["COSMOLOGICAL_PARAMETERS--OMEGA_M"], col["COSMOLOGICAL_PARAMETERS--SIGMA_8"]]
        lw = X[:, col["LOG_WEIGHT"]]
        ok = np.isfinite(lw) & np.all(np.isfinite(X[:, idx]), axis=1)
        Y = X[ok][:, idx]
        Y = np.concatenate([Y, (Y[:, 1] * np.sqrt(Y[:, 0] / 0.3))[:, None]], axis=1)
        w = np.exp(lw[ok] - lw[ok].max())
        out.append(Post(KIDS_STAT_NAMES.get(st, st), Y, ["omega_m", "sigma_8", "S8"], weights=w, kind="kids",
                        colour=EXT_COLOURS.get(st, "#777777"), role="outline", key="kids_" + st))
    return out


def load_planck(planck_dir: str, burnin: float) -> Optional[Post]:
    if not planck_dir or not os.path.isdir(planck_dir):
        print("[planck] missing %s -> skipped" % planck_dir); return None
    pn = [f for f in os.listdir(planck_dir) if f.endswith(".paramnames") and "post_" not in f]
    if not pn:
        print("[planck] no .paramnames in %s -> skipped" % planck_dir); return None
    root = os.path.join(planck_dir, pn[0][:-len(".paramnames")])
    names = [l.split()[0].rstrip("*") for l in open(root + ".paramnames") if l.strip()]
    chains = sorted(f for f in os.listdir(planck_dir) if f.startswith(os.path.basename(root) + "_")
                    and f[len(os.path.basename(root)) + 1:-4].isdigit() and f.endswith(".txt"))
    parts, ws = [], []
    for f in chains:
        C = np.loadtxt(os.path.join(planck_dir, f))
        C = C[int(burnin * len(C)):]
        parts.append(C); ws.append(C[:, 0])
    C = np.concatenate(parts); w = np.concatenate(ws)
    idx = [2 + names.index(n) for n in ("omegam", "sigma8", "S8")]
    return Post(PLANCK_NAME, C[:, idx], ["omega_m", "sigma_8", "S8"], weights=w, kind="planck",
                colour=EXT_COLOURS["planck"], role="outline", key="planck")


def load_companion(companion_dir: str) -> List[Post]:
    """Companion score-based 2-pt SBI samples (hypatia `sbi_score_{novd,vd}_real_data/`).
    Format: TODO once the files are fetched and inspected (npz / txt / h5). Until then: tries a
    `samples*.npz` with keys (`samples`, `names`/`param_names`) or a `samples*.txt` with a
    `#`-header of column names; anything else is reported and skipped."""
    out = []
    if not companion_dir or not os.path.isdir(companion_dir):
        print("[companion] missing %s -> skipped" % companion_dir); return out
    for tag in ("novd", "vd"):
        d = os.path.join(companion_dir, tag)
        if not os.path.isdir(d):
            print("[companion] no %s/ -> skipped" % tag); continue
        fs = sorted(f for f in os.listdir(d) if f.startswith("samples"))
        if not fs:
            print("[companion] %s/: no samples* file -> skipped" % tag); continue
        path = os.path.join(d, fs[0])
        try:
            if path.endswith(".npz"):
                with np.load(path, allow_pickle=True) as z:
                    keys = list(z.files)
                    x = np.asarray(z["samples"]) if "samples" in keys else np.asarray(z[keys[0]])
                    names = [str(n) for n in (z["names"] if "names" in keys else z["param_names"])]
            elif path.endswith(".npy"):
                x = np.load(path); names = ["omega_m", "sigma_8"][: x.shape[1]]
            else:
                with open(path) as fh:
                    hdr = fh.readline().lstrip("#").split()
                x = np.loadtxt(path); names = hdr
            x = x.reshape(-1, x.shape[-1])
            lower = [n.lower() for n in names]
            alias = {"omega_m": ("omega_m", "omegam", "om", "omega_matter"), "sigma_8": ("sigma_8", "sigma8", "s8_sigma"),
                     "S8": ("s8", "s_8"), "w0": ("w0", "w")}
            cols, out_names = [], []
            for n, al in alias.items():
                for a in al:
                    if a in lower:
                        cols.append(lower.index(a)); out_names.append(n); break
            X = x[:, cols]
            if "S8" not in out_names and {"omega_m", "sigma_8"} <= set(out_names):
                s8 = X[:, out_names.index("sigma_8")] * np.sqrt(X[:, out_names.index("omega_m")] / 0.3)
                X = np.concatenate([X, s8[:, None]], axis=1); out_names.append("S8")
            out.append(Post(COMPANION_NAMES[tag], X, out_names, kind="companion", colour=EXT_COLOURS[tag],
                            role="outline", key="companion_" + tag))
            print("[companion] %s: %s (%d draws, %s)" % (tag, fs[0], len(X), out_names))
        except Exception as e:  # noqa: BLE001
            print("[companion] %s/%s: could not parse (%s) -> skipped; fix load_companion" % (tag, fs[0], e))
    return out


# --------------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------------
def corner(posts: List[Post], params: Sequence[str], out: str, title: str, truth: Dict[str, float],
           rng: np.random.Generator, max_samples: int, figsize=(8, 8), no_titles=False, mock=False):
    from src.viz import style as S
    posts = [p for p in posts if p.has(params)]
    if not posts:
        print("[skip] %s: no posterior carries %s" % (os.path.basename(out), params)); return None
    labels = [S.label(p) for p in params]
    sd, cc, kw, wt = {}, {}, {}, {}
    for p in posts:
        x, w = p.thinned(max_samples, rng)
        idx = [p.names.index(q) for q in params]
        sd[p.name] = x[:, idx]; cc[p.name] = labels
        kw[p.name] = S.chain_kwargs(p.role, p.colour)
        if w is not None:
            wt[p.name] = w
    tr = {S.label(q): truth[q] for q in params if q in truth} if truth else None
    fig = S.plot_chains(sd, cc, truth=tr or None, plotting_kwargs=kw, figsize=figsize, smooth=S.CORNER_SMOOTH,
                        close=False, weights=wt or None)
    with S.context():
        if not no_titles:
            fig.suptitle(S.tex(("MOCK -- " if mock else "") + title), fontsize=13, y=1.0)
        S.save(fig, out)
    print("[fig] %s" % out)
    return out


def s8_whiskers(posts: List[Post], out: str, title: str, truth: Dict[str, float], no_titles=False, mock=False):
    from src.viz import style as S
    import matplotlib.pyplot as plt
    posts = [p for p in posts if "S8" in p.names]
    if not posts:
        return None
    with S.context():
        fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(posts) + 1.2))
        ys, labels = [], []
        for i, p in enumerate(posts):
            v = p.x[:, p.names.index("S8")]
            q16, q50, q84 = _wq(v, [0.16, 0.5, 0.84], p.weights)
            y = len(posts) - 1 - i
            ax.errorbar(q50, y, xerr=[[q50 - q16], [q84 - q50]], fmt="o", color=p.colour or "k",
                        capsize=3, lw=1.6, ms=5)
            ys.append(y); labels.append(S.tex(p.name))
        if truth and "S8" in truth:
            ax.axvline(truth["S8"], color="k", lw=1, ls="--", label="truth")
        ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=9)
        for tl, p in zip(ax.get_yticklabels(), posts):
            tl.set_color(p.colour or "k")
        ax.set_xlabel(S.label("S8")); ax.set_ylim(-0.7, len(posts) - 0.3)
        ax.grid(axis="x", alpha=0.3)
        if not no_titles:
            ax.set_title(S.tex(("MOCK -- " if mock else "") + title), fontsize=11)
        S.save(fig, out)
    print("[fig] %s" % out)
    return out


def write_table(posts: List[Post], params: Sequence[str], out_base: str, note: str = ""):
    rows = []
    for p in posts:
        rows.extend(summarise(p, params))
    if not rows:
        return
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    keys = ["posterior", "param", "mean", "std", "median", "p16", "p84", "n_draws", "weighted"]
    with open(out_base + ".csv", "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join("%.5g" % r[k] if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    with open(out_base + ".md", "w") as fh:
        if note:
            fh.write(note + "\n\n")
        fh.write("| posterior | param | mean | std | median | 68% interval | n | weighted |\n|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            fh.write("| %s | %s | %.4f | %.4f | %.4f | [%.4f, %.4f] | %d | %s |\n" % (
                r["posterior"], r["param"], r["mean"], r["std"], r["median"], r["p16"], r["p84"], r["n_draws"], r["weighted"]))
    print("[table] %s.{csv,md}" % out_base)


# --------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("mock", "real"), required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stage", choices=("1", "2", "all"), default="all")
    ap.add_argument("--label", default="A2long", help="real: blind-store label")
    ap.add_argument("--matched-root", default="/data/alex/variate_samples_kids")
    ap.add_argument("--matched-tag", default="gower_match_vdq")
    ap.add_argument("--event", type=int, default=0, help="mock: event index of the matched dump")
    ap.add_argument("--arms", nargs="+", default=["band", "nla_m"])
    ap.add_argument("--repeats", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--kids-dir", default="/data/alex/unblinding/external/kids_legacy_release/"
                    "KiDS_Legacy_cosmic_shear_data_release/chains_and_config_files")
    ap.add_argument("--kids-stats", nargs="*", default=["cosebis", "bandpowers"])
    ap.add_argument("--planck-dir", default="/data/alex/unblinding/external/planck2018/extracted/base/"
                    "plikHM_TTTEEE_lowl_lowE_lensing")
    ap.add_argument("--planck-burnin", type=float, default=0.3)
    ap.add_argument("--companion-dir", default="/data/alex/unblinding/external/companion_2pt")
    ap.add_argument("--max-samples", type=int, default=20000, help="seeded uniform thinning for the contours (tables use all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-titles", action="store_true")
    ap.add_argument("--corner-size", type=float, default=8.0)
    args = ap.parse_args(argv)

    mock = args.source == "mock"
    pre = "MOCK_" if mock else ""
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    s1 = os.path.join(args.out_dir, "stage1_2pt"); s2 = os.path.join(args.out_dir, "stage2_field")
    tdir = os.path.join(args.out_dir, "tables")
    do1, do2 = args.stage in ("1", "all"), args.stage in ("2", "all")

    heads, reps, truth, prov = load_ours(args)
    kids = load_kids(args.kids_dir, args.kids_stats) if args.kids_stats else []
    planck = load_planck(args.planck_dir, args.planck_burnin)
    comp = load_companion(args.companion_dir)
    fs = (args.corner_size, args.corner_size)
    L, W = "LCDM_fixed_w0", "kids_s8_analytic"
    tt = {"band": "2-pt", "nla_m": "field"}
    written = []

    def head(arm, prior):
        return heads.get((arm, prior))

    def group(num, name, prior, arms_in, stage_dir, externals):
        """a: headline(s); b: repeats-vs-pooled per arm; c: + externals; table."""
        params = PRIOR_PARAMS[prior]
        hs = [head(a, prior) for a in arms_in if head(a, prior) is not None]
        if not hs:
            print("[skip] group %s: no posterior for %s" % (num, prior)); return
        tag = "%s%s" % (pre, num)
        ttl = "%s, %s" % (" + ".join(tt[a] for a in arms_in), PRIOR_NAMES[prior])
        written.append(corner(hs, params, os.path.join(stage_dir, "%sa_%s_corner.png" % (tag, name)), ttl + ": headline (pooled)",
                              truth, rng, args.max_samples, fs, args.no_titles, mock))
        for a in arms_in:
            rl = reps.get((a, prior), [])
            if len(rl) > 1 and head(a, prior) is not None:
                written.append(corner(rl + [Post(head(a, prior).name, head(a, prior).x, head(a, prior).names,
                                                 colour="k", role="real", key=a)],
                                      params, os.path.join(stage_dir, "%sb_%s_repeats_%s.png" % (tag, name, a)),
                                      "%s %s: repeats vs pooled" % (tt[a], PRIOR_NAMES[prior]), truth, rng,
                                      args.max_samples, fs, args.no_titles, mock))
        ext = [e for e in externals if e is not None and e.has(params)]
        if ext:
            written.append(corner(ext + hs, params, os.path.join(stage_dir, "%sc_%s_vs_external.png" % (tag, name)),
                                  ttl + ": vs external", truth, rng, args.max_samples, fs, args.no_titles, mock))
        allp = hs + [p for a in arms_in for p in reps.get((a, prior), [])] + ext
        write_table(allp, params, os.path.join(tdir, "%s_%s" % (tag, name)), note="%s%s" % ("MOCK (truth: %s). " % {
            k: round(v, 4) for k, v in truth.items() if k in params} if mock else "", ttl))

    if do1:
        group("01", "2pt_LCDM", L, ["band"], s1, kids + comp)
        group("03", "2pt_wCDM", W, ["band"], s1, [])
    if do2:
        group("02", "2pt_field_LCDM", L, ["band", "nla_m"], s2, kids + comp)
        group("04", "2pt_field_wCDM", W, ["band", "nla_m"], s2, [])
        if planck is not None:
            hs = [head(a, L) for a in ("band", "nla_m") if head(a, L) is not None]
            if hs:
                written.append(corner([planck] + hs, PRIOR_PARAMS[L], os.path.join(s2, pre + "05_LCDM_vs_Planck_corner.png"),
                                      "%s: field + 2-pt vs Planck 2018" % PRIOR_NAMES[L], truth, rng, args.max_samples, fs,
                                      args.no_titles, mock))
                write_table(hs + [planck], PRIOR_PARAMS[L], os.path.join(tdir, pre + "05_LCDM_vs_Planck"))
        allL = [head(a, L) for a in ("nla_m", "band") if head(a, L) is not None] + comp + kids + ([planck] if planck else [])
        written.append(s8_whiskers(allL, os.path.join(s2, pre + "06_S8_summary.png"), "%s $S_8$ summary (median, 68%%)" % PRIOR_NAMES[L],
                                   truth, args.no_titles, mock))

    meta = dict(source=args.source, label=None if mock else args.label, stage=args.stage, event=args.event if mock else None,
                truth={k: v for k, v in truth.items()} if mock else None, provenance=prov, seed=args.seed,
                max_samples=args.max_samples, kids_stats=args.kids_stats, planck=planck is not None,
                companion=[c.key for c in comp], figures=[w for w in written if w])
    with open(os.path.join(args.out_dir, pre + "manifest.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=float)
    print("[done] %d figures -> %s" % (len([w for w in written if w]), args.out_dir))


if __name__ == "__main__":
    main()
