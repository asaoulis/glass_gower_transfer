#!/usr/bin/env python
"""Tier-4 REVEAL figure set: PHYSICAL-unit posteriors of the KiDS-Legacy observation, numbered in
the notebook's section-8 order, with the public external chains over-plotted.

    01  2-pt (bandpowers) ΛCDM  (+ 01e vs Planck, 01f vs DES Y3 / HSC Y3)   -- Stage 1 (the 2-pt arm alone)
    02  2-pt + field level ΛCDM                      -- Stage 2
    03  2-pt wCDM  (+ 03f vs DES Y3 wCDM)            -- Stage 1
    04  2-pt + field level wCDM                      -- Stage 2
    05  final ΛCDM: field + 2-pt vs Planck 2018      -- Stage 2
    06  S_8 summary whiskers (ours, KiDS-Legacy, companion 2-pt SBI, DES/HSC, Planck)   -- Stage 2
    07  field + 2-pt vs DES Y3 / HSC Y3 (a ΛCDM, b wCDM)   -- Stage 2
  per group: `a` = headline corner (pooled / final), `b` = repeats-vs-pooled diagnostic,
  `c` = headline + KiDS-Legacy / companion comparisons, `d` = all free parameters (+S8) with the
  sampling prior on the diagonal, `e` = vs Planck, `f` = vs the other surveys;
  `tables/<group>.{csv,md}` = mean / std / median / 68 %.

Style (one layout, the VD-report look): everything drawn in `S.context()` (18 pt serif LaTeX) and
saved by `S.save()` (PNG 200 dpi + PDF 300 dpi); corners 9 in (3 params) / 11 in (4) / 16 in (all),
whiskers 8.5 x 5 in; no titles (`--titles` opts in); short legend labels ("field level", "2-pt",
"repeat i", "pooled", "KiDS-Legacy COSEBIs", "Planck 2018", "prior", ...) with the full qualified
names in the tables and the manifest `legend` map.

Two sources, one code path:
  --source mock   a matched Gower mock (truth known, dashed grey crosshairs): the LAYOUT rehearsal.
                  Reads `<matched-root>/<exp>/external/<matched-tag>/` per arm x repeat x prior
                  (`--event K` picks the event). Pooled-equivalent = the repeats' draws concatenated
                  (equal weight, like `eval.py --mode pool`). Every file is `MOCK_`-prefixed and every
                  figure carries a grey "MOCK" tag (top left).
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
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.blind import BLIND_ROOT                                   # noqa: E402
from src.blind.standardise import _load_physical, _scaler_box, list_raw_runs, prior_of_dump  # noqa: E402
from src.ml.eval.arms import ARMS                                  # noqa: E402

from src.viz import style as S                                   # noqa: E402

PRIOR_NAMES = {"LCDM_fixed_w0": r"$\Lambda$CDM", "kids_s8_analytic": r"$w$CDM"}
PRIOR_PARAMS = {"LCDM_fixed_w0": ("omega_m", "sigma_8", "S8"),
                "kids_s8_analytic": ("omega_m", "sigma_8", "S8", "w0")}
# Long names (tables, manifest) vs short legend labels (figures). The qualifiers dropped from the
# legend -- pooling, IA model, VD, references -- live in the table rows and the manifest `legend` map.
KIDS_STAT_NAMES = {"cosebis": "KiDS-Legacy COSEBIs (Wright+25)", "bandpowers": "KiDS-Legacy band powers (Wright+25)",
                   "xipm": r"KiDS-Legacy $\xi_\pm$ (Wright+25)",
                   "cosebis_plus_external": "KiDS-Legacy COSEBIs + DES Y3 + BAO + SN (Stölzner+25)"}
KIDS_STAT_LABELS = {"cosebis": "KiDS-Legacy COSEBIs", "bandpowers": "KiDS-Legacy band powers",
                    "xipm": r"KiDS-Legacy $\xi_\pm$", "cosebis_plus_external": "KiDS-Legacy + ext."}
COMPANION_NAMES = {"novd": "companion 2-pt SBI (score, no VD)", "vd": "companion 2-pt SBI (score, VD)"}
COMPANION_LABELS = {"novd": "2-pt SBI, no VD", "vd": "2-pt SBI, VD"}
PLANCK_NAME = "Planck 2018 TT,TE,EE+lowE+lensing"
PLANCK_LABEL = "Planck 2018"
ARM_LABELS = {"nla_m": "field level", "band": "2-pt"}
# Colours: the VD-report scheme (Paul Tol muted, `S.palette()`): field level teal PAL[6], 2-pt wine
# PAL[5] (script-local overrides of the style's `nla_m`/`band` colours: the style's indigo flagship
# clashed with the wine next to it -- user 2026-10-12 -- so Tier-4 draws the field arm teal while
# the Tier-3 figures keep the style's indigo), KiDS-Legacy sand (COSEBIs, solid) / olive (band
# powers, dashed), companion cyan (no VD, dashed) / green (VD, solid), Planck the black reference
# outline (never filled), prior grey. Repeats are a `tint_ramp` of their arm's hue.
PAL = S.palette()
ARM_COLOURS = {"nla_m": PAL[6], "band": PAL[5]}
EXT_COLOURS = {"cosebis": PAL[2], "bandpowers": PAL[7], "xipm": PAL[0], "cosebis_plus_external": PAL[8],
               "novd": PAL[4], "vd": PAL[3], "planck": S.role_colour("real")}
EXT_LINESTYLE = {"cosebis": "-", "bandpowers": "--", "xipm": ":", "cosebis_plus_external": "-.",
                 "novd": "--", "vd": "-", "planck": "-"}
PRIOR_COLOUR = "#8C8C8C"
LW = {"flagship": 2.2, "arm": 2.0, "kids": 2.0, "companion": 2.0, "survey": 2.0, "planck": 1.8, "prior": 1.6,
      "repeat": 1.4, "pooled": 3.0}
SHADE_ALPHA = {"flagship": 0.75, "arm": 0.6}       # pinned: chainconsumer's default 1/sqrt(n_chains) varies per figure
TRUTH_KW = {"color": "#444444", "line_style": "--", "line_width": 1.2}
DRAW_ORDER = {"prior": -1, "planck": 0, "survey": 1, "kids": 2, "companion": 3, "ours": 4}   # externals under, our arms on top
LEGEND_ORDER = {"ours": 0, "companion": 1, "kids": 2, "survey": 3, "planck": 4, "prior": 5}   # headline first in the legend
ROLE_ORDER = {"flagship": 0, "arm": 1, "outline": 2}

# Corner geometry (inches) and type at the paper's 18 pt context (`S.context()`, CORNER_PLOTCONFIG
# 16 pt ticks/labels): (figsize, PlotConfig overrides, legend font size).
CORNER_3 = ((9.0, 9.0), {}, 14)
CORNER_4 = ((11.0, 11.0), {}, 14)
CORNER_FULL = ((16.0, 16.0), dict(tick_font_size=13, label_font_size=15, max_ticks=3), 13)
# Outer-axis ticks: (max ticks, label rotation); ticks closer than TICK_MARGIN x span to a panel
# edge are dropped so labels of neighbouring panels never touch.
TICKS_3_4 = (3, 0)
TICKS_FULL = (3, 45)
TICK_MARGIN = 0.1
# Smoothing. chainconsumer's kernel width is range / (2 * bins) whatever `smooth` is (`smooth` only
# sets the grid resolution: smooth=20 alone is the same kernel on a finer, noisier grid), so the
# lever is the per-chain `bins`. 3/4-param corners: bins 25 (range/50; chainconsumer default 35 ->
# range/70), ~1.5 % contour broadening. Full corners: ALL draws (`--full-max-samples 0`) and a
# Silverman-scaled kernel, bins = FULL_BINS x (N_eff / FULL_BINS_NREF)^(1/6) clipped to
# FULL_BINS_CLIP (20 at 125k draws = range/40, ~2 % broadening; ~15 at 25k; ~25 at 500k). The
# earlier 12/(8,20) (range/24, ~6 %) was judged too aggressive by the user (2026-10-12): it visibly
# fattened the Om / sigma_8 marginals. The histogram path reflects at the prior-box edge (correct).
# kde=True was tested: rounded box corners but jagged 68 % lines on flat-flat panels (MegKDE's hard
# 3-sigma truncation) and ~2 min per figure.
CORNER_BINS = 25
FULL_SMOOTH = 10
FULL_BINS = 20
FULL_BINS_NREF = 125000
FULL_BINS_CLIP = (12, 30)
FULL_KDE = False


def _full_bins(n_eff: float) -> int:
    b = FULL_BINS * (max(n_eff, 1.0) / FULL_BINS_NREF) ** (1.0 / 6.0)
    return int(np.clip(round(b), *FULL_BINS_CLIP))


def arm_colour(arm: str) -> str:
    return ARM_COLOURS.get(arm) or S.arm_colour(arm)


def arm_label(arm: str) -> str:
    return ARM_LABELS.get(arm, S.arm_name(arm))


def tint_ramp(hex_colour: str, n: int, lo: float = 0.62, hi: float = 0.18) -> List[str]:
    """`n` tints of one hue, light -> dark (mix with white by `lo`..`hi`): the repeats-vs-pooled
    ramp (identical runs, lightness only; the pooled line is the full-strength hue)."""
    import matplotlib.colors as mc
    base = np.array(mc.to_rgb(hex_colour))
    return [mc.to_hex(base + (1 - base) * f) for f in np.linspace(lo, hi, max(n, 1))]


# --------------------------------------------------------------------------------------------
# one posterior = physical draws + names (+ optional weights, truth)
# --------------------------------------------------------------------------------------------
class Post:
    def __init__(self, name, x, names, weights=None, kind="ours", colour=None, role="outline", key="",
                 linestyle="-", linewidth=None, label=None):
        self.name, self.x, self.names = name, np.asarray(x, dtype=np.float64), list(names)
        self.label = name if label is None else label          # short legend label; `name` = table name
        self.weights = None if weights is None else np.asarray(weights, dtype=np.float64)
        self.kind, self.colour, self.role, self.key = kind, colour, role, key
        self.linestyle, self.linewidth = linestyle, linewidth

    def chain_kwargs(self):
        kw = S.chain_kwargs(self.role, self.colour)
        if self.role == "arm":                      # the 2-pt fill sits under the headline: lighter
            kw["shade_gradient"] = 0.7
        kw["linestyle"] = self.linestyle
        if self.linewidth is not None:
            kw["linewidth"] = self.linewidth
        filled = bool(kw.get("shade"))
        kw["bar_shade"] = filled                    # 1-D 68 % fill only under our filled arms
        if filled:
            kw["shade_alpha"] = SHADE_ALPHA.get(self.role, 0.6)
        else:
            kw["shade_gradient"] = 0.0              # outlines: ONE colour for the 68 and 95 % lines (identity)
        return kw


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


def prior_post(experiment: str, prior_mode: str, n: int, seed: int) -> Optional[Post]:
    """`n` draws of the SAMPLING prior of `prior_mode` (the same `build_prior_for_mode` the chains
    used), unscaled to physical units over the free parameters (+ S8), as a grey outline.
    Under `LCDM_fixed_w0` the chains condition on w0 = -1; the overlay is the 9-param prior with
    the w0 column dropped (identical to the conditional when w0 is an independent block)."""
    try:
        import torch
        import eval as _eval
        from src.ml.eval.nle_external import build_prior_for_mode, free_param_names
        cfg, _ = _eval.load_config(experiment)
        names = list(cfg.cosmo_param_names)
        preset = dict(getattr(cfg, "scaler_options", {}).get("cosmo", {}).get("preset_overrides", {}) or {})
        prior, _fixed = build_prior_for_mode(prior_mode, names, preset_overrides=preset or None)
        torch.manual_seed(seed)
        z = prior.sample((n,)).detach().cpu().numpy().reshape(n, -1)
        _, lo, hi = _scaler_box(experiment, None)
        if z.shape[1] != len(lo):
            raise ValueError("prior sample has %d columns, the box %d" % (z.shape[1], len(lo)))
        x = z * (hi - lo) + lo
        free = free_param_names(prior_mode, names)
        x = x[:, [names.index(f) for f in free]]
        s8 = x[:, free.index("sigma_8")] * np.sqrt(x[:, free.index("omega_m")] / 0.3)
        x = np.concatenate([x, s8[:, None]], axis=1)
        return Post("prior (%s)" % PRIOR_NAMES[prior_mode], x, free + ["S8"], kind="prior", colour=PRIOR_COLOUR,
                    role="outline", key="prior_" + prior_mode, linewidth=LW["prior"], label="prior")
    except Exception as e:  # noqa: BLE001
        print("[prior] could not sample the %s prior for %s: %s -> full corners without prior" % (prior_mode, experiment, e))
        return None


def load_ours(args) -> tuple[Dict[tuple, Post], Dict[tuple, List[Post]], Dict[str, float], List[dict]]:
    """{(arm, prior): headline Post}, {(arm, prior): [repeat Posts]}, truth (mock only), provenance."""
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
            ramp = tint_ramp(arm_colour(arm), max(len(rr), 1))
            rl = []
            for i, r in enumerate(rr):
                x, names = _load_physical(r["path"], r["experiment"], event)
                ri = r["match"].rsplit("_", 1)[-1]
                rl.append(Post("%s, repeat %s" % (S.arm_name(arm), ri), x, names,
                               kind="ours", colour=ramp[i], role="outline", key="%s_%s" % (arm, r["match"]),
                               linewidth=LW["repeat"], label="repeat %s" % ri))
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
            role = "flagship" if arm == "nla_m" else "arm"
            heads[(arm, prior)] = Post(hname, x, names, kind="ours", colour=arm_colour(arm), role=role, key=arm,
                                       linewidth=LW[role], label=arm_label(arm))
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
                        colour=EXT_COLOURS.get(st, "#777777"), role="outline", key="kids_" + st,
                        linestyle=EXT_LINESTYLE.get(st, "-"), linewidth=LW["kids"],
                        label=KIDS_STAT_LABELS.get(st, st)))
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
                colour=EXT_COLOURS["planck"], role="outline", key="planck", linewidth=LW["planck"],
                label=PLANCK_LABEL)


def load_companion(companion_dir: str) -> List[Post]:
    """Companion score-based 2-pt SBI posteriors (hypatia `sbi_score_{novd,vd}_real_data/`, fetched by
    the user): `nle_real_data_samples_r3.npz`, keys `joint` [N, 10] + `params`; LCDM, unweighted.
    Parsed by `scripts/tier4_external_chains.read_companion`."""
    from tier4_external_chains import read_companion
    out = []
    for tag in ("novd", "vd"):
        r = read_companion(tag, companion_dir) if companion_dir else None
        if r is None:
            print("[companion] %s: absent -> skipped" % tag); continue
        X, names = r
        out.append(Post(COMPANION_NAMES[tag], X, names, kind="companion", colour=EXT_COLOURS[tag],
                        role="outline", key="companion_" + tag, linestyle=EXT_LINESTYLE[tag],
                        linewidth=LW["companion"], label=COMPANION_LABELS[tag]))
        print("[companion] %s: %d draws" % (tag, len(X)))
    return out


# public DES Y3 / HSC Y3 chains (scripts/tier4_external_chains.SURVEY_CHAINS): outline chains, one hue
# per survey (DES indigo PAL[1], HSC Tol-vibrant orange HSC_ORANGE -- the muted purple PAL[8] was a third
# magenta next to the 2-pt wine and the DES+KiDS-1000 rose PAL[0] in 07 -- indigo is free since the
# field arm moved to teal), linestyle per statistic. Surveys appear only in 06/07 next to the arms.
HSC_ORANGE = "#EE7733"
SURVEY_STYLE = {   # (colour, linestyle); outlines only
    "des_y3_cs_lcdm": (PAL[1], "-"), "des_y3_cs_wcdm": (PAL[1], "-"),
    "des_y3_3x2_lcdm": (PAL[1], "--"), "des_y3_3x2_wcdm": (PAL[1], "--"),
    "hsc_y3_cl": (HSC_ORANGE, "-"), "hsc_y3_xipm": (HSC_ORANGE, "--"), "hsc_y3_3x2": (HSC_ORANGE, ":"),
    "des_y3_kids1000": (PAL[0], "-"), "hsc_y3_clusters": (HSC_ORANGE, "-."),
}


def load_surveys(include_beyond2pt: bool = False) -> List[Post]:
    from tier4_external_chains import available, read_cosmosis_chain
    out = []
    for key, path, name, model, probe in available(include_beyond2pt=include_beyond2pt):
        Y, names, w = read_cosmosis_chain(path)
        col, ls = SURVEY_STYLE.get(key, (PAL[9], "-"))
        full = "%s (%s, %s)" % (name, model, probe)
        out.append(Post(full, Y, names, weights=w, kind="survey", colour=col, role="outline", key=key,
                        linestyle=ls, linewidth=LW["survey"], label=name))
        out[-1].model = model
        print("[survey] %s: %d draws (%s)" % (key, len(Y), model))
    return out


# --------------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------------
LAYOUTS = ("paper",)            # one layout: the paper's 18 pt context, figures drawn large (VD-report look)
LEGEND_LOG: Dict[str, List[List[str]]] = {}     # figure -> [[legend label, full name], ...] (manifest)


def _corner_geometry(nparams: int):
    return CORNER_FULL if nparams > 5 else (CORNER_4 if nparams > 3 else CORNER_3)


def _mock_tag(fig):
    fig.text(0.01, 0.99, "MOCK", fontsize=14, color="0.45", ha="left", va="top")


def _legend_handles(posts: List[Post], prior: Optional[Post], truth: bool):
    """Legend in reading order (headline first, then the external chains, prior, truth): filled
    chains as a filled patch, outlines as their line (colour, style, width)."""
    import matplotlib.colors as mc
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    hs = []
    for p in sorted(posts, key=lambda p: (LEGEND_ORDER.get(p.kind, 0), ROLE_ORDER.get(p.role, 2) if p.kind == "ours" else 0)):
        kw = p.chain_kwargs()
        if kw.get("shade"):
            hs.append(Patch(facecolor=mc.to_rgba(p.colour, 0.55), edgecolor=p.colour, lw=kw["linewidth"],
                            label=S.tex(p.label)))
        else:
            hs.append(Line2D([], [], color=p.colour, ls=p.linestyle, lw=kw["linewidth"], label=S.tex(p.label)))
    if prior is not None:
        hs.append(Line2D([], [], color=prior.colour, lw=prior.linewidth, label=S.tex(prior.label)))
    if truth:
        hs.append(Line2D([], [], color=TRUTH_KW["color"], ls=TRUTH_KW["line_style"], lw=TRUTH_KW["line_width"],
                         label="truth"))
    return hs


def _tidy_ticks(axes, nbins: int, rotation: float, margin: float = TICK_MARGIN):
    """Outer-axis ticks (bottom row x, left column y): round values at least `margin` x span from
    either panel edge (relaxed only if fewer than two survive)."""
    from matplotlib.ticker import FixedLocator, MaxNLocator
    n = axes.shape[0]

    def pick(lo, hi):
        span = hi - lo
        for m in (margin, margin / 2, 0.0):              # finer steps first, then a smaller margin
            for nb in (nbins, nbins + 1, nbins + 2):
                t = MaxNLocator(nbins=nb, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
                keep = [v for v in t if lo + m * span <= v <= hi - m * span]
                if len(keep) >= 2:
                    return keep
        return keep

    for j in range(n):
        ax = axes[-1, j]
        ax.xaxis.set_major_locator(FixedLocator(pick(*ax.get_xlim())))
        ax.tick_params(axis="x", which="major", labelrotation=rotation)
    for i in range(1, n):
        ax = axes[i, 0]
        ax.yaxis.set_major_locator(FixedLocator(pick(*ax.get_ylim())))
        ax.tick_params(axis="y", which="major", labelrotation=rotation)


def _prior_diagonal(axes, prior: Post, params: Sequence[str], labels: Sequence[str]):
    """The sampling prior as a grey 1-D line on the diagonal ONLY (no 2-D prior contours), drawn
    under the posteriors with chainconsumer's own smoothed-histogram 1-D estimator."""
    import pandas as pd
    from chainconsumer import Chain
    from chainconsumer.plotting.contours import plot_dist
    have = [(i, q, lab) for i, (q, lab) in enumerate(zip(params, labels)) if q in prior.names]
    if not have:
        return
    df = pd.DataFrame(prior.cols([q for _, q, _ in have]), columns=[lab for _, _, lab in have])
    ch = Chain(samples=df, name="prior", color=prior.colour, linestyle="-", linewidth=prior.linewidth,
               shade=False, bar_shade=False, smooth=S.CORNER_SMOOTH, zorder=-1)
    for i, _q, lab in have:
        ax = axes[i, i]
        xlim, (_, ytop) = ax.get_xlim(), ax.get_ylim()
        n0 = len(ax.lines)
        plot_dist(ax, ch, lab)
        ax.set_xlim(xlim)
        for ln in ax.lines[n0:]:
            x, y = np.asarray(ln.get_xdata()), np.asarray(ln.get_ydata())
            inside = (x >= xlim[0]) & (x <= xlim[1])
            if inside.any() and 1.05 * y[inside].max() > ytop:
                ytop = 1.1 * y[inside].max()
        ax.set_ylim(0, ytop)


def corner(posts: List[Post], params: Sequence[str], out: str, title: str, truth: Dict[str, float],
           rng: np.random.Generator, max_samples: int, titles=False, mock=False, prior: Optional[Post] = None,
           full=False):
    """Chains drawn externals-first (Planck, KiDS, companion) and our arms last (the headline on top).
    `full`: the all-parameter corner -- heavier smoothing (`FULL_BINS`/`FULL_KDE`) and the sampling
    `prior` as a 1-D line on the diagonal only."""
    posts = [p for p in posts if p.has(params)]
    if not posts:
        print("[skip] %s: no posterior carries %s" % (os.path.basename(out), params)); return None
    posts = sorted(posts, key=lambda p: (DRAW_ORDER.get(p.kind, 3), 0 if p.role in ("outline",) else 1))
    labels = [S.label(p) for p in params]
    sd, cc, kw, wt = {}, {}, {}, {}
    for p in posts:
        x, w = p.thinned(max_samples, rng)
        idx = [p.names.index(q) for q in params]
        sd[p.name] = x[:, idx]; cc[p.name] = labels
        kw[p.name] = p.chain_kwargs()
        kw[p.name]["bins"] = CORNER_BINS
        if full:     # heavier smoothing; no 1-D 68 % fill (chainconsumer's MAX interval is arbitrary on a flat marginal)
            n_eff = len(x) if w is None else w.sum() ** 2 / (w ** 2).sum()
            kw[p.name].update(bins=_full_bins(n_eff), kde=FULL_KDE, bar_shade=False)
        if w is not None:
            wt[p.name] = w
    tr = {S.label(q): truth[q] for q in params if q in truth} if truth else None
    figsize, pc, legend_fs = _corner_geometry(len(params))
    fig = S.plot_chains(sd, cc, truth=tr or None, plotting_kwargs=kw, figsize=figsize,
                        smooth=FULL_SMOOTH if full else S.CORNER_SMOOTH, close=False, weights=wt or None,
                        plot_config={**pc, "show_legend": False}, truth_kwargs=TRUTH_KW, prune_ticks=False)
    n = len(params)
    axes = np.asarray(fig.axes[: n * n], dtype=object).reshape(n, n)
    with S.context():
        fig.subplots_adjust(top=0.95)                      # square panels, no dead strip above the corner
        if prior is not None and prior.has([q for q in params if q != "S8"]):
            _prior_diagonal(axes, prior, params, labels)
        _tidy_ticks(axes, *(TICKS_FULL if full else TICKS_3_4))
        fig.align_ylabels(list(axes[:, 0])); fig.align_xlabels(list(axes[-1, :]))
        hs = _legend_handles(posts, prior, bool(tr))
        axes[0, -1].legend(handles=hs, loc="upper right", bbox_to_anchor=(1.0, 1.0), frameon=False,
                           fontsize=legend_fs, handlelength=1.6, handletextpad=0.6, labelspacing=0.35,
                           borderaxespad=0.0)
        if titles:
            fig.suptitle(S.tex(title), y=0.995)
        if mock:
            _mock_tag(fig)
        S.save(fig, out)
    LEGEND_LOG[os.path.basename(out)] = [[p.label, p.name] for p in posts] + ([[prior.label, prior.name]] if prior else [])
    print("[fig] %s" % out)
    return out


WHISKER_GROUPS = ("this work", "KiDS-Legacy", "other surveys", "CMB")


def s8_whiskers(groups: Sequence[List[Post]], out: str, title: str, truth: Dict[str, float], titles=False,
                mock=False):
    """Median + 68 % bar per posterior (Wright+25 Fig. 11 grammar), one block per group in
    `WHISKER_GROUPS` order (ours | KiDS-Legacy | other surveys | Planck) split by thin rules; the
    headline's and Planck's 68 % intervals are also drawn as faint vertical bands."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator
    groups = [[p for p in g if p is not None and "S8" in p.names] for g in groups]
    rows, bounds = [], []
    for g in groups:
        if not g:
            continue
        if rows:
            bounds.append(len(rows))
        rows.extend(g)
    if not rows:
        return None
    with S.context():
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        nr = len(rows)
        for i, p in enumerate(rows):
            v = p.x[:, p.names.index("S8")]
            q16, q50, q84 = _wq(v, [0.16, 0.5, 0.84], p.weights)
            y = nr - 1 - i
            if (p.kind == "ours" and p.role == "flagship") or p.kind == "planck":
                ax.axvspan(q16, q84, color=p.colour, alpha=0.15, lw=0, zorder=0)
            eb = ax.errorbar(q50, y, xerr=[[q50 - q16], [q84 - q50]], fmt="o", color=p.colour, ms=8, lw=2.4,
                             capsize=4, capthick=2.4, zorder=3)
            if p.linestyle not in ("-", "solid"):
                for lc in eb[2]:
                    lc.set_linestyle(p.linestyle)
        for b in bounds:
            ax.axhline(nr - b - 0.5, color="0.3", lw=0.8, zorder=1)
        if truth and "S8" in truth:
            ax.axvline(truth["S8"], color=TRUTH_KW["color"], ls=TRUTH_KW["line_style"], lw=TRUTH_KW["line_width"],
                       zorder=2)
        ax.set_yticks(range(nr - 1, -1, -1))
        ax.set_yticklabels([S.tex(p.label) for p in rows], fontsize="small")
        ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis="y", which="both", length=0)
        ax.set_ylim(-0.6, nr - 0.4)
        ax.set_xlabel(S.label("S8"))
        if titles:
            ax.set_title(S.tex(title))
        if mock:
            _mock_tag(fig)
        S.save(fig, out)
    LEGEND_LOG[os.path.basename(out)] = [[p.label, p.name] for p in rows]
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
    keyed = []
    for p in posts:
        if p.label != p.name and [p.label, p.name] not in keyed:
            keyed.append([p.label, p.name])
    with open(out_base + ".md", "w") as fh:
        if note:
            fh.write(note + "\n\n")
        if keyed:
            fh.write("Figure legend labels: " + "; ".join("'%s' = %s" % (a, b) for a, b in keyed) + ".\n\n")
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
    ap.add_argument("--max-samples", type=int, default=50000,
                    help="seeded uniform thinning for the 3/4-param contours (tables use all draws)")
    ap.add_argument("--full-max-samples", type=int, default=0,
                    help="seeded uniform thinning for the full-parameter corners; 0 = all draws (keep >= 50k: "
                         "smooth prior-dominated panels)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-surveys", action="store_true", help="skip the DES Y3 / HSC Y3 overlays (07 + whisker rows)")
    ap.add_argument("--beyond2pt", action="store_true", help="also include the HSC cluster-abundance chain (different S8 definition)")
    ap.add_argument("--prior-draws", type=int, default=200000, help="prior draws for the full-parameter corner diagonal")
    ap.add_argument("--titles", action="store_true",
                    help="draw figure titles (off by default: file names + captions carry them)")
    ap.add_argument("--layout", choices=LAYOUTS, default="paper",
                    help="paper = the 18 pt serif LaTeX context, corners 9/11/16 in, whiskers 8.5x5 in (the only layout)")
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
    surveys = load_surveys(args.beyond2pt) if not args.no_surveys else []
    L, W = "LCDM_fixed_w0", "kids_s8_analytic"
    tt = {"band": "2-pt", "nla_m": "field"}
    ttl_kw = dict(titles=args.titles, mock=mock)
    written = []

    def head(arm, prior):
        return heads.get((arm, prior))

    def group(num, name, prior, arms_in, stage_dir, externals):
        """a: headline(s); b: repeats-vs-pooled per arm; c: + externals; d: all parameters; table."""
        params = PRIOR_PARAMS[prior]
        hs = [head(a, prior) for a in arms_in if head(a, prior) is not None]
        if not hs:
            print("[skip] group %s: no posterior for %s" % (num, prior)); return
        tag = "%s%s" % (pre, num)
        ttl = "%s, %s" % (" + ".join(tt[a] for a in arms_in), PRIOR_NAMES[prior])
        written.append(corner(hs, params, os.path.join(stage_dir, "%sa_%s_corner.png" % (tag, name)), ttl + ": headline (pooled)",
                              truth, rng, args.max_samples, **ttl_kw))
        for a in arms_in:
            rl = reps.get((a, prior), [])
            h = head(a, prior)
            if len(rl) > 1 and h is not None:
                pooled = Post(h.name, h.x, h.names, colour=arm_colour(a), role="outline", key=a,
                              linewidth=LW["pooled"], label="pooled")
                written.append(corner(rl + [pooled], params, os.path.join(stage_dir, "%sb_%s_repeats_%s.png" % (tag, name, a)),
                                      "%s %s: repeats vs pooled" % (tt[a], PRIOR_NAMES[prior]), truth, rng,
                                      args.max_samples, **ttl_kw))
        ext = [e for e in externals if e is not None and e.has(params)]
        if ext:
            written.append(corner(ext + hs, params, os.path.join(stage_dir, "%sc_%s_vs_external.png" % (tag, name)),
                                  ttl + ": vs external", truth, rng, args.max_samples, **ttl_kw))
        # d: the FULL free-parameter posterior (+S8) with the sampling prior on the diagonal
        full = [n for n in hs[0].names if n != "S8"] + ["S8"]
        pr = prior_post(hs[0].key and ARMS[hs[0].key][0].format(r=ARMS[hs[0].key][2][0]), prior, args.prior_draws, args.seed)
        written.append(corner(hs, full, os.path.join(stage_dir, "%sd_%s_full_corner.png" % (tag, name)),
                              ttl + ": all parameters, with the prior", truth, rng, args.full_max_samples,
                              prior=pr, full=True, **ttl_kw))
        allp = hs + [p for a in arms_in for p in reps.get((a, prior), [])] + ext + ([pr] if pr else [])
        write_table(allp, full, os.path.join(tdir, "%s_%s" % (tag, name)), note="%s%s" % ("MOCK (truth: %s). " % {
            k: float(round(v, 4)) for k, v in truth.items() if k in full} if mock else "", ttl))

    sv_L = [p for p in surveys if getattr(p, "model", "LCDM") == "LCDM"]
    sv_W = [p for p in surveys if getattr(p, "model", "LCDM") == "wCDM"]

    def vs_planck_and_surveys(arms_in, stage_dir, planck_stem, surv_L_stem, surv_W_stem):
        """Our headline posteriors against Planck 2018 (LCDM) and the public DES Y3 / HSC Y3 chains
        (LCDM; wCDM vs DES Y3 only). Stage 1: the 2-pt arm alone (01e/01f/03f); stage 2: field + 2-pt
        (05/07a/07b). The KiDS-Legacy release has LCDM chains only, so no wCDM KiDS overlay exists."""
        who = " + ".join(tt[a] for a in arms_in)
        hs_L = [head(a, L) for a in arms_in if head(a, L) is not None]
        hs_W = [head(a, W) for a in arms_in if head(a, W) is not None]
        if planck is not None and hs_L:
            written.append(corner([planck] + hs_L, PRIOR_PARAMS[L], os.path.join(stage_dir, pre + planck_stem + "_corner.png"),
                                  "%s: %s vs Planck 2018" % (PRIOR_NAMES[L], who), truth, rng, args.max_samples, **ttl_kw))
            write_table(hs_L + [planck], PRIOR_PARAMS[L], os.path.join(tdir, pre + planck_stem))
        if sv_L and hs_L:
            written.append(corner(sv_L + hs_L, PRIOR_PARAMS[L], os.path.join(stage_dir, pre + surv_L_stem + "_corner.png"),
                                  "%s: %s vs DES Y3 / HSC Y3" % (PRIOR_NAMES[L], who), truth, rng, args.max_samples, **ttl_kw))
            write_table(hs_L + sv_L, PRIOR_PARAMS[L], os.path.join(tdir, pre + surv_L_stem))
        if sv_W and hs_W:
            written.append(corner(sv_W + hs_W, PRIOR_PARAMS[W], os.path.join(stage_dir, pre + surv_W_stem + "_corner.png"),
                                  "%s: %s vs DES Y3" % (PRIOR_NAMES[W], who), truth, rng, args.max_samples, **ttl_kw))
            write_table(hs_W + sv_W, PRIOR_PARAMS[W], os.path.join(tdir, pre + surv_W_stem))

    if do1:
        group("01", "2pt_LCDM", L, ["band"], s1, kids + comp)
        group("03", "2pt_wCDM", W, ["band"], s1, [])
        vs_planck_and_surveys(["band"], s1, "01e_2pt_LCDM_vs_Planck", "01f_2pt_LCDM_vs_surveys", "03f_2pt_wCDM_vs_surveys")
    if do2:
        group("02", "2pt_field_LCDM", L, ["band", "nla_m"], s2, kids + comp)
        group("04", "2pt_field_wCDM", W, ["band", "nla_m"], s2, [])
        vs_planck_and_surveys(["band", "nla_m"], s2, "05_LCDM_vs_Planck", "07a_LCDM_vs_surveys", "07b_wCDM_vs_surveys")
        # whisker groups (WHISKER_GROUPS): this work | KiDS-Legacy (companion SBI, Wright+25) | other surveys
        # (DES Y3 / HSC Y3, LCDM) | Planck
        ours = [head(a, L) for a in ("nla_m", "band") if head(a, L) is not None]
        written.append(s8_whiskers([ours, comp + kids, sv_L, [planck] if planck else []],
                                   os.path.join(s2, pre + "06_S8_summary.png"),
                                   "%s $S_8$ summary (median, 68%%)" % PRIOR_NAMES[L], truth, **ttl_kw))

    meta = dict(source=args.source, label=None if mock else args.label, stage=args.stage, event=args.event if mock else None,
                truth={k: v for k, v in truth.items()} if mock else None, provenance=prov, seed=args.seed,
                max_samples=args.max_samples, full_max_samples=args.full_max_samples, prior_draws=args.prior_draws,
                kids_stats=args.kids_stats, planck=planck is not None, companion=[c.key for c in comp],
                surveys=[p.key for p in surveys],
                figures=[w for w in written if w],
                legend=LEGEND_LOG,     # per figure: [legend label, full name (pooling / IA / VD / reference)]
                style=dict(layout=args.layout, context="S.context() 18 pt serif LaTeX; S.save PNG 200 dpi + PDF 300 dpi",
                           corner_figsize={"3": CORNER_3[0], "4": CORNER_4[0], "full": CORNER_FULL[0]},
                           legend_fontsize={"3/4": CORNER_3[2], "full": CORNER_FULL[2]},
                           smooth=S.CORNER_SMOOTH, corner_bins=CORNER_BINS, full_smooth=FULL_SMOOTH,
                           full_bins="%d x (N_eff/%d)^(1/6), clipped to %s" % (FULL_BINS, FULL_BINS_NREF, FULL_BINS_CLIP),
                           full_kde=FULL_KDE,
                           linewidths=LW, shade_alpha=SHADE_ALPHA, truth=TRUTH_KW, contour_levels="68 %, 95 %"))
    with open(os.path.join(args.out_dir, pre + "manifest.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=float)
    print("[done] %d figures -> %s" % (len([w for w in written if w]), args.out_dir))


if __name__ == "__main__":
    main()
