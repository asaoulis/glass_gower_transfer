#!/usr/bin/env python
"""Tier-4 REVEAL figure set: PHYSICAL-unit posteriors of the KiDS-Legacy observation, numbered in
the notebook's section-8 order, with the public external chains over-plotted.

    01  2-pt (bandpowers) ΛCDM  (+ 01e vs Planck, 01f vs DES Y3 / HSC Y3)   -- Stage 1 (the 2-pt arm alone)
    02  2-pt + field level ΛCDM                      -- Stage 2
    03  2-pt wCDM  (+ 03f vs DES Y3 wCDM)            -- Stage 1
    04  2-pt + field level wCDM                      -- Stage 2
    05  final ΛCDM: field + 2-pt vs Planck 2018  (05b: the wCDM arms vs the Planck ΛCDM chain)   -- Stage 2
    06  S_8 summary whiskers (ours, KiDS-Legacy, companion 2-pt SBI, DES/HSC, Planck)   -- Stage 2
        (06b: wCDM; externals without a wCDM chain keep their ΛCDM row, tagged)
    07  field + 2-pt vs DES Y3 / HSC Y3 (a ΛCDM, b wCDM)   -- Stage 2
    08  IA-variate OVERLAY: field level NLA-M (pooled, filled teal) + NLA-z + NLA (outlines), one
        contour per model (08a ΛCDM / wCDM on Om, s8, S8 [+w]; 08d on every shared parameter incl. A_IA)
                                                     -- Stage 3 (the IA variates)
    09  S_8 whiskers as 06/06b with the variate rows added (09 ΛCDM, 09b wCDM)   -- Stage 3
    10  field level NLA-z suite (a/b/c/d/e/f as for the field arm; `_LCDM` / `_wCDM` in the name)   -- Stage 3
    11  field level NLA suite (same)                                                             -- Stage 3
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
into `stage1_2pt/`; `--stage 2` the field groups (02, 04, 05, 06) into `stage2_field/`; `--stage 3`
the IA-variate groups (08-11) into `stage3_variates/`; `all` every stage.

Stage 3 (IA variates, `--variates nla_z nla`): availability is PARTIAL and grows. Each variate arm x
prior uses exactly the dumps on disk -- the cluster pooled dump when one exists (its member repeats read
from the dump's `pooled_provenance`), else the per-repeat dumps concatenated at equal weight -- and a
missing arm / prior / repeat is skipped with a printed note, never an error. The repeats behind every
headline are recorded in the manifest (`variates` block) and in the README section this stage writes.
A `--stage 3` run MERGES into an existing manifest.json (the stage-1/2 record is kept) and rewrites only
the README's stage-3 section (between `<!-- stage3-variates -->` markers).

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
from src.blind.standardise import _load_physical, _scaler_box, list_raw_runs, pooled_members, prior_of_dump  # noqa: E402
from src.ml.eval.arms import ARMS, DELAYED_ARMS, arm_experiment    # noqa: E402

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
ARM_LABELS = {"nla_m": "field level", "band": "2-pt", "nla_z": r"field level, NLA-$z$", "nla": "field level, NLA",
              "k2": r"field level, $\kappa=2$", "k2_nla": r"field level, $\kappa=2$ + NLA"}
# Colours: the VD-report scheme (Paul Tol muted, `S.palette()`): field level teal PAL[6], 2-pt wine
# PAL[5] (script-local overrides of the style's `nla_m`/`band` colours: the style's indigo flagship
# clashed with the wine next to it -- user 2026-10-12 -- so Tier-4 draws the field arm teal while
# the Tier-3 figures keep the style's indigo), KiDS-Legacy sand (COSEBIs, solid) / olive (band
# powers, dashed), companion cyan (no VD, dashed) / green (VD, solid), Planck the black reference
# outline (never filled), prior grey. Repeats are a `tint_ramp` of their arm's hue.
PAL = S.palette()
# IA variates (stage 3): Paul Tol VIBRANT blue (NLA-z) and red (NLA), script-local like the teal above.
# The style's own variate colours clash here (nla_z = PAL[5] wine = the 2-pt arm, nla = PAL[0] rose = DES
# Y3 + KiDS-1000); the muted cycle has no free hue left after the externals (sand, olive, cyan, green,
# indigo, rose, orange, purple = KiDS + ext.). Blue sits between indigo (DES) and cyan (companion) but is far from both in
# lightness; red is the vibrant partner of the HSC orange (that scheme's two warm hues are designed to
# be told apart) and redder than the rose. In the 08 overlay NLA is also dashed (identity without colour).
# kappa=2 b_g-prior variates (2026-10-17): dark gold / deep violet, script-local like the IA variates (k2 = NLA-M IA
# with the 2x broader b_g prior marginalised; k2_nla = the same with NLA IA).
# 2-pt NLA: Tol-vibrant magenta (the style's sand fallback is the KiDS-Legacy COSEBIs colour in the 09 whiskers).
ARM_COLOURS = {"nla_m": PAL[6], "band": PAL[5], "nla_z": "#0077BB", "nla": "#CC3311", "k2": "#997700", "k2_nla": "#5D3A9B",
               "band_nla": "#EE3377", "hos": "#0077BB", "hos_gmrg": "#33BBEE"}   # hos: Tol-vibrant blue (the style's magenta ~ the 2-pt wine;
                                                           # shared with NLA-z, so plot hos and nla_z in separate runs)
VARIATE_ARMS = ("nla_z", "nla")
# stage-3 legend labels (stage 1/2 keep "field level" for NLA-M: a rerun of those must not relabel them)
STAGE3_LABELS = {"nla_m": "field level, NLA-M", "nla_z": r"field level, NLA-$z$", "nla": "field level, NLA",
                 "band": "2-pt", "k2": r"field level, NLA-M, $\kappa=2$", "k2_nla": r"field level, NLA, $\kappa=2$",
                 "band_nla": "2-pt, NLA", "band_nla_z": r"2-pt, NLA-$z$", "hos": "2-pt + third moments (HOS)",
                 "hos_gmrg": "2-pt + third moments + Gmrg (HOS)"}
STAGE3_NAMES = {"nla_m": "field-level NLE, NLA-M IA, VD", "nla_z": "field-level NLE, NLA-z IA, VD",
                "nla": "field-level NLE, NLA IA, VD", "band": "2-pt (bandpowers) NLE, NLA-M IA, VD",
                "k2": "field-level NLE, NLA-M IA, VD, kappa=2 b_g prior (b_g marginalised)",
                "k2_nla": "field-level NLE, NLA IA, VD, kappa=2 b_g prior (b_g marginalised)",
                "band_nla": "2-pt (bandpowers) NLE, NLA IA (free A_IA), VD",
                "band_nla_z": "2-pt (bandpowers) NLE, NLA-z IA (a_ia + b_z), VD",
                "hos": "EE bandpowers + R1 third moments hybrid NLE (HOS), NLA-M IA, VD",
                "hos_gmrg": "EE bandpowers + R1 + Gmrg third moments hybrid NLE (HOS), NLA-M IA, VD"}
VARIATE_LINESTYLE = {"nla_z": "-", "nla": "--", "k2": ":", "k2_nla": "-.", "band_nla": ":", "band_nla_z": ":", "hos": "-", "hos_gmrg": "-"}  # 08 overlay outlines
# per-variate figure group: (number, file stem)
# 12 is the A_IA,total figure -> the kappa=2 suites are 13 / 14, the 2-pt NLA arm (T3b) 15, the 2-pt NLA-z arm 16,
# the HOS hybrid (EE + third moments, NLA-M) 17 (+ 17g: field / 2-pt / HOS on one corner).
# Opt-in: pass `--variates nla_z nla k2 [k2_nla] [band_nla] [band_nla_z] [hos]` (hos: usually with --arm-label hos=A2).
VARIATE_GROUPS = {"nla_z": ("10", "field_NLAz"), "nla": ("11", "field_NLA"), "k2": ("13", "field_k2"),
                  "k2_nla": ("14", "field_k2NLA"), "band_nla": ("15", "2pt_NLA"), "band_nla_z": ("16", "2pt_NLAz"),
                  "hos": ("17", "HOS"), "hos_gmrg": ("18", "HOS_Gmrg")}
# 18 = the R1 + Gmrg HOS hybrid (+ 18g: 2-pt / HOS / HOS+Gmrg / field on one corner, HOS+Gmrg filled; --gmrg-headline).
# Usually `--variates hos hos_gmrg --gmrg-headline` (both HOS heads are needed for 18g).
MODEL_PRIOR = {"LCDM": "LCDM_fixed_w0", "wCDM": "kids_s8_analytic"}


def _fig_model(path: str) -> str:
    return "wCDM" if "wCDM" in os.path.basename(path) else "LCDM"
EXT_COLOURS = {"cosebis": PAL[2], "bandpowers": PAL[7], "xipm": PAL[0], "cosebis_plus_external": PAL[8],
               "novd": PAL[4], "vd": PAL[3], "planck": S.role_colour("real")}
EXT_LINESTYLE = {"cosebis": "-", "bandpowers": "--", "xipm": ":", "cosebis_plus_external": "-.",
                 "novd": "--", "vd": "-", "planck": "-"}
PRIOR_COLOUR = "#8C8C8C"
LW = {"flagship": 2.2, "arm": 2.0, "kids": 2.0, "companion": 2.0, "survey": 2.0, "planck": 1.8, "prior": 1.6,
      "repeat": 1.4, "pooled": 3.0, "variate": 2.6}
SHADE_ALPHA = {"flagship": 0.75, "arm": 0.6}       # pinned: chainconsumer's default 1/sqrt(n_chains) varies per figure
TRUTH_KW = {"color": "#444444", "line_style": "--", "line_width": 1.2}
# kind "variate" = an IA-variate outline over the filled NLA-M headline (08): drawn LAST so the filled
# contour never hides it, listed right after the headline in the legend
DRAW_ORDER = {"prior": -1, "planck": 0, "survey": 1, "kids": 2, "companion": 3, "ours": 4, "variate": 5}   # externals under, our arms on top
LEGEND_ORDER = {"ours": 0, "variate": 0.5, "companion": 1, "kids": 2, "survey": 3, "planck": 4, "prior": 5}   # headline first in the legend
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


def blind_label(args, arm: str) -> str:
    """Blind-store label to read `arm` from: `--arm-label ARM=LABEL` overrides `--label` per arm."""
    return dict(s.split("=", 1) for s in (getattr(args, "arm_label", None) or [])).get(arm, args.label)


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
        for r in repeats:
            exp = arm_experiment(arm, r)
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


# (arm, prior) -> what went into the headline: per-repeat dumps on disk, the pooled dump's members,
# and how the headline was formed (manifest `variates` block + README; filled by `load_ours`)
COVERAGE: Dict[tuple, dict] = {}


def _repeat_of(arm: str, experiment: str) -> Optional[int]:
    for r in ARMS[arm][2]:
        if arm_experiment(arm, r) == experiment:
            return int(r)
    return None


def _rlist(reps: Sequence[int]) -> str:
    return ", ".join("r%d" % r for r in reps)


def load_ours(args, arms: Optional[Sequence[str]] = None, strict: bool = True, priors: Optional[Sequence[str]] = None
              ) -> tuple[Dict[tuple, Post], Dict[tuple, List[Post]], Dict[str, float], List[dict]]:
    """{(arm, prior): headline Post}, {(arm, prior): [repeat Posts]}, truth (mock only), provenance.
    `strict=False` (stage-3 variates): an arm / prior with no dump is skipped with a note."""
    heads, reps, truth, prov = {}, {}, {}, []
    priors = list(PRIOR_PARAMS) if priors is None else list(priors)
    arms = list(args.arms if arms is None else arms)
    if args.source == "mock":
        runs = mock_runs(args.matched_root, args.matched_tag, arms, args.repeats, priors)
        event = args.event
    else:
        runs = [r for r in list_raw_runs(BLIND_ROOT) if r["label"] == blind_label(args, r["arm"]) and r["arm"] in arms]
        event = 0
    if not runs:
        if strict:
            raise SystemExit("no runs found for source=%s" % args.source)
        print("[variates] no dumps at all for %s (label %s) -> skipped" % (", ".join(arms), args.label))
        return heads, reps, truth, prov
    trng = np.random.default_rng(args.seed + 7)
    for arm in arms:
        for prior in priors:
            rr = sorted([r for r in runs if r["arm"] == arm and r["prior"] == prior and not r["pooled"]],
                        key=lambda r: r["match"])
            pooled = [r for r in runs if r["arm"] == arm and r["prior"] == prior and r["pooled"]]
            if not rr and not pooled:
                if not strict:
                    print("[variates] %s %s: no dump (pooled or per-repeat) for label %s -> skipped"
                          % (arm, prior, args.label))
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
            on_disk = [int(r["match"].rsplit("_", 1)[-1]) for r in rr]
            cov = dict(per_repeat=on_disk, pooled_members=None, headline="")
            if args.source == "real" and blind_label(args, arm) != args.label:
                cov["label"] = blind_label(args, arm)
            full_set = sorted(int(r) for r in ARMS[arm][2])
            if pooled:
                x, names = _load_physical(pooled[0]["path"], pooled[0]["experiment"], 0)
                exps = pooled_members(pooled[0]["path"])
                mem = None if exps is None else sorted(r for r in (_repeat_of(arm, e) for e in exps) if r is not None)
                cov["pooled_members"] = mem
                if mem is None:
                    cov["headline"] = "cluster pooled dump (members not recorded)"
                    hname = "%s (pooled)" % S.arm_name(arm)
                else:
                    cov["headline"] = "cluster pooled dump over %s" % _rlist(mem)
                    hname = "%s (pooled)" % S.arm_name(arm) if mem == full_set else \
                        "%s (pooled %s)" % (S.arm_name(arm), _rlist(mem))
                    if on_disk and sorted(on_disk) != mem:
                        print("[note] %s %s: pooled dump covers %s but per-repeat dumps on disk are %s "
                              "(re-pool + fetch to include them)" % (arm, prior, _rlist(mem), _rlist(sorted(on_disk))))
                prov.append(dict(arm=arm, prior=prior, match="pooled", file=os.path.basename(pooled[0]["path"]),
                                 experiment=pooled[0]["experiment"], event=0, n=len(x), members=mem))
            elif len(rl) > 1:
                n_min = min(len(p.x) for p in rl)
                # equal-weight mixture needs equal draw counts: seeded uniform subsample, never a head slice
                parts = [p.x if len(p.x) == n_min else p.x[np.sort(trng.choice(len(p.x), n_min, replace=False))]
                         for p in rl]
                x = np.concatenate(parts, axis=0); names = rl[0].names
                hname = "%s (pooled %d repeats)" % (S.arm_name(arm), len(rl)) if sorted(on_disk) == full_set else \
                    "%s (pooled in-script %s)" % (S.arm_name(arm), _rlist(sorted(on_disk)))
                cov["headline"] = "per-repeat dumps concatenated in-script (equal weight, %d draws each) over %s" % (
                    n_min, _rlist(sorted(on_disk)))
            else:
                x, names = rl[0].x, rl[0].names
                hname = "%s (repeat %s only, no pool)" % (S.arm_name(arm), rr[0]["match"].rsplit("_", 1)[-1])
                cov["headline"] = "single repeat %s (no pool)" % _rlist(on_disk)
            if "label" in cov:
                cov["headline"] += " [label %s]" % cov["label"]
            COVERAGE[(arm, prior)] = cov
            role = "flagship" if arm in ("nla_m",) + VARIATE_ARMS else "arm"
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
    # an explicit Post.legend_rank (set by --hos-headline figures) orders first; everything else keeps kind/role order
    for p in sorted(posts, key=lambda p: (getattr(p, "legend_rank", None) is None, getattr(p, "legend_rank", 0) or 0,
                                          LEGEND_ORDER.get(p.kind, 0), ROLE_ORDER.get(p.role, 2) if p.kind == "ours" else 0)):
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
                mock=False, height: float = 5.0):
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
        fig, ax = plt.subplots(figsize=(8.5, height))
        nr = len(rows)
        for i, p in enumerate(rows):
            v = p.x[:, p.names.index("S8")]
            q16, q50, q84 = _wq(v, [0.16, 0.5, 0.84], p.weights)
            y = nr - 1 - i
            if (p.kind == "ours" and p.role == "flagship" and p.key == "nla_m") or p.kind == "planck":
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
# stage 3: the IA variates (08 overlay, 09 S8 whiskers, 10+ per-variate suites)
# --------------------------------------------------------------------------------------------
def _s3copy(p: Post, arm: str, legend_rank=None, **over) -> Post:
    """A stage-3 relabelled copy (same draws): "field level, NLA-M" etc. in the shared figures."""
    kw = dict(weights=p.weights, kind=p.kind, colour=p.colour, role=p.role, key=p.key, linestyle=p.linestyle,
              linewidth=p.linewidth, label=STAGE3_LABELS.get(arm, p.label))
    kw.update(over)
    q = Post(p.name, p.x, p.names, **kw)
    if legend_rank is not None:
        q.legend_rank = legend_rank
    return q


def _hos_trio(head, pr):
    """--hos-headline styling: 2-pt solid outline (legend first), HOS filled, field level dashed outline."""
    return [_s3copy(head(("band", pr)), "band", kind="variate", role="outline", linewidth=LW["variate"],
                    linestyle="-", legend_rank=0),
            _s3copy(head(("hos", pr)), "hos", kind="ours", role="flagship", linestyle="-", legend_rank=1),
            _s3copy(head(("nla_m", pr)), "nla_m", kind="variate", role="outline", linewidth=LW["variate"],
                    linestyle="--", legend_rank=2)]


def _gmrg_quartet(head, pr):
    """--gmrg-headline styling: 2-pt solid outline, HOS dash-dot outline, HOS+Gmrg FILLED (the only filled
    contour), field level dashed outline; legend in that order."""
    out = [_s3copy(head(("band", pr)), "band", kind="variate", role="outline", linewidth=LW["variate"],
                   linestyle="-", legend_rank=0),
           _s3copy(head(("hos", pr)), "hos", kind="variate", role="outline", linewidth=LW["variate"],
                   linestyle="-.", legend_rank=1),
           _s3copy(head(("hos_gmrg", pr)), "hos_gmrg", kind="ours", role="flagship", linestyle="-", legend_rank=2),
           _s3copy(head(("nla_m", pr)), "nla_m", kind="variate", role="outline", linewidth=LW["variate"],
                   linestyle="--", legend_rank=3)]
    return out


def stage3_variates(args, variates, heads, reps, truth, rng, written, group, vs_planck_and_surveys, as_lcdm,
                    kids, comp, planck, sv_L, sv_W, s3, tdir, pre, ttl_kw) -> dict:
    """Write the stage-3 figures; return the manifest `stage3_variates` block. Every figure uses only
    the variates that have a posterior for its prior (none -> the figure is skipped with a note)."""
    L, W = "LCDM_fixed_w0", "kids_s8_analytic"
    models = tuple(m for m in ((L, "LCDM"), (W, "wCDM")) if m[1] in args.models)
    head = heads.get
    fig_inputs = {}           # figure basename -> {arm: headline description}

    def note_inputs(path, arms_prior):
        if path:
            fig_inputs[os.path.basename(path)] = {
                a: COVERAGE.get((a, pr), {}).get("headline", "?") for a, pr in arms_prior}

    present = {pr: [v for v in variates if head((v, pr)) is not None] for pr, _ in models}
    for pr, model in models:
        missing = [v for v in variates if v not in present[pr]]
        if missing:
            print("[variates] %s: no posterior yet for %s -> left out of the %s figures" % (model, ", ".join(missing), model))

    # 08 overlay: NLA-M filled (the Tier-4 teal) + each variate as a thick outline, one contour per model
    for pr, model in models:
        base = head(("nla_m", pr))
        if base is None or not present[pr]:
            print("[skip] 08 overlay %s: %s" % (model, "no NLA-M headline" if base is None else "no variate posterior yet"))
            continue
        posts = [_s3copy(base, "nla_m")] + [
            _s3copy(head((v, pr)), v, kind="variate", role="outline", linewidth=LW["variate"],
                    linestyle=VARIATE_LINESTYLE.get(v, "-")) for v in present[pr]]
        if getattr(args, "hos_headline", False) and present[pr] == ["hos"]:
            posts = [p for p in _hos_trio(head, pr) if p.label != STAGE3_LABELS["band"]]
        if getattr(args, "gmrg_headline", False) and sorted(present[pr]) == ["hos", "hos_gmrg"]:
            posts = [p for p in _gmrg_quartet(head, pr) if p.label != STAGE3_LABELS["band"]]
        arms_pr = [("nla_m", pr)] + [(v, pr) for v in present[pr]]
        ttl = "%s: field level, IA variates vs NLA-M" % PRIOR_NAMES[pr]
        f = corner(posts, PRIOR_PARAMS[pr], os.path.join(s3, "%s08a_IA_overlay_%s_corner.png" % (pre, model)), ttl,
                   truth, rng, args.max_samples, **ttl_kw)
        written.append(f); note_inputs(f, arms_pr)
        shared = [n for n in posts[0].names if n != "S8" and all(n in q.names for q in posts)] + ["S8"]
        f = corner(posts, shared, os.path.join(s3, "%s08d_IA_overlay_%s_shared_corner.png" % (pre, model)),
                   ttl + " (all shared parameters)", truth, rng, args.full_max_samples, full=True, **ttl_kw)
        written.append(f); note_inputs(f, arms_pr)
        write_table(posts, shared, os.path.join(tdir, "%s08_IA_overlay_%s" % (pre, model)),
                    note="%s. Inputs: %s. NB A_IA is each model's own IA amplitude (NLA-M: at the pivot "
                         "halo mass; NLA-z: at the pivot redshift; NLA: constant), so it is comparable only "
                         "loosely." % (ttl, "; ".join("%s = %s" % (a, COVERAGE.get((a, pr), {}).get("headline", "?"))
                                                      for a, pr in arms_pr)))

    # 17g: the NLA-M trio on one corner -- field level filled, 2-pt and the HOS hybrid as outlines
    for pr, model in models:
        trio = [("nla_m", None), ("band", "--"), ("hos", "-")]
        if "hos" not in present[pr] or any(head((a, pr)) is None for a, _ in trio):
            continue
        posts = [_s3copy(head(("nla_m", pr)), "nla_m")] + [
            _s3copy(head((a, pr)), a, kind="variate", role="outline", linewidth=LW["variate"], linestyle=ls)
            for a, ls in trio[1:]]
        if getattr(args, "hos_headline", False):
            posts = _hos_trio(head, pr)
        arms_pr = [(a, pr) for a, _ in trio]
        ttl = "%s: field level vs 2-pt vs 2-pt + third moments (HOS), NLA-M" % PRIOR_NAMES[pr]
        f = corner(posts, PRIOR_PARAMS[pr], os.path.join(s3, "%s17g_HOS_vs_field_2pt_%s_corner.png" % (pre, model)),
                   ttl, truth, rng, args.max_samples, **ttl_kw)
        written.append(f); note_inputs(f, arms_pr)
        shared = [n for n in posts[0].names if n != "S8" and all(n in q.names for q in posts)] + ["S8"]
        write_table(posts, shared, os.path.join(tdir, "%s17g_HOS_vs_field_2pt_%s" % (pre, model)),
                    note="%s. Inputs: %s." % (ttl, "; ".join("%s = %s" % (a, COVERAGE.get((a, p_), {}).get("headline", "?"))
                                                           for a, p_ in arms_pr)))

    # 18g: the 4-way NLA-M comparison -- HOS+Gmrg filled; 2-pt, HOS and field level as outlines
    for pr, model in models:
        quad = ("band", "hos", "hos_gmrg", "nla_m")
        if not getattr(args, "gmrg_headline", False) or any(head((a, pr)) is None for a in quad):
            if getattr(args, "gmrg_headline", False):
                print("[skip] 18g %s: missing %s" % (model, [a for a in quad if head((a, pr)) is None]))
            continue
        posts = _gmrg_quartet(head, pr)
        arms_pr = [(a, pr) for a in quad]
        ttl = "%s: 2-pt vs HOS vs HOS + Gmrg vs field level, NLA-M" % PRIOR_NAMES[pr]
        f = corner(posts, PRIOR_PARAMS[pr], os.path.join(s3, "%s18g_HOSGmrg_vs_HOS_2pt_field_%s_corner.png" % (pre, model)),
                   ttl, truth, rng, args.max_samples, **ttl_kw)
        written.append(f); note_inputs(f, arms_pr)
        shared = [n for n in posts[0].names if n != "S8" and all(n in q.names for q in posts)] + ["S8"]
        write_table(posts, shared, os.path.join(tdir, "%s18g_HOSGmrg_vs_HOS_2pt_field_%s" % (pre, model)),
                    note="%s. Inputs: %s." % (ttl, "; ".join("%s = %s" % (a, COVERAGE.get((a, p_), {}).get("headline", "?"))
                                                           for a, p_ in arms_pr)))

    # 09 S8 whiskers: 06 / 06b with the variate rows between the NLA-M field level and the 2-pt rows
    for pr, model, stem in ((L, "LCDM", "09_S8_summary_variates"), (W, "wCDM", "09b_S8_summary_variates_wCDM")):
        if model not in args.models:
            continue
        if not present[pr]:
            print("[skip] %s: no variate posterior yet" % stem); continue
        order = (["band", "hos", "nla_m"] if getattr(args, "hos_headline", False) and present[pr] == ["hos"]
                 else ["nla_m"] + present[pr] + ["band"])
        if getattr(args, "gmrg_headline", False) and sorted(present[pr]) == ["hos", "hos_gmrg"]:
            order = ["band", "hos", "hos_gmrg", "nla_m"]
        arms_rows = [a for a in order if head((a, pr)) is not None]
        rows = [_s3copy(head((a, pr)), a) for a in arms_rows]
        if pr == L:
            groups = [rows, comp + kids, sv_L, [planck] if planck else []]
        else:
            w_keys = {q.key for q in sv_W}
            sv_only_L = [as_lcdm(q) for q in sv_L if q.key.replace("_lcdm", "_wcdm") not in w_keys]
            groups = [rows, [as_lcdm(q) for q in comp + kids], sv_W + sv_only_L, [as_lcdm(planck)] if planck else []]
        f = s8_whiskers(groups, os.path.join(s3, pre + stem + ".png"),
                        "%s $S_8$ summary with the IA variates (median, 68%%)" % PRIOR_NAMES[pr], truth,
                        height=5.0 + 0.35 * len(present[pr]), **ttl_kw)
        written.append(f); note_inputs(f, [(a, pr) for a in arms_rows])

    # 10+: the full field-arm suite per variate (a headline, b repeats vs pooled, c vs KiDS-Legacy /
    # companion (LCDM), d all parameters + prior, e vs Planck, f vs DES Y3 / HSC Y3)
    for v in variates:
        if v not in VARIATE_GROUPS:
            print("[variates] %s: no figure-group number assigned (VARIATE_GROUPS) -> no per-variate suite" % v)
            continue
        if v not in present.get(L, []) and v not in present.get(W, []):
            print("[skip] %s suite: no posterior for either model" % v); continue
        num, stem = VARIATE_GROUPS[v]
        n0 = len(written)
        if "LCDM" in args.models:
            group(num, stem + "_LCDM", L, [v], s3, kids + comp)
        if "wCDM" in args.models:
            group(num, stem + "_wCDM", W, [v], s3, [])
        vs_planck_and_surveys([v], s3, "%se_%s_LCDM_vs_Planck" % (num, stem), "%sf_%s_LCDM_vs_surveys" % (num, stem),
                              "%sf_%s_wCDM_vs_surveys" % (num, stem), planck_W_stem="%se_%s_wCDM_vs_Planck" % (num, stem))
        for f in written[n0:]:
            if f:
                pr = W if "_wCDM_" in os.path.basename(f) else L
                note_inputs(f, [(v, pr)])

    ia = ia_total_figure(heads, variates, s3, tdir, pre, args, ttl_kw)
    written.extend(ia["figures"])
    for f in ia["figures"]:
        fig_inputs[os.path.basename(f)] = ia.get("inputs", {}).get(os.path.basename(f), {})

    run_priors = [MODEL_PRIOR[m] for m in args.models]
    cov = {"%s/%s" % (a, pr): dict(c, model=PRIOR_NAMES[pr]) for (a, pr), c in COVERAGE.items()
           if (a in variates or a == "nla_m") and pr in run_priors}
    return dict(variates=list(variates), present={model: present[pr] for pr, model in models},
                coverage=cov, figure_inputs=fig_inputs,
                colours={a: arm_colour(a) for a in ["nla_m"] + list(variates)},
                labels={a: STAGE3_LABELS.get(a, a) for a in ["nla_m"] + list(variates) + ["band"]},
                full_names={a: STAGE3_NAMES.get(a, a) for a in ["nla_m"] + list(variates) + ["band"]},
                overlay=dict(nla_m="filled (flagship role)", variates="outlines, lw %.1f" % LW["variate"],
                             linestyle=VARIATE_LINESTYLE),
                ia_total={k: v for k, v in ia.items() if k != "inputs"})


# --------------------------------------------------------------------------------------------
# 12: A_IA,total(z) per IA model (Wright et al. 2025 style), from each model's prefactor
# --------------------------------------------------------------------------------------------
A_PIV = 0.769                  # NLA-z pivot scale factor (src/cosmology/nla.py:nla_z_effective_amplitude)
LOG10_M_PIVOT = 13.5           # NLA-M halo-mass pivot (src/cosmology/nla.py:kappa_ia_nla_m)
IA_TOTAL_NZ_DRAWS = 256        # dz-shift realisations of the tomographic n(z) (marginalised per draw)


def ia_geometry(data_dir: str, seed: int, n_real: int = IA_TOTAL_NZ_DRAWS) -> dict:
    """Per-bin <a>^(i) and <z>^(i) exactly as the simulator forms them: `calculate_tomo_nz` (the
    nla_vd model shifts the n(z) by one dz ~ N(Nz_biases, Nz_covariance) draw per mock) on the
    line-of-sight grid linspace(zmin, zmax, n_los_chi) of src/KiDS/simulation_config.py, then
    <a>^(i) = sum a(z) n_i(z) / sum n_i(z) (master_kids_legacy_simulator.py, the nla_z `avg_a`).
    `n_real` dz realisations, drawn with a seeded copy of the global numpy RNG (restored after)."""
    from src.KiDS.simulation_config import n_los_chi, zmax, zmin
    from src.KiDS.tomo import calculate_tomo_nz, nbins
    z = np.linspace(zmin, zmax, n_los_chi)
    a = 1.0 / (1.0 + z)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        avg_a, avg_z = np.empty((n_real, nbins)), np.empty((n_real, nbins))
        for k in range(n_real):
            nz = calculate_tomo_nz(data_dir, n_los_chi, z, shift_nz=True)
            avg_a[k] = [np.average(a, weights=nz[i]) for i in range(nbins)]
            avg_z[k] = [np.average(z, weights=nz[i]) for i in range(nbins)]
    finally:
        np.random.set_state(state)
    nz0 = calculate_tomo_nz(data_dir, n_los_chi, z, shift_nz=False)
    return dict(avg_a=avg_a, avg_z=avg_z, z_eff=avg_z.mean(axis=0),
                z_eff_unshifted=np.array([np.average(z, weights=nz0[i]) for i in range(nbins)]),
                avg_a_mean=avg_a.mean(axis=0), nbins=nbins)


def ia_total_draws(arm: str, p: Post, geom: dict, data_dir: str, rng: np.random.Generator) -> Optional[np.ndarray]:
    """[S, nbins] total IA amplitude per posterior draw and tomographic bin (the prefactor terms that
    differ between models; the common -rho_c1 Om / D(z) is excluded):
      NLA    A = a_ia (every bin)
      NLA-z  A = a_ia + b_z (<a>^(i)/a_piv - 1), <a>^(i) from a random dz realisation per draw
      NLA-M  A = a_ia f_red^(i) (10^log10M^(i) / 10^13.5)^b_ia, log10M ~ the simulator's massdep MVN
             (load_massdep_priors), drawn independently per posterior draw (not inferred)."""
    nb = geom["nbins"]
    col = lambda n: p.x[:, p.names.index(n)][:, None]
    if arm == "nla":
        return np.repeat(col("a_ia"), nb, axis=1)
    if arm == "nla_z":
        k = rng.integers(0, len(geom["avg_a"]), size=len(p.x))
        return col("a_ia") + col("b_z") * (geom["avg_a"][k] / A_PIV - 1.0)
    if arm in ("nla_m", "band"):
        from src.KiDS.systematics import f_red, load_massdep_priors
        mu, cov = load_massdep_priors(data_dir)
        logm = rng.multivariate_normal(mu, cov, size=len(p.x))
        return col("a_ia") * f_red[None, :] * 10.0 ** (col("b_ia") * (logm - LOG10_M_PIVOT))
    return None


def ia_total_figure(heads, variates, s3, tdir, pre, args, ttl_kw) -> dict:
    """12_A_IA_total_vs_z_{LCDM,wCDM}: NLA-M points (68 % bars) at the bins' effective redshifts, NLA-z
    curve + shaded 68 %, NLA flat band; whichever variates exist. Returns the manifest block."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    out = dict(figures=[], tables=[], method=None, z_eff=None)
    try:
        geom = ia_geometry(args.sim_data_dir, args.seed)
    except Exception as e:  # noqa: BLE001
        print("[skip] 12 A_IA,total: n(z) / massdep inputs unavailable under %s (%s)" % (args.sim_data_dir, e))
        return out
    zg = np.linspace(0.0, 2.0, 201)
    out["z_eff"] = geom["z_eff"].tolist(); out["z_eff_unshifted"] = geom["z_eff_unshifted"].tolist()
    out["avg_a"] = geom["avg_a_mean"].tolist()
    out["method"] = ("per draw: NLA A=a_ia; NLA-z A=a_ia+b_z(<a>_i/%.3f-1) with <a>_i from one of %d dz-shifted n(z) "
                     "realisations (simulator's calculate_tomo_nz + avg_a); NLA-M A=a_ia f_red_i (10^logM_i/10^%.1f)^b_ia "
                     "with logM ~ massdep MVN (massdep_means/cov [2:]) per draw; common -rho_c1 Om/D(z) excluded; "
                     "bins plotted at the mean over realisations of the n(z)-weighted mean z (simulator LOS grid)"
                     % (A_PIV, IA_TOTAL_NZ_DRAWS, LOG10_M_PIVOT))
    for pr, model in (("LCDM_fixed_w0", "LCDM"), ("kids_s8_analytic", "wCDM")):
        if model not in args.models:
            continue
        arms = [a for a in ["nla_m"] + [v for v in ("nla_z", "nla") if v in variates] if heads.get((a, pr)) is not None]
        if "nla_m" not in arms:
            print("[skip] 12 A_IA,total %s: no NLA-M headline" % model); continue
        rng = np.random.default_rng(args.seed + 12)
        rows, yr = [], []
        stem = "%s12_A_IA_total_vs_z_%s" % (pre, model)
        with S.context():
            fig, ax = plt.subplots(figsize=(8.0, 7.0))
            hs = []
            for a in arms:
                p = heads[(a, pr)]
                A = ia_total_draws(a, p, geom, args.sim_data_dir, rng)
                q16, q50, q84 = np.quantile(A, [0.16, 0.5, 0.84], axis=0)
                for i in range(geom["nbins"]):
                    rows.append(dict(model=STAGE3_LABELS[a].replace("$", ""), arm=a, posterior=p.name, bin=i + 1,
                                     z_eff=geom["z_eff"][i], median=q50[i], p16=q16[i], p84=q84[i],
                                     mean=A[:, i].mean(), std=A[:, i].std(), n_draws=len(A)))
                c = arm_colour(a)
                lab = S.tex(STAGE3_LABELS[a])
                if a == "nla":
                    ax.axhspan(q16[0], q84[0], color=c, alpha=0.22, lw=0, zorder=1)
                    ax.axhline(q50[0], color=c, lw=2.4, ls="--", zorder=2)
                    hs.append((Patch(facecolor=mc_rgba(c, 0.35), edgecolor=c, lw=2.4, ls="--"), lab))
                    yr += [q16[0], q84[0]]
                elif a == "nla_z":
                    th = p.thinned(100000, rng)[0]
                    az = 1.0 / (1.0 + zg)
                    Az = th[:, [p.names.index("a_ia")]] + th[:, [p.names.index("b_z")]] * (az[None, :] / A_PIV - 1.0)
                    c16, c50, c84 = np.quantile(Az, [0.16, 0.5, 0.84], axis=0)
                    ax.fill_between(zg, c16, c84, color=c, alpha=0.25, lw=0, zorder=1)
                    ax.plot(zg, c50, color=c, lw=2.4, zorder=3)
                    hs.append((Patch(facecolor=mc_rgba(c, 0.35), edgecolor=c, lw=2.4), lab))
                    yr += list(q16) + list(q84)
                else:
                    ze = np.asarray(geom["z_eff"])
                    ax.errorbar(ze, q50, yerr=[q50 - q16, q84 - q50], fmt="o", color=c, ms=8, lw=2.4, capsize=4,
                                capthick=2.4, zorder=4)
                    hs.append((Line2D([], [], color=c, marker="o", ms=8, lw=2.4), lab))
                    yr += list(q16) + list(q84)
            # reference rows (table only): the NLA-M SAMPLING-prior predictive of A_IA,total, so a reader can
            # tell a data constraint from the prior (a_ia, b_ia have informative Gaussian priors)
            pp = prior_post(arm_experiment("nla_m", ARMS["nla_m"][2][0]), pr, 200000, args.seed)
            if pp is not None:
                A = ia_total_draws("nla_m", pp, geom, args.sim_data_dir, rng)
                q16, q50, q84 = np.quantile(A, [0.16, 0.5, 0.84], axis=0)
                for i in range(geom["nbins"]):
                    rows.append(dict(model="NLA-M sampling-prior predictive (reference)", arm="nla_m_prior",
                                     posterior=pp.name, bin=i + 1, z_eff=geom["z_eff"][i], median=q50[i], p16=q16[i],
                                     p84=q84[i], mean=A[:, i].mean(), std=A[:, i].std(), n_draws=len(A)))
            lo, hi = min(yr), max(yr)
            pad = 0.25 * (hi - lo) if hi > lo else 0.5
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlim(0.0, 1.7)
            ax.axhline(0.0, color="0.6", lw=0.8, zorder=0)
            ax.set_xlabel(r"$z$")
            ax.set_ylabel(r"$A_\mathrm{IA,\,total}$")
            ax.legend([h for h, _ in hs], [l for _, l in hs], loc="best", frameon=False, fontsize=14)
            if ttl_kw.get("titles"):
                ax.set_title(S.tex("%s: total IA amplitude per tomographic bin" % PRIOR_NAMES[pr]))
            if ttl_kw.get("mock"):
                _mock_tag(fig)
            S.save(fig, os.path.join(s3, stem + ".png"))
        f = os.path.join(s3, stem + ".png")
        print("[fig] %s" % f)
        LEGEND_LOG[os.path.basename(f)] = [[STAGE3_LABELS[a], heads[(a, pr)].name] for a in arms]
        out["figures"].append(f)
        out.setdefault("inputs", {})[os.path.basename(f)] = {a: COVERAGE.get((a, pr), {}).get("headline", "?") for a in arms}
        tb = os.path.join(tdir, "%s12_A_IA_total_%s" % (pre, model))
        os.makedirs(tdir, exist_ok=True)
        keys = ["model", "arm", "bin", "z_eff", "median", "p16", "p84", "mean", "std", "n_draws", "posterior"]
        with open(tb + ".csv", "w") as fh:
            fh.write(",".join(keys) + "\n")
            for r in rows:
                fh.write(",".join("%.5g" % r[k] if isinstance(r[k], float) else '"%s"' % r[k] if k == "posterior" else str(r[k])
                                  for k in keys) + "\n")
        with open(tb + ".md", "w") as fh:
            fh.write("%s: A_IA,total per tomographic bin (median, 68%%). Method: %s. Inputs: %s.\n\n" % (
                PRIOR_NAMES[pr], out["method"], "; ".join("%s = %s" % kv for kv in out["inputs"][os.path.basename(f)].items())))
            fh.write("| model | bin | z_eff | median | 68% interval | mean | std |\n|---|---|---|---|---|---|---|\n")
            for r in rows:
                fh.write("| %s | %d | %.3f | %.3f | [%.3f, %.3f] | %.3f | %.3f |\n" % (
                    r["model"], r["bin"], r["z_eff"], r["median"], r["p16"], r["p84"], r["mean"], r["std"]))
        print("[table] %s.{csv,md}" % tb)
        out["tables"].append(tb)
    return out


def mc_rgba(c, alpha):
    import matplotlib.colors as mc
    return mc.to_rgba(c, alpha)


S3_BEGIN, S3_END = "<!-- stage3-variates:begin -->", "<!-- stage3-variates:end -->"


def write_stage3_readme(out_dir: str, v3: dict, pre: str, args):
    """(Re)write the README's stage-3 section between the markers; the rest of the README is kept."""
    import datetime
    L_, W_ = "LCDM_fixed_w0", "kids_s8_analytic"
    lines = [S3_BEGIN, "", "## Stage 3 -- IA variates (`stage3_variates/`)", "",
             "Produced by `PYTHONPATH=. python scripts/tier4_reveal_plots.py --source %s --label %s --stage 3 "
             "--out-dir <this folder>` (%s UTC; variates requested: %s). Availability is PARTIAL and grows: "
             "re-run the same command after new repeats / pools are fetched into the store "
             "(`scripts/sample_observation.py fetch --what both`); it rewrites only `stage3_variates/`, the "
             "`08`-`11` tables, this section and the manifest's `stage3_variates` block."
             % (args.source, args.label, datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
                ", ".join(args.variates)), "",
             "Inputs (what each headline is; `per-repeat` = repeat dumps on disk; the pooled dump's members "
             "come from its own `pooled_provenance`):", "",
             "| arm | model | per-repeat dumps | pooled dump members | headline used |", "|---|---|---|---|---|"]
    for arm in ["nla_m"] + list(v3["variates"]):
        for pr in (L_, W_):
            c = v3["coverage"].get("%s/%s" % (arm, pr))
            if c is None:
                lines.append("| %s | %s | none | none | -- (not available yet: left out) |" % (arm, PRIOR_NAMES[pr]))
                continue
            mem = c.get("pooled_members")
            lines.append("| %s | %s | %s | %s | %s |" % (arm, PRIOR_NAMES[pr], _rlist(c["per_repeat"]) or "none",
                                                      "--" if mem is None else _rlist(mem), c["headline"]))
    figs = [os.path.splitext(os.path.basename(f))[0] for f in v3["figures"]]
    lines += ["", "Figures written by the last stage-3 run (`.png` 200 dpi + `.pdf` 300 dpi):", ""]
    desc = {"08a": "overlay on the cosmological parameters: field level NLA-M (pooled, filled teal) + the IA variates (outlines), one contour",
            "08d": "the same overlay on every parameter the arms share (cosmology + A_IA; NB A_IA is each model's own amplitude)",
            "09_": "S8 whiskers as 06 with the variate rows (LCDM)",
            "12_": "A_IA,total vs z per IA model (Wright+25 Fig. style): NLA-M points + 68% bars at the bins' z_eff, NLA-z curve + 68% band, NLA flat band; table tables/12_A_IA_total_*", "09b": "S8 whiskers as 06b with the variate rows (wCDM; LCDM-only externals tagged)",
            "a": "headline corner (pooled)", "b": "repeats vs pooled", "c": "vs KiDS-Legacy COSEBIs / band powers + companion 2-pt SBI",
            "d": "all free parameters + S8, sampling prior on the diagonal", "e": "vs Planck 2018 (wCDM: the Planck LCDM chain, tagged)",
            "f": "vs DES Y3 / HSC Y3 public chains"}
    for f in figs:
        b = f[len(pre):]
        key = b[:3] if b[:3] in desc else b[2]
        ins = v3["figure_inputs"].get(f + ".png", {})
        lines.append("- `%s` -- %s. Inputs: %s" % (f, desc.get(key, ""), "; ".join("%s = %s" % kv for kv in ins.items()) or "--"))
    if not figs:
        lines.append("- (none yet: no variate posterior in the store)")
    lines += ["", "Groups: `08` overlay, `09` S8 whiskers, %s, `12` A_IA,total vs z; the per-variate suites mirror the field arm's "
              "02/04/05/07 letters (`a` headline, `b` repeats vs pooled, `c` vs KiDS-Legacy + companion (LCDM only), "
              "`d` all parameters with the prior, `e` vs Planck, `f` vs DES Y3 / HSC Y3). Tables: `tables/08_*`, "
              "`tables/1X_*`, `tables/12_A_IA_total_*` (mean / std / median / 68%%; each variate table's first line names its inputs)."
              % ", ".join("`%s` %s" % (n, STAGE3_LABELS[a]) for a, (n, _st) in VARIATE_GROUPS.items()), "",
              "Legend / colours (script-local, `ARM_COLOURS` in the script): \"field level, NLA-M\" = teal %s "
              "(filled, as in stages 1-2); \"field level, NLA-z\" = Tol-vibrant blue %s; \"field level, NLA\" = "
              "Tol-vibrant red %s (dashed in the 08 overlay); \"2-pt\" = wine (09 only). The style's default variate "
              "colours were not used: they are the 2-pt wine (NLA-z) and the DES+KiDS-1000 rose (NLA). In the "
              "per-variate suites the variate is the filled headline and its repeats a tint ramp of its hue. "
              "Externals as in stages 1-2. Full names: %s."
              % (arm_colour("nla_m"), arm_colour("nla_z"), arm_colour("nla"),
                 "; ".join("\"%s\" = %s" % (STAGE3_LABELS[a], STAGE3_NAMES[a]) for a in STAGE3_NAMES)), "",
              "`12` A_IA,total(z): %s. Bin effective redshifts z_eff = %s (unshifted n(z): %s); mean <a>_i = %s. "
              "NLA-k is not part of this analysis. The `12` tables also list the NLA-M SAMPLING-prior predictive "
              "(a_ia, b_ia Gaussian priors x the same massdep halo masses) as reference rows: compare before reading "
              "the NLA-M points as a constraint." % (
                  (v3.get("ia_total") or {}).get("method") or "not produced (inputs missing)",
                  ", ".join("%.3f" % z for z in (v3.get("ia_total") or {}).get("z_eff") or []) or "--",
                  ", ".join("%.3f" % z for z in (v3.get("ia_total") or {}).get("z_eff_unshifted") or []) or "--",
                  ", ".join("%.3f" % z for z in (v3.get("ia_total") or {}).get("avg_a") or []) or "--"), "",
              "Same style and smoothing as stages 1-2 (manifest `style`): 3/4-parameter corners thinned to "
              "`--max-samples` by seeded uniform thinning (never a head slice), full / shared-parameter corners "
              "on all draws with the Silverman-scaled bins.", "", S3_END]
    block = "\n".join(lines).replace("NLA-$z$", "NLA-z").replace("$\\Lambda$CDM", "LCDM").replace("$w$CDM", "wCDM") + "\n"
    path = os.path.join(out_dir, pre + "README.md")
    txt = open(path).read() if os.path.exists(path) else "# Tier-4 reveal set (label %s)\n" % args.label
    if S3_BEGIN in txt and S3_END in txt:
        a, b = txt.index(S3_BEGIN), txt.index(S3_END) + len(S3_END)
        txt = txt[:a] + block.rstrip("\n") + txt[b:]
    else:
        txt = txt.rstrip("\n") + "\n\n" + block
    with open(path, "w") as fh:
        fh.write(txt)
    print("[readme] %s (stage-3 section)" % path)


# --------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("mock", "real"), required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stage", choices=("1", "2", "3", "all"), default="all",
                    help="1 = 2-pt, 2 = field + 2-pt, 3 = IA variates (merges into an existing manifest), all")
    ap.add_argument("--label", default="A2long", help="real: blind-store label")
    ap.add_argument("--arm-label", nargs="*", default=[], metavar="ARM=LABEL",
                    help="real: read these arms from another blind-store label (e.g. band_nla_z=A2 before its A2long "
                         "dumps exist); recorded in the figure inputs as '[label X]'")
    ap.add_argument("--matched-root", default="/data/alex/variate_samples_kids")
    ap.add_argument("--matched-tag", default="gower_match_vdq")
    ap.add_argument("--event", type=int, default=0, help="mock: event index of the matched dump")
    ap.add_argument("--arms", nargs="+", default=["band", "nla_m"])
    ap.add_argument("--sim-data-dir", default=os.path.join(REPO, "kids-legacy-sbi", "data"),
                    help="stage 3 / 12: the simulator data dir (nofzs/, priors/massdep_*) for A_IA,total")
    ap.add_argument("--models", nargs="+", choices=list(MODEL_PRIOR), default=list(MODEL_PRIOR),
                    help="stage 3: cosmological models to (re)draw; the other model's stage-3 figures/records are kept")
    ap.add_argument("--variates", nargs="*", default=list(VARIATE_ARMS),
                    help="stage 3: IA-variate arms (whichever have dumps are plotted; the rest are skipped)")
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
    ap.add_argument("--hos-headline", action="store_true",
                    help="stage 3 with --variates hos: HOS filled as the headline, 2-pt solid outline first, field "
                         "level dashed outline (08 overlay, 17g trio, 09 row order 2-pt / HOS / field)")
    ap.add_argument("--gmrg-headline", action="store_true",
                    help="with --variates hos hos_gmrg: 18g 4-way corner (2-pt solid / HOS dash-dot outlines, HOS + Gmrg "
                         "filled, field level dashed), 08 overlay and 09 rows 2-pt / HOS / HOS+Gmrg / field")
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
    do3 = args.stage in ("3", "all")
    s3 = os.path.join(args.out_dir, "stage3_variates")

    variates = []
    for v in (args.variates if do3 else []):
        if v not in ARMS:
            print("[variates] %s: unknown arm -> skipped" % v)
        elif v in DELAYED_ARMS:
            print("[variates] %s: DELAYED arm (row still names a no-VD chain) -> skipped" % v)
        elif v not in variates:
            variates.append(v)
    base_arms = list(args.arms) if (do1 or do2) else [a for a in ("nla_m", "band") if a not in variates]
    heads, reps, truth, prov = load_ours(args, base_arms)
    if variates:
        vh, vr, _vt, vp = load_ours(args, variates, strict=False, priors=[MODEL_PRIOR[m] for m in args.models])
        heads.update(vh); reps.update(vr); prov.extend(vp)
    kids = load_kids(args.kids_dir, args.kids_stats) if args.kids_stats else []
    planck = load_planck(args.planck_dir, args.planck_burnin)
    comp = load_companion(args.companion_dir)
    surveys = load_surveys(args.beyond2pt) if not args.no_surveys else []
    L, W = "LCDM_fixed_w0", "kids_s8_analytic"
    tt = {"band": "2-pt", "nla_m": "field", "nla_z": "field NLA-z", "nla": "field NLA", "k2": "field kappa=2",
          "k2_nla": "field kappa=2 NLA", "band_nla": "2-pt NLA", "band_nla_z": "2-pt NLA-z", "hos": "HOS",
          "hos_gmrg": "HOS+Gmrg"}
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
                cov = COVERAGE.get((a, prior), {})
                mem, disk = cov.get("pooled_members"), cov.get("per_repeat")
                plab = "pooled" if not mem or sorted(mem) == sorted(disk or []) else "pooled (%s)" % _rlist(mem)
                pooled = Post(h.name, h.x, h.names, colour=arm_colour(a), role="outline", key=a,
                              linewidth=LW["pooled"], label=plab)
                written.append(corner(rl + [pooled], params, os.path.join(stage_dir, "%sb_%s_repeats_%s.png" % (tag, name, a)),
                                      "%s %s: repeats vs pooled" % (tt[a], PRIOR_NAMES[prior]), truth, rng,
                                      args.max_samples, **ttl_kw))
        ext = [e for e in externals if e is not None and e.has(params)]
        if ext:
            written.append(corner(ext + hs, params, os.path.join(stage_dir, "%sc_%s_vs_external.png" % (tag, name)),
                                  ttl + ": vs external", truth, rng, args.max_samples, **ttl_kw))
        # d: the FULL free-parameter posterior (+S8) with the sampling prior on the diagonal
        full = [n for n in hs[0].names if n != "S8"] + ["S8"]
        pr = prior_post(hs[0].key and arm_experiment(hs[0].key, ARMS[hs[0].key][2][0]), prior, args.prior_draws, args.seed)
        written.append(corner(hs, full, os.path.join(stage_dir, "%sd_%s_full_corner.png" % (tag, name)),
                              ttl + ": all parameters, with the prior", truth, rng, args.full_max_samples,
                              prior=pr, full=True, **ttl_kw))
        allp = hs + [p for a in arms_in for p in reps.get((a, prior), [])] + ext + ([pr] if pr else [])
        cov_note = "".join(". %s: headline = %s; per-repeat dumps: %s" % (
            a, COVERAGE[(a, prior)]["headline"], _rlist(COVERAGE[(a, prior)]["per_repeat"]) or "none")
            for a in arms_in if a in variates and (a, prior) in COVERAGE)
        write_table(allp, full, os.path.join(tdir, "%s_%s" % (tag, name)), note="%s%s%s" % ("MOCK (truth: %s). " % {
            k: float(round(v, 4)) for k, v in truth.items() if k in full} if mock else "", ttl, cov_note))

    sv_L = [p for p in surveys if getattr(p, "model", "LCDM") == "LCDM"]
    sv_W = [p for p in surveys if getattr(p, "model", "LCDM") == "wCDM"]

    def as_lcdm(p: Post) -> Post:
        """A copy of an external LCDM-only posterior for the wCDM figures: same draws, label tagged
        ($\Lambda$CDM) so the reader sees the chain carries no w (user 2026-10-12: 'add an extra row
        without w from the ones that are missing it')."""
        return Post(p.name + " [LCDM chain]", p.x, p.names, weights=p.weights, kind=p.kind, colour=p.colour,
                    role=p.role, key=p.key + "_aslcdm", linestyle=p.linestyle, linewidth=p.linewidth,
                    label=p.label + r" ($\Lambda$CDM)")

    def vs_planck_and_surveys(arms_in, stage_dir, planck_stem, surv_L_stem, surv_W_stem, planck_W_stem=None):
        """Our headline posteriors against Planck 2018 (LCDM) and the public DES Y3 / HSC Y3 chains
        (LCDM; wCDM vs DES Y3 only). Stage 1: the 2-pt arm alone (01e/01f/03f/03e); stage 2: field + 2-pt
        (05/07a/07b/05b). The KiDS-Legacy release has LCDM chains only, so no wCDM KiDS overlay exists;
        `planck_W_stem` draws the wCDM arms against the Planck LCDM chain on the (Om, s8, S8) corner."""
        who = " + ".join(tt[a] for a in arms_in)
        hs_L = [head(a, L) for a in arms_in if head(a, L) is not None]
        hs_W = [head(a, W) for a in arms_in if head(a, W) is not None]
        if planck is not None and hs_L:
            written.append(corner([planck] + hs_L, PRIOR_PARAMS[L], os.path.join(stage_dir, pre + planck_stem + "_corner.png"),
                                  "%s: %s vs Planck 2018" % (PRIOR_NAMES[L], who), truth, rng, args.max_samples, **ttl_kw))
            write_table(hs_L + [planck], PRIOR_PARAMS[L], os.path.join(tdir, pre + planck_stem))
        if planck is not None and hs_W and planck_W_stem:
            pl = as_lcdm(planck)
            written.append(corner([pl] + hs_W, PRIOR_PARAMS[L], os.path.join(stage_dir, pre + planck_W_stem + "_corner.png"),
                                  "%s: %s vs Planck 2018 (LCDM chain)" % (PRIOR_NAMES[W], who), truth, rng, args.max_samples,
                                  **ttl_kw))
            write_table(hs_W + [pl], PRIOR_PARAMS[W], os.path.join(tdir, pre + planck_W_stem))
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
        vs_planck_and_surveys(["band"], s1, "01e_2pt_LCDM_vs_Planck", "01f_2pt_LCDM_vs_surveys", "03f_2pt_wCDM_vs_surveys",
                              planck_W_stem="03e_2pt_wCDM_vs_Planck")
    if do2:
        group("02", "2pt_field_LCDM", L, ["band", "nla_m"], s2, kids + comp)
        group("04", "2pt_field_wCDM", W, ["band", "nla_m"], s2, [])
        vs_planck_and_surveys(["band", "nla_m"], s2, "05_LCDM_vs_Planck", "07a_LCDM_vs_surveys", "07b_wCDM_vs_surveys",
                              planck_W_stem="05b_wCDM_vs_Planck")
        # whisker groups (WHISKER_GROUPS): this work | KiDS-Legacy (companion SBI, Wright+25) | other surveys
        # (DES Y3 / HSC Y3, LCDM) | Planck
        ours = [head(a, L) for a in ("nla_m", "band") if head(a, L) is not None]
        written.append(s8_whiskers([ours, comp + kids, sv_L, [planck] if planck else []],
                                   os.path.join(s2, pre + "06_S8_summary.png"),
                                   "%s $S_8$ summary (median, 68%%)" % PRIOR_NAMES[L], truth, **ttl_kw))
        # 06b: the wCDM whisker. Externals with a wCDM chain (DES Y3 cosmic shear / 3x2pt) appear as such;
        # those without one (KiDS-Legacy, the companion SBI, HSC Y3, DES Y3 + KiDS-1000, Planck) keep their
        # LCDM row, tagged (LCDM).
        ours_W = [head(a, W) for a in ("nla_m", "band") if head(a, W) is not None]
        if ours_W:
            w_keys = {p.key for p in sv_W}
            sv_only_L = [as_lcdm(p) for p in sv_L if p.key.replace("_lcdm", "_wcdm") not in w_keys]
            written.append(s8_whiskers([ours_W, [as_lcdm(p) for p in comp + kids], sv_W + sv_only_L,
                                        [as_lcdm(planck)] if planck else []],
                                       os.path.join(s2, pre + "06b_S8_summary_wCDM.png"),
                                       "%s $S_8$ summary (median, 68%%); rows tagged LCDM carry no $w$" % PRIOR_NAMES[W],
                                       truth, **ttl_kw))

    v3 = None
    if do3:
        n0 = len(written)
        v3 = stage3_variates(args, variates, heads, reps, truth, rng, written, group, vs_planck_and_surveys, as_lcdm,
                             kids, comp, planck, sv_L, sv_W, s3, tdir, pre, ttl_kw)
        v3["figures"] = [w for w in written[n0:] if w]

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
    mpath = os.path.join(args.out_dir, pre + "manifest.json")
    if v3 is not None:
        meta["stage3_variates"] = v3
        if args.stage == "3" and os.path.exists(mpath):
            # MERGE: keep the stage-1/2 record, replace only what stage 3 owns
            with open(mpath) as fh:
                old = json.load(fh)
            # a `--models LCDM` run keeps the earlier wCDM stage-3 records (and vice versa)
            ov3 = old.get("stage3_variates") or {}
            keep = [f for f in ov3.get("figures", []) if _fig_model(f) not in args.models]
            v3["figures"] = keep + v3["figures"]
            run_priors = [MODEL_PRIOR[m] for m in args.models]
            v3["coverage"] = {**{k: c for k, c in (ov3.get("coverage") or {}).items() if k.split("/", 1)[1] not in run_priors},
                              **v3["coverage"]}
            v3["figure_inputs"] = {**{k: c for k, c in (ov3.get("figure_inputs") or {}).items() if _fig_model(k) not in args.models},
                                   **v3["figure_inputs"]}
            v3["present"] = {**{k: c for k, c in (ov3.get("present") or {}).items() if k not in args.models}, **v3["present"]}
            v3["models_last_run"] = list(args.models)
            old["figures"] = [f for f in old.get("figures", []) if os.sep + "stage3_variates" + os.sep not in f] + v3["figures"]
            old.setdefault("legend", {}).update(LEGEND_LOG)
            old["provenance"] = [p for p in old.get("provenance", []) if p.get("arm") not in variates] + \
                [p for p in prov if p.get("arm") in variates]
            old["stage3_variates"] = v3
            meta = old
        write_stage3_readme(args.out_dir, v3, pre, args)
    with open(mpath, "w") as fh:
        json.dump(meta, fh, indent=2, default=float)
    print("[done] %d figures -> %s" % (len([w for w in written if w]), args.out_dir))


if __name__ == "__main__":
    main()
