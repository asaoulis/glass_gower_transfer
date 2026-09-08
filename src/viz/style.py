"""Publication style for every figure in the KiDS-Legacy multifidelity-SBI project.

ONE source of truth. Everything here is lifted from the options the paper figures were rendered
with (`.claude/runs/paper-review/last-runs-and-plots/artifacts/{plot_common,chain_common}.py`,
`fig3_fig4_posteriors.py`, `ml_results_plotting.ipynb` cells 0/6/28,
`ml_experiments_9param_saved.ipynb` cells 3/5/6/11) so new figures match the published ones:

  * `scienceplots` ``["science", "muted"]`` style context: serif, LaTeX text (``usetex``),
    ticks in, no grid, frameless legends, Paul Tol *muted* cycle.
  * A GLOBAL ``font.size = 18`` before drawing (the notebooks did this; the science style does
    not reset it). Figures are drawn at ~6x4.5 in and scaled to column width by LaTeX, so 18 pt
    lands at ~9 pt on the page. Keep it, or the new figures come out with smaller text.
  * Colour identity pinned BY KEY into the cycle (the paper's ``EXPERIMENT_COLORS`` pattern), not
    consumed from the prop cycle in draw order.
  * Corner plots through ChainConsumer 1.3 with the notebook's ``PlotConfig`` (flip=False,
    tick/label 16 pt, contour labels 20 pt, ``max_ticks=2``, serif, usetex, no summaries) and the
    cell-11 fill (``shade=True, shade_gradient=0.9``), ``ChainConfig(smooth=10)`` for the 4-param
    cosmology corner, edge ticks pruned; saved as SVG 300 dpi + PNG 200 dpi.

Palette OPTIONS (`PALETTES`) are alternative cycles for the same roles. `tol-muted` is the
published default; the others are offered so a choice can be made on real figures
(`scripts/unblinding_figures.py --palette`). Every palette carries the SAME role table
(`ROLES`) -- real data is always black, the mock ensemble always light grey, the stop rule
always a low-alpha red -- so swapping the palette never changes what a figure says.

Blind-analysis rule: nothing in this module prints, summarises or annotates chain statistics.
`plot_chains` forces ``summarise=False``; do not add `Truth` markers to a chain from the blind
store (there is none to add).
"""
from __future__ import annotations

import contextlib
import importlib
import os
import re
from typing import Iterable, Mapping, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.ticker import MaxNLocator

# --------------------------------------------------------------------------------------------
# scienceplots: importing registers the styles, but on a fresh interpreter the style library is
# sometimes built before the entry points are read -- the notebook works around that with an
# importlib.reload. Keep the same trick.
# --------------------------------------------------------------------------------------------
NOTEBOOK_FONT_SIZE = 18          # the paper's global font size (notebook cell 0)
PNG_DPI = 200                    # paper: png 200 dpi, svg/pdf 300 dpi
VECTOR_DPI = 300

# A&A widths, for figures that must be drawn at final size instead of scaled by LaTeX.
AA_COLUMN_IN = 88 / 25.4         # \columnwidth  (3.46 in)
AA_TEXT_IN = 180 / 25.4          # \textwidth    (7.09 in)

# Paper figure sizes (inches) -- what the published panels were drawn at.
FIGSIZE_LINE = (6, 4.5)          # fig1/fig2 metric-vs-N panels
FIGSIZE_LINE_WIDE = (7, 5)       # plot_results default
FIGSIZE_CORNER = (6, 6)          # 4-param cosmology corner (fig3_cosmo)
FIGSIZE_CORNER_FULL = (10, 10)   # 9/10-param corner (fig3_full)
FIGSIZE_TWO_PANEL = (7.4, 9.6)   # fidelity two-panel (stacked)


def use_scienceplots() -> list[str]:
    """Register the scienceplots styles (with the notebook's reload trick) and set the paper's
    global font size. Returns the style list the paper used: ``["science", "muted"]``."""
    import scienceplots  # noqa: F401

    if "science" not in plt.style.available:
        importlib.reload(scienceplots)
        import scienceplots  # noqa: F401,F811
    if "science" not in plt.style.available:
        raise RuntimeError("scienceplots styles are not registered; `pip install scienceplots`.")
    plt.rcParams.update({"font.size": NOTEBOOK_FONT_SIZE})
    return ["science", "muted"]


def muted_colors() -> list[str]:
    """The `muted` cycle as the notebooks index it (0 rose, 1 indigo, 2 sand, 3 green, 4 cyan,
    5 wine, 6 teal, 7 olive, 8 purple, 9 pale grey)."""
    use_scienceplots()
    with plt.style.context(["science", "muted"]):
        return plt.rcParams["axes.prop_cycle"].by_key()["color"]


# --------------------------------------------------------------------------------------------
# Palette options. Each is an ordered cycle; roles below index into it BY KEY.
# --------------------------------------------------------------------------------------------
PALETTES: dict[str, dict] = {
    # The published default (scienceplots `muted` == Paul Tol muted, same order).
    "tol-muted": {
        "style": "muted",
        "cycle": ["#CC6677", "#332288", "#DDCC77", "#117733", "#88CCEE", "#882255", "#44AA99",
                  "#999933", "#AA4499", "#DDDDDD"],
        "names": ["rose", "indigo", "sand", "green", "cyan", "wine", "teal", "olive", "purple",
                  "pale grey"],
        "note": "Paul Tol muted -- the paper's cycle (fig1-4). Colour-blind safe; prints well.",
    },
    # Paul Tol bright: brighter, fewer colours, still CVD-safe (scienceplots `bright`).
    "tol-bright": {
        "style": "bright",
        "cycle": ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"],
        "names": ["blue", "red", "green", "yellow", "cyan", "purple", "grey"],
        "note": "Paul Tol bright. Higher chroma than muted; 6 hues + grey.",
    },
    # Paul Tol high-contrast + medium-contrast extension: 3 saturated hues + their light/dark
    # partners. Best for real-vs-mock-vs-reference figures with few classes.
    "tol-high-contrast": {
        "style": "high-contrast",
        "cycle": ["#004488", "#DDAA33", "#BB5566", "#6699CC", "#997700", "#994455", "#EECC66",
                  "#EE99AA"],
        "names": ["blue", "yellow", "red", "light blue", "dark yellow", "dark red",
                  "light yellow", "light red"],
        "note": "Paul Tol high/medium contrast. Pairs separated by lightness -- survives greyscale.",
    },
    # Survey-flavoured deep tones (KiDS/DES-style contour colours): navy, vermilion, teal, amber,
    # plum, slate. Not a Tol set; checked for deuteranopia separation (worst pair navy/plum).
    "survey": {
        "style": None,
        "cycle": ["#1F4E79", "#C8502B", "#2A9D8F", "#E9B44C", "#6D3F7A", "#5B6770", "#7FB3D5",
                  "#B8B8B8"],
        "names": ["navy", "vermilion", "teal", "amber", "plum", "slate", "sky", "grey"],
        "note": "Deep survey tones. Navy/vermilion read as the classic 'data vs model' pair.",
    },
}
DEFAULT_PALETTE = "tol-muted"

# Semantic roles -> index into the palette cycle (or a fixed hex). The role table is the same
# for every palette so a figure keeps its meaning when the palette is swapped.
#   real       the observation / real-data posterior (ALWAYS black, thick)
#   flagship   the headline nla_m BGP arm
#   mock       matched-mock / null ensembles (ALWAYS light grey, thin)
#   reference  guide lines (analytic nulls, medians): dark grey, dashed/dotted
#   stop       stop-rule regions: the palette's red at low alpha
#   pass       "in distribution" tints: the palette's green
#   T / S      the two mock labels (T = known-OOD control, S = in-distribution control)
#   arms       the six analysis arms of `src/ml/eval/arms.py`
ROLES: dict[str, dict] = {
    "tol-muted": {
        "real": "#000000", "flagship": 1, "mock": "#B0B0B0", "reference": "#444444",
        "stop": 0, "pass": 3, "T": 0, "S": 1,
        "arms": {"nla_m": 1, "nla_m_nobgp": 4, "nla": 0, "nla_z": 5, "vd": 6, "k2": 7},
    },
    "tol-bright": {
        "real": "#000000", "flagship": 0, "mock": "#B0B0B0", "reference": "#444444",
        "stop": 1, "pass": 2, "T": 1, "S": 0,
        "arms": {"nla_m": 0, "nla_m_nobgp": 4, "nla": 1, "nla_z": 5, "vd": 2, "k2": 3},
    },
    "tol-high-contrast": {
        "real": "#000000", "flagship": 0, "mock": "#B0B0B0", "reference": "#444444",
        "stop": 2, "pass": 4, "T": 2, "S": 0,
        "arms": {"nla_m": 0, "nla_m_nobgp": 3, "nla": 2, "nla_z": 5, "vd": 1, "k2": 4},
    },
    "survey": {
        "real": "#000000", "flagship": 0, "mock": "#B0B0B0", "reference": "#444444",
        "stop": 1, "pass": 2, "T": 1, "S": 0,
        "arms": {"nla_m": 0, "nla_m_nobgp": 6, "nla": 1, "nla_z": 4, "vd": 2, "k2": 3},
    },
}

# Sequential ramps for ordered classes (tomographic bins, seeds): the paper used viridis
# (`plt.cm.viridis(np.linspace(0.12, 0.82, n))`, fidelity + coverage figures).
SEQUENTIAL_CMAP = "viridis"
DIVERGING_CMAP = "RdBu_r"        # z-score heatmaps, symmetric about 0
ARM_ORDER = ("nla_m", "nla_m_nobgp", "nla", "nla_z", "vd", "k2")


def _check_arm_drift() -> None:
    """The role table must name exactly the arms in `src.ml.eval.arms.ARMS` (lockstep guard)."""
    try:
        from src.ml.eval.arms import ARMS
    except Exception:            # pragma: no cover - importable without the ML stack
        return
    for name, roles in ROLES.items():
        missing = set(ARMS) - set(roles["arms"])
        extra = set(roles["arms"]) - set(ARMS)
        if missing or extra:
            raise RuntimeError(f"style.ROLES[{name}]['arms'] out of step with ARMS: "
                               f"missing={sorted(missing)} extra={sorted(extra)}")


def palette(name: str = DEFAULT_PALETTE) -> list[str]:
    return list(PALETTES[name]["cycle"])


def _resolve(spec, name: str) -> str:
    return PALETTES[name]["cycle"][spec] if isinstance(spec, int) else str(spec)


def role_colour(role: str, name: str = DEFAULT_PALETTE) -> str:
    """Colour for a semantic role ('real', 'flagship', 'mock', 'reference', 'stop', 'pass',
    'T', 'S')."""
    return _resolve(ROLES[name][role], name)


def arm_colour(arm: str, name: str = DEFAULT_PALETTE) -> str:
    _check_arm_drift()
    return _resolve(ROLES[name]["arms"][arm], name)


def arm_colours(name: str = DEFAULT_PALETTE) -> dict[str, str]:
    return {a: arm_colour(a, name) for a in ARM_ORDER}


def label_colour(label: str, name: str = DEFAULT_PALETTE) -> str:
    """Label colour: the mock controls T and S are pinned; ANY other label is an observation and
    takes the `real` role (black). Never derived from a hash (process-salted, non-deterministic)."""
    if label in ("T", "S"):
        return role_colour(label, name)
    return role_colour("real", name)


def sequential(n: int, cmap: str = SEQUENTIAL_CMAP, lo: float = 0.12, hi: float = 0.82):
    """`n` ordered colours from a perceptual ramp, the paper's viridis window."""
    return [matplotlib.colormaps[cmap](x) for x in np.linspace(lo, hi, n)]


def styles(name: str = DEFAULT_PALETTE, usetex: bool | None = None) -> list[str]:
    """Style list for `plt.style.context`: science (+ the Tol style when scienceplots has it,
    + `no-latex` when usetex=False)."""
    use_scienceplots()
    out = ["science"]
    if PALETTES[name]["style"]:
        out.append(PALETTES[name]["style"])
    if usetex is False:
        out.append("no-latex")
    return out


@contextlib.contextmanager
def context(name: str = DEFAULT_PALETTE, font_size: float = NOTEBOOK_FONT_SIZE,
            usetex: bool | None = None, **rc):
    """`with style.context(): ...` -- the paper's style context with the chosen palette cycle.

    ``font_size`` defaults to the paper's 18 pt (draw at ~6x4.5 in, LaTeX scales to column
    width). Pass ``font_size=9`` with an A&A-width figsize to draw at final size instead.
    ``usetex=None`` keeps the science style's LaTeX text; ``False`` selects `no-latex` (+ STIX
    math) for environments without TeX or for strings with unescaped specials.
    """
    with plt.style.context(styles(name, usetex)):
        plt.rcParams["axes.prop_cycle"] = cycler(color=palette(name))
        plt.rcParams["font.size"] = font_size
        if usetex is False:
            plt.rcParams["mathtext.fontset"] = "stix"
        plt.rcParams.update(rc)
        yield


_TEX_SPECIALS = {"%": r"\%", "&": r"\&", "#": r"\#", "_": r"\_"}
# In LaTeX TEXT mode (OT1) `<`, `>` and `|` render as inverted punctuation / dashes: wrap them
# in math outside `$...$`.
_TEXT_TO_MATH = {"<": "$<$", ">": "$>$", "|": "$|$"}
_UNICODE_TO_TEX = {
    "χ²": r"$\chi^2$", "χ": r"$\chi$", "σ": r"$\sigma$", "Ω": r"$\Omega$", "→": r"$\rightarrow$",
    "≥": r"$\geq$", "≤": r"$\leq$", "≈": r"$\approx$", "±": r"$\pm$", "×": r"$\times$",
    "—": "--", "–": "--", "…": r"\ldots", "ℓ": r"$\ell$", "µ": r"$\mu$", "°": r"$^\circ$",
    "∈": r"$\in$", "′": r"$'$", "″": r"$''$",
}


def tex(s: str) -> str:
    """Make a data-derived string safe for LaTeX text rendering: escape specials OUTSIDE math
    mode and convert the unicode symbols the figure titles use. Idempotent on already-escaped
    input; leaves ``$...$`` untouched."""
    for u, t in _UNICODE_TO_TEX.items():
        s = s.replace(u, t)
    parts = re.split(r"(\$[^$]*\$)", s)
    out = []
    for i, p in enumerate(parts):
        if i % 2 == 1:
            out.append(p)
            continue
        for ch, rep in _TEX_SPECIALS.items():
            p = re.sub(r"(?<!\\)" + re.escape(ch), lambda _m, r=rep: r, p)
        for ch, rep in _TEXT_TO_MATH.items():
            p = p.replace(ch, rep)
        out.append(p)
    return "".join(out)


def probe_tex() -> bool:
    """Render one string with usetex to fail fast when TeX is broken (returns False instead of
    raising a late draw-time error)."""
    try:
        with context():
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.text(0.5, 0.5, tex(r"probe $\Omega_\mathrm{m}$ 68% χ² nla_m"))
            fig.canvas.draw()
            plt.close(fig)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------------------------
# Labels (paper conventions: `$w_0$` -> `$w$`, \mathrm subscripts)
# --------------------------------------------------------------------------------------------
LABELS = {
    "omega_m": r"$\Omega_\mathrm{m}$", "sigma_8": r"$\sigma_8$", "S8": r"$S_8$", "s8": r"$S_8$",
    "w0": r"$w$", "mnu": r"$m_\nu$", "h": r"$h$", "ns": r"$n_\mathrm{s}$",
    "ombh2": r"$\Omega_\mathrm{b} h^2$", "a_ia": r"$A_\mathrm{IA}$", "b_ia": r"$\beta_\mathrm{IA}$",
    "b_z": r"$\beta_z$", "log10_m_eff": r"$\log_{10} M_\mathrm{eff}$",
}
LABELS.update({f"b_g_bin{i}": r"$b_{g,%d}$" % i for i in range(1, 7)})
ARM_NAMES = {
    "nla_m": r"NLA-M (BGP)", "nla_m_nobgp": r"NLA-M (no BGP)", "nla": r"NLA", "nla_z": r"NLA-$z$",
    "vd": r"variable depth", "k2": r"$\kappa = 2$",
}
LABEL_NAMES = {"T": r"T (GLASS $b_g{=}1$ catalogue: known-OOD control)",
               "S": r"S (held-out $N$-body mock: in-distribution control)"}


def label(param: str) -> str:
    return LABELS.get(param, tex(param))


def arm_name(arm: str) -> str:
    return ARM_NAMES.get(arm, tex(arm))


# --------------------------------------------------------------------------------------------
# Saving (paper: svg 300 dpi + png 200 dpi, tight bbox)
# --------------------------------------------------------------------------------------------
def save(fig, path: str | os.PathLike, formats: Sequence[str] = ("png", "pdf"),
         transparent: bool = False, close: bool = True) -> list[str]:
    """Save `fig` as every format in `formats` next to `path` (extension replaced). PNG at
    200 dpi, vector formats at 300 dpi, tight bbox -- the paper's settings."""
    base, _ = os.path.splitext(str(path))
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    written = []
    # Render INSIDE the style context. Text objects remember `usetex` from creation, but the
    # LaTeX font family (\rmfamily vs \sffamily) is read from rcParams at DRAW time -- a
    # savefig after the `with context():` block exits silently sets every word in CM Sans.
    with plt.style.context(styles()), plt.rc_context({"font.family": "serif"}):
        for fmt in formats:
            p = f"{base}.{fmt}"
            fig.savefig(p, dpi=PNG_DPI if fmt == "png" else VECTOR_DPI, bbox_inches="tight",
                        transparent=transparent)
            written.append(p)
    if close:
        plt.close(fig)
    return written


def panel_label(ax, text: str, x: float = 0.02, y: float = 0.96, **kw):
    """'(a)'-style panel letter, top-left inside the axes (the user's reference layout)."""
    kw = {"ha": "left", "va": "top", "fontsize": plt.rcParams["font.size"], **kw}
    return ax.text(x, y, text, transform=ax.transAxes, **kw)


def legend(ax, **kw):
    """Frameless legend (science style default) with the paper's small font unless overridden."""
    kw = {"frameon": False, **kw}
    return ax.legend(**kw)


# --------------------------------------------------------------------------------------------
# Corner plots -- `plot_chains`, from chain_common.py (cells 3/5/6/11), truth made optional
# --------------------------------------------------------------------------------------------
CORNER_PLOTCONFIG = dict(flip=False, tick_font_size=16, label_font_size=16,
                         contour_label_font_size=20, serif=True, usetex=True, max_ticks=2)
CORNER_SMOOTH = 10               # cell 11: the 4-param cosmology corner
SHADE_GRADIENT = 0.9             # fig3_cosmo fill contrast


def chain_kwargs(role: str, colour: str | None = None, name: str = DEFAULT_PALETTE, **over) -> dict:
    """ChainConsumer `Chain` kwargs per role, all from the paper's cells:
      flagship / arm : filled, ``shade_gradient=0.9`` (cell 11)
      real           : black outline, thick, no fill (the observation)
      mock           : thin light-grey outline, no fill (an ensemble member)
      outline        : coloured outline, no fill (cell 8: the 10-param corner)
    """
    base = {
        "flagship": dict(shade=True, shade_gradient=SHADE_GRADIENT, linewidth=1.6),
        "arm": dict(shade=True, shade_gradient=SHADE_GRADIENT, linewidth=1.4),
        "real": dict(shade=False, linewidth=2.4, color=role_colour("real", name)),
        "mock": dict(shade=False, linewidth=0.9, color=role_colour("mock", name)),
        "outline": dict(shade=False, linewidth=1.6),
    }[role]
    if colour is not None:
        base["color"] = colour
    base.update(over)
    return base


def _prune_edge_ticks(fig, nbins: int = 3) -> None:
    """Drop the first and last major tick on every axis (hand-set `extents` put a round tick a
    hair inside a limit and the rotated label overprints its neighbour's). Cosmetic only."""
    for ax in fig.get_axes():
        for axis in (ax.xaxis, ax.yaxis):
            if isinstance(axis.get_major_locator(), MaxNLocator):
                axis.set_major_locator(MaxNLocator(nbins=nbins, prune="both"))


def plot_chains(samples_dict: Mapping[str, np.ndarray], columns, truth: Mapping | None = None,
                colors: Mapping[str, str] | None = None, plotting_kwargs: Mapping[str, dict] | None = None,
                figsize=FIGSIZE_CORNER, extents: Mapping | None = None, savefig=None,
                smooth: int | None = CORNER_SMOOTH, prune_ticks: bool = True,
                legend_loc: str = "upper left", usetex: bool = True, palette_name: str = DEFAULT_PALETTE,
                formats: Sequence[str] = ("png", "pdf"), close: bool = True):
    """The paper's corner plot. `samples_dict` = {name: (n_draws, n_params) array}; `columns` a
    list of TeX labels or {name: list}. Chains are drawn in insertion order (later on top).
    `truth` is optional (None for anything from the blind store). Summaries are never drawn."""
    import logging

    import pandas as pd
    from chainconsumer import Chain, ChainConfig, ChainConsumer, PlotConfig, Truth

    # ChainConsumer logs "Parameter X in chain Y is not constrained" while drawing: a coarse width
    # statement about the chain. Silence it (stdout audit rule for the blind analysis).
    logging.getLogger("chainconsumer").setLevel(logging.ERROR)
    c = ChainConsumer()
    colors = colors or {}
    plotting_kwargs = plotting_kwargs or {}
    extents = extents or {}
    cycle = palette(palette_name)

    for zorder, (name, samples) in enumerate(samples_dict.items()):
        cols = columns.get(name, columns[next(iter(columns))]) if isinstance(columns, dict) else columns
        df = pd.DataFrame(np.asarray(samples), columns=cols)
        kw = dict(plotting_kwargs.get(name, {}))
        if name in colors:
            kw["color"] = colors[name]
        kw.setdefault("color", cycle[zorder % len(cycle)])
        c.add_chain(Chain(samples=df, parameters=list(cols), name=tex(str(name)), zorder=zorder, **kw))

    if smooth is not None:
        c.set_override(ChainConfig(smooth=smooth))
    c.set_plot_config(PlotConfig(**{**CORNER_PLOTCONFIG, "usetex": usetex},
                                 legend_kwargs={"loc": legend_loc}, extents=dict(extents),
                                 summarise=False))
    if truth is not None:
        c.add_truth(Truth(location=dict(truth), color="black", label="Truth"))

    with context(palette_name, usetex=usetex):
        fig = c.plotter.plot(figsize=figsize)
        if prune_ticks:
            _prune_edge_ticks(fig)
        if savefig is not None:
            save(fig, savefig, formats=formats, close=False)
    if close and savefig is not None:
        plt.close(fig)
    return fig


def standardised_extents(columns: Iterable[str], half_width: float = 4.0) -> dict:
    """Fixed +-4 windows for standardised (unit-normal) posteriors so panels align across the
    whole figure set (Tier 3 plots A/B/B'/D)."""
    return {c: [-half_width, half_width] for c in columns}


__all__ = [
    "AA_COLUMN_IN", "AA_TEXT_IN", "ARM_NAMES", "ARM_ORDER", "DEFAULT_PALETTE", "DIVERGING_CMAP",
    "FIGSIZE_CORNER", "FIGSIZE_CORNER_FULL", "FIGSIZE_LINE", "FIGSIZE_LINE_WIDE",
    "FIGSIZE_TWO_PANEL", "LABELS", "LABEL_NAMES", "NOTEBOOK_FONT_SIZE", "PALETTES", "ROLES",
    "SEQUENTIAL_CMAP", "arm_colour", "arm_colours", "arm_name", "chain_kwargs", "context",
    "label", "label_colour", "legend", "muted_colors", "palette", "panel_label", "plot_chains",
    "probe_tex", "role_colour", "save", "sequential", "standardised_extents", "styles", "tex",
    "use_scienceplots",
]
