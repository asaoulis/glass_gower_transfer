# Figure style guide (`src/viz/style.py`)

One module, one look, for every figure in the KiDS-Legacy multifidelity-SBI project: the
unblinding figure set, the Tier-3 corner plots, the notebook, and the paper battery. Everything
here is the option set the published paper figures were rendered with
(`.claude/runs/paper-review/last-runs-and-plots/artifacts/{plot_common,chain_common}.py`,
`fig3_fig4_posteriors.py`, `ml_results_plotting.ipynb`), promoted into an importable module so
it cannot drift.

```python
from src.viz import style as S

with S.context():                      # science + muted, serif, LaTeX, 18 pt (the paper)
    fig, ax = plt.subplots(figsize=S.FIGSIZE_LINE)
    ax.plot(x, y, color=S.arm_colour("nla_m"), label=S.arm_name("nla_m"))
    ax.set_xlabel(S.tex("High fidelity sims ($N$)"))
    S.panel_label(ax, "(a)"); S.legend(ax)
S.save(fig, "figures/my_figure")       # -> .png (200 dpi) + .pdf (300 dpi), tight bbox
```

## 1. The fixed choices (do not re-decide per figure)

| what | value | origin |
|---|---|---|
| matplotlib style | `plt.style.context(["science", "muted"])` (scienceplots) | notebook cells 6/10, `plot_common.use_scienceplots` |
| text | LaTeX (`text.usetex`), serif (`\rmfamily`), amsmath/amssymb preamble | science style |
| global font size | **18 pt**, set before drawing; figures drawn at ~6 x 4.5 in and scaled to column width by LaTeX (lands at ~9 pt on the page) | notebook cell 0 |
| line plots | `fmt='-o'`, `linewidth=2`, dashed grid alpha 0.6, legend `x-small`, figsize (6, 4.5) or (7, 5) | `plot_results` |
| corner plots | ChainConsumer 1.3; `PlotConfig(flip=False, tick 16, label 16, contour labels 20, max_ticks=2, serif, usetex, summarise=False)`; 4-param cosmology corner `ChainConfig(smooth=10)`, `shade=True, shade_gradient=0.9`, figsize (6, 6), edge ticks pruned; full corner (10, 10), no smoothing, outlines only | `chain_common.plot_chains`, `fig3_fig4_posteriors` |
| ordered classes (tomo bins, seeds) | viridis window `np.linspace(0.12, 0.82, n)` -> `S.sequential(n)` | fidelity / coverage figures |
| diverging maps (z-scores) | `RdBu_r`, symmetric limits | Tier-1 heatmaps |
| labels | `$\Omega_\mathrm{m}$`, `$\sigma_8$`, `$S_8$`, **`$w$` (never `$w_0$`)**, `$A_\mathrm{IA}$`, `$\beta_\mathrm{IA}$`, `$b_{g,i}$` -> `S.label(name)` | `chain_common` |
| output | PNG 200 dpi + PDF/SVG 300 dpi, `bbox_inches="tight"` | `chain_common`, `fig1_fig2_vs_N` |
| colour identity | pinned **by key** (arm, label, role), never consumed from the prop cycle in draw order | `plot_common.EXPERIMENT_COLORS` |

Two size contexts:

* **`context()` (default, 18 pt)** — the paper's convention. Draw at the sizes above; LaTeX scales.
* **`context(font_size=9)` + `figsize=(S.AA_COLUMN_IN, …)`** — draw at final A&A size (88 mm
  column, 180 mm text width) when a figure must not be scaled. Do not mix the two in one figure set.

## 2. Roles (the same in every palette)

| role | meaning | colour |
|---|---|---|
| `real` | the observation / real-data posterior | **black**, thick (lw 2.4) |
| `flagship` | the headline NLA-M (BGP) arm | palette colour, filled contours |
| `mock` | matched-mock / null ensembles | light grey `#B0B0B0`, thin (lw 0.9), no fill |
| `reference` | guide lines (analytic nulls, medians, chance) | dark grey `#444444`, dashed/dotted |
| `stop` | stop-rule regions | palette red at alpha 0.10–0.18 |
| `pass` | "in distribution" / identity gates | palette green |
| `T`, `S` | the two mock controls (T known-OOD, S in-distribution) | palette red / palette blue |
| arms | `nla_m, nla_m_nobgp, nla, nla_z, vd, k2` | six fixed cycle entries (`S.arm_colours()`); a drift check against `src/ml/eval/arms.py:ARMS` |

Corner-plot roles via `S.chain_kwargs(role, colour)`: `flagship`/`arm` filled (`shade_gradient=0.9`),
`real` black outline, `mock` grey outline, `outline` coloured outline (the 10-param corner of cell 8).
Standardised (Tier-3) corners use fixed `extents = ±4 sigma` on every panel so figures align.

## 3. Palette options

All four carry the role table above, so swapping the option changes nothing a figure says.
`00_style_palette_options.png` in the figure set shows them side by side on synthetic data;
`figures/palette_options/<name>/` shows the Tier-1/2 panels in each.

| name | cycle | when |
|---|---|---|
| **`tol-muted` (default)** | `#CC6677 #332288 #DDCC77 #117733 #88CCEE #882255 #44AA99 #999933 #AA4499 #DDDDDD` (Paul Tol muted = scienceplots `muted`) | the paper's cycle (fig 1–4). Choosing anything else means regenerating the paper battery for consistency. |
| `tol-bright` | `#4477AA #EE6677 #228833 #CCBB44 #66CCEE #AA3377 #BBBBBB` | higher chroma, six hues; screens / talks |
| `tol-high-contrast` | `#004488 #DDAA33 #BB5566` + medium-contrast partners `#6699CC #997700 #994455 #EECC66 #EE99AA` | few classes; survives greyscale printing |
| `survey` | `#1F4E79 #C8502B #2A9D8F #E9B44C #6D3F7A #5B6770 #7FB3D5 #B8B8B8` | KiDS/DES-flavoured deep tones (navy/vermilion "data vs model") |

Arm assignment per palette is in `style.ROLES[name]["arms"]`.

## 4. TeX safety (usetex is on)

Every data-derived string (arm keys, dataset names, anything with `_ % & # < > |` or unicode)
goes through `S.tex()`: it escapes specials outside `$…$`, wraps `<`, `>`, `|` in math (text-mode
OT1 renders them as inverted punctuation), and converts `χ² σ Ω → ≥ ≤ ≈ ± × — …` to LaTeX.
`S.probe_tex()` renders one string up front so a broken TeX install fails fast. For an
environment without TeX use `context(usetex=False)` (`no-latex` + STIX math).

**`font.size` leaks.** `use_scienceplots()` sets the paper's 18 pt globally before the style
context snapshots rcParams, so `S.context()` leaves 18 pt behind on exit (the notebooks' behaviour).
Reset it yourself (`plt.rcParams["font.size"] = 11`) in a notebook cell drawn afterwards.

**Save inside the context.** Text objects remember `usetex` from creation, but LaTeX reads the
font *family* from `rcParams` at draw time: a `fig.savefig` after the `with` block exits renders
every word in CM Sans. `S.save()` re-enters the context, so use it (or save inside the block).

## 5. Layout rules (the user's reference look)

* Short axis labels; panel letters `(a)`, `(b)` inside the axes (`S.panel_label`); frameless
  in-panel legends; **no suptitles** — the explanation goes in the caption / README.
* One idea per panel; ≤ 2 panels side by side at 18 pt (each ≥ 6 in wide), otherwise stack.
* Real data on top (drawn last), ensembles underneath.
* Reference lines dashed/dotted in grey, never a second saturated colour.
* Stop-rule regions as light tints, the boundary as a dashed line in the same colour.

## 6. Retrofit recipe for an existing script

1. `from src.viz import style as S`; delete the local `rcParams`/palette blocks.
2. Replace `_style()`/`use_scienceplots()` with `S.styles()` or wrap drawing in `with S.context():`.
3. Replace hex constants with roles: seeds -> `S.sequential(n)`, pooled/accent -> `S.role_colour("flagship")`,
   null floor -> `S.role_colour("reference")`, arms -> `S.arm_colour(arm)`.
4. Route titles/labels through `S.tex`, parameter names through `S.label`.
5. Save with `S.save(fig, path)`.
6. Add `--palette` (choices `list(S.PALETTES)`) if the figure is part of an option comparison.

Demo: `.claude/runs/training-runs/production-training-runs/artifacts/variate_diagnostics/plot_tarp_ensembled_restyled.py`
writes `figures_ensembled_restyled/<palette>/` without touching the originals.

## 7. Blind-analysis rules for plotting code

* Chain names are passed through `S.tex()` inside `plot_chains`; an observation label is never
  coloured by its name (`label_colour` -> black for anything but the mock controls T/S).
* `plot_chains` forces `summarise=False` and silences ChainConsumer's "parameter X is not
  constrained" log; never add `Truth` markers to a chain from the blind store.
* Nothing in `src.viz` prints chain statistics. Keep it that way (stdout audit rule).
