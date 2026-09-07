"""Shared naming/palette/order for the Gower BGP misspecification figures.

ONE definition of colour + label + order for every figure in this task, so the three
"money plots" stay visually comparable across their three x-axis versions (cross-repeat
KL, single-encoder OOD, multi-encoder OOD). Import this rather than re-declaring a
COLORS dict per script — the previous campaign carried five divergent copies.

Variate names are the keys of `GOWER_BGP_VARIATES` in src/ml/eval/misspec.py, i.e. the
directory names under checkpoints/<exp>/misspec/.
"""

# In-distribution reference (the physics the encoders were trained on). Kept as a tuple
# because the plotting scripts test membership against both GLASS and Gower reference names.
IN_DIST = ("gower_bgp_nla_m", "nla_m", "glass_nla_m", "glass_bgp_sc8a1", "glass_dn_sc8a1")

# Palette. Qualitative, colour-blind safe (Okabe-Ito derived), and grouped by MECHANISM so
# the reader can see the two families at a glance:
#   - intrinsic alignment (IA) mis-modelling  -> warm  (orange / red)
#   - source-clustering (galaxy bias b_g)     -> cool  (blue / purple)
#   - survey depth                            -> green
# The in-distribution reference is black in every figure.
COLORS = {
    "gower_bgp_nla_m": "black",
    "gower_nla":       "#E69F00",   # orange
    "gower_nla_z":     "#D55E00",   # vermillion
    "gower_vd":        "#009E73",   # green
    "gower_gb1p0":     "#0072B2",   # blue
    "gower_gb1p3":     "#CC79A7",   # purple
    # GLASS-side names, so the same module drives the validation figures built on the
    # local sc8a1 ladder (identical schema, different variate names).
    "glass_bgp_sc8a1": "black",
    "glass_gb0p7":     "#CC79A7",
    "glass_gb1p0":     "#0072B2",
    "glass_gb1p5":     "#56B4E9",
}

# Paper labels: minimal, no underscores, maths where it is a symbol. No "(in-dist)" noise
# on the reference beyond the one word that identifies it.
LABELS = {
    "gower_bgp_nla_m": "in-distribution",
    "gower_nla":       "NLA",
    "gower_nla_z":     "NLA-$z$",
    "gower_vd":        "variable depth",
    "gower_gb1p0":     r"$b_g = 1.0$ fixed",
    "gower_gb1p3":     r"$b_g = 1.3$ fixed",
    "glass_bgp_sc8a1": "in-distribution",
    "glass_gb0p7":     r"$b_g = 0.7$",
    "glass_gb1p0":     r"$b_g = 1.0$",
    "glass_gb1p5":     r"$b_g = 1.5$",
}

# Legend/plot order: reference first, then source clustering, then depth, then IA. This is
# roughly increasing detectability, which makes the ladder read left-to-right in the bars.
ORDER = [
    "gower_bgp_nla_m",
    "gower_gb1p0",
    "gower_gb1p3",
    "gower_vd",
    "gower_nla",
    "gower_nla_z",
    "glass_bgp_sc8a1", "glass_gb1p0", "glass_gb1p5", "glass_gb0p7",
]

# Variates whose IA parametrisation DIFFERS from the nla_m training suite, so `a_ia` has a
# different meaning and is excluded from calibration (mirrors `exclude_params` in
# src/ml/eval/misspec.py). Figures must not silently compare a_ia across these.
AIA_EXCLUDED = ("gower_nla", "gower_nla_z", "nla", "nla_z", "glass_nla", "glass_nla_z")

# The 9-param nla_m inference vector, in the order the flow samples it.
PARAMS = ["omega_m", "sigma_8", "w0", "mnu", "h", "ns", "ombh2", "a_ia", "b_ia"]
KEY3 = ["omega_m", "sigma_8", "w0"]


def order_variates(names):
    """Sort `names` into ORDER, with any unknown name appended alphabetically at the end."""
    known = [n for n in ORDER if n in set(names)]
    rest = sorted(n for n in names if n not in set(ORDER))
    return known + rest


def color(name):
    return COLORS.get(name, "0.5")


def label(name):
    return LABELS.get(name, name.replace("_", " "))


def is_in_dist(name):
    return name in IN_DIST
