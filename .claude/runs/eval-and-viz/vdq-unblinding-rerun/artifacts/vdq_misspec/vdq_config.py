"""ONE source of truth for the VDQ model-misspecification tables + paper figures (AMENDMENT 2026-10-19).

Pack dirs, repeats, match templates, the variate list, display names and colours. Every script in
this folder imports from here instead of re-declaring a dict (the production-model-misspec task
carried five divergent copies of COLORS).

Colours: the four physics variates reuse the project arm colours (`src.viz.style.arm_colour`, so a
variate has the same colour here as its arm has in the Tier-3/4 figures): nla -> `nla`, nla_z ->
`nla_z`, gbk2 -> `k2`, vdqk2_nla -> `k2_nla`. The two fixed-b_g slices get two fixed hexes chosen
to be distinct from every tol-muted cycle entry. The in-distribution reference is black (lines /
markers); the real observation is drawn as a black DASHED vertical line, last.
"""
from __future__ import annotations

import os
import sys

REPO = "/home/alex/work/glass_gower_transfer"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

CKPT = "/data/alex/ml-checkpoints"
R = "/data/alex/unblinding/real"
OBS_DIR = f"{R}/inputs/cluster_fetch/obs_kids_A2_sc8a1"
OUT_TIER2 = f"{R}/work/tier2"                        # <pack>_vdq_full/, <pack>_vdq_phase6/
OUT_MISSPEC = f"{R}/work/misspec_vdq"                # figures/, TABLES.md, numbers.{json,tex}
FIG_DIR = f"{OUT_MISSPEC}/figures"
ROWS_DIR = f"{R}/inputs/tier2_rows"

IN_DIST = "gower_vdq_nla_m"
# order = reference, source clustering (b_g), IA, double misspec
VARIATES = ["gower_vdq_nla_m", "gower_vdq_gb1p0", "gower_vdq_gb1p3", "gower_vdq_gbk2",
            "gower_vdq_nla", "gower_vdq_nla_z", "gower_vdqk2_nla"]
OOD = [v for v in VARIATES if v != IN_DIST]
EXCLUDE = {"gower_vdq_nla": ["a_ia"], "gower_vdq_nla_z": ["a_ia"], "gower_vdqk2_nla": ["a_ia"]}
NLA_FAMILY = ("gower_vdq_nla", "gower_vdq_nla_z", "gower_vdqk2_nla")     # a_ia = A_IA^total
CONTROL = "gower_vdq_nla"
HARD = ["gower_vdq_gb1p0", "gower_vdq_gb1p3", "gower_vdq_gbk2"]
GB_PAIR = ("gower_vdq_gb1p0", "gower_vdq_gb1p3")
DOUBLE = "gower_vdqk2_nla"           # kappa=2 + NLA: own row; pooled table with AND without (D2)

PACKS = {
    "field": {
        "exp": "gower_npe_finetune_nla_m_vdq_z8_ens1",
        "repeats": [0, 2, 3, 4, 5],
        "match": "ncosmo300_{r}",               # misspec moments/samples per repeat
        "summ_match": "ncosmo300_{r}",          # summaries (ens1: bare file)
        "kl": "om_s8_fullcov",                  # adopted Tier-2 KL (tier2.ADOPTED_KL_BY_PACK)
        "obs_label": "A2vdq",
        "title": "field level",
    },
    "band": {
        "exp": "gower_npe_finetune_band_nla_m_vdq_ens9",
        "repeats": [0, 1, 2, 3, 4],
        "match": "ncosmo300_{r}",
        "summ_match": "ncosmo300_{r}_m0",       # ens9: member-0 representative (DECISIONS X-1)
        "kl": "full",
        "obs_label": "A2vdqband",
        "title": "2-pt (bandpowers)",
    },
}
for _k, _p in PACKS.items():
    _p["key"] = _k
    _p["base"] = f"{CKPT}/{_p['exp']}"
    _p["misspec"] = f"{_p['base']}/misspec"
    _p["summaries"] = f"{_p['base']}/summaries"
    _p["obs_json"] = f"{OBS_DIR}/obs_score_{_p['obs_label']}.json"
    _p["matches"] = [_p["match"].format(r=r) for r in _p["repeats"]]
    _p["summ_matches"] = [_p["summ_match"].format(r=r) for r in _p["repeats"]]
    _p["tier2_dir"] = f"{OUT_TIER2}/{_k}_vdq_full"
    _p["phase6_dir"] = f"{OUT_TIER2}/{_k}_vdq_phase6"
    _p["rows_cache"] = f"{ROWS_DIR}/{_k}_vdq_rows.npz"
    _p["meanp_npz"] = f"{OUT_TIER2}/{_k}_vdq_meanp_knn/meanp_knn.npz"

NAMES = {
    "gower_vdq_nla_m": r"in-distribution",
    "gower_vdq_gb1p0": r"$b_g = 1.0$",
    "gower_vdq_gb1p3": r"$b_g = 1.3$",
    "gower_vdq_gbk2": r"$b_g$ prior $\kappa = 2$",
    "gower_vdq_nla": r"NLA",
    "gower_vdq_nla_z": r"NLA-$z$",
    "gower_vdqk2_nla": r"$\kappa = 2$ + NLA",
}
SHORT = {"gower_vdq_nla_m": "in-dist", "gower_vdq_gb1p0": "gb1p0", "gower_vdq_gb1p3": "gb1p3",
         "gower_vdq_gbk2": "gbk2", "gower_vdq_nla": "nla", "gower_vdq_nla_z": "nla_z",
         "gower_vdqk2_nla": "k2_nla"}
_ARM_OF = {"gower_vdq_nla": "nla", "gower_vdq_nla_z": "nla_z", "gower_vdq_gbk2": "k2",
           "gower_vdqk2_nla": "k2_nla"}
_FIXED = {"gower_vdq_nla_m": "#000000", "gower_vdq_gb1p0": "#4477AA", "gower_vdq_gb1p3": "#EE7733"}
OBS_COLOUR = "#000000"
OBS_LS = (0, (5, 2.5))               # dashed: distinguishes the observation from the black in-dist marks


def colour(v: str) -> str:
    if v in _FIXED:
        return _FIXED[v]
    from src.viz import style as S
    return S.arm_colour(_ARM_OF[v])


def name(v: str) -> str:
    return NAMES.get(v, v.replace("_", r"\_"))


def is_in_dist(v: str) -> bool:
    return v == IN_DIST


def scaler_box(exp: str):
    """(names, lo, hi) of the pack's min-max box (the npz samples / theta0s live in [0,1])."""
    from src.observation.checks.tier2 import _scaler_box
    return _scaler_box(exp)


def present(pack: dict, kind: str = "misspec") -> list:
    """Variates (in VARIATES order) whose misspec/ or summaries/ dir exists locally."""
    root = pack[kind]
    return [v for v in VARIATES if os.path.isdir(os.path.join(root, v))]


if __name__ == "__main__":
    from src.viz import style as S
    cols = {v: colour(v) for v in VARIATES}
    print(cols)
    assert len(set(cols.values())) == len(VARIATES), "colours not distinct"
    cyc = set(c.upper() for c in S.palette())
    for v in ("gower_vdq_gb1p0", "gower_vdq_gb1p3"):
        assert cols[v].upper() not in cyc, v
    print("probe_tex:", S.probe_tex())
    for k, p in PACKS.items():
        print(k, p["exp"], p["repeats"], "misspec present:", present(p), "summaries present:", present(p, "summaries"))
