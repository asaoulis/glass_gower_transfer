"""The production NLE arm table: arm name -> (experiment-name template, bake arm, repeats).

One copy, imported by everything that resolves `--pool-arm nla_m --repeat-indices 0 1 2` into
experiment directories. `bake` is the baked observation store suffix each arm reads.
DELAYED_ARMS lists arms registered ahead of their Stage B, and ACTIVE_ARMS is the rest.
"""
from __future__ import annotations

# arm -> (experiment name template with {r}, bake-arm suffix, repeats that exist)
# New arms go at the end: sample_observation derives per-arm seeds from list(ARMS).index(arm).
ARMS = {
    "nla_m":       ("gower_nle_finetune_nla_m_vdq_z8_r{r}_ens9",              "sc8a1", (0, 1, 2, 3, 4)),
    "nla_m_nobgp": ("gower_nle_finetune_nla_m_z8_r{r}_ens9",                  "a0_tagged", (0, 1, 2, 3, 4)),
    "nla":         ("gower_nle_finetune_nla_vdq_z8_k6_r{r}_ens9_e150",        "sc8a1", (0, 1, 2, 4, 5)),
    "nla_z":       ("gower_nle_finetune_nla_z_vdq_z8_k6_r{r}_ens9_e150",      "sc8a1", (0, 1, 2, 3, 4)),
    "vd":          ("gower_nle_finetune_nla_m_vd_bgp_z8_hf_r{r}_ens9_e150",   "sc8a1", (0, 1, 2, 3, 4)),
    "k2":          ("gower_nle_finetune_nla_m_vdqk2_z8_cosmo_k6_r{r}_ens9_e150", "sc8a1", (0, 1, 2, 3, 4)),
    "band":        ("gower_nle_finetune_band_nla_m_vdq_k8_r{r}_ens9_e150",      "sc8a1", (0, 1, 2, 3, 4)),
    "k2_nla":      ("gower_nle_finetune_nla_vdqk2_z8_cosmo_k6_r{r}_ens9_e150",    "sc8a1", (0, 1, 2, 4, 5)),
    "band_nla":    ("gower_nle_finetune_band_nla_vdq_k8_r{r}_ens9_e150",          "sc8a1", (0, 1, 2)),
    "band_nla_z":  ("gower_nle_finetune_band_nla_z_vdq_k6_r{r}_ens9_e150",        "sc8a1", (0, 1, 2, 3, 4)),
    # Bake `sc8a1_hos` is the HOS sidecars of the sc8a1 observation store (nle_external.store_glob).
    "hos":         ("gower_nle_finetune_hos_nla_m_vdq_k8_r{r}_ens9_e150_xo",      "sc8a1_hos", (0, 1, 2, 3, 4)),
    "hos_gmrg":    ("gower_nle_finetune_hosx_gmrg_nla_m_vdq_k8_r{r}_ens9_e150_xo", "sc8a1_hosx", (0, 1, 2, 3, 4)),
    # vd2pr is the VD v2 (--vd-table v2pr) variate. vdqctl is its matched quadcap control.
    "vd2pr":       ("gower_nle_finetune_nla_m_vd2pr_z8_k6_r{r}_ens9_e150",        "sc8a1", (0, 1, 2, 3, 4)),
    "vd2pr_band":  ("gower_nle_finetune_band_nla_m_vd2pr_k6_r{r}_ens9_e150",      "sc8a1", (0, 1, 2, 3, 4)),
    "vdqctl":      ("gower_nle_finetune_nla_m_vdqctl_z8_k6_r{r}_ens9_e150",       "sc8a1", (0, 1, 2, 3, 4)),
    "vdqctl_band": ("gower_nle_finetune_band_nla_m_vdqctl_k6_r{r}_ens9_e150",     "sc8a1", (0, 1, 2, 3, 4)),
    # Stage B r0 on the partial Gower VD2PR store. Never pool with `vd2pr`.
    "vd2pr_p10k":      ("gower_nle_finetune_nla_m_vd2pr_p10k_z8_k6_r{r}_ens9_e150",   "sc8a1", (0,)),
    "vd2pr_p10k_band": ("gower_nle_finetune_band_nla_m_vd2pr_p10k_k6_r{r}_ens9_e150", "sc8a1", (0,)),
}

# Per-(arm, repeat) experiment names that do not follow the arm's template. Resolve names with
# arm_experiment, never by formatting the template for these repeats.
EXPERIMENT_OVERRIDES = {
    ("nla_z", 3): "gower_nle_finetune_nla_z_vdq_z8_k6_r3b_ens9_e150",
    ("nla", 4): "gower_nle_finetune_nla_vdq_z8_rr_k6_r4_ens9_e150",
    ("nla", 5): "gower_nle_finetune_nla_vdq_z8_rr_k6_r5_ens9_e150",
    ("band_nla_z", 1): "gower_nle_finetune_band_nla_z_vdq_rr_k6_r1_ens9_e150",
    ("band_nla_z", 2): "gower_nle_finetune_band_nla_z_vdq_rr_k6_r2_ens9_e150",
}

DELAYED_ARMS = frozenset({"nla_m_nobgp", "vd",
                          # Registered ahead of their Stage B. Drop each once live.
                          "vd2pr", "vd2pr_band",
                          "vd2pr_p10k"})
ACTIVE_ARMS = tuple(a for a in ARMS if a not in DELAYED_ARMS)

DEFAULT_PRIORS = ("kids_s8_analytic", "LCDM_fixed_w0")


def assert_active(arm):
    """Raise for an arm that is registered but in DELAYED_ARMS."""
    if arm in DELAYED_ARMS:
        raise ValueError("arm %r is DELAYED: its row still names the no-VD chain %r. Repoint it at its "
                         "VDQ chain and drop it from DELAYED_ARMS in src/ml/eval/arms.py before "
                         "sampling or pooling it." % (arm, ARMS[arm][0]))


def assert_matches_sample_observation(path=None):
    """Raise if this table and the copy in scripts/sample_observation.py disagree.

    The script is parsed with ast, not imported. Returns the list of differences (empty when in
    lockstep) and raises only on a missing or unparseable table.
    """
    import ast
    import os

    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "..", "..", "scripts", "sample_observation.py")
    path = os.path.normpath(path)
    with open(path) as fh:
        tree = ast.parse(fh.read())
    other = None
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "src.ml.eval.arms" and any(
                a.name == "ARMS" for a in node.names):
            return []          # the script imports THIS table: lockstep by construction
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "ARMS" for t in node.targets):
            other = ast.literal_eval(node.value)
            break
    if other is None:
        raise ValueError("no ARMS table found in %s" % path)

    diffs = []
    for arm in sorted(set(ARMS) | set(other)):
        a, b = ARMS.get(arm), other.get(arm)
        if a is None or b is None:
            diffs.append("%s: only in %s" % (arm, "arms.py" if b is None else "sample_observation.py"))
        elif tuple(a) != tuple(b):
            diffs.append("%s: arms.py=%s vs sample_observation.py=%s" % (arm, tuple(a), tuple(b)))
    return diffs


def arm_experiments(arm, repeats=None):
    """-> [(repeat, experiment_name)] for `arm`, restricted to repeats that exist.

    Raises for a requested repeat the arm does not have.
    """
    if arm not in ARMS:
        raise KeyError("unknown arm %r; known: %s" % (arm, sorted(ARMS)))
    assert_active(arm)
    template, _bake, available = ARMS[arm]
    want = tuple(available) if repeats is None else tuple(int(r) for r in repeats)
    missing = [r for r in want if r not in available]
    if missing:
        raise ValueError("arm %r has no repeat(s) %s (it has %s)" % (arm, missing, list(available)))
    return [(r, arm_experiment(arm, r)) for r in want]


def arm_experiment(arm, repeat):
    """The experiment name of one (arm, repeat): the arm's template, unless EXPERIMENT_OVERRIDES says
    otherwise. Every consumer must use this instead of formatting `ARMS[arm][0]` itself."""
    if arm not in ARMS:
        raise KeyError("unknown arm %r; known: %s" % (arm, sorted(ARMS)))
    r = int(repeat)
    return EXPERIMENT_OVERRIDES.get((arm, r), ARMS[arm][0].format(r=r))


# The eval match_string cannot be built from the repeat index for the nla_m-era chains. It is
# looked up per arm.
MATCH_TEMPLATES = {
    "nla_m": "ncosmo300_{r}",          # VDQ row also max_trainval_cosmos=[300] -> ncosmo300
    "nla_m_nobgp": "ncosmo300_{r}",
    "nla": "ncosmoNone_{r}",
    "nla_z": "ncosmoNone_{r}",
    "vd": "ncosmoNone_{r}",
    "k2": "ncosmoNone_{r}",
    "k2_nla": "ncosmoNone_{r}",
    # max_trainval_cosmos=[300], so nle_external.match_string_for derives ncosmo300_{r}.
    "band": "ncosmo300_{r}",
    "band_nla": "ncosmoNone_{r}",      # max_trainval_cosmos=None (the 200-id gower_vdq_nla store)
    "band_nla_z": "ncosmoNone_{r}",    # max_trainval_cosmos=None (the 200-id gower_vdq_nla_z store)
    "hos": "ncosmo300_{r}",            # the band M16 split (max_trainval_cosmos=[300]); verified vs match_string_for
    "hos_gmrg": "ncosmo300_{r}",       # same Stage-B split as hos
    "vd2pr": "ncosmoNone_{r}",         # max_trainval_cosmos=None (the 200-id gower_vd2pr_nla_m store, 100-id lock)
    "vd2pr_band": "ncosmoNone_{r}",
    "vdqctl": "ncosmoNone_{r}",        # same split recipe on the 200-id subset of the main Gower VDQ bake
    "vdqctl_band": "ncosmoNone_{r}",
    "vd2pr_p10k": "ncosmoNone_{r}",    # same split recipe as vd2pr on the partial `_p10k` Gower bake
    "vd2pr_p10k_band": "ncosmoNone_{r}",
}
assert set(MATCH_TEMPLATES) == set(ARMS), "MATCH_TEMPLATES and ARMS must cover the same arms"
assert all(a in ARMS and r in ARMS[a][2] for a, r in EXPERIMENT_OVERRIDES), "override for an unknown (arm, repeat)"


def arm_match_string(arm, repeat):
    """The eval `match_string` for one (arm, repeat)."""
    if arm not in MATCH_TEMPLATES:
        raise KeyError("unknown arm %r; known: %s" % (arm, sorted(ARMS)))
    return MATCH_TEMPLATES[arm].format(r=int(repeat))
