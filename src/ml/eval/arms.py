"""The production NLE arm table: arm name -> (experiment-name template, bake arm, repeats).

ONE copy, imported by everything that needs it. It previously lived in
`scripts/sample_observation.py`; `eval.py --mode pool` needs the same mapping to resolve
`--pool-arm nla_m --repeat-indices 0 1 2` into the three experiment directories, and a second copy
of a table whose repeat tuples differ per arm (kappa=2 has FOUR repeats, everything else five)
would drift the moment one arm gains a seed.

⚠️ These are the `_hf` RETRAIN rows where the `e890aec` encoder-cutover bug applied (see
`.claude/runs/training-runs/production-training-runs/BUG_kappa2_prior.md` §7.16/§7.19). The
same-shaped non-`_hf` names for `nla` / `nla_z` / `vd` / `k2` ran on the wrong source encoder and
must never be scored — do not "simplify" the templates back to them.

`bake` is the baked observation store suffix each arm reads: `nla_m_nobgp` was trained on the
plain-counts product and reads `_a0_tagged`; every other arm reads the smoothed-counts `_sc8a1`.
⚠️ This table is DUPLICATED in `scripts/sample_observation.py` (not yet repointed here). Run
`assert_matches_sample_observation()` before trusting either -- they have already drifted once.
"""
from __future__ import annotations

# arm -> (experiment name template with {r}, bake-arm suffix, repeats that exist)
#
# ⭐ VDQ CUTOVER (2026-10-09, user-approved; eval-and-viz/vdq-unblinding-rerun V6): the two MAIN arms
# `nla_m` (field-level hybrid) and `band` (2-pt) now name the VD-quadcap chains trained on VD mocks.
# Their no-VD predecessors were `gower_nle_finetune_nla_m_bgp_z8_r{r}_ens9` and
# `gower_nle_finetune_band_nla_m_bgp_k8_r{r}_ens9_e150` (repeats 0,2,3,4). The five variate arms below
# still name no-VD chains: they are DELAYED, not removed (user) -- kept so every table keeps its rows,
# but listed in DELAYED_ARMS so nothing can sample/pool them against a VDQ headline until each is
# repointed at its VDQ chain (then drop it from DELAYED_ARMS).
ARMS = {
    "nla_m":       ("gower_nle_finetune_nla_m_vdq_z8_r{r}_ens9",              "sc8a1", (0, 1, 2, 3, 4)),
    "nla_m_nobgp": ("gower_nle_finetune_nla_m_z8_r{r}_ens9",                  "a0_tagged", (0, 1, 2, 3, 4)),
    "nla":         ("gower_nle_finetune_nla_bgp_z8_hf_r{r}_ens9_e150",        "sc8a1", (0, 1, 2, 3, 4)),
    "nla_z":       ("gower_nle_finetune_nla_z_bgp_z8_hf_r{r}_ens9_e150",      "sc8a1", (0, 1, 2, 3, 4)),
    "vd":          ("gower_nle_finetune_nla_m_vd_bgp_z8_hf_r{r}_ens9_e150",   "sc8a1", (0, 1, 2, 3, 4)),
    "k2":          ("gower_nle_finetune_nla_m_bgpk2_z16_k5_hf_r{r}_ens9_e150", "sc8a1", (0, 1, 2, 3)),
    # The 2-pt-ONLY (bandpower) NLE chain (M16 procedure, VDQ stores). Same flagship Gower store,
    # same 200-id test lock, so "what do the maps buy over the 2-point function?" is single-variable.
    # `arm_experiments` RAISES on a requested-but-missing repeat rather than silently dropping it.
    "band":        ("gower_nle_finetune_band_nla_m_vdq_k8_r{r}_ens9_e150",      "sc8a1", (0, 1, 2, 3, 4)),
}

# Variate arms whose rows still name no-VD chains (see the VDQ CUTOVER note above).
DELAYED_ARMS = frozenset({"nla_m_nobgp", "nla", "nla_z", "vd", "k2"})
ACTIVE_ARMS = tuple(a for a in ARMS if a not in DELAYED_ARMS)

DEFAULT_PRIORS = ("kids_s8_analytic", "LCDM_fixed_w0")


def assert_active(arm):
    """Raise for an arm that is registered but DELAYED (its row still names a no-VD chain)."""
    if arm in DELAYED_ARMS:
        raise ValueError("arm %r is DELAYED: its row still names the no-VD chain %r. Repoint it at its "
                         "VDQ chain and drop it from DELAYED_ARMS in src/ml/eval/arms.py before "
                         "sampling or pooling it." % (arm, ARMS[arm][0]))


def assert_matches_sample_observation(path=None):
    """⚠️ ANTI-DRIFT GUARD. Compare this table with the one still in `scripts/sample_observation.py`.

    That script has not been repointed at this module yet (it was being edited concurrently when
    this file was written), so for now TWO copies exist. They drifted within the hour: the bake
    suffix for `nla_m_nobgp` moved `a1` -> `a0_tagged` in the script while this copy still said
    `a1`. A stale bake suffix sends an arm at the WRONG baked observation store, which produces a
    perfectly plausible posterior from the wrong product -- silent, and in the unblinding.

    Parsed with `ast`, not imported: `scripts/` is not a package and the script has side effects.
    Returns the list of differences (empty == in lockstep); raises only on a missing/unparseable
    table, so a caller can decide whether a difference is fatal.
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

    Asking for a repeat an arm does not have is an ERROR, not a silent drop: a pooled posterior
    quietly built from 3 of the 4 requested seeds is a different object from the one requested.
    """
    if arm not in ARMS:
        raise KeyError("unknown arm %r; known: %s" % (arm, sorted(ARMS)))
    assert_active(arm)
    template, _bake, available = ARMS[arm]
    want = tuple(available) if repeats is None else tuple(int(r) for r in repeats)
    missing = [r for r in want if r not in available]
    if missing:
        raise ValueError("arm %r has no repeat(s) %s (it has %s)" % (arm, missing, list(available)))
    return [(r, template.format(r=r)) for r in want]


# ⚠️ The eval `match_string` is NOT constructible from the repeat index: the two `nla_m`-era
# chains carry `ncosmo300_<r>` while every later chain carries `ncosmoNone_<r>`. This is the same
# fact `artifacts/variate_diagnostics/variates.py` records ("match strings are NOT constructed"),
# and it is tabulated rather than pattern-matched off the experiment name for exactly that reason.
MATCH_TEMPLATES = {
    "nla_m": "ncosmo300_{r}",          # VDQ row also max_trainval_cosmos=[300] -> ncosmo300 (verified 2026-10-09)
    "nla_m_nobgp": "ncosmo300_{r}",
    "nla": "ncosmoNone_{r}",
    "nla_z": "ncosmoNone_{r}",
    "vd": "ncosmoNone_{r}",
    "k2": "ncosmoNone_{r}",
    # M16 carries max_trainval_cosmos=[300] (length 1), so `nle_external.match_string_for` derives
    # `ncosmo300_{r}` -- VERIFIED against that resolver, not inferred from the experiment name.
    "band": "ncosmo300_{r}",
}
assert set(MATCH_TEMPLATES) == set(ARMS), "MATCH_TEMPLATES and ARMS must cover the same arms"


def arm_match_string(arm, repeat):
    """The eval `match_string` for one (arm, repeat)."""
    if arm not in MATCH_TEMPLATES:
        raise KeyError("unknown arm %r; known: %s" % (arm, sorted(ARMS)))
    return MATCH_TEMPLATES[arm].format(r=int(repeat))
