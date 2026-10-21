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
    # ⭐ VDQ variates (2026-10-13, user: "run the variate observation long sampling"): repointed from the
    # no-VD `_bgp_z8_hf` chains to the VDQ k6 Stage-B rows (training-runs/VD-final-train-models G-PLAN).
    # nla_z repeat 3 is the `_r3b` row -- see EXPERIMENT_OVERRIDES; never format this template at r=3.
    # nla repeats 4 and 5 are the `_rr` encoder-reroll rows (EXPERIMENT_OVERRIDES); r3 is RETIRED (its split's GLASS
    # ceiling, user 2026-10-14 -> replaced by index 5), so the tuple is deliberately non-contiguous.
    "nla":         ("gower_nle_finetune_nla_vdq_z8_k6_r{r}_ens9_e150",        "sc8a1", (0, 1, 2, 4, 5)),
    "nla_z":       ("gower_nle_finetune_nla_z_vdq_z8_k6_r{r}_ens9_e150",      "sc8a1", (0, 1, 2, 3, 4)),
    "vd":          ("gower_nle_finetune_nla_m_vd_bgp_z8_hf_r{r}_ens9_e150",   "sc8a1", (0, 1, 2, 3, 4)),
    # ⭐ kappa=2 (2026-10-17, user "final model rows"): repointed from the no-VD `_bgpk2_z16_k5_hf` chain to the VDQ
    # COSMOLOGY-ONLY chain (b_g marginalised by the kappa=2 sims; training-runs/VD-final-train-models AMENDMENT 14).
    "k2":          ("gower_nle_finetune_nla_m_vdqk2_z8_cosmo_k6_r{r}_ens9_e150", "sc8a1", (0, 1, 2, 3, 4)),
    # The 2-pt-ONLY (bandpower) NLE chain (M16 procedure, VDQ stores). Same flagship Gower store,
    # same 200-id test lock, so "what do the maps buy over the 2-point function?" is single-variable.
    # `arm_experiments` RAISES on a requested-but-missing repeat rather than silently dropping it.
    "band":        ("gower_nle_finetune_band_nla_m_vdq_k8_r{r}_ens9_e150",      "sc8a1", (0, 1, 2, 3, 4)),
    # kappa=2 + NLA IA (VDQ, cosmology-only, b_g marginalised). APPENDED LAST on purpose: sample_observation derives
    # per-arm seeds from list(ARMS).index(arm), so inserting earlier would shift every later arm's seeds.
    "k2_nla":      ("gower_nle_finetune_nla_vdqk2_z8_cosmo_k6_r{r}_ens9_e150",    "sc8a1", (0, 1, 2, 4, 5)),
    # 2-pt (bandpower) NLE, NLA IA (free a_ia ~ U[-6,6]), VDQ stores (2026-10-17, unblinding-investigation-2 T3b).
    # APPENDED LAST for the same seed-stability reason as k2_nla. Stage B uses the field NLA variate split (the
    # gower_vdq_nla store is the 200-id fixed set: all trainval 80/20 + 100-id lock) -> ncosmoNone_{r}.
    "band_nla":    ("gower_nle_finetune_band_nla_vdq_k8_r{r}_ens9_e150",          "sc8a1", (0, 1, 2)),
    # 2-pt (bandpower) NLE, NLA-z IA (a_ia ~ U[-6,6] + b_z), VDQ stores (2026-10-20, NLA-z PPC counterpart of band_nla).
    # APPENDED LAST (seed stability). Same 200-id Gower store split as nla_z -> ncosmoNone_{r}.
    "band_nla_z":  ("gower_nle_finetune_band_nla_z_vdq_k8_r{r}_ens9_e150",        "sc8a1", (0, 1, 2)),
    # Beyond-2-pt HOS hybrid NLE (2026-10-21, eval-and-viz/hos-hybrid-training): frozen band + R1 third-moment MLP,
    # the band P3b chain on the HOS sidecar stores. APPENDED LAST (seed stability). Bake `sc8a1_hos` = the HOS sidecars
    # of the sc8a1 observation store (nle_external.store_glob resolves the `_hos` suffix).
    "hos":         ("gower_nle_finetune_hos_nla_m_vdq_k8_r{r}_ens9_e150",         "sc8a1_hos", (0, 1, 2, 3, 4)),
}

# Per-(arm, repeat) experiment names that do NOT follow the arm's template. nla_z repeat 3 was re-run
# as `_r3b` on the re-rolled r3 encoder (same repeat index / split seed; the superseded `_r3` Stage A
# was fit on the old encoder and must never be scored). ALWAYS resolve names via `arm_experiment`.
EXPERIMENT_OVERRIDES = {
    ("nla_z", 3): "gower_nle_finetune_nla_z_vdq_z8_k6_r3b_ens9_e150",
    # nla encoder reroll (training-runs/VD-final-train-models AMENDMENT 10, user 2026-10-14): r4 and the new
    # index 5 run on `glass_encoder_finetune_nla_vdq_z8_rr` (own frames/scalers/whitener). Stage-B gates:
    # r4 passed guard-c 50; r5 tripped 50 (gap 52.66) and passed under the user-raised 150 (2026-10-15).
    # Bundled evals in family: FoM(Om,s8) 6.79 / 7.05, S8 w68 0.060 / 0.057, TARP .025 / .026.
    ("nla", 4): "gower_nle_finetune_nla_vdq_z8_rr_k6_r4_ens9_e150",
    ("nla", 5): "gower_nle_finetune_nla_vdq_z8_rr_k6_r5_ens9_e150",
}

# Variate arms whose rows still name no-VD chains (see the VDQ CUTOVER note above).
# k2 went live 2026-10-17 00:45Z; k2_nla went live 2026-10-17 once its Gower Stage B r0-4 existed (r3 under the
# user-raised guard-c 800, rev 7a26762). k2_nla r3 RETIRED -> r5 (2026-10-17): r3 per-member test NLL 3.9-7.3 on the shared
# 100-id lock vs siblings 0.15-2.26 (out of family); r5 (new split, guard-c 150 for an ep0 gap of 60.25) = 1.98-2.45 (in family).
# band_nla went live 2026-10-19 once its Gower Stage B r0-2 existed (r0/r2 under the user-rule guard-c 150, rev 96bffe9;
# member frames recovered per repeat before external scoring).
# band_nla_z went live 2026-10-21 once its Gower Stage B r0-2 existed and were evaluated (task eval-and-viz/band-nla-z-2pt-nle;
# all three repeats under the user-rule guard-c 150 -- ep0 gaps 50.55 / 95.59 / 54.08 nats -- rev 3b04364/e21c28a; per-member test
# -log p(z|theta) -4.9..-8.1 on the 100-id lock; TARP 0.041-0.050; member frames must be recovered per repeat before external scoring).
DELAYED_ARMS = frozenset({"nla_m_nobgp", "vd"})
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
    return [(r, arm_experiment(arm, r)) for r in want]


def arm_experiment(arm, repeat):
    """The experiment name of one (arm, repeat): the arm's template, unless EXPERIMENT_OVERRIDES says
    otherwise. Every consumer must use this instead of formatting `ARMS[arm][0]` itself."""
    if arm not in ARMS:
        raise KeyError("unknown arm %r; known: %s" % (arm, sorted(ARMS)))
    r = int(repeat)
    return EXPERIMENT_OVERRIDES.get((arm, r), ARMS[arm][0].format(r=r))


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
    "k2_nla": "ncosmoNone_{r}",
    # M16 carries max_trainval_cosmos=[300] (length 1), so `nle_external.match_string_for` derives
    # `ncosmo300_{r}` -- VERIFIED against that resolver, not inferred from the experiment name.
    "band": "ncosmo300_{r}",
    "band_nla": "ncosmoNone_{r}",      # max_trainval_cosmos=None (the 200-id gower_vdq_nla store)
    "band_nla_z": "ncosmoNone_{r}",    # max_trainval_cosmos=None (the 200-id gower_vdq_nla_z store)
    "hos": "ncosmo300_{r}",            # the band M16 split (max_trainval_cosmos=[300]); verified vs match_string_for
}
assert set(MATCH_TEMPLATES) == set(ARMS), "MATCH_TEMPLATES and ARMS must cover the same arms"
assert all(a in ARMS and r in ARMS[a][2] for a, r in EXPERIMENT_OVERRIDES), "override for an unknown (arm, repeat)"


def arm_match_string(arm, repeat):
    """The eval `match_string` for one (arm, repeat)."""
    if arm not in MATCH_TEMPLATES:
        raise KeyError("unknown arm %r; known: %s" % (arm, sorted(ARMS)))
    return MATCH_TEMPLATES[arm].format(r=int(repeat))
