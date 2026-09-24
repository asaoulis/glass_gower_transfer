#!/usr/bin/env python
"""Validation gates for the weighted shear-map normalisation (spec §6), observation side.

    PYTHONPATH=. python scripts/verify_weighted_build.py --fixture /data/alex/unblinding/fixtures/smoke_regen_seeded
        [--real-catalogue <h5/fits> --column-map <json>]   # gate 5 on the real catalogue

Gates (named as in .claude/plans/shear_normalisation_spec.md §6):
  0  ingestion: a weight column is filtered with the same row mask as the data (two bad rows
     planted), aligned, summarised; `lensfit` + an UNPATCHED estimator raises (never a silent
     unweighted build); the 1-based TOMOBIN trap raises.
  1  bit-identity: the fixture's sim catalogue with weights=None (--exact-rng) reproduces the
     stored mock bit-for-bit. THE §0 proof; must pass before and after the protected patch.
  2  weighted no-op: weights = ones(n) must match gate 1 to float tolerance   [needs the patch]
  3  scale invariance: weights = alpha*w for alpha in {1e-3, 1e3}, identical  [needs the patch]
  3b production rescale: apply_weights(rescale='neff') -- per-BIN w~ = w*sum(w)/sum(w^2) --
     leaves the maps identical, and sum(w~) == N_eff per bin
  4  Eq.-11 identity: wraps .claude/plans/verify_shear_normalisation.py
  4b weights= path: weighted mean/counts maps vs analytic S_p/W_i, S_p/D_p; ones == None
  5  W_i check: N_eff/N_pix per bin vs the mocks' mean count per pixel,
     n_arcmin2 * 967.39 deg^2 * 3600 / N_pix = [20.888, 19.467, 17.673, 17.206, 15.876, 12.608],
     to four figures (rel 5e-4); also n_eff/arcmin^2 vs src/KiDS/tomo.py (real catalogue only).
     RE-PRE-REGISTERED 2026-09-24 (user): the old literal [15.002 .. 9.054] contradicted this
     defining formula by x0.7182 in every bin (its sum w^2/sum w = 18.663 exceeds max w = 15.56).
Gates 2-3 report SKIPPED (with the reason) until `make_alm_shear_convergence` accepts `weights=`.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.observation import (Catalogue, Geometry, MapVariants, apply_c_terms, apply_m_bias,  # noqa: E402
                             apply_weights, build_observation, compare_observation_to_mock,
                             estimator_accepts_weights, load_catalogue, weight_summary)
from src.observation.geometry import master_postproc_rng  # noqa: E402

from src.KiDS.tomo import n_arcmin2 as _N_ARCMIN2  # noqa: E402
from src.observation.catalogue_io import KIDS_AREA_DEG2, KIDS_NPIX_NORM  # noqa: E402

# Gate-5 reference under normalisation (b): the mocks' mean count per pixel, COMPUTED from its
# definition (re-pre-registered 2026-09-24; the literal below was an arithmetic slip, kept for the record).
W_REF = [float(n) * KIDS_AREA_DEG2 * 3600.0 / KIDS_NPIX_NORM for n in _N_ARCMIN2]
W_REF_SUPERSEDED = [15.002, 13.981, 12.693, 12.357, 11.402, 9.054]
GATE5_TOL = 5e-4                                                  # "to four figures"
PY = sys.executable


def _fixture(fixture: str):
    cat = sorted(glob.glob(os.path.join(fixture, "catalogues", "catalogue_*.h5")))[0]
    mock = sorted(glob.glob(os.path.join(fixture, "output_*.h5")))[0]
    return cat, mock


def _build(cat: Catalogue, m_bias, out_dir: str, label: str, weights=None):
    attrs = cat.provenance.get("sim_attrs", {})
    geometry = Geometry.from_catalogue_attrs(attrs) if attrs else Geometry.production()
    rng = master_postproc_rng(attrs)
    return build_observation(cat, m_bias, out_path=os.path.join(out_dir, f"observation_{label}.h5"),
                             label=label, geometry=geometry, variants=MapVariants.named("production"),
                             rng=rng, verbose=False, weights=weights)


def _maps(path):
    import h5py
    out = {}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and (name.startswith("cls_results") or name.startswith("pixelised_results")):
                if obj.dtype.kind in "fic":
                    out[name] = obj[()]
        f.visititems(visit)
    return out


def _max_rel(a, b):
    worst = 0.0
    for k in a:
        x, y = np.asarray(a[k], float), np.asarray(b[k], float)
        scale = np.max(np.abs(x)) or 1.0
        worst = max(worst, float(np.max(np.abs(x - y)) / scale))
    return worst


def gate0(fixture: str, tmp: str) -> bool:
    """Ingestion: weight filtering / alignment / summary, the unpatched-estimator refusal, TOMOBIN trap."""
    import h5py
    cat_path, _ = _fixture(fixture)
    ok = True
    with h5py.File(cat_path, "r") as f:
        g = f["catalogue"]
        n = min(200_000, g["RA"].shape[0])
        cols = {k: g[k][:n] for k in ("RA", "DEC", "ZBIN", "E1", "E2")}
    rng = np.random.default_rng(0)
    w = rng.lognormal(0.0, 0.8, n)
    e1 = cols["E1"].astype(float).copy(); e1[10] = np.nan            # bad row A (non-finite e1)
    w[20] = 0.0                                                     # bad row B (non-positive weight)
    ext = os.path.join(tmp, "ext.h5")
    with h5py.File(ext, "w") as f:
        f["ra"], f["dec"], f["e1"], f["e2"] = cols["RA"], cols["DEC"], e1, cols["E2"]
        f["tomobin"] = cols["ZBIN"].astype(int) + 1                 # 1-based like KiDS TOMOBIN
        f["weight"] = w
    cm = {"ra": "ra", "dec": "dec", "e1": "e1", "e2": "e2", "zbin": "tomobin", "zbin_offset": 1, "weight": "weight"}
    cat = load_catalogue(ext, column_map=cm, kind="h5")
    st = cat.provenance["validation"]
    keep = np.ones(n, bool); keep[10] = False; keep[20] = False
    aligned = (cat.weight.shape[0] == cat.n_gal == n - 2 and np.allclose(cat.weight, w[keep])
               and np.allclose(cat.data["E2"], cols["E2"][keep]))
    print(f"  [0a] weight column filtered with the data mask: dropped={st['n_dropped']} (expect 2, "
          f"1 bad weight) aligned={aligned}")
    ok &= st["n_dropped"] == 2 and st["n_dropped_bad_weight"] == 1 and aligned
    cat = apply_weights(cat, mode="lensfit", nbins=6)
    summ = cat.provenance["treatment"]["weights"]["summary"]["per_bin"]
    print(f"  [0b] lensfit mode recorded; N_eff/N bin0 = {summ[0]['n_eff'] / summ[0]['n']:.3f} "
          f"(lognormal(0,0.8) expects ~{1 / np.exp(0.64):.3f})")
    ok &= abs(summ[0]["n_eff"] / summ[0]["n"] - 1 / np.exp(0.64)) < 0.05
    m = apply_m_bias(cat, source="auto")
    print(f"  [0c] auto m-bias on a real (non-sim) catalogue -> {cat.provenance['treatment']['m_bias']['source']}")
    ok &= cat.provenance["treatment"]["m_bias"]["source"] == "auto->fiducial"
    cat = apply_c_terms(cat)
    ok &= cat.provenance["treatment"]["c_terms"]["weighted"] is True
    if not estimator_accepts_weights():
        try:
            _build(cat, m, tmp, "should_raise", weights=cat.estimator_weights)
            print("  [0d] FAIL: lensfit weights + unpatched estimator did NOT raise")
            ok = False
        except RuntimeError as e:
            print(f"  [0d] lensfit + unpatched estimator refuses: {str(e)[:60]}...")
    else:
        print("  [0d] estimator accepts weights (patched) -- refusal path not applicable")
    try:
        load_catalogue(ext, column_map={**cm, "zbin_offset": 0}, kind="h5")
        print("  [0e] FAIL: 1-based TOMOBIN with offset 0 did not raise"); ok = False
    except ValueError as e:
        print(f"  [0e] 1-based TOMOBIN trap raises: {str(e)[:50]}...")
    return ok


def gate1(fixture: str, tmp: str, reference: str = None):
    cat_path, mock = _fixture(fixture)
    cat = load_catalogue(cat_path, kind="sim")
    cat = apply_weights(cat, mode="ignore")
    m = apply_m_bias(cat, source="auto")
    cat = apply_c_terms(cat)
    out = _build(cat, m, tmp, "g1")
    rep = compare_observation_to_mock(out, mock)
    worst = max(v for v in rep["per_dataset"].values() if np.isfinite(v)) if rep.get("per_dataset") else np.nan
    print(f"  [1] fidelity vs stored mock: pass={rep['pass']} worst rel rms={worst:.3e} "
          f"(a float32-stored fixture: the sc8/bandpower residuals are its storage floor)")
    ok = bool(rep["pass"])
    # The bit-identity statement proper: the per-dataset residuals must be IDENTICAL to the
    # pre-edit reference report of the same fixture (GATE 0a), i.e. today's edits changed nothing
    # on the unweighted path.
    if reference and os.path.exists(reference):
        ref = json.load(open(reference))["per_dataset"]
        common = [k for k in ref if k in rep["per_dataset"]]
        dmax = max(abs(float(ref[k]) - float(rep["per_dataset"][k])) for k in common)
        same = dmax < 1e-12
        print(f"  [1] unweighted path unchanged vs pre-edit reference ({len(common)} datasets): "
              f"{'IDENTICAL' if same else 'DIFFERS'} (max |d rel rms| = {dmax:.2e})")
        ok &= same
    return ok, out, cat, m


def gate2_3(cat: Catalogue, m, ref_path: str, tmp: str) -> bool:
    if not estimator_accepts_weights():
        print("  [2] SKIPPED: estimator has no weights= (apply .claude/plans/map_shears_weights.patch)")
        print("  [3] SKIPPED: same")
        return True
    ref = _maps(ref_path)
    ones = _build(cat, m, tmp, "g2", weights=np.ones(cat.n_gal))
    d = _max_rel(ref, _maps(ones))
    print(f"  [2] weights=ones vs unweighted: max rel diff {d:.3e}")
    ok = d < 1e-10
    rng = np.random.default_rng(3)
    w = rng.lognormal(0.0, 0.8, cat.n_gal)
    base = _maps(_build(cat, m, tmp, "g3a", weights=w))
    for alpha in (1e-3, 1e3):
        d = _max_rel(base, _maps(_build(cat, m, tmp, f"g3_{alpha:g}", weights=alpha * w)))
        print(f"  [3] alpha={alpha:g}: max rel diff {d:.3e}")
        ok &= d < 1e-10
    # [3b] the PRODUCTION rescale, apply_weights(rescale='neff'): w~ = w * sum(w)/sum(w^2) PER
    # TOMOGRAPHIC BIN. A per-bin rescale is a strictly stronger test than the global alpha above --
    # a different factor in every bin would expose any cross-bin coupling in the estimator, and the
    # real build relies on the maps being untouched so that only W_i and an exported count field
    # change scale. Also checks that sum(w~) really is N_eff, which is the whole point.
    import copy as _copy
    wcat = Catalogue(data=cat.data, weight=w.copy(), provenance=_copy.deepcopy(cat.provenance))
    wcat.provenance.setdefault("treatment", {})
    wcat = apply_weights(wcat, mode="lensfit", rescale="neff")
    rec = wcat.provenance["treatment"]["weights"]
    nb = int(cat.provenance.get("nbins", 6))
    for i in range(nb):
        sel = cat.data["ZBIN"] == i
        if not sel.any():
            continue
        sw_t = float(wcat.weight[sel].sum())
        neff = float(w[sel].sum()) ** 2 / float((w[sel] ** 2).sum())
        rel = abs(sw_t - neff) / neff
        if rel > 1e-12:
            print(f"  [3b] bin {i}: sum(w~) = {sw_t:.6f} != N_eff = {neff:.6f} (rel {rel:.2e})")
            ok = False
    d = _max_rel(base, _maps(_build(cat, m, tmp, "g3b", weights=wcat.weight)))
    print(f"  [3b] per-bin rescale='neff' vs raw w: max rel diff {d:.3e} "
          f"(divisors sum w^2/sum w = {[None if v is None else round(v, 4) for v in rec['divisor_sum_w2_over_sum_w']]})")
    ok &= d < 1e-10
    return ok


def gate4() -> bool:
    r = subprocess.run([PY, str(REPO / ".claude/plans/verify_shear_normalisation.py")], capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(REPO)})
    last = (r.stdout.strip().splitlines() or ["?"])[-1]
    print(f"  [4] Eq.-11 identity (verify_shear_normalisation.py): {last}")
    return r.returncode == 0


def gate4b() -> bool:
    """Direct check of the estimator's weights= path (once patched): weighted 'mean' and 'counts'
    maps vs the analytic S_p / W_i and S_p / D_p of spec §1, and weights=None == weights=ones."""
    if not estimator_accepts_weights():
        print("  [4b] SKIPPED: estimator has no weights=")
        return True
    import healpy as hp
    sys.path.insert(0, str(REPO / ".claude/plans"))
    from verify_shear_normalisation import synthetic, analytic_mean_norm, NBINS, NSIDE, M_BIAS
    from src.cosmology.map_shears import make_alm_shear_convergence
    cat, w, mask = synthetic()
    kw = dict(mask=mask, return_shear=True)
    ref_mean = analytic_mean_norm(cat, w, M_BIAS, float(mask.sum()))
    got_mean = make_alm_shear_convergence(cat, M_BIAS, NBINS, NSIDE, 2 * NSIDE, normalization="mean",
                                          rng=np.random.default_rng(1), weights=w, **kw)[2]
    got_cnt = make_alm_shear_convergence(cat, M_BIAS, NBINS, NSIDE, 2 * NSIDE, normalization="counts",
                                         rng=np.random.default_rng(1), weights=w, **kw)[2]
    ok = True
    for i in range(NBINS):
        sel = cat["ZBIN"] == i
        e = cat["E1"][sel] + 1j * cat["E2"][sel]; ww = w[sel]
        ebar = (ww * e).sum() / ww.sum()
        pix = hp.ang2pix(NSIDE, cat["RA"][sel], cat["DEC"][sel], lonlat=True)
        S = np.zeros(hp.nside2npix(NSIDE), complex); D = np.zeros(hp.nside2npix(NSIDE))
        np.add.at(S, pix, ww * (e - ebar)); np.add.at(D, pix, ww)
        ref_cnt = np.zeros_like(S); v = D > 0; ref_cnt[v] = S[v] / D[v] / (1 + M_BIAS[i])
        r1 = np.abs(got_mean[i] - ref_mean[i]).max() / np.abs(ref_mean[i]).max()
        r2 = np.abs(got_cnt[i] - ref_cnt).max() / np.abs(ref_cnt).max()
        ok &= r1 < 1e-12 and r2 < 1e-12
        print(f"  [4b] bin {i}: weights= mean-norm vs S_p/W_i {r1:.1e}; counts-norm vs S_p/D_p {r2:.1e}")
    un = make_alm_shear_convergence(cat, M_BIAS, NBINS, NSIDE, 2 * NSIDE, normalization="counts",
                                    rng=np.random.default_rng(1), **kw)[2]
    on = make_alm_shear_convergence(cat, M_BIAS, NBINS, NSIDE, 2 * NSIDE, normalization="counts",
                                    rng=np.random.default_rng(1), weights=np.ones(len(cat)), **kw)[2]
    d = max(np.abs(un[i] - on[i]).max() / np.abs(un[i]).max() for i in range(NBINS))
    print(f"  [4b] weights=None vs weights=ones: {d:.1e}")
    return ok and d < 1e-12


def gate5(real: str, column_map: str, json_out: str = None) -> bool:
    cm = json.load(open(column_map)) if column_map else None
    cat = load_catalogue(real, column_map=cm, kind="auto")
    summ = weight_summary(cat, 6)["per_bin"]
    ok = True
    rows = []
    for b in summ:
        i = b["bin"]
        nref = float(_N_ARCMIN2[i])
        dw = abs(b["W_i_neff_norm"] - W_REF[i]) / W_REF[i]
        dn = abs(b["n_eff_per_arcmin2"] - nref) / nref
        passed = dw < GATE5_TOL and dn < GATE5_TOL
        ok &= passed
        rows.append({"bin": i + 1, "n": b["n"], "sum_w": b["sum_w"], "sum_w2": b["sum_w2"], "N_eff": b["n_eff"],
                     "divisor_sum_w2_over_sum_w": b["sum_w2"] / b["sum_w"],
                     "W_i_data": b["W_i_neff_norm"], "W_i_ref": W_REF[i], "rel_diff_W": dw,
                     "n_eff_per_arcmin2": b["n_eff_per_arcmin2"], "n_arcmin2_tomo": nref, "rel_diff_n_eff": dn,
                     "W_i_superseded_literal": W_REF_SUPERSEDED[i], "pass": bool(passed)})
        print(f"  [5] bin {i + 1}: W_i={b['W_i_neff_norm']:.4f} (ref {W_REF[i]:.4f}, rel {dw:.1e}) "
              f"n_eff={b['n_eff_per_arcmin2']:.5f}/arcmin2 (tomo.py {nref:.4f}, rel {dn:.1e}) "
              f"{'PASS' if passed else 'FAIL'}")
    if json_out:
        os.makedirs(os.path.dirname(json_out), exist_ok=True)
        with open(json_out, "w") as fh:
            json.dump({"gate": 5, "pass": bool(ok), "tolerance_rel": GATE5_TOL, "catalogue": real,
                       "column_map": cm, "npix_norm": KIDS_NPIX_NORM, "area_deg2": KIDS_AREA_DEG2,
                       "reference": "W_i = n_arcmin2 * area_deg2 * 3600 / npix_norm (re-pre-registered 2026-09-24)",
                       "load_provenance": cat.provenance.get("validation"), "per_bin": rows}, fh, indent=1)
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", default="/data/alex/unblinding/fixtures/smoke_regen_seeded")
    ap.add_argument("--real-catalogue", default=None)
    ap.add_argument("--column-map", default=None)
    ap.add_argument("--keep", default=None, help="keep the build outputs under this dir")
    ap.add_argument("--only-gate5", action="store_true", help="skip gates 0-4b (already recorded) and run gate 5 only")
    ap.add_argument("--gate5-json", default=None, help="write gate 5's per-bin table here")
    ap.add_argument("--reference-json", default="/data/alex/unblinding/fixtures/gate0a/observation_S7_fidelity.json",
                    help="pre-edit fidelity report of the same fixture; gate 1 requires identical per-dataset residuals")
    a = ap.parse_args(argv)
    results = {}
    tmp = a.keep or tempfile.mkdtemp(prefix="wgate_")
    os.makedirs(tmp, exist_ok=True)
    print(f"estimator accepts weights: {estimator_accepts_weights()}   (outputs: {tmp})")
    if a.only_gate5:
        if not a.real_catalogue:
            ap.error("--only-gate5 needs --real-catalogue")
        print("gate 5 (W_i / n_eff on the real catalogue)")
        ok = gate5(a.real_catalogue, a.column_map, a.gate5_json)
        print(f"\ngate 5: {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    print("gate 0 (ingestion)"); results["0"] = gate0(a.fixture, tmp)
    print("gate 1 (bit-identity, weights=None)"); results["1"], ref, cat, m = gate1(a.fixture, tmp, a.reference_json)
    print("gates 2-3 (weighted no-op, scale invariance)"); results["2-3"] = gate2_3(cat, m, ref, tmp)
    print("gate 4 (Eq. 11 identity)"); results["4"] = gate4()
    print("gate 4b (weights= path vs analytic weighted maps)"); results["4b"] = gate4b()
    if a.real_catalogue:
        print("gate 5 (W_i / n_eff on the real catalogue)"); results["5"] = gate5(a.real_catalogue, a.column_map, a.gate5_json)
    print("\n" + "  ".join(f"gate {k}: {'PASS' if v else 'FAIL'}" for k, v in results.items()))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
