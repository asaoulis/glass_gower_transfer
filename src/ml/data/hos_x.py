"""Candidate superset of noise-texture-robust third-order map statistics beyond R1 (`hos_x/vector`).

Task eval-and-viz/hos-3pt-information. Same conventions as R1 (`src/ml/data/hos.py`): the STORED meter-normalised
sc8a1 E map cast to float64 BEFORE filtering, anisotropic plate-carree filters with reflect boundary, footprint pixels
(frac > 0.5) pooled N + S, the pooled mean subtracted per channel, moment = mean of the product over the pooled pixels.
Every statistic is an ODD (cubic) polynomial of the map, so sign-symmetric shape noise of any variance texture has
zero pure-noise expectation.

A *channel* is (bin or merged bin, filter, scale); a block holds every unordered channel triple a <= b <= c of its
channel list (lexicographic in channel order) unless stated.
  R1     m3x3, 6 bins, Gaussian FWHM 8, 16 (112). Computed by `hos.third_moments`, bit-identical to `hos/vector`.
  Geq    m3x3, 6 bins, Gaussian FWHM 4, 12, 24, 32 (4 x 56).
  Gpool  kbar = mean of the 6 bins at Gaussian {4, 8, 12, 16, 24, 32}, all scale triples (56).
  Gmrg   merged bins M1 = (b1 + b2)/2, M2 = (b3 + b4)/2, M3 = (b5 + b6)/2 x Gaussian {8, 16, 24, 32} (12 ch, 364).
  Ceq    compensated (Crittenden) aperture, theta {4, 8, 16, 32}, 6 bins, equal-theta m3x3 (4 x 56).
  Cmrg   M1..M3 x Crittenden theta {4, 8, 16, 32} (12 ch, 364).
  G6u    (optional) 6 bins x Gaussian {8, 16, 32} (18 ch), only triples spanning >= 2 distinct scales (972).
Scales are NOMINAL total FWHM t: the stored map already carries a 4' FWHM Gaussian (t = 4 is the stored map), extra
FWHM sqrt(t^2 - 16) (hos.extra_fwhm). Crittenden: U(l) = (l theta)^2 / 2 exp(-l^2 theta^2 / 2), i.e.
Map_theta = -(theta^2 / 2) Laplacian(G_{sigma=theta} * kappa); the stored 4' FWHM (sigma0 = 4 / 2.3548 arcmin) is
folded in, so the applied Gaussian has sigma_extra = sqrt(theta^2 - sigma0^2). The filter is applied as its analytic
symbol in the DCT-II domain (exact reflect-boundary convolution) with the anisotropic spacings dy = PIX,
dx = PIX cos dec_c (see `aperture`).
"""
import itertools

import numpy as np
from src.ml.data.hos import (DEC_C, NB, PIX, R1_NAMES, SIDES, TRIPLES, extra_fwhm, load_footprint, smooth,
                             third_moments)

HOSX_VERSION = "hosx-v1"
HOSX_SCALES_G = (4, 8, 12, 16, 24, 32)
HOSX_GEQ_SCALES = (4, 12, 24, 32)
HOSX_MRG_SCALES = (8, 16, 24, 32)
HOSX_G6U_SCALES = (8, 16, 32)
HOSX_THETA_C = (4, 8, 16, 32)
MERGE = ((0, 1), (2, 3), (4, 5))
SIGMA0 = 4.0 / (2 * np.sqrt(2 * np.log(2)))          # arcmin, the stored map's Gaussian
BLOCKS_DEFAULT = ("R1", "Geq", "Gpool", "Gmrg", "Ceq", "Cmrg")
BLOCKS_ALL = BLOCKS_DEFAULT + ("G6u",)


def aperture_symbol(theta, side, shape):
    """(H, W) DCT-II-domain symbol of -(theta^2/2) Lap G_sigma_extra: (theta^2/2) k^2 exp(-k^2 sigma_extra^2/2) at
    k_y = pi m / (H dy), k_x = pi n / (W dx) (1/arcmin)."""
    H, W = shape
    se = np.sqrt(theta ** 2 - SIGMA0 ** 2)
    dy, dx = PIX, PIX * np.cos(np.radians(DEC_C[side]))
    ky = np.pi * np.arange(H) / (H * dy)
    kx = np.pi * np.arange(W) / (W * dx)
    k2 = ky[:, None] ** 2 + kx[None, :] ** 2
    return (theta ** 2 / 2.0) * k2 * np.exp(-k2 * se ** 2 / 2.0)


def aperture(x, theta, side):
    """x (..., H, W) float64 -> Crittenden compensated aperture -(theta^2/2) Lap(G_sigma_extra * x), theta arcmin.

    Applied as the analytic symbol in the DCT-II domain, which is EXACT convolution with the untruncated kernel under
    the reflect (half-sample symmetric) boundary used by every other filter here. A sampled scipy
    gaussian_filter(order=2) kernel is not usable: at theta = 4 sigma_extra = 0.53 px and the sampled kernel is not
    compensated (non-zero DC response; A2 gate (iii) measured 8.6x the analytic response at l = 150)."""
    from scipy.fft import dctn, idctn
    assert x.dtype == np.float64, f"aperture needs float64 input, got {x.dtype}"
    S = aperture_symbol(theta, side, x.shape[-2:])
    return idctn(dctn(x, type=2, axes=(-2, -1), norm="ortho") * S, type=2, axes=(-2, -1), norm="ortho")


def _pool(F, foot):
    """{side: (n, H, W)} -> (n, npix) pooled N + S footprint pixels, mean subtracted per channel."""
    D = np.concatenate([F[s][:, foot[s]] for s in SIDES], axis=1)
    return D - D.mean(axis=1, keepdims=True)


def _merge(F):
    """(6, ...) per-bin field -> (3, ...) merged bins."""
    return np.stack([(F[a] + F[b]) / 2.0 for a, b in MERGE])


def hosx_channels(maps, foot, optional=False):
    """maps {side: (6, H, W) float64} -> dict of pooled, mean-subtracted channel matrices (n, npix):
    'G{t}' (6 bins at Gaussian t), 'C{theta}' (6 bins, aperture theta), 'kbar' (6 scales), 'Gm' (12, scale-major
    M1..M3 per scale), 'Cm' (12, theta-major)."""
    for s in SIDES:
        assert maps[s].dtype == np.float64, maps[s].dtype
    sm = {t: {s: (smooth(maps[s], extra_fwhm(t), s) if t != 4 else maps[s]) for s in SIDES} for t in HOSX_SCALES_G}
    ap = {th: {s: aperture(maps[s], th, s) for s in SIDES} for th in HOSX_THETA_C}
    ch = {}
    for t in HOSX_SCALES_G:
        ch[f"G{t}"] = _pool(sm[t], foot)
    for th in HOSX_THETA_C:
        ch[f"C{th}"] = _pool(ap[th], foot)
    # linear combinations of pooled, mean-subtracted channels == pooling / mean-subtracting the combined field
    ch["kbar"] = np.stack([ch[f"G{t}"].mean(axis=0) for t in HOSX_SCALES_G])
    ch["Gm"] = np.concatenate([_merge(ch[f"G{t}"]) for t in HOSX_MRG_SCALES])
    ch["Cm"] = np.concatenate([_merge(ch[f"C{th}"]) for th in HOSX_THETA_C])
    return ch


def triples_of(n):
    return [(a, b, c) for a in range(n) for b in range(a, n) for c in range(b, n)]


def all_triples(D):
    """(n, npix) -> {(a, b, c): mean(D_a D_b D_c)} for a <= b <= c, via one BLAS product per pair."""
    n, npix = D.shape
    pairs = [(a, b) for a in range(n) for b in range(a, n)]
    P = np.stack([D[a] * D[b] for a, b in pairs])
    T = (P @ D.T) / npix
    return {(a, b, c): T[k, c] for k, (a, b) in enumerate(pairs) for c in range(b, n)}


# ---------------------------------------------------------------- block specs (channel labels + triple selection)
def _bin_lab(i):
    return f"b{i + 1}"


def _mrg_lab(m):
    return f"M{m + 1}"


def _eq_block(prefix, key_fmt, scales):
    names, idx = [], []
    for t in scales:
        for (i, j, l) in TRIPLES:
            names.append(f"{prefix}/{key_fmt}{t}/b{i + 1}b{j + 1}b{l + 1}")
            idx.append((t, (i, j, l)))
    return names, idx


def _multi_block(prefix, labels, keep=None):
    """labels: per channel (scale_label, bin_label). Triples a <= b <= c, optionally filtered."""
    names, trips = [], []
    for (a, b, c) in triples_of(len(labels)):
        if keep is not None and not keep(labels[a], labels[b], labels[c]):
            continue
        names.append(f"{prefix}/" + ".".join(f"{labels[k][0]}{labels[k][1]}" for k in (a, b, c)))
        trips.append((a, b, c))
    return names, trips


GPOOL_LABELS = [(f"s{t}", "K") for t in HOSX_SCALES_G]
GMRG_LABELS = [(f"s{t}", _mrg_lab(m)) for t in HOSX_MRG_SCALES for m in range(3)]
CMRG_LABELS = [(f"t{th}", _mrg_lab(m)) for th in HOSX_THETA_C for m in range(3)]
G6U_LABELS = [(f"s{t}", _bin_lab(i)) for t in HOSX_G6U_SCALES for i in range(NB)]
_G6U_KEEP = (lambda x, y, z: len({x[0], y[0], z[0]}) >= 2)

_GEQ_NAMES, _GEQ_IDX = _eq_block("Geq", "s", HOSX_GEQ_SCALES)
_CEQ_NAMES, _CEQ_IDX = _eq_block("Ceq", "t", HOSX_THETA_C)
_GPOOL_NAMES, _GPOOL_TR = _multi_block("Gpool", GPOOL_LABELS)
_GMRG_NAMES, _GMRG_TR = _multi_block("Gmrg", GMRG_LABELS)
_CMRG_NAMES, _CMRG_TR = _multi_block("Cmrg", CMRG_LABELS)
_G6U_NAMES, _G6U_TR = _multi_block("G6u", G6U_LABELS, keep=_G6U_KEEP)

BLOCK_NAMES = {"R1": list(R1_NAMES), "Geq": _GEQ_NAMES, "Gpool": _GPOOL_NAMES, "Gmrg": _GMRG_NAMES,
               "Ceq": _CEQ_NAMES, "Cmrg": _CMRG_NAMES, "G6u": _G6U_NAMES}
assert [len(BLOCK_NAMES[b]) for b in BLOCKS_ALL] == [112, 224, 56, 364, 224, 364, 972], \
    [len(BLOCK_NAMES[b]) for b in BLOCKS_ALL]


def hosx_names(blocks=BLOCKS_DEFAULT):
    return [n for b in blocks for n in BLOCK_NAMES[b]]


def hosx_blocks(blocks=BLOCKS_DEFAULT):
    """{block: index list into hosx_names(blocks)}."""
    out, k = {}, 0
    for b in blocks:
        out[b] = list(range(k, k + len(BLOCK_NAMES[b])))
        k += len(BLOCK_NAMES[b])
    return out


HOSX_NAMES = hosx_names(BLOCKS_DEFAULT)
HOSX_BLOCKS = hosx_blocks(BLOCKS_DEFAULT)
HOSX_NAMES_ALL = hosx_names(BLOCKS_ALL)
HOSX_BLOCKS_ALL = hosx_blocks(BLOCKS_ALL)


def _legs(name):
    """name -> list of (scale_label, bin_label) legs, e.g. 's8', 'b1' / 't4', 'M2' / 's12', 'K'."""
    blk, rest = name.split("/", 1)
    if blk.startswith("s") and "/m3x3/" in name:             # R1 's8/m3x3/b1b2b3'
        t, _, bins = name.split("/")
        return [(t, f"b{c}") for c in bins[1:].split("b")]
    if blk in ("Geq", "Ceq"):
        t, bins = rest.split("/")
        return [(t, f"b{c}") for c in bins[1:].split("b")]
    out = []
    for leg in rest.split("."):
        k = 1
        while leg[k].isdigit():
            k += 1
        out.append((leg[:k], leg[k:]))
    return out


def has_leg4(name):
    return any(t in ("s4", "t4") for t, _ in _legs(name))


def is_distinct(name):
    """No repeated bin / merged bin (kbar-only triples are never distinct)."""
    b = [x for _, x in _legs(name)]
    return "K" not in b and len(set(b)) == 3


def index_sets(names):
    """Helper index sets into `names`: 'no4', 'distinct', and per scale / theta for the equal-scale blocks."""
    S = {"no4": [k for k, n in enumerate(names) if not has_leg4(n)],
         "distinct": [k for k, n in enumerate(names) if is_distinct(n)]}
    for k, n in enumerate(names):
        blk = n.split("/")[0]
        if blk in ("Geq", "Ceq"):
            S.setdefault(f"{blk}/{n.split('/')[1]}", []).append(k)
        elif blk.startswith("s") and "/m3x3/" in n:
            S.setdefault(f"R1/{blk}", []).append(k)
    return S


def hosx_vector(maps, foot, optional=False):
    """maps {side: (6, H, W) float64 (meter-normalised)} -> (vec, names). Block order R1 | Geq | Gpool | Gmrg | Ceq |
    Cmrg [| G6u]. R1 comes from the EXISTING `hos.third_moments` path (bit-identical to `hos/vector`)."""
    blocks = BLOCKS_ALL if optional else BLOCKS_DEFAULT
    ch = hosx_channels(maps, foot)
    parts = [third_moments(maps, foot).ravel()]
    eq = {}
    for t in HOSX_GEQ_SCALES:
        tr = all_triples(ch[f"G{t}"])
        eq[("G", t)] = [tr[x] for x in TRIPLES]
    for th in HOSX_THETA_C:
        tr = all_triples(ch[f"C{th}"])
        eq[("C", th)] = [tr[x] for x in TRIPLES]
    parts.append(np.concatenate([eq[("G", t)] for t in HOSX_GEQ_SCALES]))
    trp = all_triples(ch["kbar"])
    parts.append(np.array([trp[x] for x in _GPOOL_TR]))
    trg = all_triples(ch["Gm"])
    parts.append(np.array([trg[x] for x in _GMRG_TR]))
    parts.append(np.concatenate([eq[("C", th)] for th in HOSX_THETA_C]))
    trc = all_triples(ch["Cm"])
    parts.append(np.array([trc[x] for x in _CMRG_TR]))
    if optional:
        D6 = np.concatenate([ch[f"G{t}"] for t in HOSX_G6U_SCALES])
        tr6 = all_triples(D6)
        parts.append(np.array([tr6[x] for x in _G6U_TR]))
    vec = np.concatenate([np.asarray(p, dtype=np.float64) for p in parts])
    names = hosx_names(blocks)
    assert vec.shape == (len(names),), (vec.shape, len(names))
    return vec, names


# ---------------------------------------------------------------- sidecar bake (hos.bake_one(variant="x"))
# Versioned bake list (Phase A, artifacts/PHASE_A.md): the default blocks; G6u NOT baked (redundant, adds nothing over
# Cmrg at n = 511). Ceq theta = 8' and every 4' / theta = 4 leg ARE baked; analysis / production vectors select by index.
HOSX_BAKE_BLOCKS = BLOCKS_DEFAULT
HOSX_BAKE_NAMES = hosx_names(HOSX_BAKE_BLOCKS)
HOSX_BAKE_BLOCK_INDEX = hosx_blocks(HOSX_BAKE_BLOCKS)
FILTER_SPEC = ("Gaussian: anisotropic plate-carree, extra FWHM sqrt(t^2-16) (t=4 stored map), reflect, truncate 4 "
               "(hos.smooth); Crittenden: analytic DCT-II symbol (theta^2/2) k^2 exp(-k^2 (theta^2-sigma0^2)/2), "
               "sigma0 = 4'/2.3548, dy = PIX, dx = PIX cos dec_c; merged bins (b1+b2)/2, (b3+b4)/2, (b5+b6)/2; "
               "kbar = mean of 6 bins; pooled N+S footprint, mean-subtracted per channel")


def group_index(groups, names=None):
    """Sorted union of feature indices (into `names`, default HOSX_BAKE_NAMES) of the group keys in `groups`:
    a block ('R1', 'Geq', 'Gpool', 'Gmrg', 'Ceq', 'Cmrg'[, 'G6u']), a block subset '<block>/no4' | '/distinct' |
    '/no4distinct', or a per-scale key 'R1/s8', 'Geq/s24', 'Ceq/t16'."""
    names = list(HOSX_BAKE_NAMES if names is None else names)
    blk_of = [n.split("/")[0] if not ("/m3x3/" in n and n.split("/")[0].startswith("s")) else "R1" for n in names]
    S = index_sets(names)
    out = set()
    for g in groups:
        base, _, sub = g.partition("/")
        if not sub:
            idx = [k for k, b in enumerate(blk_of) if b == base]
        elif sub in ("no4", "distinct", "no4distinct"):
            want = {"no4": set(S["no4"]), "distinct": set(S["distinct"]),
                    "no4distinct": set(S["no4"]) & set(S["distinct"])}[sub]
            idx = [k for k, b in enumerate(blk_of) if b == base and k in want]
        else:
            idx = S.get(g, [])
        if not idx:
            raise KeyError(f"hos_x group {g!r} matches no feature")
        out.update(idx)
    return sorted(out)


def hosx_bake_vector(maps, foot):
    """The baked `hos_x/vector` (HOSX_BAKE_BLOCKS order) of one meter-normalised map dict."""
    assert HOSX_BAKE_BLOCKS == BLOCKS_DEFAULT
    vec, names = hosx_vector(maps, foot, optional=False)
    assert names == HOSX_BAKE_NAMES
    return vec


def bake_attrs(mask_sha="", git_rev="unknown", source=""):
    import json
    return dict(names=json.dumps(HOSX_BAKE_NAMES), blocks=json.dumps(HOSX_BAKE_BLOCK_INDEX), version=HOSX_VERSION,
                mask_sha256=mask_sha, git_rev=git_rev, filter_spec=FILTER_SPEC, source=source,
                map_convention="stored sc8a1 E (divided by noise_std_*/all at bake), float64 before filtering")


__all__ = ["HOSX_VERSION", "HOSX_SCALES_G", "HOSX_THETA_C", "MERGE", "BLOCKS_DEFAULT", "BLOCKS_ALL", "BLOCK_NAMES",
           "HOSX_NAMES", "HOSX_BLOCKS", "HOSX_NAMES_ALL", "HOSX_BLOCKS_ALL", "aperture", "hosx_channels",
           "hosx_vector", "hosx_names", "hosx_blocks", "index_sets", "has_leg4", "is_distinct", "all_triples",
           "load_footprint", "HOSX_BAKE_BLOCKS", "HOSX_BAKE_NAMES", "HOSX_BAKE_BLOCK_INDEX", "hosx_bake_vector",
           "bake_attrs", "group_index"]
