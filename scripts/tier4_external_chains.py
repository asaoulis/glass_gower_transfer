"""Public survey chains for the Tier-4 reveal figures (CosmoSIS text format, header-driven).

Every file is tab-separated text with a `#`-prefixed first line of column names and a `weight`
column (PolyChord / MultiNest importance weights). Columns are matched by NAME (case-insensitive),
never by index; S8 is derived as sigma_8 sqrt(Omega_m/0.3) when the file carries no S_8 column
(checked to a ratio of 1.0000 on every file that has one).

Registry: `SURVEY_CHAINS` = ordered list of (key, path, display name, model, probe) for the rows
the user asked for (DES Y3 and HSC Y3, 2-pt and beyond-2-pt). Chains that are NOT public (DES Y3
field-level / SBI Jeffrey+24, Gatti+24, Zuercher+22 peaks; HSC Y3 peaks / PDFs) are listed in
`NOT_PUBLIC` so the README can say so.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

EXT_ROOT = "/data/alex/unblinding/external"

# key, relative path, legend name, model ('LCDM' | 'wCDM'), probe class ('2pt' | 'beyond2pt' | '3x2pt')
SURVEY_CHAINS: List[Tuple[str, str, str, str, str]] = [
    ("des_y3_cs_lcdm", "des_y3_2pt/chain_1x2pt_lcdm_SR_maglim.txt", "DES Y3 cosmic shear", "LCDM", "2pt"),
    ("des_y3_cs_wcdm", "des_y3_2pt/chain_1x2pt_wcdm_SR_maglim.txt", "DES Y3 cosmic shear", "wCDM", "2pt"),
    ("hsc_y3_cl", "hsc_y3_dalal23_li23_cosmic_shear/hsc_y3_fourier.txt", r"HSC Y3 $C_\ell$", "LCDM", "2pt"),
    ("hsc_y3_xipm", "hsc_y3_dalal23_li23_cosmic_shear/hsc_y3_real.txt", r"HSC Y3 $\xi_\pm$", "LCDM", "2pt"),
    ("des_y3_kids1000", "des_y3_kids1000_joint/chain_desy3_and_kids1000_hybrid_analysis.txt",
     "DES Y3 + KiDS-1000", "LCDM", "2pt"),
    ("des_y3_3x2_lcdm", "des_y3_2pt/chain_3x2pt_lcdm_SR_maglim.txt", r"DES Y3 3$\times$2pt", "LCDM", "3x2pt"),
    ("des_y3_3x2_wcdm", "des_y3_2pt/chain_3x2pt_wcdm_SR_maglim.txt", r"DES Y3 3$\times$2pt", "wCDM", "3x2pt"),
    ("hsc_y3_3x2", "hsc_y3_zhang_2x2pt_3x2pt/3x2pt_chain_b0_dnnz_all_fiducial.txt", r"HSC Y3 3$\times$2pt", "LCDM", "3x2pt"),
    ("hsc_y3_clusters", "hsc_y3_shear_selected_clusters/cnstwlcspz_multinest_results.txt",
     "HSC Y3 shear-selected clusters", "LCDM", "beyond2pt"),
]

NOT_PUBLIC = [
    "DES Y3 field-level / SBI wCDM (Jeffrey+24): chains promised 'upon publication', none online (2026-10)",
    "DES Y3 moments / peaks / wavelets SBI (Gatti+24): no data-availability release",
    "DES Y3 peaks (Zuercher+22): code only, no chains",
    "HSC Y3 peaks / PDFs (Thiele+23, Marques+23, Novaes+): no public chains on the HSC release pages",
    "HSC Y3 Dalal+23 / Li+23 official files need an HSC login; GitHub copies used instead",
]

_ALIASES = {
    "omega_m": ("cosmological_parameters--omega_m",),
    "sigma_8": ("cosmological_parameters--sigma_8",),
    "S8": ("cosmological_parameters--s_8", "cosmological_parameters--s8"),
    "w0": ("cosmological_parameters--w",),
    "weight": ("weight",),
}


def read_cosmosis_chain(path: str, want=("omega_m", "sigma_8", "S8", "w0")):
    """-> (X [N, len(names)], names, weights) of a CosmoSIS text chain, columns by name.
    Rows with non-finite values or zero weight are dropped; S8 derived when absent."""
    with open(path) as fh:
        hdr = [h.strip().lower() for h in fh.readline().lstrip("#").strip().split()]
    X = np.loadtxt(path, comments="#")
    if X.ndim == 1:
        X = X[None, :]
    col = {h: i for i, h in enumerate(hdr)}

    def find(name):
        for a in _ALIASES[name]:
            if a in col:
                return col[a]
        return None

    iw = find("weight")
    w = X[:, iw] if iw is not None else np.ones(len(X))
    cols, names = [], []
    for n in want:
        i = find(n)
        if i is not None and np.isfinite(X[:, i]).any():     # an all-NaN derived column counts as absent
            cols.append(i); names.append(n)
    Y = X[:, cols]
    ok = np.isfinite(w) & (w > 0) & np.all(np.isfinite(Y), axis=1)
    Y, w = Y[ok], w[ok]
    if "S8" not in names and {"omega_m", "sigma_8"} <= set(names):
        s8 = Y[:, names.index("sigma_8")] * np.sqrt(Y[:, names.index("omega_m")] / 0.3)
        Y = np.concatenate([Y, s8[:, None]], axis=1); names.append("S8")
    return Y, names, w / w.max()


def available(root: str = EXT_ROOT, include_beyond2pt: bool = False):
    """The registry rows whose file exists. The cluster-abundance chain (the only public HSC
    beyond-2-pt chain) is OFF by default: its quoted S8 uses sigma_8 (Omega_m/0.3)^0.25 and Omega_m
    is unconstrained there, so a sqrt-definition S8 row is not comparable."""
    return [(k, os.path.join(root, rel), name, model, probe) for k, rel, name, model, probe in SURVEY_CHAINS
            if os.path.isfile(os.path.join(root, rel)) and (include_beyond2pt or probe != "beyond2pt")]


COMPANION_ROOT = os.path.join(EXT_ROOT, "companion_2pt")
COMPANION_FILE = "nle_real_data_samples_r3.npz"      # keys: joint [N, 10], params [10] (omega_m, s8, sigma_8, ...)
COMPANION_NAMES = {"novd": "2-pt SBI, no VD", "vd": "2-pt SBI, VD"}


def read_companion(tag: str, root: str = COMPANION_ROOT):
    """-> (X [N, 3], ['omega_m', 'sigma_8', 'S8']) of the companion score-based 2-pt SBI posterior
    (hypatia `sbi_score_{novd,vd}_real_data/`, LCDM, unweighted draws), or None when absent."""
    path = os.path.join(root, tag, COMPANION_FILE)
    if not os.path.isfile(path):
        return None
    with np.load(path, allow_pickle=False) as d:
        X = np.asarray(d["joint"], dtype=np.float64)
        names = [str(n) for n in d["params"]]
    cols = {"omega_m": "omega_m", "sigma_8": "sigma_8", "S8": "s8"}
    idx = [names.index(v) for v in cols.values()]
    Y = X[:, idx]
    Y = Y[np.all(np.isfinite(Y), axis=1)]
    return Y, list(cols)


if __name__ == "__main__":
    for tag in COMPANION_NAMES:
        r = read_companion(tag)
        print("companion %-5s %s" % (tag, "absent" if r is None else "%d draws, cols=%s" % (len(r[0]), r[1])))
    for k, p, name, model, probe in available(include_beyond2pt=True):
        Y, names, w = read_cosmosis_chain(p)
        s8 = Y[:, names.index("S8")]
        m = np.average(s8, weights=w); sd = np.sqrt(np.average((s8 - m) ** 2, weights=w))
        print("%-18s %-30s %-5s %-9s n=%6d  S8 = %.3f +- %.3f  cols=%s" % (k, name, model, probe, len(Y), m, sd, names))
