import numpy as np

def compute_glass_cls(lp, ws, ell):
    """
    Compute and reorder Cls for GLASS from a LevinPower object.

    Args:
        lp: LevinPower object.
        ws: Shell window functions.
        n_glass_shells: Initial number of redshift shells.

    Returns:
        glass_cls: Reordered angular power spectra in GLASS format.
        ws: Updated shell windows with last shell removed.
        n_glass_shells: Updated shell count (reduced by 1).
    """
    print("Computing angular power spectra...")
    n_glass_shells = len(ws)
    Cl_gg, Cl_gs, Cl_ss = lp.compute_C_ells(ell)
    Cl_gg = np.array(Cl_gg)

    # Build index mappings
    idx_ls = np.array([[i, j] for i in range(n_glass_shells) for j in range(i + 1)])  # i >= j
    idx_sl = np.array([[i, j + i] for i in range(n_glass_shells) for j in range(n_glass_shells - i)])  # i <= j

    # Reorder Cls from (j, i) to (i, j)
    new_order = [np.where((idx_sl == pair).all(axis=1))[0][0] for pair in idx_ls[:, [1, 0]]]
    Cl_gg_reordered = Cl_gg[new_order]

    # Drop final shell
    n_glass_shells -= 1
    ws = ws[:-1]

    # Build dictionary of blocks
    counter = 0
    cl_gg_blocks = {}
    for i in range(n_glass_shells):
        for j in range(i + 1):
            c_ell = Cl_gg_reordered[counter]
            cl_gg_blocks[f'W{i+1}xW{j+1}'] = c_ell
            counter += 1

    # Repack blocks into array for GLASS
    glass_cls = np.array([
        cl_gg_blocks[f'W{i}xW{j}']
        for i in range(1, n_glass_shells + 1)
        for j in range(i, 0, -1)
    ])

    return glass_cls, ws, n_glass_shells
