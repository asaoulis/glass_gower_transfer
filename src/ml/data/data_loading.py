from typing import Dict, List, Tuple, Union
import numpy as np
import h5py

def _resolve_h5(obj: h5py.Group, path: Union[Tuple[str, ...], List[str], str]):
    if isinstance(path, str):
        path = (path,)
    cur = obj
    for key in path:
        if key not in cur:
            raise KeyError(f"Path {'/'.join(path)} not found under {cur.name}. Available: {list(cur.keys())}")
        cur = cur[key]
    return cur

def _stack_group_datasets(group: h5py.Group) -> np.ndarray:
    arrays = []
    order = []
    def dfs(g, prefix=()):
        for k in sorted(g.keys()):
            node = g[k]
            if isinstance(node, h5py.Dataset):
                arrays.append(np.asarray(node[()]))
                order.append(prefix + (k,))
            elif isinstance(node, h5py.Group):
                dfs(node, prefix + (k,))
    dfs(group)
    if not arrays:
        raise ValueError(f"Group {group.name} contains no datasets to load.")
    try:
        return np.stack(arrays, axis=0)
    except ValueError as e:
        shapes = [a.shape for a in arrays]
        raise ValueError(f"Cannot stack datasets under {group.name}; shapes: {shapes}") from e

def load_cosmo_params(file_path: str, cosmo_params: List[str], as_torch: bool = True, dtype=np.float32):
    with h5py.File(file_path, "r") as f:
        grp = f["cosmo_dict"]
        vals = [float(np.asarray(grp[p][()])) for p in cosmo_params]
    arr = np.asarray(vals, dtype=dtype)
    if as_torch:
        import torch
        arr = torch.from_numpy(arr)
    return arr

def list_patches(file_path: str, map_type: str = "E") -> List[str]:
    with h5py.File(file_path, "r") as f:
        base = f["pixelised_results"][map_type]
        return sorted(list(base.keys()))

def unpack_data(
    file_path: str,
    nested_keys: Dict[str, Tuple[str, ...]],
    cosmo_params: List[str],
    as_torch: bool = True,
    dtype=np.float32,
    stack_groups: bool = False,  # default: do NOT stack; require explicit patch (e.g., 'north')
):
    data = {}
    with h5py.File(file_path, "r") as f:
        for out_key, path in nested_keys.items():
            node = _resolve_h5(f, path)
            if isinstance(node, h5py.Dataset):
                arr = np.asarray(node[()], dtype=dtype)
            elif isinstance(node, h5py.Group):
                if not stack_groups:
                    raise ValueError(
                        f"Path {'/'.join(path)} is a group. Specify a patch like "
                        f"{path + ('north',)} or {path + ('south',)}, or set stack_groups=True."
                    )
                arr = _stack_group_datasets(node).astype(dtype, copy=False)
            else:
                raise TypeError(f"Unsupported HDF5 node type at {'/'.join(path)}: {type(node)}")
            data[out_key] = arr

    cosmo = load_cosmo_params(file_path, cosmo_params, as_torch=as_torch, dtype=dtype)

    if as_torch:
        import torch
        data = {k: torch.from_numpy(v) if not isinstance(v, torch.Tensor) else v for k, v in data.items()}

    return data, cosmo