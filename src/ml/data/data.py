
import os
import re
import glob
import random
from typing import Dict, List, Sequence, Tuple, Union, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .data_loading import unpack_data


_COSMO_RE = re.compile(r"output_(\d+)_\d+\.h5$", re.IGNORECASE)


def extract_cosmo_index(path: str) -> int:
    """
    Extract cosmology index from a filename like .../output_<cosmo>_<sample>.h5
    """
    m = _COSMO_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Could not parse cosmology index from: {path}")
    return int(m.group(1))


def collect_paths(patterns: Union[str, Sequence[str]]) -> List[str]:
    """
    Expand one or multiple glob patterns into a sorted list of files.
    """
    if isinstance(patterns, str):
        patterns = [patterns]
    paths: List[str] = []
    for p in patterns:
        paths.extend(glob.glob(p))
    # Deduplicate and sort
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"No files matched the provided pattern(s): {patterns}")
    return paths


def split_by_cosmology(
    patterns: Union[str, Sequence[str]],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Glob files, group by cosmology index, shuffle cosmologies, and split without leakage.
    Returns lists of file paths for train/val/test.
    """
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    all_paths = collect_paths(patterns)
    by_cosmo: Dict[int, List[str]] = {}
    for p in all_paths:
        cidx = extract_cosmo_index(p)
        by_cosmo.setdefault(cidx, []).append(p)

    # Sort files within each cosmology for stability
    for k in by_cosmo:
        by_cosmo[k].sort()

    cosmologies = list(by_cosmo.keys())
    rng = random.Random(seed)
    rng.shuffle(cosmologies)

    n = len(cosmologies)
    if n == 0:
        raise ValueError("No cosmologies found.")

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val  # remainder

    train_cosmos = set(cosmologies[:n_train])
    val_cosmos = set(cosmologies[n_train:n_train + n_val])
    test_cosmos = set(cosmologies[n_train + n_val:])

    train_paths: List[str] = []
    val_paths: List[str] = []
    test_paths: List[str] = []

    for c in train_cosmos:
        train_paths.extend(by_cosmo[c])
    for c in val_cosmos:
        val_paths.extend(by_cosmo[c])
    for c in test_cosmos:
        test_paths.extend(by_cosmo[c])

    return train_paths, val_paths, test_paths


class H5CosmoDataset(Dataset):
    """
    Dataset that loads items via unpack_data for given HDF5 paths.
    __getitem__ returns (data_dict, cosmo_vector) as unpacked by unpack_data.
    """

    def __init__(
        self,
        paths: Sequence[str],
        nested_keys: Dict[str, Tuple[str, ...]],
        cosmo_params: List[str],
        *,
        as_torch: bool = True,
        dtype=np.float32,
        stack_groups: bool = False,
        transform=None,
    ):
        self.paths = list(paths)
        self.nested_keys = nested_keys
        self.cosmo_params = cosmo_params
        self.as_torch = as_torch
        self.dtype = dtype
        self.stack_groups = stack_groups
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        data, cosmo = unpack_data(
            path,
            self.nested_keys,
            self.cosmo_params,
            as_torch=self.as_torch,
            dtype=self.dtype,
            stack_groups=self.stack_groups,
        )
        if self.transform is not None:
            data = self.transform(data)
        return data, cosmo


def build_datasets(
    patterns: Union[str, Sequence[str]],
    nested_keys: Dict[str, Tuple[str, ...]],
    cosmo_params: List[str],
    *,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    as_torch: bool = True,
    dtype=np.float32,
    stack_groups: bool = False,
) -> Tuple[H5CosmoDataset, H5CosmoDataset, H5CosmoDataset]:
    """
    Convenience: split by cosmology and return three datasets.
    """
    train_paths, val_paths, test_paths = split_by_cosmology(
        patterns, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac, seed=seed
    )
    train_ds = H5CosmoDataset(
        train_paths, nested_keys, cosmo_params,
        as_torch=as_torch, dtype=dtype, stack_groups=stack_groups
    )
    val_ds = H5CosmoDataset(
        val_paths, nested_keys, cosmo_params,
        as_torch=as_torch, dtype=dtype, stack_groups=stack_groups
    )
    test_ds = H5CosmoDataset(
        test_paths, nested_keys, cosmo_params,
        as_torch=as_torch, dtype=dtype, stack_groups=stack_groups
    )
    return train_ds, val_ds, test_ds


def build_dataloaders(
    patterns: Union[str, Sequence[str]],
    nested_keys: Dict[str, Tuple[str, ...]],
    cosmo_params: List[str],
    *,
    batch_size: int = 4,
    val_batch_size: Optional[int] = None,
    test_batch_size: Optional[int] = None,
    shuffle_train: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: Optional[bool] = None,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    as_torch: bool = True,
    dtype=np.float32,
    stack_groups: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Return DataLoaders for train/val/test ensuring no cosmology leakage.
    """
    train_ds, val_ds, test_ds = build_datasets(
        patterns, nested_keys, cosmo_params,
        train_frac=train_frac, val_frac=val_frac, test_frac=test_frac,
        seed=seed, as_torch=as_torch, dtype=dtype, stack_groups=stack_groups
    )

    if val_batch_size is None:
        val_batch_size = batch_size
    if test_batch_size is None:
        test_batch_size = val_batch_size

    # Default for persistent_workers if not given
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader, test_loader


# Example usage:
# nested_keys = {
#     "E": ("pixelised_results", "E", "north"),
#     "B": ("pixelised_results", "B", "north"),
#     "bandpowers": ("cls_results", "full", "bandpowers"),
#     "bandpower_cls": ("cls_results", "full", "bandpower_ls"),
#     "cls": ("cls_results", "full", "cls"),
# }
# cosmo_params = ["Omega_c", "Omega_b", "n_s", "sigma_8", "h0"]
# train_loader, val_loader, test_loader = build_dataloaders(
#     "/share/gpu5/asaoulis/transfer_datasets/gower_full_only_mocks/output_*.h5",
#     nested_keys,
#     cosmo_params,
#     batch_size=8,
#     as_torch=False,  # or True if you want tensors
# )
