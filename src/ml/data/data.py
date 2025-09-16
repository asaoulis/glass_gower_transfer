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


# Default mapping from simple quantity names to HDF5 nested key paths
DEFAULT_QUANTITY_PATHS = {
    "E_north": ("pixelised_results", "E", "north"),
    "E_south": ("pixelised_results", "E", "south"),
    "B_north": ("pixelised_results", "B", "north"),
    "B_south": ("pixelised_results", "B", "south"),
    "bandpowers": ("cls_results", "full", "bandpowers"),
    "bandpower_ls": ("cls_results", "full", "bandpower_ls"),
    "cls": ("cls_results", "full", "cls"),
}

# Keys that may contain E/B patches in the data dict
EB_MAP_KEYS: Tuple[str, str, str, str] = (
    "E_north", "E_south", "B_north", "B_south"
)


class RandomEBPatchAugment:
    """
    Random E/B patch augmentation with flips and 180° rotation.
    - Operates only on keys in EB_MAP_KEYS that are present in the data dict.
    - Groups by patch suffix ("north"/"south") so that E and B of the same patch
      receive identical random ops.
    - If only one of E/B exists for a patch, it is augmented alone.
    - Augmentations: vertical flip, horizontal flip, 180° rotation (2^3 combos).
    """
    def __init__(self):
        # Define augmentation functions operating on the last two spatial dims
        self._augs = [
            ("vflip", self._vflip),
            ("hflip", self._hflip),
            ("rot180", self._rot180),
        ]

    @staticmethod
    def _vflip(x):
        if torch.is_tensor(x):
            return torch.flip(x, dims=[-2])
        return np.flip(x, axis=-2)

    @staticmethod
    def _hflip(x):
        if torch.is_tensor(x):
            return torch.flip(x, dims=[-1])
        return np.flip(x, axis=-1)

    @staticmethod
    def _rot180(x):
        if torch.is_tensor(x):
            return torch.rot90(x, k=2, dims=(-2, -1))
        return np.rot90(x, k=2, axes=(-2, -1))

    @staticmethod
    def _rand_bool(use_torch: bool) -> bool:
        if use_torch:
            return bool(torch.randint(0, 2, ()).item())
        return bool(np.random.randint(0, 2))

    def __call__(self, data: Dict[str, Union[np.ndarray, torch.Tensor]]):
        # Collect available E/B keys per patch suffix
        present = [k for k in EB_MAP_KEYS if k in data]
        if not present:
            return data
        by_patch: Dict[str, List[str]] = {}
        for k in present:
            parts = k.split("_", 1)
            if len(parts) != 2:
                continue
            patch = parts[1]  # 'north' or 'south'
            by_patch.setdefault(patch, []).append(k)

        # Apply independent random combo per patch
        for patch, keys in by_patch.items():
            # Determine RNG backend from first tensor in this patch
            first_val = data[keys[0]]
            use_torch = torch.is_tensor(first_val)
            flags = [self._rand_bool(use_torch) for _ in self._augs]
            # Apply selected augs in order to all keys of this patch
            for k in keys:
                x = data[k]
                for flag, (_, fn) in zip(flags, self._augs):
                    if flag:
                        x = fn(x)
                data[k] = x
        return data


def build_nested_keys_from_quantities(quantities: Sequence[str]) -> Dict[str, Tuple[str, ...]]:
    """
    Convert a list of dataset quantity names into the nested_keys mapping
    expected by the H5CosmoDataset/build_dataloaders utilities.

    Known quantities and their default locations are defined in DEFAULT_QUANTITY_PATHS.
    """
    nested: Dict[str, Tuple[str, ...]] = {}
    unknown = [q for q in quantities if q not in DEFAULT_QUANTITY_PATHS]
    if unknown:
        known = ", ".join(sorted(DEFAULT_QUANTITY_PATHS.keys()))
        raise KeyError(f"Unknown dataset_quantities: {unknown}. Known options: {known}")
    for q in quantities:
        nested[q] = DEFAULT_QUANTITY_PATHS[q]
    return nested


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
    transform: Optional[object] = None,
) -> Tuple[H5CosmoDataset, H5CosmoDataset, H5CosmoDataset]:
    """
    Convenience: split by cosmology and return three datasets.
    """
    train_paths, val_paths, test_paths = split_by_cosmology(
        patterns, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac, seed=seed
    )
    train_ds = H5CosmoDataset(
        train_paths, nested_keys, cosmo_params,
        as_torch=as_torch, dtype=dtype, stack_groups=stack_groups, transform=transform
    )
    val_ds = H5CosmoDataset(
        val_paths, nested_keys, cosmo_params,
        as_torch=as_torch, dtype=dtype, stack_groups=stack_groups, transform=None
    )
    test_ds = H5CosmoDataset(
        test_paths, nested_keys, cosmo_params,
        as_torch=as_torch, dtype=dtype, stack_groups=stack_groups, transform=None
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
    augment_eb_patches: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Return DataLoaders for train/val/test ensuring no cosmology leakage.
    Optionally applies random E/B patch augmentations to the training set only.
    """
    transform = RandomEBPatchAugment() if augment_eb_patches else None

    train_ds, val_ds, test_ds = build_datasets(
        patterns, nested_keys, cosmo_params,
        train_frac=train_frac, val_frac=val_frac, test_frac=test_frac,
        seed=seed, as_torch=as_torch, dtype=dtype, stack_groups=stack_groups,
        transform=transform,
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
#     augment_eb_patches=True,
# )
