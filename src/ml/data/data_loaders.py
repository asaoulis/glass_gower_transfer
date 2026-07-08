import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data_augmentations import RandomEBPatchAugment
from .data_selection import SelectionStrategy, StratifiedBins, split_by_cosmology, extract_cosmo_index

# Track corrupt/unreadable h5 files so we warn once each (not every epoch/worker).
_BAD_PATHS = set()


def _warn_bad_file(path, err):
    if path not in _BAD_PATHS:
        _BAD_PATHS.add(path)
        import warnings
        warnings.warn(f"[dataloader] skipping unreadable file ({type(err).__name__}): {path}")


class H5CosmoDataset(Dataset):
    """Dataset that loads items via unpack_data for given HDF5 paths."""

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
        return_cosmo_id: bool = False,
        allow_missing_cosmo_params: bool = False,
    ):
        self.paths = list(paths)
        self.nested_keys = nested_keys
        self.cosmo_params = cosmo_params
        self.as_torch = as_torch
        self.dtype = dtype
        self.stack_groups = stack_groups
        self.transform = transform
        # Cross-variate eval: NaN-fill cosmo params absent from a dataset instead of treating
        # the file as corrupt (a missing cosmo_dict key would otherwise KeyError-skip EVERY
        # file of an NLA/NLA-z variate). Data-group KeyErrors still skip as before.
        self.allow_missing_cosmo_params = allow_missing_cosmo_params
        # Opt-in: also return the integer cosmology id as a 3rd batch element. Default OFF so every
        # non-VICReg path keeps the unchanged (data_dict, theta) 2-tuple contract. Used by the VICReg
        # train loader to group same-cosmology samples for the invariance term.
        self.return_cosmo_id = return_cosmo_id

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        from .data_loading import unpack_data

        # Resilience for large/partially-generated out-of-core datasets: a corrupt/truncated
        # h5 file (OSError "file signature not found") or one missing the requested E/B variant
        # group (KeyError) must NOT crash training. Skip to the next valid sample (bounded), and
        # warn ONCE per bad path. A handful of bad files in ~10^4 is a negligible sampling change.
        n = len(self.paths)
        last_err = None
        for attempt in range(min(n, 64)):
            j = (idx + attempt) % n
            path = self.paths[j]
            try:
                data, cosmo = unpack_data(
                    path,
                    self.nested_keys,
                    self.cosmo_params,
                    as_torch=self.as_torch,
                    dtype=self.dtype,
                    stack_groups=self.stack_groups,
                    allow_missing_cosmo=self.allow_missing_cosmo_params,
                )
                if self.transform is not None:
                    data = self.transform(data)
                if self.return_cosmo_id:
                    # CRITICAL: derive the id from the file ACTUALLY returned (post skip-bad-file
                    # fallback `path = self.paths[j]`), never the requested idx — so a corrupt-file
                    # swap groups the sample under its true cosmology, not a mislabelled slot.
                    cid = extract_cosmo_index(path)
                    return data, cosmo, cid
                return data, cosmo
            except (OSError, KeyError) as e:
                last_err = e
                _warn_bad_file(path, e)
                continue
        raise RuntimeError(
            f"Could not read a valid sample within 64 of idx {idx} (last error: {last_err})"
        )


def build_nested_keys_from_quantities(quantities: Sequence[str], eb_variant=None):
    from .data_augmentations import build_nested_keys_from_quantities as _build

    return _build(quantities, eb_variant=eb_variant)


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
    max_trainval_cosmos: Optional[int] = None,
    selection_strategy: SelectionStrategy = "random",
    selection_cosmo_params: Optional[List[str]] = None,
    stratified_bins: StratifiedBins = 5,
    test_shape_noise_idx: Optional[Sequence[int]] = None,
    ensemble_seed: Optional[int] = None,
    N_extra_test_cosmologies: Optional[int] = None,
    fixed_test_sim_ids: Optional[Union[str, Sequence[int]]] = None,
    N_test_cosmologies: Optional[int] = None,
    return_cosmo_id: bool = False,
) -> Tuple[H5CosmoDataset, H5CosmoDataset, H5CosmoDataset]:
    train_paths, val_paths, test_paths = split_by_cosmology(
        patterns,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
        max_trainval_cosmos=max_trainval_cosmos,
        selection_strategy=selection_strategy,
        selection_cosmo_params=selection_cosmo_params,
        stratified_bins=stratified_bins,
        test_shape_noise_idx=test_shape_noise_idx,
        ensemble_seed=ensemble_seed,
        N_extra_test_cosmologies=N_extra_test_cosmologies,
        fixed_test_sim_ids=fixed_test_sim_ids,
        N_test_cosmologies=N_test_cosmologies,
    )
    train_ds = H5CosmoDataset(
        train_paths, nested_keys, cosmo_params,
        as_torch=as_torch, dtype=dtype, stack_groups=stack_groups, transform=transform,
        return_cosmo_id=return_cosmo_id,
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
    num_workers: Optional[int] = None,
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
    max_trainval_cosmos: Optional[int] = None,
    selection_strategy: SelectionStrategy = "random",
    selection_cosmo_params: Optional[List[str]] = None,
    stratified_bins: StratifiedBins = 5,
    test_shape_noise_idx: Optional[Sequence[int]] = None,
    ensemble_seed: Optional[int] = None,
    N_extra_test_cosmologies: Optional[int] = None,
    fixed_test_sim_ids: Optional[Union[str, Sequence[int]]] = None,
    N_test_cosmologies: Optional[int] = None,
    use_cosmo_batch_sampler: bool = False,
    m_per_cosmo: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return DataLoaders for train/val/test ensuring no cosmology leakage.

    When ``use_cosmo_batch_sampler`` (VICReg only) the TRAIN loader uses an
    :class:`MPerCosmoBatchSampler` (k distinct cosmologies × ``m_per_cosmo`` realisations per
    batch) and the train dataset returns the per-sample cosmology id as a 3rd batch element. The
    val/test loaders are unchanged (plain random/sequential, 2-tuple batches)."""
    if num_workers is None:
        try:
            cpu_count = os.cpu_count() or 1
        except Exception:
            cpu_count = 1
        num_workers = max(0, min(8, max(1, cpu_count - 1)))

    transform = RandomEBPatchAugment() if augment_eb_patches else None

    train_ds, val_ds, test_ds = build_datasets(
        patterns,
        nested_keys,
        cosmo_params,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
        as_torch=as_torch,
        dtype=dtype,
        stack_groups=stack_groups,
        transform=transform,
        max_trainval_cosmos=max_trainval_cosmos,
        selection_strategy=selection_strategy,
        selection_cosmo_params=selection_cosmo_params,
        stratified_bins=stratified_bins,
        test_shape_noise_idx=test_shape_noise_idx,
        ensemble_seed=ensemble_seed,
        N_extra_test_cosmologies=N_extra_test_cosmologies,
        fixed_test_sim_ids=fixed_test_sim_ids,
        N_test_cosmologies=N_test_cosmologies,
        return_cosmo_id=use_cosmo_batch_sampler,
    )

    if val_batch_size is None:
        val_batch_size = batch_size
    if test_batch_size is None:
        test_batch_size = val_batch_size

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    if use_cosmo_batch_sampler:
        # VICReg: m-per-cosmology batches. batch_sampler is mutually exclusive with
        # batch_size/shuffle/drop_last (they are encoded by the sampler itself).
        from .cosmo_batch_sampler import MPerCosmoBatchSampler

        train_batch_sampler = MPerCosmoBatchSampler(
            train_ds.paths,
            m_per_cosmo,
            batch_size,
            shuffle=shuffle_train,
            seed=seed,
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
    else:
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