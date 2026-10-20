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


def _read_for_cache(args):
    path, nested_keys, cosmo_params, dtype, stack_groups, allow_missing = args
    from .data_loading import unpack_data
    try:
        return unpack_data(path, nested_keys, cosmo_params, as_torch=False, dtype=dtype, stack_groups=stack_groups,
                           allow_missing_cosmo=allow_missing)
    except (OSError, KeyError):
        return None


def _dir_cache(directory: str, ds: "H5CosmoDataset", workers: int):
    """(basename->row, keys, {key: (N, ...) array}, cosmo (N, P), ok (N,)) for every output_*.h5 in `directory`."""
    import glob
    import hashlib
    import json
    keys = sorted(ds.nested_keys)
    tag = json.dumps([[k, list(ds.nested_keys[k])] for k in keys] + [list(ds.cosmo_params), np.dtype(ds.dtype).str,
                     bool(ds.stack_groups), bool(ds.allow_missing_cosmo_params)])
    files = sorted(glob.glob(os.path.join(directory, "output_*.h5")))
    if not files:
        return None
    key = hashlib.sha1((tag + "|" + str(len(files)) + "|" + os.path.basename(files[0]) + "|"
                        + os.path.basename(files[-1])).encode()).hexdigest()[:16]
    cdir = os.path.join(directory, ".preload_cache")
    cpath = os.path.join(cdir, f"{key}.npz")
    if not os.path.exists(cpath):
        from concurrent.futures import ProcessPoolExecutor
        args = [(f, ds.nested_keys, ds.cosmo_params, ds.dtype, ds.stack_groups, ds.allow_missing_cosmo_params)
                for f in files]
        with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
            items = list(ex.map(_read_for_cache, args, chunksize=256))
        ref = next(it for it in items if it is not None)
        ok = np.array([it is not None for it in items])
        arrays = {k: np.stack([it[0][k] if it is not None else np.zeros_like(ref[0][k]) for it in items]) for k in keys}
        cosmo = np.stack([it[1] if it is not None else np.zeros_like(ref[1]) for it in items])
        os.makedirs(cdir, exist_ok=True)
        tmp = cpath + f".tmp{os.getpid()}.npz"
        np.savez(tmp, names=np.array([os.path.basename(f) for f in files]), ok=ok, cosmo=cosmo, tag=tag,
                 **{f"k_{i}": arrays[k] for i, k in enumerate(keys)})
        os.replace(tmp, cpath)
        print(f"[H5CosmoDataset] wrote dir cache {cpath} ({len(files)} files, {int(ok.sum())} ok)")
    d = np.load(cpath, allow_pickle=False)
    if str(d["tag"]) != tag:
        return None
    arrays = {k: d[f"k_{i}"] for i, k in enumerate(keys)}
    return {n: i for i, n in enumerate(d["names"])}, keys, arrays, d["cosmo"], d["ok"]


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
        side_input_fn=None,
        preload: bool = False,
        preload_workers: int = 8,
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
        # Opt-in: per-PATH extra conditioning vector stored under data["side_input"] (e.g. the true
        # cosmology for an oracle-conditioned b_g encoder; see src/ml/data/side_inputs.py). Default
        # None leaves every existing item byte-identical.
        self.side_input_fn = side_input_fn
        # Opt-in in-memory cache of the raw (pre-transform) items, for small per-mock vectors (e.g. HOS sidecars)
        # where an epoch would otherwise be ~10^5 HDF5 opens for a tiny model. Default OFF: unchanged behaviour.
        # Filled eagerly with a thread pool; unreadable files are not cached and keep the skip-bad-file policy.
        self._mem: Dict[int, tuple] = {}
        if preload:
            self._preload(int(preload_workers))

    def _read(self, path):
        from .data_loading import unpack_data
        return unpack_data(path, self.nested_keys, self.cosmo_params, as_torch=False, dtype=self.dtype,
                           stack_groups=self.stack_groups, allow_missing_cosmo=self.allow_missing_cosmo_params)

    def _preload(self, workers: int):
        """Fill the RAM cache. When every path lives in ONE directory, go through a directory-level on-disk cache
        (`<dir>/.preload_cache/<key>.npz`: every output_*.h5 of the dir, read once with the same `_read` in a process
        pool -- h5py serialises threads on the GIL -- and written atomically), so repeated runs on the same store
        (sweeps, ensemble members, train/val/test of one run) skip ~10^5 HDF5 opens. Otherwise read directly."""
        dirs = {os.path.dirname(os.path.abspath(p)) for p in self.paths}
        cache = _dir_cache(dirs.pop(), self, workers) if len(dirs) == 1 else None
        if cache is not None:
            index, keys, arrays, cosmo, ok = cache
            for j, p in enumerate(self.paths):
                i = index.get(os.path.basename(p))
                if i is not None and ok[i]:
                    self._mem[j] = ({k: arrays[k][i] for k in keys}, cosmo[i])
        else:
            for j, p in enumerate(self.paths):
                try:
                    self._mem[j] = self._read(p)
                except (OSError, KeyError):
                    pass
        print(f"[H5CosmoDataset] preloaded {len(self._mem)}/{len(self.paths)} items"
              f"{' (dir cache)' if cache is not None else ''}")

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
            if j in self._mem:
                import torch
                data, cosmo = self._mem[j]
                if self.as_torch:
                    data = {k: torch.from_numpy(np.array(v, copy=True)) for k, v in data.items()}
                    cosmo = torch.from_numpy(np.array(cosmo, copy=True))
                else:
                    data = {k: np.array(v, copy=True) for k, v in data.items()}
                    cosmo = np.array(cosmo, copy=True)
                if self.transform is not None:
                    data = self.transform(data)
                if self.side_input_fn is not None:
                    data["side_input"] = self.side_input_fn(path)
                if self.return_cosmo_id:
                    return data, cosmo, extract_cosmo_index(path)
                return data, cosmo
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
                if self.side_input_fn is not None:
                    # From the file ACTUALLY returned (post skip-bad-file fallback), never idx.
                    data["side_input"] = self.side_input_fn(path)
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
    side_input_fn=None,
    preload: bool = False,
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
        return_cosmo_id=return_cosmo_id, side_input_fn=side_input_fn, preload=preload,
    )
    val_ds = H5CosmoDataset(
        val_paths, nested_keys, cosmo_params,
        as_torch=as_torch, dtype=dtype, stack_groups=stack_groups, transform=None,
        side_input_fn=side_input_fn, preload=preload,
    )
    test_ds = H5CosmoDataset(
        test_paths, nested_keys, cosmo_params,
        as_torch=as_torch, dtype=dtype, stack_groups=stack_groups, transform=None,
        side_input_fn=side_input_fn, preload=preload,
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
    augment_eb_ra_roll: bool = False,
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
    side_input_fn=None,
    preload: bool = False,
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

    transform = RandomEBPatchAugment(ra_roll=augment_eb_ra_roll) if augment_eb_patches else None

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
        side_input_fn=side_input_fn,
        preload=preload,
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