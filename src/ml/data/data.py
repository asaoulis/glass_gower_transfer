import os
import re
import glob
import random
from typing import Dict, List, Sequence, Tuple, Union, Optional, Literal

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .data_loading import unpack_data, load_cosmo_params


SelectionStrategy = Literal["random", "kcenter", "stratified"]


StratifiedBins = Union[int, Literal["auto"]]


def _load_cosmo_vectors_for_cosmos(
    *,
    by_cosmo: Dict[int, List[str]],
    cosmos: Sequence[int],
    cosmo_params: Optional[List[str]],
    dtype=np.float64,
) -> np.ndarray:
    """Load one cosmology parameter vector per cosmology id.

    Uses the first path for each cosmology as a representative file.
    """
    vectors: List[np.ndarray] = []
    for c in cosmos:
        paths = by_cosmo.get(c)
        if not paths:
            raise KeyError(f"No file paths found for cosmology {c}")
        v, _ = load_cosmo_params(
            paths[0],
            cosmo_params=cosmo_params,
            as_torch=False,
            dtype=dtype,
        )
        vectors.append(np.asarray(v, dtype=dtype))
    return np.stack(vectors, axis=0)


def _kcenter_greedy_indices(x: np.ndarray, k: int) -> List[int]:
    """Greedy k-center (farthest-point) selection.

    Args:
        x: Array of shape (n_points, n_dims).
        k: Number of points to select.

    Returns:
        Indices into x (length k), selected to cover the space.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected x to be 2D (n,d), got shape {x.shape}")
    n = int(x.shape[0])
    if k <= 0:
        raise ValueError("k must be positive")
    if k > n:
        raise ValueError(f"k={k} cannot exceed number of points n={n}")
    if k == n:
        return list(range(n))

    # Start with the point farthest from the centroid.
    centroid = x.mean(axis=0)
    d2_centroid = np.sum((x - centroid) ** 2, axis=1)
    first = int(np.argmax(d2_centroid))
    selected = [first]

    # Maintain distance to nearest selected center for each point.
    min_d2 = np.sum((x - x[first]) ** 2, axis=1)
    min_d2[first] = -np.inf  # never re-select

    for _ in range(1, k):
        nxt = int(np.argmax(min_d2))
        selected.append(nxt)
        d2_new = np.sum((x - x[nxt]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2_new)
        min_d2[nxt] = -np.inf
    return selected


def _assign_quantile_bins_1d(x: np.ndarray, n_bins: int) -> Tuple[np.ndarray, int]:
    """Assign values in x to quantile bins.

    Returns bin indices in [0, n_eff_bins-1] and the effective number of bins.
    If x has too many repeated values, n_eff_bins may be < n_bins.
    """
    if n_bins <= 1:
        return np.zeros_like(x, dtype=int), 1

    q = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges = np.quantile(x, q)
    edges = np.unique(edges)

    # edges length m defines (m-1) bins
    if edges.size <= 2:
        return np.zeros_like(x, dtype=int), 1

    # digitize using internal edges; produces indices 0..(m-2)
    internal = edges[1:-1]
    idx = np.digitize(x, internal, right=False).astype(int)
    n_eff = int(edges.size - 1)
    return idx, n_eff


def _apportion_proportional(k: int, sizes: Dict[Tuple[int, ...], int]) -> Dict[Tuple[int, ...], int]:
    """Allocate integer samples per stratum proportional to stratum size.

    Uses floor + largest remainder; caps at stratum size.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    n_total = int(sum(sizes.values()))
    if n_total <= 0:
        raise ValueError("No candidates to apportion")

    raw = {h: (k * sizes[h]) / n_total for h in sizes}
    alloc = {h: min(sizes[h], int(np.floor(raw[h]))) for h in sizes}
    remaining = k - sum(alloc.values())

    if remaining <= 0:
        return alloc

    # Distribute remaining according to largest remainder among strata with capacity
    remainders = {h: raw[h] - np.floor(raw[h]) for h in sizes}
    # deterministic ordering for ties
    order = sorted(sizes.keys(), key=lambda h: (-remainders[h], h))
    for h in order:
        if remaining <= 0:
            break
        if alloc[h] < sizes[h]:
            alloc[h] += 1
            remaining -= 1

    # If still remaining due to pathological caps, fill arbitrarily but deterministically.
    if remaining > 0:
        order2 = sorted(sizes.keys())
        for h in order2:
            if remaining <= 0:
                break
            cap = sizes[h] - alloc[h]
            if cap <= 0:
                continue
            take = min(cap, remaining)
            alloc[h] += take
            remaining -= take

    return alloc


def _resolve_stratified_bins(stratified_bins: StratifiedBins, *, k: int, n_dims: int) -> int:
    """Resolve a stratified bin spec to an integer bin count.

    Auto mode chooses bins so that the number of joint strata (~bins**n_dims)
    scales sensibly with k, without becoming too sparse at small k.
    """
    if n_dims <= 0:
        raise ValueError("n_dims must be positive")

    if stratified_bins == "auto":
        # Choose bins approximately so that bins**n_dims ~ k, then clamp.
        # For n_dims=3 and k in [10, 100], this yields bins in [2, 5].
        bins = int(round(float(k) ** (1.0 / float(n_dims))))
    else:
        bins = int(stratified_bins)

    # Sensible clamps: must be >=2 to define bins, and avoid too many strata.
    bins = max(2, min(5, bins))
    return bins


def _select_trainval_cosmos(
    *,
    by_cosmo: Dict[int, List[str]],
    candidate_cosmos: Sequence[int],
    k: int,
    strategy: SelectionStrategy,
    selection_cosmo_params: Optional[List[str]],
    stratified_bins: StratifiedBins = 5,
    seed: int = 42,
    dtype=np.float64,
) -> List[int]:
    """Select a subset of cosmology ids from candidates.

    This only chooses WHICH cosmologies to include. It does not decide the
    train/val split; that is handled later based on ordering/reshuffling.
    """
    candidate_cosmos = list(candidate_cosmos)
    if k <= 0:
        raise ValueError("k must be positive")
    if k > len(candidate_cosmos):
        raise ValueError(f"k={k} cannot exceed available candidates={len(candidate_cosmos)}")

    if strategy == "random":
        return candidate_cosmos[:k]

    if strategy == "kcenter":
        x = _load_cosmo_vectors_for_cosmos(
            by_cosmo=by_cosmo,
            cosmos=candidate_cosmos,
            cosmo_params=selection_cosmo_params,
            dtype=dtype,
        )

        # Normalise dimensions to avoid one parameter dominating distances.
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std > 0, std, 1.0)
        xz = (x - mean) / std

        sel_idx = _kcenter_greedy_indices(xz, k)
        selected_set = set(candidate_cosmos[i] for i in sel_idx)

        # Preserve the existing candidate ordering (already seed-shuffled upstream)
        # so train/val splitting stays consistent with prior behavior.
        return [c for c in candidate_cosmos if c in selected_set]

    if strategy == "stratified":
        if selection_cosmo_params is None or len(selection_cosmo_params) == 0:
            raise ValueError(
                "selection_cosmo_params must be provided for stratified selection "
                "(e.g. ['omega_m', 'sigma_8', 'w0'])."
            )
        if len(selection_cosmo_params) > 3:
            raise ValueError(
                f"Stratified selection expects 1-3 params; got {len(selection_cosmo_params)}: {selection_cosmo_params}"
            )
        resolved_bins = _resolve_stratified_bins(stratified_bins, k=k, n_dims=len(selection_cosmo_params))

        x = _load_cosmo_vectors_for_cosmos(
            by_cosmo=by_cosmo,
            cosmos=candidate_cosmos,
            cosmo_params=selection_cosmo_params,
            dtype=dtype,
        )
        if x.ndim != 2 or x.shape[1] != len(selection_cosmo_params):
            raise ValueError(
                f"Unexpected cosmology vector shape {x.shape} for selection_cosmo_params={selection_cosmo_params}"
            )

        # Assign quantile bins per dimension
        per_dim_bins: List[np.ndarray] = []
        for j in range(x.shape[1]):
            idx_j, _ = _assign_quantile_bins_1d(x[:, j], resolved_bins)
            per_dim_bins.append(idx_j)

        strata: Dict[Tuple[int, ...], List[int]] = {}
        for i, c in enumerate(candidate_cosmos):
            key = tuple(int(per_dim_bins[j][i]) for j in range(len(per_dim_bins)))
            strata.setdefault(key, []).append(c)

        sizes = {h: len(v) for h, v in strata.items()}
        alloc = _apportion_proportional(k, sizes)

        selected_set = set()
        base_seed = int(seed)
        for h in sorted(strata.keys()):
            k_h = int(alloc.get(h, 0))
            if k_h <= 0:
                continue
            items = list(strata[h])

            # Deterministic per-stratum shuffle derived from seed and stratum key.
            mix = base_seed & 0xFFFFFFFF
            for v in h:
                mix = (mix * 1000003 + int(v) + 1) & 0xFFFFFFFF
            rng_h = random.Random(mix)
            rng_h.shuffle(items)
            selected_set.update(items[:k_h])

        # Preserve candidate ordering
        return [c for c in candidate_cosmos if c in selected_set]

    if strategy not in {"random", "kcenter", "stratified"}:
        raise ValueError(f"Unknown selection strategy: {strategy}")

    # defensive fallback
    raise ValueError(f"Unhandled selection strategy: {strategy}")


def extract_cosmo_index(path: str) -> int:
    """
    Extract cosmology index from a filename like .../output_<cosmo_id>_<stuff>_<sample>.h5
    """
    # split and select first number
    base = os.path.basename(path)
    match = re.search(r"output_(\d+)_", base)
    if match is None:
        raise ValueError(f"Could not extract cosmology index from path: {path}")
    return int(match.group(1))


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


def _filter_paths_by_shape_noise_idx(paths: List[str], test_shape_noise_idx: Optional[Sequence[int]]) -> List[str]:
    """Optionally filter file paths by shape-noise repeat index.

    Expected filename pattern (before extension) is something like
    ..._SN<idx>.h5 or ..._<idx>.h5; we conservatively extract the final
    integer before the extension and keep only those whose index is in
    ``test_shape_noise_idx``.
    """
    if not test_shape_noise_idx:
        return paths

    allowed = set(int(i) for i in test_shape_noise_idx)
    filtered: List[str] = []
    for p in paths:
        base = os.path.basename(p)
        m = re.search(r"(\d+)(?=\.h5$)", base)
        if m is None:
            # If we cannot parse an index, keep the file to avoid
            # accidentally dropping data.
            filtered.append(p)
            continue
        idx = int(m.group(1))
        if idx in allowed:
            filtered.append(p)
    return filtered


def split_by_cosmology(
    patterns: Union[str, Sequence[str]],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    max_trainval_cosmos: Optional[int] = None,
    test_shape_noise_idx: Optional[Sequence[int]] = None,
    ensemble_seed: Optional[int] = None,
    selection_strategy: SelectionStrategy = "random",
    selection_cosmo_params: Optional[List[str]] = None,
    stratified_bins: StratifiedBins = 5,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Glob files, group by cosmology index, shuffle cosmologies, and split without leakage.

    The test set is defined as the last ``test_frac`` fraction of the cosmologies in
    a stable shuffled order (so that test is fixed across different train/val seeds).

    Optionally, a fixed number of cosmologies for training+validation
    (``max_trainval_cosmos``) can be requested: in that case, exactly that many
    cosmologies (if available) are chosen from the remaining pool. By default this
    selection is random (via shuffling), but can be switched to a space-filling
    selection strategy (e.g. greedy k-center) using ``selection_strategy``.

    Optionally, ``test_shape_noise_idx`` can be used to further restrict which
    samples (e.g. shape-noise repeats) are kept in the *test* split only, based on
    an index parsed from the filename.

    ensemble_seed:
        If provided, reshuffles the *order* of the selected train+val cosmology
        subset after it has been selected (i.e. after applying max_trainval_cosmos).
        This does NOT change which cosmologies are included, only which ones fall
        into train vs val when splitting.
    """
    print(
        f"[split_by_cosmology] seed={seed} ensemble_seed={ensemble_seed} "
        f"max_trainval_cosmos={max_trainval_cosmos} selection_strategy={selection_strategy} stratified_bins={stratified_bins}",
        flush=True,
    )
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
    rng = random.Random(42)
    rng.shuffle(cosmologies)

    n_total = len(cosmologies)
    if n_total == 0:
        raise ValueError("No cosmologies found.")

    # First, define a fixed test set as the last test_frac of all cosmologies
    n_test = int(round(n_total * test_frac))
    n_test = max(1, n_test) if test_frac > 0 else 0
    if n_test > n_total:
        n_test = n_total

    test_cosmos = set(cosmologies[-n_test:]) if n_test > 0 else set()
    remaining_cosmos = cosmologies[:-n_test] if n_test > 0 else cosmologies
    if seed != 42:
        # If a different seed is provided, reshuffle the remaining cosmologies
        rng.seed(seed)
        rng.shuffle(remaining_cosmos)
    n_remaining = len(remaining_cosmos)

    # Optionally limit the total number of training+validation cosmologies
    if max_trainval_cosmos is not None:
        if max_trainval_cosmos <= 0:
            raise ValueError("max_trainval_cosmos must be positive if provided.")
        if max_trainval_cosmos > n_remaining:
            raise ValueError(
                f"Requested max_trainval_cosmos={max_trainval_cosmos} but only "
                f"{n_remaining} cosmologies are available after reserving the test set."
            )
        trainval_cosmos = _select_trainval_cosmos(
            by_cosmo=by_cosmo,
            candidate_cosmos=remaining_cosmos,
            k=max_trainval_cosmos,
            strategy=selection_strategy,
            selection_cosmo_params=selection_cosmo_params,
            stratified_bins=stratified_bins,
            seed=seed,
        )
    else:
        trainval_cosmos = remaining_cosmos

    # Optional ensemble reshuffle AFTER selection
    if ensemble_seed is not None:
        rng_ens = random.Random(int(ensemble_seed))
        trainval_cosmos = list(trainval_cosmos)
        rng_ens.shuffle(trainval_cosmos)

    n_trainval = len(trainval_cosmos)
    if n_trainval == 0:
        raise ValueError("No cosmologies left for training/validation after reserving test set.")

    # Compute train/val counts within the selected train+val pool
    # Use relative fractions renormalised to (train_frac + val_frac)
    trainval_total_frac = train_frac + val_frac
    if trainval_total_frac <= 0:
        raise ValueError("train_frac + val_frac must be positive.")

    rel_train_frac = train_frac / trainval_total_frac
    rel_val_frac = val_frac / trainval_total_frac

    n_train = int(n_trainval * rel_train_frac)
    n_val = n_trainval - n_train  # remainder goes to validation
    # ensure there is at least one cosmology for valdiation
    if n_val == 0 and val_frac > 0:
        n_train = max(0, n_train - 1)
        n_val = 1

    train_cosmos = set(trainval_cosmos[:n_train])
    val_cosmos = set(trainval_cosmos[n_train:n_train + n_val])
    print(f"Total cosmologies: {n_total}, Train: {len(train_cosmos)}, Val: {len(val_cosmos)}, Test: {len(test_cosmos)}")
    train_paths: List[str] = []
    val_paths: List[str] = []
    test_paths: List[str] = []

    for c in train_cosmos:
        train_paths.extend(by_cosmo[c])
    for c in val_cosmos:
        val_paths.extend(by_cosmo[c])
    for c in test_cosmos:
        test_paths.extend(by_cosmo[c])

    # Optionally restrict test paths by shape-noise index
    test_paths = _filter_paths_by_shape_noise_idx(test_paths, test_shape_noise_idx)

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
    "mixed_bandpowers": ("cls_results", "full", "mixed_bandpowers"),
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
    max_trainval_cosmos: Optional[int] = None,
    selection_strategy: SelectionStrategy = "random",
    selection_cosmo_params: Optional[List[str]] = None,
    stratified_bins: int = 5,
    test_shape_noise_idx: Optional[Sequence[int]] = None,
    ensemble_seed: Optional[int] = None,
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
    stratified_bins: int = 5,
    test_shape_noise_idx: Optional[Sequence[int]] = None,
    ensemble_seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Return DataLoaders for train/val/test ensuring no cosmology leakage.
    Optionally applies random E/B patch augmentations to the training set only.

    ``test_shape_noise_idx`` can be used to further filter the test-set files to
    specific shape-noise repeats, while leaving train/val untouched.

    ``num_workers``: if None (default), choose a heuristic based on available
    CPU cores, capped to 8 to avoid oversubscribing small machines.
    """
    # Decide number of workers if not explicitly given
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
