import glob
import os
import random
import re
from typing import Dict, List, Optional, Sequence, Tuple, Literal, Union

import numpy as np


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
    from .data_loading import load_cosmo_params

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
    """Greedy k-center (farthest-point) selection."""
    if x.ndim != 2:
        raise ValueError(f"Expected x to be 2D (n,d), got shape {x.shape}")
    n = int(x.shape[0])
    if k <= 0:
        raise ValueError("k must be positive")
    if k > n:
        raise ValueError(f"k={k} cannot exceed number of points n={n}")
    if k == n:
        return list(range(n))

    centroid = x.mean(axis=0)
    d2_centroid = np.sum((x - centroid) ** 2, axis=1)
    first = int(np.argmax(d2_centroid))
    selected = [first]

    min_d2 = np.sum((x - x[first]) ** 2, axis=1)
    min_d2[first] = -np.inf

    for _ in range(1, k):
        nxt = int(np.argmax(min_d2))
        selected.append(nxt)
        d2_new = np.sum((x - x[nxt]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2_new)
        min_d2[nxt] = -np.inf
    return selected


def _assign_quantile_bins_1d(x: np.ndarray, n_bins: int) -> Tuple[np.ndarray, int]:
    """Assign values in x to quantile bins."""
    if n_bins <= 1:
        return np.zeros_like(x, dtype=int), 1

    q = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges = np.quantile(x, q)
    edges = np.unique(edges)

    if edges.size <= 2:
        return np.zeros_like(x, dtype=int), 1

    internal = edges[1:-1]
    idx = np.digitize(x, internal, right=False).astype(int)
    n_eff = int(edges.size - 1)
    return idx, n_eff


def _apportion_proportional(k: int, sizes: Dict[Tuple[int, ...], int]) -> Dict[Tuple[int, ...], int]:
    """Allocate integer samples per stratum proportional to stratum size."""
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

    remainders = {h: raw[h] - np.floor(raw[h]) for h in sizes}
    order = sorted(sizes.keys(), key=lambda h: (-remainders[h], h))
    for h in order:
        if remaining <= 0:
            break
        if alloc[h] < sizes[h]:
            alloc[h] += 1
            remaining -= 1

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
    """Resolve a stratified bin spec to an integer bin count."""
    if n_dims <= 0:
        raise ValueError("n_dims must be positive")

    if stratified_bins == "auto":
        bins = int(round(float(k) ** (1.0 / float(n_dims))))
    else:
        bins = int(stratified_bins)

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
    """Select a subset of cosmology ids from candidates."""
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
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std > 0, std, 1.0)
        xz = (x - mean) / std
        sel_idx = _kcenter_greedy_indices(xz, k)
        selected_set = set(candidate_cosmos[i] for i in sel_idx)
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

            mix = base_seed & 0xFFFFFFFF
            for v in h:
                mix = (mix * 1000003 + int(v) + 1) & 0xFFFFFFFF
            rng_h = random.Random(mix)
            rng_h.shuffle(items)
            selected_set.update(items[:k_h])

        return [c for c in candidate_cosmos if c in selected_set]

    if strategy not in {"random", "kcenter", "stratified"}:
        raise ValueError(f"Unknown selection strategy: {strategy}")

    raise ValueError(f"Unhandled selection strategy: {strategy}")


def extract_cosmo_index(path: str) -> int:
    """Extract cosmology index from a filename like .../output_<cosmo_id>_<stuff>_<sample>.h5"""
    base = os.path.basename(path)
    match = re.search(r"output_(\d+)_", base)
    if match is None:
        raise ValueError(f"Could not extract cosmology index from path: {path}")
    return int(match.group(1))


def collect_paths(patterns: Union[str, Sequence[str]]) -> List[str]:
    """Expand one or multiple glob patterns into a sorted list of files."""
    if isinstance(patterns, str):
        patterns = [patterns]
    paths: List[str] = []
    for p in patterns:
        paths.extend(glob.glob(p))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"No files matched the provided pattern(s): {patterns}")
    return paths


def _filter_paths_by_shape_noise_idx(paths: List[str], test_shape_noise_idx: Optional[Sequence[int]]) -> List[str]:
    """Optionally filter test file paths by suffix indices."""
    if not test_shape_noise_idx:
        return paths

    if len(test_shape_noise_idx) == 2:
        rot_idx, shape_idx = (int(v) for v in test_shape_noise_idx)
        rot_shape_pat = re.compile(r"out(\d+)_rot(\d+)_(\d+)\.h5$")

        filtered: List[str] = []
        for p in paths:
            base = os.path.basename(p)
            m = rot_shape_pat.search(base)
            if m is None:
                continue
            rot_v, shape_v = int(m.group(2)), int(m.group(3))
            if rot_v == rot_idx and shape_v == shape_idx:
                filtered.append(p)
        return filtered

    if len(test_shape_noise_idx) == 3:
        out_idx, rot_idx, shape_idx = (int(v) for v in test_shape_noise_idx)
        strict_pat = re.compile(r"out(\d+)_rot(\d+)_(\d+)\.h5$")

        filtered: List[str] = []
        for p in paths:
            base = os.path.basename(p)
            m = strict_pat.search(base)
            if m is None:
                continue
            out_v, rot_v, shape_v = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            if out_v == out_idx and rot_v == rot_idx and shape_v == shape_idx:
                filtered.append(p)
        return filtered

    allowed = set(int(i) for i in test_shape_noise_idx)
    filtered: List[str] = []
    for p in paths:
        base = os.path.basename(p)
        m = re.search(r"(\d+)(?=\.h5$)", base)
        if m is None:
            filtered.append(p)
            continue
        idx = int(m.group(1))
        if idx in allowed:
            filtered.append(p)
    return filtered


def _resolve_forced_test_cosmos(
    fixed_test_sim_ids: Optional[Union[str, Sequence[int]]],
    by_cosmo: Dict[int, List[str]],
) -> Optional[List[int]]:
    """Resolve the opt-in fixed test set to the on-disk cosmologies to force into test.

    Returns a sorted list of cosmology ids to force into the test split, or ``None`` to
    use the normal split. ``None`` is returned when the feature is off, when none of the
    locked ids are present on disk, or when forcing them would leave train/val empty
    (graceful fallback — keeps small datasets, e.g. the smoke mini-dataset, working).
    """
    if fixed_test_sim_ids is None:
        return None
    from .fixed_test_set import resolve_fixed_test_ids

    requested = resolve_fixed_test_ids(fixed_test_sim_ids)
    if not requested:
        return None

    n_present = len(by_cosmo)
    forced = sorted(set(requested) & set(by_cosmo.keys()))
    if not forced:
        print(
            f"[split_by_cosmology] fixed_test_sim_ids: none of the {len(requested)} locked "
            f"sim_ids are present on disk; falling back to the normal split.",
            flush=True,
        )
        return None
    if n_present - len(forced) < 1:
        print(
            f"[split_by_cosmology] fixed_test_sim_ids: forcing {len(forced)} sim_ids into "
            f"test would leave no train/val cosmologies (of {n_present} on disk); falling "
            f"back to the normal split.",
            flush=True,
        )
        return None
    print(
        f"[split_by_cosmology] fixed_test_sim_ids: forced {len(forced)} cosmologies into test "
        f"(of {len(requested)} requested, {n_present} on disk).",
        flush=True,
    )
    return forced


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
    N_extra_test_cosmologies: Optional[int] = None,
    fixed_test_sim_ids: Optional[Union[str, Sequence[int]]] = None,
    N_test_cosmologies: Optional[int] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """Glob files, group by cosmology index, shuffle cosmologies, and split without leakage."""
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

    for k in by_cosmo:
        by_cosmo[k].sort()

    cosmologies = list(by_cosmo.keys())
    rng = random.Random(42)
    rng.shuffle(cosmologies)

    n_total = len(cosmologies)
    if n_total == 0:
        raise ValueError("No cosmologies found.")

    # Opt-in fixed test set: force the on-disk subset of the locked sim_ids into test and
    # remove them from the trainval pool (else use the normal last-slice test selection).
    forced_test_cosmos = _resolve_forced_test_cosmos(fixed_test_sim_ids, by_cosmo)
    if forced_test_cosmos is not None:
        test_cosmos = set(forced_test_cosmos)
        remaining_cosmos = [c for c in cosmologies if c not in test_cosmos]
    else:
        n_test = int(round(n_total * test_frac))
        n_test = max(1, n_test) if test_frac > 0 else 0
        if n_test > n_total:
            n_test = n_total

        test_cosmos = set(cosmologies[-n_test:]) if n_test > 0 else set()
        remaining_cosmos = cosmologies[:-n_test] if n_test > 0 else cosmologies
    if seed != 42:
        rng.seed(seed)
        rng.shuffle(remaining_cosmos)
    n_remaining = len(remaining_cosmos)

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

    if ensemble_seed is not None:
        rng_ens = random.Random(int(ensemble_seed))
        trainval_cosmos = list(trainval_cosmos)
        rng_ens.shuffle(trainval_cosmos)

    n_trainval = len(trainval_cosmos)
    if n_trainval == 0:
        raise ValueError("No cosmologies left for training/validation after reserving test set.")

    trainval_total_frac = train_frac + val_frac
    if trainval_total_frac <= 0:
        raise ValueError("train_frac + val_frac must be positive.")

    rel_train_frac = train_frac / trainval_total_frac
    n_train = int(n_trainval * rel_train_frac)
    n_val = n_trainval - n_train
    if n_val == 0 and val_frac > 0:
        n_train = max(0, n_train - 1)
        n_val = 1

    train_cosmos = set(trainval_cosmos[:n_train])
    val_cosmos = set(trainval_cosmos[n_train:n_train + n_val])

    # Eval-time sub-selection: trim the TEST set to the first N cosmologies by sorted sim_id,
    # AFTER train/val are fully resolved. Deliberately applied last so the trainval pool — and
    # therefore the fitted scalers — stay byte-identical to training regardless of N. (Contrast
    # with shrinking fixed_test_sim_ids, which would return cosmologies to the trainval pool.)
    if N_test_cosmologies:
        n_keep = int(N_test_cosmologies)
        if n_keep <= 0:
            raise ValueError("N_test_cosmologies must be positive if provided.")
        kept = sorted(test_cosmos)[:n_keep]
        print(
            f"[split_by_cosmology] N_test_cosmologies={n_keep}: trimmed test set "
            f"{len(test_cosmos)} -> {len(kept)} cosmologies (first by sorted sim_id).",
            flush=True,
        )
        test_cosmos = set(kept)

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
    if N_extra_test_cosmologies:
        extra_cosmos = set(cosmologies) - train_cosmos - val_cosmos - test_cosmos
        extra_cosmos = sorted(extra_cosmos)
        extra_selected = extra_cosmos[:N_extra_test_cosmologies]
        for c in extra_selected:
            test_paths.extend(by_cosmo[c])
        print(f"Added {len(extra_selected)} extra cosmologies to test set, new test count: {len(test_paths)}")

    test_paths = _filter_paths_by_shape_noise_idx(test_paths, test_shape_noise_idx)
    return train_paths, val_paths, test_paths