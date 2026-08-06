import re
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch


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


EB_MAP_KEYS: Tuple[str, str, str, str] = (
    "E_north", "E_south", "B_north", "B_south"
)

# E/B pixelised-map quantity names: an E or B prefix, an OPTIONAL smoothing tag
# (e.g. "fwhm8", "fwhm6_lcut1024", "fwhm6_lmin76_lcut1024"), then the patch side. Matches both the
# logical names ("E_north") and explicit variant-tagged names ("E_fwhm6_lmin76_lcut1024_north").
_EB_QUANTITY_RE = re.compile(r"^(E|B)(?:_(.+))?_(north|south)$")


def _resolve_eb_quantity(quantity: str, eb_variant: Optional[str]) -> Optional[Tuple[str, str, str]]:
    """Resolve an E/B map quantity name to its ('pixelised_results', group, side) path.

    - Logical names (no tag, e.g. 'E_north') resolve to group 'E' when ``eb_variant`` is
      None/empty (legacy bare groups), else 'E_{eb_variant}' (the new multi-variant schema).
    - Explicit tagged names (e.g. 'E_fwhm8_north') resolve directly to 'E_fwhm8',
      independent of ``eb_variant`` (lets you load several variants at once).
    Returns None if ``quantity`` is not an E/B map name.
    """
    m = _EB_QUANTITY_RE.match(quantity)
    if m is None:
        return None
    eb, tag, side = m.group(1), m.group(2), m.group(3)
    if tag is None:
        group = eb if not eb_variant else f"{eb}_{eb_variant}"
    else:
        group = f"{eb}_{tag}"
    return ("pixelised_results", group, side)


class RandomEBPatchAugment:
    """Random E/B patch augmentation with flips and 180° rotation."""

    def __init__(self):
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
        present = [k for k in data if _EB_QUANTITY_RE.match(k)]
        if not present:
            return data

        # Group by patch side (north/south) so E and B -- and every smoothing variant -- on
        # the same side receive the SAME random flips, keeping the E/B pair and all variants
        # spatially aligned. (For the legacy bare keys this is identical to the old grouping.)
        by_patch: Dict[str, List[str]] = {}
        for k in present:
            side = "north" if k.endswith("north") else "south"
            by_patch.setdefault(side, []).append(k)

        for patch, keys in by_patch.items():
            first_val = data[keys[0]]
            use_torch = torch.is_tensor(first_val)
            flags = [self._rand_bool(use_torch) for _ in self._augs]
            for k in keys:
                x = data[k]
                for flag, (_, fn) in zip(flags, self._augs):
                    if flag:
                        x = fn(x)
                data[k] = x
        return data


class EBNoiseNormTransform:
    """Per-sample E/B map noise normalisation (shear-estimator hardening, Track B).

    Removes the source-clustering-modulated noise-amplitude channel from the map branch
    (the b_g leakage measured in .claude/runs/kids-preparation/improved-shear-processing):

    - mode='self'  (B1_selfstd): each E/B map bin is standardised over its own footprint
      ((v - mean)/std on pixels != 0; zeros outside the footprint stay 0). The map branch
      carries no per-mock amplitude; the bandpower branch (noise-debiased) keeps it.

    Mirrors scripts/shear_replay/replay.py map_post 'self_std', validated on the paired
    b_g catalogues (residual < 0.2 sigma vs baseline 7-9 sigma). Stats are per
    (bin, patch-side) here — the loader never sees the joint N+S footprint.
    (A 'divB' mode — divide E by the B-map std — also passed the paired test but is
    deliberately NOT implemented: B-modes are poorly modelled on real data and the main
    analysis must not rely on them; user decision 2026-08-05.)
    """

    def __init__(self, mode: str, eps: float = 1e-12):
        if mode != "self":
            raise ValueError(f"eb_noise_norm mode must be 'self', got {mode!r}")
        self.mode = mode
        self.eps = eps

    @staticmethod
    def _bin_stats(x):
        """Footprint (nonzero-pixel) mean/std per bin for an (nbins, H, W) array/tensor."""
        if torch.is_tensor(x):
            flat = x.reshape(x.shape[0], -1)
            mask = flat != 0
            n = mask.sum(dim=1).clamp(min=1).to(flat.dtype)
            mean = (flat * mask).sum(dim=1) / n
            var = (((flat - mean[:, None]) * mask) ** 2).sum(dim=1) / n
            return mean, torch.sqrt(var)
        flat = np.asarray(x).reshape(x.shape[0], -1)
        mask = flat != 0
        n = np.maximum(mask.sum(axis=1), 1).astype(flat.dtype)
        mean = (flat * mask).sum(axis=1) / n
        var = (((flat - mean[:, None]) * mask) ** 2).sum(axis=1) / n
        return mean, np.sqrt(var)

    def __call__(self, data: Dict[str, Union[np.ndarray, torch.Tensor]]):
        eb_keys = [k for k in data if _EB_QUANTITY_RE.match(k)]
        if not eb_keys:
            return data
        out = dict(data)
        for k in eb_keys:
            x = out[k]
            mask = x != 0
            mean, sd = self._bin_stats(x)
            sd = sd + self.eps
            out[k] = (x - mean[:, None, None]) / sd[:, None, None] * mask
        return out


class ChainedDataTransform:
    """Apply a sequence of data-dict transforms in order (picklable, DataLoader-worker safe)."""

    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


def build_nested_keys_from_quantities(
    quantities: Sequence[str], eb_variant: Optional[str] = None
) -> Dict[str, Tuple[str, ...]]:
    """Convert a list of dataset quantity names into a nested_keys mapping.

    E/B pixelised-map quantities are resolved by pattern (honouring ``eb_variant`` for the
    logical names E_north/E_south/B_north/B_south); all other quantities come from
    DEFAULT_QUANTITY_PATHS.
    """
    nested: Dict[str, Tuple[str, ...]] = {}
    for q in quantities:
        eb_path = _resolve_eb_quantity(q, eb_variant)
        if eb_path is not None:
            nested[q] = eb_path
        elif q in DEFAULT_QUANTITY_PATHS:
            nested[q] = DEFAULT_QUANTITY_PATHS[q]
        else:
            known = ", ".join(sorted(DEFAULT_QUANTITY_PATHS.keys()))
            raise KeyError(
                f"Unknown dataset_quantities: {q!r}. Known options: {known} "
                f"(or E/B map keys like 'E_north' / 'E_<tag>_north')."
            )
    return nested