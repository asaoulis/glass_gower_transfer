"""Map quantity names, paths and the train-time transforms of the E/B patches.

DEFAULT_QUANTITY_PATHS maps a dataset quantity to its HDF5 path. RandomEBPatchAugment flips and
rotates patches, EBNoiseNormTransform standardises each E/B map per bin, and
build_nested_keys_from_quantities builds the loader keys.
"""
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
    # HOS sidecars (src/ml/data/hos.py): R1 third-moment vector (112) and the (4, 77) moment superset
    "hos": ("hos", "vector"),
    "hos_x": ("hos_x", "vector"),
    "hos_ext": ("hos_ext", "moments"),
}


EB_MAP_KEYS: Tuple[str, str, str, str] = (
    "E_north", "E_south", "B_north", "B_south"
)

# E/B map quantity names: E or B, an optional smoothing tag, then the patch side. Matches both
# "E_north" and "E_fwhm6_lmin76_lcut1024_north".
_EB_QUANTITY_RE = re.compile(r"^(E|B)(?:_(.+))?_(north|south)$")


def _resolve_eb_quantity(quantity: str, eb_variant: Optional[str]) -> Optional[Tuple[str, str, str]]:
    """Resolve an E/B map quantity name to its ('pixelised_results', group, side) path.

    An untagged name maps to group 'E' (eb_variant empty) or 'E_{eb_variant}'. A tagged name
    such as 'E_fwhm8_north' maps to 'E_fwhm8' whatever eb_variant is. Returns None for a
    non-E/B quantity.
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
    """Random E/B patch augmentation with flips, 180° rotation and (opt-in) RA cyclic roll."""

    def __init__(self, ra_roll: bool = False):
        self._augs = [
            ("vflip", self._vflip),
            ("hflip", self._hflip),
            ("rot180", self._rot180),
        ]
        # Cyclic RA roll along the last axis: valid only for a patch spanning the full 360 deg of RA.
        self.ra_roll = bool(ra_roll)

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
    def _roll(x, shift: int):
        if torch.is_tensor(x):
            return torch.roll(x, shifts=int(shift), dims=-1)
        return np.roll(x, int(shift), axis=-1)

    @staticmethod
    def _rand_bool(use_torch: bool) -> bool:
        if use_torch:
            return bool(torch.randint(0, 2, ()).item())
        return bool(np.random.randint(0, 2))

    @staticmethod
    def _rand_shift(width: int, use_torch: bool) -> int:
        if width <= 1:
            return 0
        if use_torch:
            return int(torch.randint(0, int(width), ()).item())
        return int(np.random.randint(0, int(width)))

    def __call__(self, data: Dict[str, Union[np.ndarray, torch.Tensor]]):
        present = [k for k in data if _EB_QUANTITY_RE.match(k)]
        if not present:
            return data

        by_patch: Dict[str, List[str]] = {}
        for k in present:
            side = "north" if k.endswith("north") else "south"
            by_patch.setdefault(side, []).append(k)

        # One RA shift per sample, applied to both sides: they are halves of one band.
        ra_shift = None
        if self.ra_roll:
            first_any = data[present[0]]
            ra_shift = self._rand_shift(first_any.shape[-1], torch.is_tensor(first_any))

        for patch, keys in by_patch.items():
            first_val = data[keys[0]]
            use_torch = torch.is_tensor(first_val)
            flags = [self._rand_bool(use_torch) for _ in self._augs]
            for k in keys:
                x = data[k]
                for flag, (_, fn) in zip(flags, self._augs):
                    if flag:
                        x = fn(x)
                if ra_shift:
                    x = self._roll(x, ra_shift)
                data[k] = x
        return data


class EBNoiseNormTransform:
    """Per-sample E/B map noise normalisation.

    mode='self' standardises each map bin over its own footprint: (v - mean) / std on the
    nonzero pixels, zeros outside stay 0. Stats are per (bin, patch side). Mirrors map_post
    'self_std' in scripts/shear_replay/replay.py.
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

    E/B map quantities resolve by pattern, honouring eb_variant for the untagged names. All
    other quantities come from DEFAULT_QUANTITY_PATHS.
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