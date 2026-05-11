from typing import Dict, List, Sequence, Tuple, Union

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
        present = [k for k in EB_MAP_KEYS if k in data]
        if not present:
            return data

        by_patch: Dict[str, List[str]] = {}
        for k in present:
            parts = k.split("_", 1)
            if len(parts) != 2:
                continue
            patch = parts[1]
            by_patch.setdefault(patch, []).append(k)

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


def build_nested_keys_from_quantities(quantities: Sequence[str]) -> Dict[str, Tuple[str, ...]]:
    """Convert a list of dataset quantity names into a nested_keys mapping."""
    nested: Dict[str, Tuple[str, ...]] = {}
    unknown = [q for q in quantities if q not in DEFAULT_QUANTITY_PATHS]
    if unknown:
        known = ", ".join(sorted(DEFAULT_QUANTITY_PATHS.keys()))
        raise KeyError(f"Unknown dataset_quantities: {unknown}. Known options: {known}")
    for q in quantities:
        nested[q] = DEFAULT_QUANTITY_PATHS[q]
    return nested