from __future__ import annotations

from collections.abc import Mapping
from typing import Dict

import torch
import torch.nn as nn


def load_partial_weights(
    target_module: nn.Module,
    source_state_dict: Dict[str, torch.Tensor],
    prefix: str = "",
    freeze: bool = False,
    verbose: bool = True,
    error_on_mismatch: bool = False,
):
    """Safely load a subset of weights into ``target_module``.

    - Optionally strips a prefix from keys in ``source_state_dict``.
    - Only loads keys that exist in ``target_module.state_dict()`` and whose
      shapes match.
    - Reports unused / missing / mismatched keys.
    - Can raise if weights are not fully consumed.
    """

    target_state = target_module.state_dict()

    # torch.compile inserts an `._orig_mod.` segment into a wrapped submodule's state_dict keys
    # (e.g. ...shared_cnn.backbone._orig_mod.patch_embed.weight). The compile decision can differ
    # between the SAVED source (e.g. a compile-trained hybrid encoder) and the as-yet-UNcompiled
    # target at load time, so a literal key comparison silently drops the compiled submodule's
    # weights. Match each source key to the target's ACTUAL key on the `._orig_mod.`-stripped
    # ("canonical") name — per-key and works whichever side is compiled (mirrors the alignment in
    # npe.py:load_from_checkpoint).
    def _canonical(key: str) -> str:
        return key.replace("._orig_mod.", ".")

    canonical_to_target = {_canonical(tk): tk for tk in target_state.keys()}

    loaded_weights: Dict[str, torch.Tensor] = {}
    used_source_keys: set[str] = set()
    skipped_shape: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    skipped_missing: list[str] = []

    for k, v in source_state_dict.items():
        if prefix:
            if not k.startswith(prefix):
                continue
            local_key = k[len(prefix) :]
        else:
            local_key = k

        target_key = canonical_to_target.get(_canonical(local_key))
        if target_key is None:
            skipped_missing.append(local_key)
            continue

        if v.shape != target_state[target_key].shape:
            skipped_shape.append(
                (target_key, tuple(v.shape), tuple(target_state[target_key].shape))
            )
            continue

        loaded_weights[target_key] = v
        used_source_keys.add(k)

    missing, unexpected = target_module.load_state_dict(loaded_weights, strict=False)

    unused_source = set(source_state_dict.keys()) - used_source_keys

    if verbose:
        print(f"[load_partial_weights] {target_module.__class__.__name__}")
        print(f"  Loaded keys: {len(loaded_weights)}")
        print(f"  Missing in target after load: {len(missing)}")
        print(f"  Unexpected during load: {len(unexpected)}")
        print(f"  Unused source keys: {len(unused_source)}")
        print(f"  Shape mismatches: {len(skipped_shape)}")
        print(f"  Missing target keys: {len(skipped_missing)}")

        if len(loaded_weights) == 0:
            print("⚠️  WARNING: No weights were loaded!")

        if skipped_shape:
            print("⚠️  Shape mismatches:")
            for kk, s1, s2 in skipped_shape[:10]:
                print(f"   {kk}: {s1} vs {s2}")

        if unused_source and prefix:
            print("⚠️  Some source keys not used — prefix may be wrong.")

    if error_on_mismatch:
        problems = (
            len(unused_source)
            + len(skipped_shape)
            + len(skipped_missing)
            + len(missing)
        )
        if problems > 0:
            raise RuntimeError(
                "load_partial_weights mismatch detected: "
                f"{len(loaded_weights)} loaded, "
                f"{len(unused_source)} unused, "
                f"{len(skipped_shape)} shape mismatch, "
                f"{len(missing)} missing"
            )

    if freeze:
        for p in target_module.parameters():
            p.requires_grad = False

    if hasattr(target_module, "only_return_mu"):
        target_module.only_return_mu = True

    # Return diagnostics so callers can convert a *silent* partial load into a hard failure when
    # correctness demands it (e.g. whitened-embedding warm starts, where a shape mismatch would
    # otherwise leave flow layers randomly initialised with a finite loss). `unused_source` is
    # intentionally excluded from the "hard" fields below because the source is usually a whole-model
    # state_dict and only a prefixed submodule is targeted.
    return {
        "loaded": len(loaded_weights),
        "missing": list(missing),
        "unexpected": list(unexpected),
        "unused_source": list(unused_source),
        "skipped_shape": skipped_shape,
        "skipped_missing": skipped_missing,
    }


class ConditionDict(dict):
    """Dict-like that additionally exposes a ``.shape`` property.

    Some downstream sbi utilities expect ``x.shape[0]`` to yield the batch
    dimension. When conditions are passed as a mapping, this wrapper makes the
    mapping look array-like by delegating shape to the first value.
    """

    def __init__(self, data: Mapping):
        if not isinstance(data, Mapping):
            raise TypeError("ConditionDict expects a mapping.")
        super().__init__(data)

    @property
    def shape(self):
        try:
            first_val = next(iter(self.values()))
        except StopIteration as exc:
            raise ValueError("ConditionDict is empty; cannot infer .shape[0].") from exc
        if not hasattr(first_val, "shape"):
            raise AttributeError("First value has no .shape attribute.")
        return first_val.shape

    def copy(self):
        return ConditionDict(self)


def _move_nested_to_device(x, device):
    """Move nested tensors/structures to ``device`` while preserving container type."""

    if hasattr(x, "to"):
        return x.to(device)
    if isinstance(x, Mapping):
        return type(x)({k: _move_nested_to_device(v, device) for k, v in x.items()})
    if isinstance(x, tuple):
        return tuple(_move_nested_to_device(v, device) for v in x)
    if isinstance(x, list):
        return type(x)([_move_nested_to_device(v, device) for v in x])
    return x


class _BatchableTransform:
    """Adapter to make an sbi/torch transform work on leading batch dims."""

    def __init__(self, base):
        self._base = base

    def __getattr__(self, name):
        return getattr(self._base, name)

    def _reshape_call(self, fn, x):
        if hasattr(x, "ndim") and x.ndim > 2:
            leading = x.shape[:-1]
            flat = x.reshape(-1, x.shape[-1])
            y = fn(flat)
            return y.reshape(*leading, y.shape[-1])
        return fn(x)

    def inv(self, y):
        return self._reshape_call(self._base.inv, y)

    def forward(self, x):
        if hasattr(self._base, "forward"):
            return self._reshape_call(self._base.forward, x)
        return self._reshape_call(self._base.__call__, x)

    def __call__(self, x):
        if hasattr(self._base, "__call__"):
            return self._reshape_call(self._base.__call__, x)
        return self.forward(x)
