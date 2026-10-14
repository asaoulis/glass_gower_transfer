"""Optional extra conditioning vectors for an encoder's density head (config key ``side_input``).

The summary the flow conditions on becomes ``[encoder(x), side]``. Two sources:

* ``{"type": "true_theta", "params": [...], "preset_overrides": {...}}`` -- the TRUE values of
  ``params`` read from each file's ``cosmo_dict``, min/max-scaled with the same preset boxes as the
  cosmo scaler. An ORACLE: it needs the truth, so such a model only exists to measure a ceiling
  (e.g. "how well can b_g be read off the maps if cosmology were known exactly?").
* ``{"type": "frozen_encoder", "ckpt_path": <experiment folder>}`` -- the summary of a separately
  trained, FROZEN encoder of the same architecture, run on the same (augmented) batch. Built on the
  model side (``src/ml/models/side_input.py``); the dataset is untouched.

⚠️ A model with a side input emits ``[z, side]`` from its embedding net, so ``compute_embeddings``
would cache the side vector too. Never use such a row as an NLE / embeddings SOURCE.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch

SIDE_INPUT_KEY = "side_input"


def side_input_config(config) -> Optional[dict]:
    side = getattr(config, "side_input", None)
    if not side:
        return None
    side = dict(side.to_dict() if hasattr(side, "to_dict") else side)
    if side.get("type") not in ("true_theta", "frozen_encoder"):
        raise ValueError(f"side_input.type must be 'true_theta' or 'frozen_encoder', got {side.get('type')!r}")
    return side


class TrueThetaSideInput:
    """Per-path callable: scaled true parameters as a float32 tensor (picklable for loader workers)."""

    def __init__(self, params: Sequence[str], preset: Dict[str, Tuple[float, float]]):
        from ..utils import _build_cosmo_preset_scaler

        self.params = list(params)
        self.scaler = _build_cosmo_preset_scaler(preset, self.params)

    def __call__(self, path: str) -> torch.Tensor:
        from .data_loading import load_cosmo_params

        theta, _ = load_cosmo_params(path, self.params, as_torch=True, dtype="float32")
        return self.scaler.transform(theta).to(torch.float32)


def build_side_input_fn(config):
    """Dataset-side provider for ``config.side_input`` (None unless type == 'true_theta')."""
    side = side_input_config(config)
    if side is None or side["type"] != "true_theta":
        return None
    from .constants import COSMO_PARAM_PRESET_MINMAX

    preset = dict(COSMO_PARAM_PRESET_MINMAX)
    preset.update({k: tuple(v) for k, v in (side.get("preset_overrides") or {}).items()})
    return TrueThetaSideInput(side["params"], preset)
