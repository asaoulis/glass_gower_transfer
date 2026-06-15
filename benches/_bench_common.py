"""Shared helpers for the glass model-optimization benches.

Cross-repo note: this benches/ dir lives in glass_gower_transfer; the optimization task is
tracked in seismo-sbi/.claude/runs/architectures/model-optimization/. Run everything with the
glass interpreter:  /data/alex/glass/env/bin/python benches/<script>.py ...
"""
from __future__ import annotations

import sys
import statistics
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SMOKE_GLOB = str(REPO_ROOT / ".claude" / "cluster" / "smoke_data" / "*.h5")
REAL_GLOB = "/data/alex/glass_mocks/*.h5"

# Pretrained-checkpoint keys to null so the bench never reaches for cluster weights
_PRETRAINED_KEYS = (
    "checkpoint_path", "pretrained_band_ckpt_path", "pretrained_backbone_ckpt_path",
    "pretrained_embedding_ckpt_path", "pretrained_flow_ckpt_path",
    "pretrained_npe_flow_ckpt_path", "pretrained_nle_flow_ckpt_path",
)


def build_bench_config(experiment, data_patterns, batch_size, *, eb_variant=None,
                       num_workers=0, pin_memory=False, max_cosmos=None,
                       augment=True):
    """Build the experiment config like train.py, but bench-shaped (no cluster ckpts)."""
    from config.default import get_default_config
    from config.experiments import experiments
    from config.ablations import ablation_experiments

    experiments.update(ablation_experiments)
    if experiment not in experiments:
        raise KeyError(f"experiment '{experiment}' not in experiments/ablations")
    exp = experiments[experiment]
    cfg = get_default_config()
    cfg.experiment_name = experiment
    for k, v in exp.items():
        if k == "max_trainval_cosmos":
            continue
        setattr(cfg, k, v)

    cfg.data_patterns = data_patterns
    cfg.batch_size = batch_size
    cfg.val_batch_size = batch_size
    cfg.test_batch_size = batch_size
    cfg.num_workers = num_workers
    cfg.pin_memory = pin_memory
    cfg.persistent_workers = (num_workers > 0)
    cfg.repeats = 1
    cfg.repeat_indices = None
    cfg.ensemble_repeats = 1
    cfg.match_string = ""
    cfg.max_trainval_cosmos = max_cosmos
    cfg.augment_eb_patches = augment
    if eb_variant is not None:
        cfg.eb_map_variant = eb_variant
    for k in _PRETRAINED_KEYS:
        if hasattr(cfg, k):
            setattr(cfg, k, None)
    cfg.load_pretrained_flow = False
    cfg.freeze_band = False
    return cfg


def tile_batch(batch, target_B, device):
    """Take a (data_dict, theta) batch and tile/truncate to exactly target_B on device."""
    import torch
    data_dict, theta = batch

    def _fit(t):
        n = t.shape[0]
        if n < target_B:
            reps = (target_B + n - 1) // n
            t = t.repeat((reps,) + (1,) * (t.ndim - 1))
        return t[:target_B].contiguous().to(device)

    data_dict = {k: _fit(v) for k, v in data_dict.items()}
    theta = _fit(theta)
    return data_dict, theta


def timed(fn, iters, warmup, sync=True):
    """Median/min/p90 ms over `iters` calls of fn() after `warmup` warmups (CUDA-synced)."""
    import torch
    for _ in range(warmup):
        fn()
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return {
        "median_ms": statistics.median(ts),
        "min_ms": ts[0],
        "p90_ms": ts[min(len(ts) - 1, int(0.9 * len(ts)))],
        "n": iters,
    }
