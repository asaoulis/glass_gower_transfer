#!/usr/bin/env python
"""Dataloader throughput bench for the glass training pipeline.

Times iterating the REAL train DataLoader (scaling + RandomEBPatchAugment in the workers),
so a storage-format / dtype / worker-count change can be A/B'd by samples/s. Compare stores
by pointing --data-patterns at different pre-baked dirs (f32 / f16 / compressed).

  /data/alex/glass/env/bin/python benches/bench_dataloader.py \
      --data-patterns '/data/alex/glass_mocks_f32/*.h5' --workers 8 --pin-memory \
      --prefetch 4 --batches 40

NB locally the whole corpus fits page cache => this is a WARM-cache proxy; the true cold
out-of-core wall is cluster-only (see plan Phase 1.5).
"""
from __future__ import annotations

import argparse
import time

from _bench_common import build_bench_config, SMOKE_GLOB


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-e", "--experiment", default="ablation_glass_no_side")
    ap.add_argument("-B", "--batch-size", type=int, default=100)
    ap.add_argument("--data-patterns", default="/data/alex/glass_mocks_f32/*.h5")
    ap.add_argument("--eb-variant", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pin-memory", action="store_true")
    ap.add_argument("--prefetch", type=int, default=None, help="prefetch_factor (workers>0)")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--batches", type=int, default=40, help="timed batches")
    ap.add_argument("--warmup-batches", type=int, default=4)
    ap.add_argument("--epochs-iter", type=int, default=1, help="passes over the loader")
    args = ap.parse_args()

    import torch
    from src.ml.utils import prepare_data_parameters

    cfg = build_bench_config(args.experiment, args.data_patterns, args.batch_size,
                             eb_variant=args.eb_variant, num_workers=args.workers,
                             pin_memory=args.pin_memory, augment=not args.no_augment)
    if args.prefetch is not None:
        cfg.prefetch_factor = args.prefetch

    t_build = time.perf_counter()
    _scalers, train_loader, _val, _test = prepare_data_parameters(cfg)
    build_s = time.perf_counter() - t_build
    n_train = len(train_loader.dataset)

    # time iterating batches (warm a few first)
    it = iter(train_loader)
    for _ in range(args.warmup_batches):
        try:
            next(it)
        except StopIteration:
            it = iter(train_loader)
            next(it)

    n_done = 0
    n_samples = 0
    t0 = time.perf_counter()
    for _pass in range(args.epochs_iter):
        for batch in train_loader:
            data_dict, theta = batch
            n_samples += theta.shape[0]
            n_done += 1
            if n_done >= args.batches:
                break
        if n_done >= args.batches:
            break
    dt = time.perf_counter() - t0
    smp_s = n_samples / dt
    ms_batch = dt / n_done * 1e3

    store = args.data_patterns.split("/")[-2] if "/" in args.data_patterns else args.data_patterns
    print(f"\n=== dataloader  store={store}  w={args.workers}  pin={args.pin_memory}  "
          f"prefetch={args.prefetch}  aug={not args.no_augment} ===")
    print(f"  build (scaler-fit) {build_s:6.1f} s   train_ds={n_train}")
    print(f"  {n_done} batches / {n_samples} samples in {dt:.2f} s")
    print(f"  throughput   {smp_s:8.1f} smp/s   ({ms_batch:.1f} ms/batch, B={args.batch_size})")


if __name__ == "__main__":
    main()
