#!/usr/bin/env python
"""End-to-end per-epoch training-loop bench (real dataloader + model + ml_perf from config).

Iterates the REAL train DataLoader (HDF5 read + scaling + augment in workers, H2D copy) and
runs the real train step (model.model.log_prob -> -mean -> backward -> clip -> opt). Reports
per-epoch TRAIN-batches-only walltime + smp/s. Epoch 0 is dropped (compile warmup + cache
warm). ml_perf (amp/compile) is picked up from the experiment config by build_model, so:

  baseline:   -e ablation_glass_no_side            (fp32, B=100)
  optimized:  -e ablation_glass_no_side_fast       (amp+compile, B=200)

  /data/alex/glass/env/bin/python benches/bench_e2e.py -e ablation_glass_no_side_fast \
      --data-patterns '/data/alex/glass_mocks_f32/*.h5' --epochs 4
"""
from __future__ import annotations

import argparse
import statistics
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-e", "--experiment", default="ablation_glass_no_side")
    ap.add_argument("--data-patterns", default="/data/alex/glass_mocks_f32/*.h5")
    ap.add_argument("--eb-variant", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pin-memory", action="store_true", default=True)
    ap.add_argument("--prefetch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=None, help="override config batch_size")
    ap.add_argument("--steady-batches", type=int, default=0,
                    help="if >0, cycle the loader and time this many batches (steady state) "
                         "instead of per-epoch — avoids the tiny-local-epoch artifact")
    ap.add_argument("--steady-warmup", type=int, default=10)
    ap.add_argument("--override-compile", default=None,
                    help="force ml_perf.compile to this mode (none|default|reduce-overhead|backbone)")
    ap.add_argument("--override-amp", default=None, choices=["0", "1"],
                    help="force ml_perf.amp off(0)/on(1)")
    args = ap.parse_args()

    import torch
    from _bench_common import build_bench_config
    from src.ml.utils import prepare_data_and_model

    dev = "cuda"
    # build_bench_config nulls pretrained ckpts + keeps experiment's ml_perf/batch_size
    from config.default import get_default_config
    from config.experiments import experiments
    from config.ablations import ablation_experiments
    experiments.update(ablation_experiments)
    exp = experiments[args.experiment]
    bs = args.batch_size or exp.get("batch_size", 100)
    cfg = build_bench_config(args.experiment, args.data_patterns, bs,
                             eb_variant=args.eb_variant, num_workers=args.workers,
                             pin_memory=args.pin_memory)
    cfg.prefetch_factor = args.prefetch
    if args.override_compile is not None or args.override_amp is not None:
        mp = getattr(cfg, "ml_perf", {})
        mp = dict(mp.to_dict() if hasattr(mp, "to_dict") else (mp or {}))
        if args.override_compile is not None:
            mp["compile"] = args.override_compile
        if args.override_amp is not None:
            mp["amp"] = (args.override_amp == "1")
        cfg.ml_perf = mp
    # restore the experiment's ml_perf (build_bench_config copied it via setattr already)
    ml_perf = getattr(cfg, "ml_perf", {})
    ml_perf = ml_perf.to_dict() if hasattr(ml_perf, "to_dict") else dict(ml_perf or {})

    (train_loader, _val, _test), model, _ = prepare_data_and_model(cfg)
    model = model.to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=3.53e-7, betas=(0.5, 0.999))

    def run_epoch():
        n_samples = 0
        t0 = time.perf_counter()
        for data_dict, theta in train_loader:
            data_dict = {k: v.to(dev, non_blocking=True) for k, v in data_dict.items()}
            theta = theta.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = -model.model.log_prob(theta, data_dict).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            n_samples += theta.shape[0]
        torch.cuda.synchronize()
        return time.perf_counter() - t0, n_samples

    print(f"\n=== e2e  exp={args.experiment}  B={bs}  w={args.workers}  ml_perf={ml_perf}  store={args.data_patterns.split('/')[-2]} ===")

    if args.steady_batches > 0:
        # cycle the loader and time a fixed number of batches at steady state
        def batch_iter():
            while True:
                for b in train_loader:
                    yield b
        gen = batch_iter()
        step_ms = []
        done = 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        while done < args.steady_warmup + args.steady_batches:
            # time the WHOLE iteration incl. waiting for the next batch (so a dataloader stall
            # shows up): t0 was set after the previous step's sync.
            data_dict, theta = next(gen)
            data_dict = {k: v.to(dev, non_blocking=True) for k, v in data_dict.items()}
            theta = theta.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = -model.model.log_prob(theta, data_dict).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1e3
            if done >= args.steady_warmup:
                step_ms.append(dt)
            done += 1
            t0 = time.perf_counter()
        step_ms.sort()
        med = statistics.median(step_ms)
        smps = bs / (med / 1e3)
        print(f"  steady: {len(step_ms)} batches  median {med:.1f} ms/batch  {smps:.1f} smp/s  "
              f"(B={bs}); epoch-equiv over {len(train_loader.dataset)} = {len(train_loader.dataset)/smps:.2f} s")
        return

    times = []
    for ep in range(args.epochs):
        dt, n = run_epoch()
        tag = "(warmup, dropped)" if ep == 0 else ""
        print(f"  epoch {ep}: {dt:6.2f} s  {n/dt:7.1f} smp/s  ({n} samples) {tag}")
        if ep > 0:
            times.append(dt)
    if times:
        med = statistics.median(times)
        n_train = len(train_loader.dataset)
        print(f"  --> median train-epoch {med:.2f} s   {n_train/med:.1f} smp/s  (n_train={n_train})")


if __name__ == "__main__":
    main()
