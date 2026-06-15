#!/usr/bin/env python
"""Batch-size scaling probe with AMP + compile(backbone) — find throughput-optimal B.

With amp+compile the peak mem at B=100 drops to ~19 GB (of 48), so larger batches fit and
should raise throughput (the encoder is compute/bandwidth-bound). Pure batch increase changes
the optimisation slightly (fewer steps/epoch) => a mild quality A/B, but it's a legit
throughput lever. Reports smp/s vs the fp32 B=100 baseline.

  /data/alex/glass/env/bin/python benches/bench_batch_scaling.py --batches 100,150,200,300
"""
from __future__ import annotations

import argparse

from _bench_common import build_bench_config, tile_batch, timed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-e", "--experiment", default="ablation_glass_no_side")
    ap.add_argument("--data-patterns", default="/data/alex/glass_mocks_f32/*.h5")
    ap.add_argument("--batches", default="100,150,200,300")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    import torch
    from src.ml.utils import prepare_data_and_model, build_model

    dev = "cuda"
    sizes = [int(b) for b in args.batches.split(",")]
    maxB = max(sizes)
    cfg = build_bench_config(args.experiment, args.data_patterns, maxB)
    (train_loader, _v, test_loader), _m0, _ = prepare_data_and_model(cfg)
    raw = next(iter(train_loader))

    BACKBONE = "embedding_net.patch_encoder.shared_cnn.backbone"

    print(f"\n=== batch scaling  amp{'' if args.no_compile else '+compile(backbone)'} ===")
    print(f"{'B':>5s} {'step ms':>9s} {'smp/s':>8s} {'x vs fp32@100':>14s} {'peak GB':>8s}")
    fp32_100_smps = 88.4  # measured fp32 B=100 baseline (1131 ms) for the x column
    for B in sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        data_dict, theta = tile_batch(raw, B, dev)
        model = build_model(cfg, test_dataloader=test_loader).to(dev)
        # encoder-scoped amp
        enc = model.model.embedding_net
        _orig = enc.forward

        def _f(*a, **k):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = _orig(*a, **k)
            return out.float() if isinstance(out, torch.Tensor) else out

        enc.forward = _f
        if not args.no_compile:
            obj = model.model
            parts = BACKBONE.split(".")
            for a in parts[:-1]:
                obj = getattr(obj, a)
            setattr(obj, parts[-1], torch.compile(getattr(obj, parts[-1]), mode="reduce-overhead"))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

        def step():
            opt.zero_grad(set_to_none=True)
            loss = -model.model.log_prob(theta, data_dict).mean()
            loss.backward()
            opt.step()
            return loss

        try:
            l0 = float(step().detach())
            r = timed(step, args.iters, args.warmup)
            peak = torch.cuda.max_memory_allocated() / 1e9
            smps = B / (r["median_ms"] / 1e3)
            print(f"{B:5d} {r['median_ms']:9.1f} {smps:8.1f} {smps/fp32_100_smps:14.2f} {peak:8.2f}")
        except RuntimeError as ex:
            print(f"{B:5d} {'OOM/err':>9s}  {str(ex)[:40]}")
        del model, opt
        if not args.no_compile:
            torch._dynamo.reset()


if __name__ == "__main__":
    main()
