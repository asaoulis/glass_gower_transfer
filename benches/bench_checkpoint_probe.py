#!/usr/bin/env python
"""Gradient-checkpointing + big-batch probe (the 3x-stretch lever).

amp+compile cut peak mem to 19 GB @B100 / 39 GB @B200; B>=300 OOMs. Gradient checkpointing
(use_checkpoint=True in the UNet, NUMERICALLY IDENTICAL — just recomputes activations in
backward) trades compute for memory so a bigger batch fits. Bigger batch raises model
throughput AND amortizes the fixed per-batch H2D/overhead => lifts BOTH model-only and e2e.
This is a bracketed lever (batch increase changes the optimisation => quality A/B / LR retune).

  /data/alex/glass/env/bin/python benches/bench_checkpoint_probe.py --batches 200,400,600,800
"""
from __future__ import annotations

import argparse

from _bench_common import build_bench_config, tile_batch, timed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-e", "--experiment", default="ablation_glass_no_side")
    ap.add_argument("--data-patterns", default="/data/alex/glass_mocks_f32/*.h5")
    ap.add_argument("--batches", default="200,400,600,800")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--no-checkpoint", action="store_true")
    args = ap.parse_args()

    import torch
    from src.ml.utils import prepare_data_and_model, build_model

    dev = "cuda"
    sizes = [int(b) for b in args.batches.split(",")]
    cfg = build_bench_config(args.experiment, args.data_patterns, max(sizes))
    # inject gradient checkpointing into the UNet via map_kwargs (-> KidsO3 **kwargs -> UNet)
    mk = cfg.model_kwargs
    mk = mk.to_dict() if hasattr(mk, "to_dict") else dict(mk)
    if not args.no_checkpoint:
        mk.setdefault("map_kwargs", {})
        mk["map_kwargs"] = {**mk["map_kwargs"], "use_checkpoint": True}
    cfg.model_kwargs = mk

    (train_loader, _v, test_loader), _m0, _ = prepare_data_and_model(cfg)
    raw = next(iter(train_loader))

    BACKBONE = "embedding_net.patch_encoder.shared_cnn.backbone"
    fp32_100_smps = 84.3  # e2e steady baseline (for the x column on smp/s)
    print(f"\n=== checkpoint probe  ckpt={not args.no_checkpoint}  amp+compile(backbone) ===")
    print(f"{'B':>5s} {'step ms':>9s} {'smp/s':>8s} {'x vs base':>10s} {'peak GB':>8s}")
    for B in sizes:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        data_dict, theta = tile_batch(raw, B, dev)
        model = build_model(cfg, test_dataloader=test_loader).to(dev)
        enc = model.model.embedding_net
        _orig = enc.forward

        def mkamp(o):
            def f(*a, **k):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = o(*a, **k)
                return out.float() if isinstance(out, torch.Tensor) else out
            return f
        enc.forward = mkamp(_orig)
        obj = model.model
        parts = BACKBONE.split(".")
        for a in parts[:-1]:
            obj = getattr(obj, a)
        setattr(obj, parts[-1], torch.compile(getattr(obj, parts[-1]), mode="reduce-overhead"))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

        def step():
            opt.zero_grad(set_to_none=True)
            loss = -model.model.log_prob(theta, data_dict).mean()
            loss.backward(); opt.step(); return loss

        try:
            l0 = float(step().detach())
            r = timed(step, args.iters, args.warmup)
            peak = torch.cuda.max_memory_allocated() / 1e9
            smps = B / (r["median_ms"] / 1e3)
            print(f"{B:5d} {r['median_ms']:9.1f} {smps:8.1f} {smps/fp32_100_smps:10.2f} {peak:8.2f}")
        except RuntimeError as ex:
            print(f"{B:5d} {'OOM/err':>9s}  {str(ex)[:50]}")
        del model, opt
        torch._dynamo.reset()


if __name__ == "__main__":
    main()
