#!/usr/bin/env python
"""In-process torch.compile probe for the kids_hybrid encoder (fast iteration).

Builds dataloaders ONCE (scaler-fit), pulls one fixed batch, then for each trial builds a
FRESH model, optionally applies encoder-scoped AMP + torch.compile at a chosen module path,
and times the full fwd+bwd+opt step. Catches per-trial compile errors so one bad strategy
doesn't abort the sweep.

  /data/alex/glass/env/bin/python benches/bench_compile_probe.py -B 100 [--iters 12]
"""
from __future__ import annotations

import argparse
import time
import traceback

from _bench_common import build_bench_config, tile_batch, timed


def scope_amp(model):
    """Wrap embedding_net.forward in bf16 autocast, cast outputs to fp32 (flow stays fp32)."""
    import torch
    enc = model.model.embedding_net
    orig = enc.forward

    def _f(*a, **k):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = orig(*a, **k)
        if isinstance(out, torch.Tensor):
            return out.float()
        if isinstance(out, (tuple, list)):
            return type(out)(o.float() if isinstance(o, torch.Tensor) else o for o in out)
        return out

    enc.forward = _f


def get_by_path(model, path):
    obj = model.model
    for a in path.split("."):
        obj = getattr(obj, a)
    return obj


def set_by_path(model, path, val):
    obj = model.model
    parts = path.split(".")
    for a in parts[:-1]:
        obj = getattr(obj, a)
    setattr(obj, parts[-1], val)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-e", "--experiment", default="ablation_glass_no_side")
    ap.add_argument("-B", "--batch-size", type=int, default=100)
    ap.add_argument("--data-patterns", default="/data/alex/glass_mocks_f32/*.h5")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=6)
    args = ap.parse_args()

    import torch
    from src.ml.utils import prepare_data_and_model, build_model

    dev = "cuda"
    cfg = build_bench_config(args.experiment, args.data_patterns, args.batch_size)
    (train_loader, _v, test_loader), model0, scalers = prepare_data_and_model(cfg)
    batch = next(iter(train_loader))
    data_dict, theta = tile_batch(batch, args.batch_size, dev)

    BACKBONE = "embedding_net.patch_encoder.shared_cnn.backbone"
    SHARED = "embedding_net.patch_encoder.shared_cnn"
    PATCHENC = "embedding_net.patch_encoder"
    ENC = "embedding_net"

    # (label, amp, compile_path|None, compile_mode)
    trials = [
        ("fp32 baseline", False, None, None),
        ("amp", True, None, None),
        ("amp + compile backbone (default)", True, BACKBONE, "default"),
        ("amp + compile backbone (reduce-overhead)", True, BACKBONE, "reduce-overhead"),
        ("amp + compile shared_cnn (default)", True, SHARED, "default"),
        ("amp + compile patch_encoder (default)", True, PATCHENC, "default"),
        ("amp + compile embedding_net (default)", True, ENC, "default"),
        ("compile backbone (default), no amp", False, BACKBONE, "default"),
    ]

    print(f"\n=== compile probe  B={args.batch_size}  iters={args.iters} ===")
    print(f"{'trial':48s} {'step ms':>9s} {'x':>5s} {'peak GB':>8s}  status")
    base_ms = None
    for label, amp, cpath, cmode in trials:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model = build_model(cfg, test_dataloader=test_loader).to(dev)
            if amp:
                scope_amp(model)
            if cpath is not None:
                tgt = get_by_path(model, cpath)
                set_by_path(model, cpath, torch.compile(tgt, mode=cmode))
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=3.53e-7,
                                    betas=(0.5, 0.999))

            def step():
                opt.zero_grad(set_to_none=True)
                lp = model.model.log_prob(theta, data_dict)
                loss = -lp.mean()
                loss.backward()
                opt.step()
                return loss

            l0 = float(step().detach())
            assert l0 == l0, "non-finite"
            r = timed(step, args.iters, args.warmup)
            peak = torch.cuda.max_memory_allocated() / 1e9
            ms = r["median_ms"]
            if base_ms is None:
                base_ms = ms
            x = base_ms / ms
            print(f"{label:48s} {ms:9.1f} {x:5.2f} {peak:8.2f}  ok (loss {l0:.2f})")
            del model, opt
        except Exception as ex:  # noqa
            msg = f"{type(ex).__name__}: {str(ex)[:80]}"
            print(f"{label:48s} {'—':>9s} {'—':>5s} {'—':>8s}  FAIL {msg}")
            if "--verbose" in __import__("sys").argv:
                traceback.print_exc()


if __name__ == "__main__":
    main()
