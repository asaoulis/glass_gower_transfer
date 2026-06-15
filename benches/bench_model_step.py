#!/usr/bin/env python
"""Model-only fwd / fwd+bwd / full-step bench for the glass kids_hybrid_bandpowers_maps arch.

Primary metric: model-only fwd+bwd+opt median ms/step on a FIXED production-shaped batch
(isolates GPU compute, the target of AMP / channels_last / compile / fused-adam).

Usage:
  /data/alex/glass/env/bin/python benches/bench_model_step.py \
      -e ablation_glass_no_side -B 100 [--amp] [--channels-last] [--tf32] \
      [--compile {none,encoder,backbone}] [--fused-adam] [--iters 30] [--warmup 8]

The batch is pulled once from the smoke mini-dataset (shapes identical to production) and
tiled to size B, so timing never touches the dataloader.
"""
from __future__ import annotations

import argparse
import json

from _bench_common import build_bench_config, tile_batch, timed, SMOKE_GLOB


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-e", "--experiment", default="ablation_glass_no_side")
    ap.add_argument("-B", "--batch-size", type=int, default=100)
    ap.add_argument("--data-patterns", default=SMOKE_GLOB)
    ap.add_argument("--eb-variant", default=None, help="e.g. fwhm8 for the real corpus")
    ap.add_argument("--amp", action="store_true", help="bf16 autocast around the forward")
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--compile", choices=["none", "encoder", "backbone", "shared", "full"], default="none")
    ap.add_argument("--fused-adam", action="store_true")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--json", action="store_true", help="emit a json result line")
    args = ap.parse_args()

    import torch
    from src.ml.utils import prepare_data_and_model

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)

    cfg = build_bench_config(args.experiment, args.data_patterns, args.batch_size,
                             eb_variant=args.eb_variant, num_workers=0)
    (train_loader, _val, _test), model, _ = prepare_data_and_model(cfg)
    model = model.to(dev)

    # one real batch -> fixed tiled batch (timing never touches the loader)
    batch = next(iter(train_loader))
    data_dict, theta = tile_batch(batch, args.batch_size, dev)

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
        data_dict = {k: (v.to(memory_format=torch.channels_last) if v.ndim == 4 else v)
                     for k, v in data_dict.items()}

    # optional compile of the map-encoder CNN. backbone = the UNetStyleEncoder convs (the
    # compute); shared_cnn/encoder/full also wrap the PoolProj head (currently breaks inductor
    # with a max_pool2d lowering NameError) -> backbone is the clean, shippable target.
    _COMPILE_PATHS = {
        "backbone": "embedding_net.patch_encoder.shared_cnn.backbone",
        "shared": "embedding_net.patch_encoder.shared_cnn",
        "encoder": "embedding_net",
    }
    if args.compile != "none":
        path = _COMPILE_PATHS.get(args.compile)
        if args.compile == "full" or path is None:
            model.model = torch.compile(model.model, mode="reduce-overhead")
            print("[compile] compiled full model.model")
        else:
            obj = model.model
            parts = path.split(".")
            for a in parts[:-1]:
                obj = getattr(obj, a)
            setattr(obj, parts[-1], torch.compile(getattr(obj, parts[-1]),
                                                  mode="reduce-overhead"))
            print(f"[compile] compiled model.model.{path}")

    opt_kwargs = dict(lr=1e-4, weight_decay=3.53e-7, betas=(0.5, 0.999))
    if args.fused_adam:
        try:
            opt = torch.optim.AdamW(model.parameters(), fused=True, **opt_kwargs)
        except Exception as ex:  # noqa
            print(f"[fused-adam] unavailable ({ex}); foreach")
            opt = torch.optim.AdamW(model.parameters(), foreach=True, **opt_kwargs)
    else:
        opt = torch.optim.AdamW(model.parameters(), **opt_kwargs)

    # Encoder-SCOPED AMP: autocast bf16 around the embedding_net (the big UNet), cast its
    # output back to fp32 so the nflows RQS spline stays fp32 (full-flow bf16 crashes at the
    # spline index_put). This mirrors what ml_perf.amp will ship in src.
    if args.amp:
        enc = model.model.embedding_net
        _orig_fwd = enc.forward

        def _amp_fwd(*a, **k):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = _orig_fwd(*a, **k)
            if isinstance(out, torch.Tensor):
                return out.float()
            if isinstance(out, (tuple, list)):
                return type(out)(o.float() if isinstance(o, torch.Tensor) else o for o in out)
            return out

        enc.forward = _amp_fwd

    def fwd_only():
        with torch.no_grad():
            model.model.log_prob(theta, data_dict)

    def full_step():
        opt.zero_grad(set_to_none=True)
        lp = model.model.log_prob(theta, data_dict)
        loss = -lp.mean()
        loss.backward()
        opt.step()
        return loss

    # correctness / finiteness
    model.train()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    loss0 = float(full_step().detach())
    assert loss0 == loss0, "non-finite loss"

    r_fwd = timed(fwd_only, args.iters, args.warmup)
    r_step = timed(full_step, args.iters, args.warmup)
    peak_gb = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
    smp_s = args.batch_size / (r_step["median_ms"] / 1e3)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    flags = []
    for f in ("amp", "channels_last", "tf32", "fused_adam"):
        if getattr(args, f):
            flags.append(f)
    if args.compile != "none":
        flags.append(f"compile:{args.compile}")
    tag = "+".join(flags) if flags else "eager-fp32"

    print(f"\n=== model-step  exp={args.experiment}  B={args.batch_size}  [{tag}] ===")
    print(f"  fwd-only      median {r_fwd['median_ms']:8.1f} ms  (min {r_fwd['min_ms']:.1f})")
    print(f"  FULL step     median {r_step['median_ms']:8.1f} ms  (min {r_step['min_ms']:.1f}, p90 {r_step['p90_ms']:.1f})")
    print(f"  throughput    {smp_s:8.1f} smp/s")
    print(f"  peak mem      {peak_gb:8.2f} GB")
    print(f"  params        {n_params/1e6:.3f}M total / {n_train/1e6:.3f}M trainable")
    print(f"  loss0={loss0:.4f}")
    if args.json:
        print("JSON " + json.dumps({
            "tag": tag, "B": args.batch_size, "exp": args.experiment,
            "fwd_ms": r_fwd["median_ms"], "step_ms": r_step["median_ms"],
            "step_min_ms": r_step["min_ms"], "smp_s": smp_s, "peak_gb": peak_gb,
            "params_m": n_params / 1e6, "loss0": loss0,
        }))


if __name__ == "__main__":
    main()
