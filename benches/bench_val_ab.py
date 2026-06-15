#!/usr/bin/env python
"""Quality A/B: train an experiment for N epochs on a fixed seed/data, report the val-loss curve.

Used to prove a numerics/capacity-touching speed option (amp, compile, f16 store, bigger batch)
does NOT degrade the posterior. Run twice (baseline vs optimized) with the SAME seed/data and
compare best val_log_prob. amp+compile at the SAME batch is numerics-only ⇒ expect ~equal; a
batch increase changes the optimisation ⇒ compare at matched wall-clock / with LR retune.

  /data/alex/glass/env/bin/python benches/bench_val_ab.py -e ablation_glass_no_side \
      --data-patterns '/data/alex/glass_mocks_f32/*.h5' --epochs 20
"""
from __future__ import annotations

import argparse
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-e", "--experiment", default="ablation_glass_no_side")
    ap.add_argument("--data-patterns", default="/data/alex/glass_mocks_f32/*.h5")
    ap.add_argument("--eb-variant", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    import pytorch_lightning as pl
    from _bench_common import build_bench_config
    from src.ml.utils import prepare_data_and_model
    from config.experiments import experiments
    from config.ablations import ablation_experiments

    experiments.update(ablation_experiments)
    exp = experiments[args.experiment]
    bs = args.batch_size or exp.get("batch_size", 100)
    pl.seed_everything(args.seed, workers=True)

    cfg = build_bench_config(args.experiment, args.data_patterns, bs,
                             eb_variant=args.eb_variant, num_workers=args.workers,
                             pin_memory=True)
    cfg.split_seed = args.seed
    (train_loader, val_loader, _t), model, _ = prepare_data_and_model(cfg)
    ml_perf = getattr(cfg, "ml_perf", {})
    ml_perf = ml_perf.to_dict() if hasattr(ml_perf, "to_dict") else dict(ml_perf or {})
    amp_on = bool(ml_perf.get("amp"))
    # Always fp32-Lightning: scoped AMP autocasts the encoder itself; whole-forward bf16-mixed
    # crashes the nflows spline (the bug this task fixes) so it's never a valid baseline here.
    precision = "32"

    class ValHist(pl.Callback):
        def __init__(self):
            self.best = float("inf")
            self.curve = []

        def on_validation_epoch_end(self, trainer, pl_module):
            v = trainer.callback_metrics.get("val_log_prob")
            if v is not None:
                v = float(v)
                self.curve.append(v)
                self.best = min(self.best, v)

    hist = ValHist()
    trainer = pl.Trainer(
        max_epochs=args.epochs, accelerator="auto", devices=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=False, num_sanity_val_steps=0,
        gradient_clip_val=0.5, precision=precision, callbacks=[hist],
        check_val_every_n_epoch=1, log_every_n_steps=5,
    )
    t0 = time.perf_counter()
    trainer.fit(model, train_loader, val_loader)
    wall = time.perf_counter() - t0

    print(f"\n=== val A/B  exp={args.experiment}  B={bs}  precision={precision}  "
          f"ml_perf={ml_perf}  store={args.data_patterns.split('/')[-2]} ===")
    print(f"  epochs={args.epochs}  wall={wall:.1f}s  seed={args.seed}")
    print(f"  best val_log_prob = {hist.best:.4f}")
    print(f"  curve = {[round(c, 3) for c in hist.curve]}")


if __name__ == "__main__":
    main()
