"""Train an NDE on frozen embeddings: python train_embeddings.py <target_exp> <source_exp...>.

The source experiments' encoders produce cached embeddings, and the target experiment's flow is
trained on them. RUN_TRAINING / RUN_EVALUATION (env, 0 or 1) override the target row's flags.
"""
import os
import sys

from config.experiments import experiments
from config.paper_2026 import paper_2026_experiments
from config.kids_legacy import kids_legacy_experiments
from config.kids_legacy_novd import kids_legacy_novd_experiments
from config.euclid import euclid_experiments
from config.kids_legacy_bgp import kids_legacy_bgp_experiments
from config.kids_legacy_vdq import kids_legacy_vdq_experiments
from config.kids_legacy_hos import kids_legacy_hos_experiments
from src.ml.embeddings.train import EmbeddingTrainArgs, train_embeddings_experiment
from src.ml.models.custom_sbi import NeuralSplineFlow

# Merge the config modules so target and source names resolve inside the embeddings pipeline.
experiments.update(paper_2026_experiments)
experiments.update(kids_legacy_experiments)
experiments.update(kids_legacy_novd_experiments)
experiments.update(euclid_experiments)
experiments.update(kids_legacy_bgp_experiments)
experiments.update(kids_legacy_vdq_experiments)
experiments.update(kids_legacy_hos_experiments)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python train_embeddings.py <target_experiment_name> <source_exp1> [<source_exp2> ...]")
        raise SystemExit(1)

    target_experiment = sys.argv[1]
    source_experiments = sys.argv[2:]

    # RUN_EVALUATION / RUN_TRAINING (0 or 1) replace the target row's run_evaluation / run_training.
    for _env, _key in (("RUN_EVALUATION", "run_evaluation"), ("RUN_TRAINING", "run_training")):
        _v = os.environ.get(_env, "").strip()
        if _v:
            if _v not in ("0", "1"):
                raise SystemExit(f"{_env} must be 0 or 1, got {_v!r}")
            experiments[target_experiment][_key] = _v == "1"
            print(f"[train_embeddings.py] {_env} override -> {_key}={_v == '1'}")

    train_embeddings_experiment(EmbeddingTrainArgs(target_experiment, source_experiments))
