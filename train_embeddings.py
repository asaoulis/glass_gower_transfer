import sys

from config.experiments import experiments
from config.ablations import ablation_experiments
from config.kids_legacy import kids_legacy_experiments
from src.ml.embeddings.train import EmbeddingTrainArgs, train_embeddings_experiment
from src.ml.models.custom_sbi import NeuralSplineFlow

# Make ablation + KiDS-Legacy experiment configs visible to the embeddings pipeline.
# `train_embeddings_experiment` / `load_pretrained_models` resolve target AND source experiments
# against the SAME `config.experiments.experiments` dict object, so mutating it in place here
# (mirroring train.py) is what makes the kids_legacy_* hybrid SOURCES resolvable. A reassignment
# would NOT propagate to the already-imported reference, so use .update().
experiments.update(ablation_experiments)
experiments.update(kids_legacy_experiments)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python train_embeddings.py <target_experiment_name> <source_exp1> [<source_exp2> ...]")
        raise SystemExit(1)

    target_experiment = sys.argv[1]
    source_experiments = sys.argv[2:]

    train_embeddings_experiment(EmbeddingTrainArgs(target_experiment, source_experiments))
