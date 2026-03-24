import sys

from src.ml.embeddings.train import EmbeddingTrainArgs, train_embeddings_experiment


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python train_embeddings.py <target_experiment_name> <source_exp1> [<source_exp2> ...]")
        raise SystemExit(1)

    target_experiment = sys.argv[1]
    source_experiments = sys.argv[2:]

    train_embeddings_experiment(EmbeddingTrainArgs(target_experiment, source_experiments))
