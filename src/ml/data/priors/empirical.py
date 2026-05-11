import os
import pickle

import numpy as np
import torch


def train_or_load_gower_prior(
    csv_path,
    variables,
    scaler,
    drop_first=192,
    device="cpu",
    epochs=2000,
    lr=1e-3,
    batch_size=128,
    val_fraction=0.2,
    patience=50,
    retrain=False,
    save_path=None,
):
    from src.cosmology.gower_street import GowerStPrior
    from ...models.custom_sbi import NeuralSplineFlow

    if save_path is None:
        safe_name = "_".join([str(v) for v in variables])
        save_path = f"./data/gower_prior_{safe_name}.pkl"

    if os.path.exists(save_path) and not retrain:
        with open(save_path, "rb") as f:
            flow = pickle.load(f)
        flow.to(device)
        return flow

    gower = GowerStPrior.from_csv(
        csv_path,
        drop_first=drop_first,
    )

    theta_np = np.stack([gower.series_true[v] for v in variables], axis=1)

    if getattr(scaler, "min", None) is None or getattr(scaler, "max", None) is None:
        scaler.fit(theta_np)

    theta_scaled = scaler.transform(theta_np)
    theta = torch.tensor(theta_scaled, dtype=torch.float32, device=device)
    dim = theta.shape[1]

    N = len(theta)
    perm = torch.randperm(N)
    n_val = int(val_fraction * N)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    theta_train = theta[train_idx]
    theta_val = theta[val_idx]

    flow = NeuralSplineFlow(
        features=dim,
        hidden_features=32,
        num_layers=4,
        num_blocks_per_layer=2,
    ).to(device)
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)

    best_val = np.inf
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        flow.train()
        perm = torch.randperm(len(theta_train), device=device)

        for i in range(0, len(theta_train), batch_size):
            idx = perm[i:i + batch_size]
            batch = theta_train[idx]
            loss = -flow.log_prob(batch).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        flow.eval()
        with torch.no_grad():
            val_loss = -flow.log_prob(theta_val).mean().item()

        print(f"epoch {epoch} train {loss.item():.4f} val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = flow.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping")
            break

    if best_state is not None:
        flow.load_state_dict(best_state)

    os.makedirs("./data", exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(flow, f)

    return flow
