from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.basic_nn import MLP
import copy
from data.preprocess import load_dataset, preprocess_dataset
from matplotlib import pyplot as plt
from train.pipeline import (
    evaluate,
    train_full,
    train_val_test_split_loader,
)
from data import single_ticker_lagged, single_ticker_minimal


def train() -> None:
    # Hyperparameters
    config = {
        "batch_size": 64,
        "lr": 1e-3,
        "epochs": 50,
        "hidden_dim": 128,
        "out_dim": 1,
        "dropout": 0.4,
        "weight_decay": 1e-5,
        "patience": 10,  # early stopping
        "seed": 87,
    }

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    ticker: str = "^GSPC"
    df = load_dataset(ticker)
    df = preprocess_dataset(df)

    cols, df = single_ticker_lagged.compute_features_and_labels(
        df, {"HORIZON": 20}
    )
    df = df.dropna()
    X: NDArray[np.float32] = df[[f"{c}_z" for c in cols]].to_numpy()
    y: NDArray[np.float32] = df["target_scaled_z"].to_numpy()
    # %%

    # Load data
    train_loader, val_loader, test_loader, input_dim = (
        train_val_test_split_loader(
            X,
            y,
        )
    )
    # Model, loss, optimizer

    model = MLP(
        input_dim,
        1,
        params={
            "n_layers": 3,
            "hidden_dim": 64,
            "hidden_dim_decay": 0.75,
            "dropout_rate": 0.1,
        },
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10000, factor=0.5
    )

    fixed_inputs, fixed_targets = next(iter(train_loader))
    fixed_inputs, fixed_targets = (
        fixed_inputs.to(device),
        fixed_targets.to(device),
    )

    # Loss in train mode (dropout on, as during train_epoch)
    model.train()
    with torch.no_grad():
        train_mode_loss = criterion(model(fixed_inputs), fixed_targets).item()
    # Loss in eval mode (as evaluate would see)
    model.eval()
    with torch.no_grad():
        eval_mode_loss = criterion(model(fixed_inputs), fixed_targets).item()

    print(
        f"Fixed batch loss – train mode: {train_mode_loss:.6f}, eval mode: {eval_mode_loss:.6f}"
    )


# Freeze a small batch from the training set for comparison
