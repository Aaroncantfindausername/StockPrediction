from numpy.typing import NDArray
from pandas import DatetimeIndex, Timestamp
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
from data import multiple_ticker, single_ticker_lagged, single_ticker_minimal


def train() -> None:
    # Hyperparameters
    config = {
        "batch_size": 256,
        "lr": 1e-4,
        "epochs": 50,
        "n_layers": 1,
        "hidden_dim": 16,
        "hidden_dim_decay": 0.75,
        "out_dim": 1,
        "dropout": 0.4,
        "weight_decay": 1e-3,
        "seed": 87,
        "scheduler_patience": 30,
    }

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    with open("datasets/subset_tickers.txt", "r") as f:
        tickers = [line.strip() for line in f if line.strip()]

    df, cols = multiple_ticker.get_feature_target_df(tickers, {"HORIZON": 20})
    df = df.dropna()
    X: NDArray[np.float32] = df[[f"{c}_z" for c in cols]].to_numpy()
    y: NDArray[np.float32] = df["target_return_z"].to_numpy()
    # %%

    # Load data
    train_loader, val_loader, test_loader, input_dim = (
        train_val_test_split_loader(
            X,
            y,
            batch_size=config["batch_size"],
            val_ratio=0.10,
            test_ratio=0.10,
        )
    )
    # Model, loss, optimizer

    model = MLP(
        input_dim,
        1,
        params={
            "n_layers": config["n_layers"],
            "hidden_dim": config["hidden_dim"],
            "hidden_dim_decay": config["hidden_dim_decay"],
            "dropout_rate": config["dropout"],
        },
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=config["scheduler_patience"], factor=0.5
    )
    train_full(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        10000,
        10000,
    )
    # Load best weights for final evaluation
    test_loss = evaluate(model, test_loader, criterion, device)

    print(f"\nTest Loss: {test_loss:.10f}")
