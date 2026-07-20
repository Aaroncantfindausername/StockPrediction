from numpy.typing import NDArray
from pandas import DatetimeIndex, Timestamp
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
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
        "batch_size": 256,
        "lr": 1e-3,
        "epochs": 1000,
        "n_layers": 3,
        "hidden_dim": 512,
        "hidden_dim_decay": 0.5,
        "out_dim": 1,
        "dropout": 0.7,
        "weight_decay": 1e-3,
        "seed": 87,
        "scheduler_patience": 1000,
        "early_stop_patience": 200000,
        "horizon": 20,
        "target": "target_return_over_atr_z",
        "ticker": "^GSPC",
        "shuffle_train": True,
    }

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    ticker: str = config["ticker"]
    df = load_dataset(ticker)
    df = preprocess_dataset(df)

    cols, df = single_ticker_minimal.compute_features_and_labels(
        df, {"HORIZON": config["horizon"]}
    )
    df[f"{config['target']}_mean"] = (
        df[config["target"]]
        .shift(config["horizon"])
        .rolling(252, min_periods=50)
        .mean()
        .astype(np.float32)
    )
    df = df.dropna()
    X: NDArray[np.float32] = df[cols].to_numpy()
    y: NDArray[np.float32] = df[config["target"]].to_numpy()
    # %%

    # Load data
    train_loader, val_loader, test_loader, input_dim = (
        train_val_test_split_loader(
            X,
            y,
            batch_size=config["batch_size"],
            val_ratio=0.10,
            test_ratio=0.10,
            shuffle_train=config["shuffle_train"],
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
    criterion = nn.L1Loss()
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
        config["epochs"],
        config["early_stop_patience"],
        load_best_val_loss=False,
    )
    # Load best weights for final evaluation
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.10f}")
    print(
        f"Baseline rolling mean prediction: {criterion(torch.tensor(y), torch.tensor(df[f'{config["target"]}_mean'].to_numpy(copy=True)))}"
    )
    model_config = {
        "input_dim": input_dim,
        "n_layers": config["n_layers"],
        "hidden_dim": config["hidden_dim"],
        "hidden_dim_decay": config["hidden_dim_decay"],
        "horizon": config["horizon"],
        "target": config["target"],
        "ticker": config["ticker"],
    }
    torch.save(model.state_dict(), "weights/model.pth")
    torch.save(model_config, "weights/config.pth")
    print("Model weights and config stored")
