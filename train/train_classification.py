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
from data import single_ticker_lagged, single_ticker_minimal


def train() -> None:
    # Hyperparameters
    config = {
        "batch_size": 256,
        "lr": 1e-4,
        "epochs": 2000,
        "n_layers": 4,
        "hidden_dim": 256,
        "hidden_dim_decay": 0.5,
        "out_dim": 2,
        "dropout": 0.3,
        "weight_decay": 1e-3,
        "seed": 87,
        "scheduler_patience": 50,
        "early_stop_patience": 200,
        "horizon": 30,
        "target": "target_return_binary",
        "ticker": "^GSPC",
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
        )
    )
    # Model, loss, optimizer

    model = MLP(
        input_dim,
        config["out_dim"],
        params={
            "n_layers": config["n_layers"],
            "hidden_dim": config["hidden_dim"],
            "hidden_dim_decay": config["hidden_dim_decay"],
            "dropout_rate": config["dropout"],
        },
    ).to(device)
    criterion = nn.CrossEntropyLoss()
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
    )
    # Load best weights for final evaluation
    test_loss = evaluate(model, test_loader, criterion, device)

    print(f"\nTest Loss: {test_loss:.10f}")
    model_config = {
        "input_dim": input_dim,
        "n_layers": config["n_layers"],
        "hidden_dim": config["hidden_dim"],
        "hidden_dim_decay": config["hidden_dim_decay"],
        "horizon": config["horizon"],
        "target": config["target"],
    }
    torch.save(model.state_dict(), "weights/model.pth")
    torch.save(model_config, "weights/config.pth")
    print("Model weights and config stored")
