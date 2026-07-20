from numpy.typing import NDArray
from pandas import DatetimeIndex, Timestamp
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from models.basic_nn import MLP
import copy
from data.preprocess import load_dataset, preprocess_dataset
from matplotlib import pyplot as plt
from models.embedded_mlp import EmbeddedMLP
from train.pipeline import (
    evaluate,
    evaluate_with_embedding,
    train_full,
    train_full_with_embedding,
    train_val_test_split_loader,
)
from data import multiple_ticker, single_ticker_lagged, single_ticker_minimal
from train.ticker_embedding_dataset import TickerEmbeddingDataset


def train() -> None:
    # Hyperparameters
    config = {
        "batch_size": 512,
        "lr": 5e-3,
        "epochs": 500,
        "n_layers": 3,
        "hidden_dim": 64,
        "hidden_dim_decay": 0.5,
        "embedding_dim": 4,
        "out_dim": 1,
        "dropout": 0.5,
        "weight_decay": 1e-3,
        "seed": 87,
        "scheduler_patience": 10,
        "early_stop_patience": 20,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "horizon": 20,
        "target": "target_return_z",
    }

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    with open("datasets/subset_tickers.txt", "r") as f:
        tickers = [line.strip() for line in f if line.strip()]
        f.close()

    df, cols = multiple_ticker.get_feature_target_df(
        tickers, {"HORIZON": config["horizon"], "lag_features": True}
    )
    df = df.dropna()

    ticker_to_id = {t: i for i, t in enumerate(tickers)}
    df["ticker_id"] = df["Ticker"].map(ticker_to_id)
    num_tickers = len(tickers)
    num_days = df.index.nunique()
    train_end = df.index[
        int(num_days * (1 - config["val_ratio"] - config["test_ratio"]))
        - config["horizon"]
        - 1
    ]  # Purging
    val_start = df.index[
        int(num_days * (1 - config["val_ratio"] - config["test_ratio"]))
    ]
    val_end = df.index[
        int(num_days * (1 - config["test_ratio"])) - config["horizon"] - 1
    ]
    test_start = df.index[int(num_days * (1 - config["test_ratio"]))]
    train_dataset = TickerEmbeddingDataset(
        df[df.index <= train_end], cols, config["target"]
    )
    val_dataset = TickerEmbeddingDataset(
        df[(df.index >= val_start) & (df.index <= val_end)],
        cols,
        config["target"],
    )
    test_dataset = TickerEmbeddingDataset(
        df[df.index >= test_start], cols, config["target"]
    )
    train_loader = DataLoader(train_dataset, config["batch_size"], True)
    val_loader = DataLoader(val_dataset, config["batch_size"], False)
    test_loader = DataLoader(test_dataset, config["batch_size"], False)
    # Model, loss, optimizer

    model = EmbeddedMLP(
        len(cols),
        1,
        config["embedding_dim"],
        num_tickers,
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
    train_full_with_embedding(
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
    test_loss = evaluate_with_embedding(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.10f}")
    model_config = {
        "input_dim": len(cols),
        "n_layers": config["n_layers"],
        "hidden_dim": config["hidden_dim"],
        "hidden_dim_decay": config["hidden_dim_decay"],
        "horizon": config["horizon"],
        "target": config["target"],
        "ticker_to_id": ticker_to_id,
        "num_unique_embeddings": num_tickers,
        "embedding_dim": config["embedding_dim"],
    }
    torch.save(model.state_dict(), "weights/embedded_model.pth")
    torch.save(model_config, "weights/embedded_config.pth")
    print("Model weights and config stored")
