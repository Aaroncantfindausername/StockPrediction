from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data.dataset import TensorDataset
from models.basic_nn import MLP
import copy
from data.preprocess import load_dataset, preprocess_dataset
from matplotlib import pyplot as plt
from train.pipeline import (
    create_sequential_windows_single_tickers,
    create_sequential_windows_multiple_tickers,
    evaluate,
    train_full,
    train_val_test_split_loader,
)
from data import single_ticker_lagged, single_ticker_minimal, multiple_ticker


def test():
    config = {
        "batch_size": 256,
        "lr": 1e-5,
        "epochs": 1000,
        "n_layers": 3,
        "hidden_dim": 256,
        "hidden_dim_decay": 0.5,
        "out_dim": 2,
        "dropout": 0.5,
        "weight_decay": 1e-3,
        "seed": 87,
        "scheduler_patience": 10,
        "early_stop_patience": 30,
        "val_ratio": 0.2,
        "test_ratio": 0.1,
        "horizon": 10,
        "target": "target_return_binary",
        "ticker": "^GSPC",
        "lag_features": False,
        "lags": [1, 2, 5],
        "shuffle_train": True,
        "classification": True,
        "transformer": True,
        "seq_len": 50,
        "d_model": 64,
        "activation_fn": "relu",
        "encoder_layers": 2,
        "n_atten_head": 8,  # Even n.o heads
    }
    with open("datasets/subset_tickers.txt", "r") as f:
        tickers = [line.strip() for line in f if line.strip()]
        f.close()

    df, cols = multiple_ticker.get_feature_target_df(
        tickers,
        {
            "HORIZON": config["horizon"],
            "lag_features": config["lag_features"],
            "lags": config["lags"],
        },
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
    if config["transformer"]:
        X_train, ticker_train, y_train = (
            create_sequential_windows_multiple_tickers(
                df[df.index <= train_end],
                cols,
                config["target"],
                config["seq_len"],
            )
        )
        train_dataset = TensorDataset(X_train, ticker_train, y_train)

        X_val, ticker_val, y_val = create_sequential_windows_multiple_tickers(
            df[(df.index >= val_start) & (df.index <= val_end)],
            cols,
            config["target"],
            config["seq_len"],
        )
        val_dataset = TensorDataset(X_val, ticker_val, y_val)

        X_test, ticker_test, y_test = (
            create_sequential_windows_multiple_tickers(
                df[df.index >= test_start],
                cols,
                config["target"],
                config["seq_len"],
            )
        )
        test_dataset = TensorDataset(X_test, ticker_test, y_test)
