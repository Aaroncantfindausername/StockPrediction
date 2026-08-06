from numpy.typing import NDArray
from pandas import DatetimeIndex, Timestamp
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models.EncoderTransformer import EncoderTransformer
import copy
from data.preprocess import load_dataset, preprocess_dataset
from matplotlib import pyplot as plt
from models.mlp import MLP
from train.pipeline import (
    create_sequential_windows_multiple_tickers,
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
    # %%
    features: list[str] = [
        # "roc_slow",
        "roc_fast",
        # "rsi_slow",
        # "rsi_fast",
        # # "macd_hist",
        # "stochk",
        # # "stochd",
        # "adx",
        # # "pct_b",
        # # "atr_norm",
        # "dist_ema_fast",
        # "dist_ema_slow",
        # "dist_sma_fast",
        "dist_sma_slow",
        # # "obv_roc",
        # "volume_ratio",
        # "ema_slow",
        # "Close",
        # "Open",
        # "High",
        # "Low",
        # "Volume",
    ]
    config = {
        "batch_size": 2048,
        "lr": 1e-4,
        "epochs": 500,
        "out_dim": 2,
        "n_layers": 3,
        "hidden_dim": 1024,
        "hidden_dim_decay": 0.5,
        "embedding_dim": 4,
        "dropout": 0.3,
        "weight_decay": 1e-3,
        "seed": 67,
        "scheduler_patience": 2,
        "early_stop_patience": 10,
        "val_ratio": 0.2,
        "test_ratio": 0.1,
        "columns": features,
        "cross_sectional_z_score": True,
        "horizon": 20,
        "target": "target_return_binary",
        "lag_features": False,
        "lags": [1, 2, 5, 10, 50, 100, 200],
        "classification": True,
        "class_threshold": 0.7,
        # Transformer config
        "transformer": False,
        "seq_len": 20,
        "d_model": 64,
        "activation_fn": "relu",
        "encoder_layers": 4,
        "n_atten_head": 4,  # Even n.o heads
        "feature_z_score_window": 252,
    }

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    with open("datasets/ETFs.txt", "r") as f:
        tickers = [line.strip() for line in f if line.strip()]
        num_tickers = len(tickers)
        f.close()

    df, cols = multiple_ticker.get_feature_target_df(
        tickers,
        {
            "HORIZON": config["horizon"],
            "lag_features": config["lag_features"],
            "lags": config["lags"],
            "columns": config["columns"],
            "class_threshold": config["class_threshold"],
            "feature_z_score_window": config["feature_z_score_window"],
            "cross_sectional_z_score": config["cross_sectional_z_score"],
        },
    )
    # %%
    df = df.groupby("Date").filter(
        lambda g: len(g) == num_tickers and g[cols].notna().all(axis=1).all()
    )
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
    else:
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
    out_dim = config["out_dim"] if config["classification"] else 1
    if config["transformer"]:
        model = EncoderTransformer(
            len(cols),
            out_dim,
            config["d_model"],
            config["encoder_layers"],
            config["n_atten_head"],
            num_tickers,
            True,
            config["embedding_dim"],
            config["hidden_dim"],
            config["dropout"],
            config["activation_fn"],
            config["seq_len"],
            device,
        ).to(device)
    else:
        model = MLP(
            len(cols),
            out_dim,
            True,
            config["embedding_dim"],
            num_tickers,
            params={
                "n_layers": config["n_layers"],
                "hidden_dim": config["hidden_dim"],
                "hidden_dim_decay": config["hidden_dim_decay"],
                "dropout_rate": config["dropout"],
            },
        ).to(device)
    criterion = (
        nn.MSELoss() if not config["classification"] else nn.CrossEntropyLoss()
    )
    optimizer = optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=config["scheduler_patience"],
        factor=0.5,
        threshold=0.0,
    )

    val_loss_baseline = evaluate_with_embedding(
        model, val_loader, criterion, device, config["classification"]
    )
    if config["classification"]:
        print("^^^ Random model accuracy;")
    print(f"Random val loss = {val_loss_baseline}")

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
        classification=config["classification"],
    )
    # Load best weights for final evaluation
    test_loss = evaluate_with_embedding(
        model, test_loader, criterion, device, config["classification"]
    )
    print(f"\nTest Loss: {test_loss:.5f}")
    model_config = {
        "input_dim": len(cols),
        "n_layers": config["n_layers"],
        "hidden_dim": config["hidden_dim"],
        "hidden_dim_decay": config["hidden_dim_decay"],
        "horizon": config["horizon"],
        "target": config["target"],
        "ticker_to_id": ticker_to_id,
        "columns": config["columns"],
        "num_unique_embeddings": num_tickers,
        "embedding_dim": config["embedding_dim"],
        "lag_features": config["lag_features"],
        "lags": config["lags"],
        "classification": config["classification"],
        "out_dim": config["out_dim"],
        "transformer": config["transformer"],
        "seq_len": config["seq_len"],
        "d_model": config["d_model"],
        "activation_fn": config["activation_fn"],
        "encoder_layers": config["encoder_layers"],
        "n_atten_head": config["n_atten_head"],
        "val_start": val_start,
    }
    torch.save(model.state_dict(), "weights/embedded_model.pth")
    torch.save(model_config, "weights/embedded_config.pth")
    print("Model weights and config stored")
