from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
from models.EncoderTransformer import EncoderTransformer
from models.mlp import MLP
from data.preprocess import load_dataset, preprocess_dataset
from train.pipeline import (
    create_sequential_windows_single_tickers,
    evaluate,
    train_full,
    train_val_test_split_loader,
)
from data import single_ticker_lagged, single_ticker_minimal


def train() -> None:
    # Hyperparameters
    features: list[str] = [
        # "roc_slow",
        # "roc_fast",
        # "rsi_slow",
        # "rsi_fast",
        # "macd_hist",
        # "stochk",
        # "stochd",
        # "adx",
        # "pct_b",
        # "atr_norm",
        # "dist_ema_fast",
        # "dist_ema_slow",
        # "obv_roc",
        # "volume_ratio",
        "Close",
        # "Open",
        # "High",
        # "Low",
        # "Volume",
    ]
    config = {
        "batch_size": 256,
        "lr": 1e-5,
        "epochs": 1000,
        "n_layers": 3,
        "hidden_dim": 128,
        "hidden_dim_decay": 0.5,
        "out_dim": 2,
        "dropout": 0.5,
        "weight_decay": 1e-3,
        "seed": 87,
        "scheduler_patience": 2,
        "early_stop_patience": 30,
        "val_ratio": 0.2,
        "test_ratio": 0.1,
        "columns": features,
        "horizon": 10,
        "target": "target_return_binary",
        "ticker": "^GSPC",
        "lag_features": False,
        "lags": [1, 2, 5],
        "shuffle_train": True,
        "classification": True,
        "transformer": True,
        "seq_len": 50,
        "d_model": 32,
        "activation_fn": "relu",
        "encoder_layers": 2,
        "n_atten_head": 8,  # Even n.o heads
    }

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    ticker: str = config["ticker"]
    df = load_dataset(ticker)
    df = preprocess_dataset(df)

    cols, df = (
        single_ticker_lagged.compute_features_and_labels(
            df, {"HORIZON": config["horizon"], "lags": config["lags"]}
        )
        if config["lag_features"]
        else single_ticker_minimal.compute_features_and_labels(
            df, {"HORIZON": config["horizon"]}
        )
    )
    df = df.dropna()

    if config["transformer"]:
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
        X_train, y_train = create_sequential_windows_single_tickers(
            df[df.index <= train_end],
            cols,
            config["target"],
            config["seq_len"],
        )
        train_dataset = TensorDataset(X_train, y_train)

        X_val, y_val = create_sequential_windows_single_tickers(
            df[(df.index >= val_start) & (df.index <= val_end)],
            cols,
            config["target"],
            config["seq_len"],
        )
        val_dataset = TensorDataset(X_val, y_val)

        X_test, y_test = create_sequential_windows_single_tickers(
            df[df.index >= test_start],
            cols,
            config["target"],
            config["seq_len"],
        )
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, config["batch_size"], True)
        val_loader = DataLoader(val_dataset, config["batch_size"], False)
        test_loader = DataLoader(test_dataset, config["batch_size"], False)
    else:
        X: NDArray[np.float32] = df[cols].to_numpy()
        y: NDArray[np.float32] = df[config["target"]].to_numpy()
        # %%

        # Load data
        train_loader, val_loader, test_loader, input_dim = (
            train_val_test_split_loader(
                X,
                y,
                batch_size=config["batch_size"],
                val_ratio=config["val_ratio"],
                test_ratio=config["test_ratio"],
                shuffle_train=config["shuffle_train"],
            )
        )
    # Model, loss, optimizer
    input_dim = len(cols)
    out_dim = config["out_dim"] if config["classification"] else 1
    if config["transformer"]:
        model = EncoderTransformer(
            len(cols),
            out_dim,
            config["d_model"],
            config["encoder_layers"],
            config["n_atten_head"],
            1,
            False,
            0,
            config["hidden_dim"],
            config["dropout"],
            config["activation_fn"],
            config["seq_len"],
            device,
        ).to(device)
    else:
        model = MLP(
            input_dim,
            out_dim,
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
        factor=0.2,
        threshold=0.0,
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
        classification=config["classification"],
    )
    test_loss = evaluate(
        model, test_loader, criterion, device, config["classification"]
    )
    print(f"\nTest Loss: {test_loss:.5f}")
    model_config = {
        "input_dim": input_dim,
        "n_layers": config["n_layers"],
        "hidden_dim": config["hidden_dim"],
        "hidden_dim_decay": config["hidden_dim_decay"],
        "horizon": config["horizon"],
        "target": config["target"],
        "ticker": config["ticker"],
        "columns": config["columns"],
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
    }
    torch.save(model.state_dict(), "weights/model.pth")
    torch.save(model_config, "weights/config.pth")
    print("Model weights and config stored")
