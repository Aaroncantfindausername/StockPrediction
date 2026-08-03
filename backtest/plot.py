from numpy.typing import NDArray
import torch
import matplotlib.pyplot as plt
import numpy as np
from data import multiple_ticker, single_ticker_lagged, single_ticker_minimal
from data.preprocess import (
    inverse_rolling_z_score,
    load_dataset,
    preprocess_dataset,
)
from models.EncoderTransformer import EncoderTransformer
from models.mlp import MLP
from train.pipeline import create_sequential_windows_multiple_tickers, evaluate
import pandas as pd


def plot_predictions(unzscore: bool = False) -> None:
    config = torch.load("weights/config.pth")
    model = MLP(
        config["input_dim"],
        1 if not config["classification"] else config["out_dim"],
        config,
    )

    # Load dataset
    ticker: str = config.get("ticker")
    df = load_dataset(ticker)
    df = preprocess_dataset(df)

    cols, df = (
        single_ticker_minimal.compute_features_and_labels(
            df, {"HORIZON": config["horizon"], "columns": config["columns"]}
        )
        if not config["lag_features"]
        else single_ticker_lagged.compute_features_and_labels(
            df,
            {
                "HORIZON": config["horizon"],
                "lags": config["lags"],
                "columns": config["columns"],
            },
        )
    )
    target = config["target"][:-2]
    df[f"{target}_std"] = df[target].rolling(252, min_periods=50).std()
    df[f"{target}_mean"] = df[target].rolling(252, min_periods=50).mean()
    df = df.dropna()
    X: NDArray[np.float32] = df[cols].to_numpy()
    state_dict = torch.load("weights/model.pth")
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        outputs = model(torch.tensor(X))
    outputs_z = pd.Series(outputs.squeeze(), index=df.index)
    outputs_unzscored = outputs_z * df[f"{target}_std"] + df[f"{target}_mean"]

    if unzscore:
        line1 = outputs_unzscored
        line2 = df[config["target"][:-2]]
    else:
        line1 = outputs_z
        line2 = df[config["target"]]
    fig, ax = plt.subplots()
    ax.plot(
        df.index,
        line1,
        label="Outputs",
    )
    ax.plot(df.index, line2, label="Targets")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("plots/fig")


def plot_predictions_embd(
    ticker: str = "^GSPC", plot_features: bool = False
) -> None:
    device = "cpu"
    config = torch.load("weights/embedded_config.pth", weights_only=False)
    out_dim = config["out_dim"] if config["classification"] else 1
    if config["transformer"]:
        model = EncoderTransformer(
            config["input_dim"],
            out_dim,
            config["d_model"],
            config["encoder_layers"],
            config["n_atten_head"],
            config["num_unique_embeddings"],
            True,
            config["embedding_dim"],
            config["hidden_dim"],
            0.0,
            config["activation_fn"],
            config["seq_len"],
            device,
        ).to(device)
    else:
        model = MLP(
            config["input_dim"],
            1 if not config["classification"] else config["out_dim"],
            config["embedding_dim"],
            config["num_unique_embeddings"],
            config,
        )

    # with open("datasets/subset_tickers.txt", "r") as f:
    #     tickers = [line.strip() for line in f if line.strip()]
    #     f.close()
    #
    # df, cols = multiple_ticker.get_feature_target_df(
    #     tickers, {"HORIZON": config["horizon"]}
    # )

    df, cols = multiple_ticker.get_feature_target_df(
        [ticker],
        {
            "HORIZON": config["horizon"],
            "lag_features": config["lag_features"],
            "columns": config["columns"],
        },
    )

    df["ticker_id"] = df["Ticker"].map(config["ticker_to_id"])
    df = df.dropna()
    if config["transformer"]:
        X_tensor, ticker_tensor, y_tensor = (
            create_sequential_windows_multiple_tickers(
                df,
                cols,
                config["target"],
                config["seq_len"],
            )
        )
        X_tensor = X_tensor.to(device)
        y_tensor = y_tensor.to(device)
        ticker_tensor = ticker_tensor.to(device)
        if config["seq_len"] > 1:
            df = df.iloc[: -config["seq_len"] + 1]
    else:
        X: NDArray[np.float32] = df[cols].to_numpy()
        X_tensor = torch.tensor(X, device=device)
        ticker_to_id = config["ticker_to_id"]
        df["ticker_id"] = df["Ticker"].map(ticker_to_id)
        ticker_tensor = (torch.tensor(df["ticker_id"].values, device=device),)
    state_dict = torch.load("weights/embedded_model.pth")
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        outputs = model(
            X_tensor,
            ticker_tensor,
        )
    fig, ax = plt.subplots()
    ax.plot(df.index, outputs.detach().cpu().numpy(), label="Outputs")
    ax.plot(df.index, df[config["target"]], label="Targets")
    if plot_features:
        for i in range(len(config["columns"])):
            ax.plot(
                df.index,
                df[f"{config['columns'][i]}_z"],
                label=config["columns"][i],
            )
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("plots/predictions_targets")
