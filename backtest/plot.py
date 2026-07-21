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
from models.basic_nn import MLP
from models.embedded_mlp import EmbeddedMLP
from train.pipeline import evaluate
import pandas as pd


def plot_predictions() -> None:
    config = torch.load("weights/config.pth")
    model = MLP(config["input_dim"], 1, config)

    # Load dataset
    ticker: str = config.get("ticker")
    df = load_dataset(ticker)
    df = preprocess_dataset(df)

    cols, df = (
        single_ticker_minimal.compute_features_and_labels(
            df, {"HORIZON": config["horizon"]}
        )
        if not config["lag_features"]
        else single_ticker_lagged.compute_features_and_labels(
            df, {"HORIZON": config["horizon"], "lags": config["lags"]}
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
    outputs_unzscored = (
        pd.Series(outputs.squeeze()) * df[f"{target}_std"]
        + df[f"{target}_mean"]
    )
    fig, ax = plt.subplots()
    ax.plot(
        df.index,
        outputs_unzscored,
        label="Outputs",
    )
    ax.plot(df.index, df[config["target"][:-2]], label="Targets")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("plots/fig")


def plot_predictions_embd(ticker: str = "^GSPC") -> None:
    config = torch.load("weights/embedded_config.pth")
    model = EmbeddedMLP(
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
        {"HORIZON": config["horizon"], "lag_features": config["lag_features"]},
    )

    df = df.dropna()
    X: NDArray[np.float32] = df[cols].to_numpy()
    ticker_to_id = config["ticker_to_id"]
    df["ticker_id"] = df["Ticker"].map(ticker_to_id)
    state_dict = torch.load("weights/embedded_model.pth")
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        outputs = model(torch.tensor(X), torch.tensor(df["ticker_id"].values))
    fig, ax = plt.subplots()
    ax.plot(
        df.index, inverse_rolling_z_score(pd.Series(outputs)), label="Outputs"
    )
    ax.plot(df.index, df[config["target"][:-2]], label="Targets")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("plots/predictions_targets")
