from backtesting import Backtest
from numpy.typing import NDArray
from data.preprocess import load_dataset, preprocess_dataset
import pandas as pd
import torch
from models.EncoderTransformer import EncoderTransformer
from models.mlp import MLP
from strategy import regression
from strategy.classification import Classification
from strategy.rank_regression import RankRegression
from strategy.regression import Regression
from strategy.simple_nn_regression import SimpleNNRegression
from data import single_ticker_minimal, multiple_ticker
import numpy as np
from train.pipeline import create_sequential_windows_multiple_tickers


def precompute_outputs(ticker: str = "^GSPC") -> None:
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
            True,
            config["embedding_dim"],
            config["num_unique_embeddings"],
            config,
        ).to(device)
    df = pd.read_parquet("datasets/dataframe.parquet")
    df = df[df["Ticker"] == ticker]
    cols = config["columns"]
    # df, cols = multiple_ticker.get_feature_target_df(
    #     [ticker],
    #     {
    #         "HORIZON": config["horizon"],
    #         "lag_features": config["lag_features"],
    #         "lags": config["lags"],
    #         "columns": config["columns"],
    #         "cross_sectional_z_score": config["cross_sectional_z_score"],
    #     },
    # )
    #
    # df["ticker_id"] = df["Ticker"].map(config["ticker_to_id"])
    # df = df.dropna()
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
        ticker_tensor = torch.tensor(df["ticker_id"].values, device=device)
    state_dict = torch.load("weights/embedded_model.pth")
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        outputs = model(
            X_tensor,
            ticker_tensor,
        )
    if config["classification"]:
        outputs = torch.argmax(outputs, 1)
    torch.save(outputs.squeeze().cpu().numpy(), "backtest/predictions.pth")
    torch.save(df.index[0], "backtest/start_date.pth")
    torch.save(df.index[-1], "backtest/end_date.pth")
    print("Outputs precomputed and saved")


# %%
def backtest(ticker: str = "^GSPC") -> None:
    df = load_dataset(ticker)
    df = preprocess_dataset(df)
    config_path = "weights/config.pth"
    config = torch.load("weights/embedded_config.pth", weights_only=False)
    start = torch.load("backtest/start_date.pth", weights_only=False)
    end = torch.load("backtest/end_date.pth", weights_only=False)

    df = df[start:end]
    strategy = (
        Classification
        if config["classification"]
        else RankRegression
        if config["cross_sectional_z_score"]
        else Regression
    )
    bt = Backtest(
        df, strategy, cash=100_000, commission=0.0, finalize_trades=True
    )

    stats = bt.run()
    print(stats)
    bt.plot(
        filename="plots/plot",
        resample=True,
        smooth_equity=True,
        plot_volume=False,
        open_browser=False,
    )
