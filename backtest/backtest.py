from typing import Tuple
import matplotlib.pyplot as plt
from backtesting import Backtest
import json
import bt
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


def precompute_outputs(
    start_date: pd.DatetimeIndex,
    end_date: pd.DatetimeIndex,
    ticker: str = "^GSPC",
    df: pd.DataFrame | None = None,
    save: bool = True,
) -> NDArray | None:
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
    if df is None:
        df = pd.read_parquet("datasets/dataframe.parquet")
    df = df[df["Ticker"] == ticker]
    df = df[start_date:end_date]
    cols = config["columns"]
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
    if save:
        torch.save(
            outputs.squeeze().cpu().numpy(),
            f"backtest/precomputed_data/{ticker}_predictions.pth",
        )
        print("Outputs precomputed and saved")
        return
    else:
        return outputs.squeeze().cpu().numpy()


# %%
def backtest(
    ticker: str, start_date: pd.DatetimeIndex, end_date: pd.DatetimeIndex
) -> None:
    df = load_dataset(ticker)
    df = preprocess_dataset(df)
    config = torch.load("weights/embedded_config.pth", weights_only=False)
    start = torch.load(f"backtest/{ticker}_start_date.pth", weights_only=False)
    end = torch.load(f"backtest/{ticker}_end_date.pth", weights_only=False)

    df = df[start:end]
    strategy = (
        Classification
        if config["classification"]
        else RankRegression
        if config["cross_sectional_z_score"]
        else Regression
    )
    bt = Backtest(
        df,
        strategy,
        cash=100_000,
        commission=0.0,
        finalize_trades=True,
    )

    stats = bt.run(predictions_path=f"backtest/{ticker}_predictions.pth")
    print(ticker)
    print(stats)
    bt.plot(
        filename=f"plots/{ticker}_backtest",
        resample=True,
        smooth_equity=True,
        plot_volume=False,
        open_browser=False,
    )


def backtest_tickers(
    ticker_list_path: str,
    start_date: pd.DatetimeIndex,
    end_date: pd.DatetimeIndex,
) -> None:

    with open(ticker_list_path, "r") as f:
        tickers = [line.strip() for line in f if line.strip()]
        f.close()
    for t in tickers:
        backtest(t, start_date, end_date)


def precompute_tickers(
    ticker_list_path: str,
    start_date: pd.DatetimeIndex,
    end_date: pd.DatetimeIndex,
) -> None:
    with open(ticker_list_path, "r") as f:
        tickers = [line.strip() for line in f if line.strip()]
    for t in tickers:
        precompute_outputs(start_date, end_date, t)


def multi_ticker_backtest(
    ticker_list_path: str,
    start: pd.DatetimeIndex,
    end: pd.DatetimeIndex,
    n=5,
) -> None:
    with open(ticker_list_path, "r") as f:
        tickers = [line.strip() for line in f if line.strip()]
        f.close()
    config = torch.load("weights/embedded_config.pth", weights_only=False)
    dfs = []
    scores = []
    for ticker in tickers:
        df = load_dataset(ticker)
        df = preprocess_dataset(df)
        df.rename(columns={"Close": ticker}, inplace=True)
        predictions = torch.load(
            f"backtest/precomputed_data/{ticker}_predictions.pth",
            weights_only=False,
        )
        score = pd.Series(predictions, index=df[start:end].index, name=ticker)
        scores.append(score)
        df = df[start:end]
        dfs.append(df[ticker])
    df_wide = pd.concat(dfs, axis=1, join="inner")
    scores_df = pd.concat(scores, axis=1, join="inner")
    weights = pd.DataFrame(0.0, index=df_wide.index, columns=df_wide.columns)

    for date in df_wide.index:
        s = scores_df.loc[date]
        assert len(s) > n * 2, "Scores should not have any missing values"
        top_n = s.nlargest(n).index
        weights.loc[date, top_n] = 1.0 / n  # Equal allocation

    strategy = bt.Strategy(
        "Top 5 monthly rotation",
        [
            bt.algos.RunWeekly(),
            bt.algos.SelectAll(),
            bt.algos.WeighTarget(weights),
            bt.algos.Rebalance(),
        ],
    )
    backtest = bt.Backtest(
        strategy,
        df_wide,
        initial_capital=100_000,
        integer_positions=False,
        # commissions=lambda q, p: max(1, abs(q) * 0.0005),
    )
    result = bt.run(backtest)
    result.display()
    result.plot()
    plt.show()


def get_common_dates(
    df: pd.DataFrame,
) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    start_dates = []
    end_dates = []
    for t in df["Ticker"].unique():
        d = df.query("Ticker == @t").index
        start_dates.append(d[0])
        end_dates.append(d[-1])
    return (max(start_dates), min(end_dates))
