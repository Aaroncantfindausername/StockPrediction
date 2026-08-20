from operator import xor
from typing import Any, Dict, Tuple
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
    model: torch.nn.Module | None = None,
    config: Dict[str, Any] | None = None,
    save: bool = True,
) -> NDArray | None:
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model is None
        else next(model.parameters()).device
    )
    if config is None:
        config = torch.load("weights/embedded_config.pth", weights_only=False)

    assert config is not None
    if model is None:
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
    tickers: list[str],
    start_date: pd.DatetimeIndex,
    end_date: pd.DatetimeIndex,
) -> None:

    for t in tickers:
        backtest(t, start_date, end_date)


def precompute_tickers(
    tickers: list[str],
    start_date: pd.DatetimeIndex,
    end_date: pd.DatetimeIndex,
    save: bool = True,
    df: pd.DataFrame | None = None,
    model: torch.nn.Module | None = None,
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    outputs_dict: Dict[str, Any] = {}
    for t in tickers:
        outputs = precompute_outputs(
            start_date, end_date, t, df=df, save=save, model=model
        )
        if outputs is not None:
            outputs_dict[t] = outputs
    if not save:
        return outputs_dict


def multi_ticker_backtest(
    start: pd.Timestamp,
    end: pd.Timestamp,
    ticker_list_path: str | None = None,
    outputs_dict: Dict[str, Any] | None = None,
    n: int = 5,
    plot: bool = False,
    save_path: str | None = None,
    baseline: bool = False,
) -> pd.DataFrame:
    dfs = []
    scores = []
    if ticker_list_path is not None:
        with open(ticker_list_path, "r") as f:
            tickers = [line.strip() for line in f if line.strip()]
            f.close()
    elif outputs_dict is not None:
        tickers = outputs_dict.keys()
    else:
        assert False, "Function accepts one of ticker_list_path or outputs_dict"
    for ticker in tickers:
        df = load_dataset(ticker)
        df = preprocess_dataset(df)
        df.rename(columns={"Close": ticker}, inplace=True)
        if ticker_list_path is not None:
            predictions = torch.load(
                f"backtest/precomputed_data/{ticker}_predictions.pth",
                weights_only=False,
            )
        elif outputs_dict is not None:
            predictions = outputs_dict[ticker]
        else:
            assert False, (
                "Function accepts one of ticker_list_path or outputs_dict"
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
    if baseline:
        baseline_strategy = bt.Strategy(
            "Random baseline",
            [
                bt.algos.RunWeekly(),
                bt.algos.SelectRandomly(5),
                bt.algos.WeighEqually(),
                bt.algos.Rebalance(),
            ],
        )
        result = bt.backtest.benchmark_random(
            backtest, baseline_strategy, nsim=100
        )
    else:
        result = bt.run(backtest)
    result.display()
    if plot:
        if baseline:
            result.plot_histogram()
        else:
            result.plot()
        plt.show()
    if save_path is not None:
        result.stats.to_pickle(f"{save_path}.pkl")
    return result.stats


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
