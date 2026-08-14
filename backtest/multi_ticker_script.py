from functools import partial

from optuna.pruners import MedianPruner
import pandas as pd
from sklearn.model_selection import train_test_split
from backtest.backtest import (
    get_common_dates,
    multi_ticker_backtest,
    precompute_tickers,
)
from data import multiple_ticker
from train import nn_objective
from train.pipeline import walk_forward_validation_outer_embd
import torch
import optuna

from train.ticker_embedding_dataset import TickerEmbeddingDataset


def run():
    df = pd.read_parquet("datasets/dataframe.parquet")
    start, end = get_common_dates(df)
    with open("datasets/ETFs.txt", "r") as f:
        tickers = [line.strip() for line in f if line.strip()]
    out_dict = precompute_tickers(tickers, start, end, False)
    multi_ticker_backtest(start, end, outputs_dict=out_dict, n=5, plot=True)


def walk_forward_validation():
    features: list[str] = [
        "roc_slow",
        "roc_fast",
        "rsi_slow",
        "rsi_fast",
        "macd_hist",
        "stochk",
        "stochd",
        "adx",
        "pct_b",
        "atr_norm",
        "dist_ema_fast",
        "dist_ema_slow",
        "dist_sma_fast",
        "dist_sma_slow",
        "obv_roc",
        "volume_ratio",
        # "Close",
        # "Open",
        # "High",
        # "Low",
        # "Volume",
    ]
    df = pd.read_parquet("datasets/dataframe.parquet")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start, end = get_common_dates(df)
    with open("datasets/ETFs.txt", "r") as f:
        tickers = [line.strip() for line in f if line.strip()]
    walk_forward_validation_outer_embd(
        tickers,
        start,
        end,
        10,
        20,
        1,
        20,
        features,
        "target_return_cross_rank",
        device,
        batch_size=256,
        embedding_dim=3,
        epochs=1000,
        trials_timeout=600,
        n_trials=4,
    )
