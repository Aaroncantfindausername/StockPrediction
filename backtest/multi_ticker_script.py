import pandas as pd
import pickle
from backtest.backtest import (
    get_common_dates,
    multi_ticker_backtest,
    precompute_tickers,
)
from train.pipeline import walk_forward_validation_outer_embd
import torch


def run():
    df = pd.read_parquet("datasets/dataframe.parquet")
    start, end = get_common_dates(df)
    with open("datasets/ETFs.txt", "r") as f:
        tickers = [line.strip() for line in f if line.strip()]
    out_dict = precompute_tickers(tickers, start, end, False)
    multi_ticker_backtest(start, end, outputs_dict=out_dict, n=5, plot=True)


def backtest_full():
    print("BEGINNING BACKTEST FULL")
    start = pd.Timestamp("2017-02-13")
    end = pd.Timestamp("2026-02-12")
    with open("backtest/precomputed_data/data.pkl", "rb") as f:
        outputs_dict = pickle.load(f)
    multi_ticker_backtest(
        start, end, outputs_dict=outputs_dict, plot=True, baseline=False
    )
    print("BACKTEST COMPLETE")


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
        30,
        features,
        "target_return_cross_rank",
        device,
        batch_size=256,
        embedding_dim=3,
        epochs=1000,
        trials_timeout=1000,
        n_trials=20,
        save=True,
    )
    print("WALK FORWARD VALIDATION SCRIPT COMPLETE")
