import yfinance as yf
import pandas as pd
import numpy as np
import os


def download_dataset(ticker: str) -> None:
    if os.path.exists(f"datasets/{ticker}.csv"):
        print(f"{ticker}.csv already exists skipping")
        return
    data = yf.download(ticker, start="2000-01-01", end="2026-8-01")
    if data is None:
        raise ValueError("Failed to download dataset")
    data.to_csv(f"datasets/{ticker}.csv")


def download_list_of_tickers(tickers: list[str]) -> None:
    for t in tickers:
        download_dataset(t)


def load_dataset(data: str) -> pd.DataFrame:
    return pd.read_csv(f"datasets/{data}.csv")


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(0, inplace=True)
    df.drop(1, inplace=True)
    df.rename(columns={"Price": "Date"}, inplace=True)
    df.index = pd.to_datetime(df["Date"])
    # df = df.drop("Price", axis=1)
    df = df[["Close", "Open", "Low", "High", "Volume"]].astype(np.float32)
    assert len(df) > 10, "df should have more than 10 rows"
    return df


def rolling_z_score(series: pd.Series, window=252, min_periods=50) -> pd.Series:
    rolling_mean = series.rolling(window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window, min_periods=min_periods).std()
    z = (series - rolling_mean) / rolling_std
    return z.astype(np.float32)


def robust_z_score(x, clip_threshold=3):
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0:
        return pd.Series(0.0, index=x.index)
    return (
        ((x - med) / (mad * 1.4826))
        .clip(lower=-clip_threshold, upper=clip_threshold)
        .astype(np.float32)
    )


def inverse_rolling_z_score(
    series: pd.Series, window=252, min_periods=50
) -> pd.Series:
    rolling_mean = series.rolling(window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window, min_periods=min_periods).std()
    z = series * rolling_std + rolling_mean
    return z.astype(np.float32)
