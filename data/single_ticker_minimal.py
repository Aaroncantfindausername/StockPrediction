from typing import Tuple, final
import talib
from numpy.typing import NDArray
import pandas as pd
import numpy as np

from data.preprocess import load_dataset, preprocess_dataset


def rolling_z_score(series: pd.Series, window=252, min_periods=50) -> pd.Series:
    rolling_mean = series.rolling(window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window, min_periods=min_periods).std()
    z = (series - rolling_mean) / rolling_std
    return z.astype(np.float32)


def get_features_labels(
    df: pd.DataFrame, HORIZON: int = 5
) -> Tuple[NDArray, NDArray]:
    columns: list[str] = [
        "rsi_7",
        "rsi_14",
        "macd_hist",
        "stochk",
        "stochd",
        "adx_14",
        "pct_b_20",
        "atr_14",
        "dist_ema_50",
        "dist_ema_200",
        "obv_roc_5",
        "volume_ratio",
    ]
    # Adding indicators
    df["rsi_7"] = talib.RSI(df["Close"], timeperiod=7)

    df["rsi_14"] = talib.RSI(df["Close"], timeperiod=14)

    df["macd_hist"] = talib.MACD(
        df["Close"], fastperiod=12, slowperiod=26, signalperiod=9
    )[2]
    df["stochk"], df["stochd"] = talib.STOCH(df["High"], df["Low"], df["Close"])
    df["adx_14"] = talib.ADX(df["High"], df["Low"], df["Close"])
    df["bbands_20_upper"], df["bbands_20_middle"], df["bbands_20_lower"] = (
        talib.BBANDS(df["Close"], timeperiod=20)
    )
    df["pct_b_20"] = (df["Close"] - df["bbands_20_lower"]) / (
        df["bbands_20_upper"] - df["bbands_20_lower"]
    )
    df["atr_14"] = talib.ATR(df["High"], df["Low"], df["Close"])
    df["atr_14_norm"] = df["atr_14"] / df["Close"]
    df["ema_50"] = talib.EMA(df["Close"], timeperiod=50)
    df["ema_200"] = talib.EMA(df["Close"], timeperiod=200)
    df["dist_ema_50"] = (df["Close"] - df["ema_50"]) / df["ema_50"]
    df["dist_ema_200"] = (df["Close"] - df["ema_200"]) / df["ema_200"]
    df["obv"] = talib.OBV(df["Close"], df["Volume"])
    df["obv_roc_5"] = talib.ROC(df["obv"], timeperiod=5)
    df["vol_sma_20"] = talib.SMA(df["Volume"], timeperiod=20)
    df["volume_ratio"] = df["Volume"] / df["vol_sma_20"]
    # Calculate forward return
    df["future_close_5"] = df["Close"].shift(-HORIZON)
    df["target_return"] = (df["future_close_5"] - df["Close"]) / df["Close"]
    for col in columns:
        df[f"{col}_z"] = rolling_z_score(df[col])
    df.dropna(inplace=True)
    return df[[f"{c}_z" for c in columns]].to_numpy(), df[
        "target_return"
    ].to_numpy()
