from typing import Any, Dict, Tuple, final
import talib
from numpy.typing import NDArray
import pandas as pd
import numpy as np

from data.preprocess import rolling_z_score


def compute_features_and_labels(
    df: pd.DataFrame, params: Dict[str, Any] = {}
) -> Tuple[list[str], pd.DataFrame]:
    df = df.copy()
    HORIZON: int = params.get("HORIZON", 5)
    columns: list[str] = [
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
        "obv_roc",
        "volume_ratio",
    ]
    # Adding indicators
    rsi_fast_t = params.get("rsi_fast_t", 7)
    rsi_slow_t = params.get("rsi_slow_t", 14)
    df[f"rsi_slow"] = talib.RSI(df["Close"], timeperiod=rsi_fast_t)

    df[f"rsi_fast"] = talib.RSI(df["Close"], timeperiod=rsi_slow_t)

    macd_fastperiod = params.get("macd_fastperiod", 12)
    macd_slowperiod = params.get("macd_slowperiod", 26)
    macd_signalperiod = params.get("macd_signalperiod", 9)
    df["macd_hist"] = talib.MACD(
        df["Close"],
        fastperiod=macd_fastperiod,
        slowperiod=macd_slowperiod,
        signalperiod=macd_signalperiod,
    )[2]

    stoch_fastk = params.get("stoch_fastk", 14)
    stoch_slowk = params.get("stoch_slowk", 3)
    stoch_slowd = params.get("stoch_slowd", 3)
    df["stochk"], df["stochd"] = talib.STOCH(
        df["High"],
        df["Low"],
        df["Close"],
        stoch_fastk,
        stoch_slowk,
        slowd_period=stoch_slowd,
    )

    adx_t = params.get("adx_t", 14)
    df["adx"] = talib.ADX(df["High"], df["Low"], df["Close"], adx_t)
    bbands_t = params.get("bbands_t", 20)
    df["bbands_upper"], df["bbands_middle"], df["bbands_lower"] = talib.BBANDS(
        df["Close"], timeperiod=bbands_t
    )
    df["pct_b"] = (df["Close"] - df["bbands_lower"]) / (
        df["bbands_upper"] - df["bbands_lower"]
    )
    atr_t = params.get("atr", 14)
    df["atr"] = talib.ATR(df["High"], df["Low"], df["Close"], atr_t)
    df["atr_norm"] = df["atr"] / df["Close"]

    ema_fast_t = params.get("ema_fast_t", 50)
    ema_slow_t = params.get("ema_slow_t", 200)
    df["ema_fast"] = talib.EMA(df["Close"], timeperiod=ema_fast_t)
    df["ema_slow"] = talib.EMA(df["Close"], timeperiod=ema_slow_t)
    df["dist_ema_fast"] = (df["Close"] - df["ema_fast"]) / df["ema_fast"]
    df["dist_ema_slow"] = (df["Close"] - df["ema_slow"]) / df["ema_slow"]
    obv_roc_t = params.get("obv_roc_t", 5)
    df["obv"] = talib.OBV(df["Close"], df["Volume"])
    df["obv_roc"] = talib.ROC(df["obv"], timeperiod=obv_roc_t)
    volume_sma_t = params.get("volume_sma_t", 20)
    df["vol_sma"] = talib.SMA(df["Volume"], timeperiod=volume_sma_t)
    df["volume_ratio"] = df["Volume"] / df["vol_sma"]
    # Calculate forward return
    df["future_close"] = df["Close"].shift(-HORIZON)
    df["target_return"] = (df["future_close"] - df["Close"]) / df["Close"]
    df["target_scaled"] = df["target_return"] / df["atr"]
    for col in columns:
        df[f"{col}_z"] = rolling_z_score(df[col])
    df["target_scaled_z"] = rolling_z_score(df["target_scaled"])
    df["target_return_z"] = rolling_z_score(df["target_return"])
    # Lag features
    lags = params.get("lags", [1, 2, 3, 5])
    lagged_features = [
        "rsi_fast",
        "macd_hist",
        "pct_b",
        "atr_norm",
        "dist_ema_fast",
        "volume_ratio",
    ]
    for feat in lagged_features:
        for lag in lags:
            df[f"{feat}_lag{lag}_z"] = df[f"{feat}_z"].shift(lag)
            columns.append(f"{feat}_lag{lag}")
    return columns, df
