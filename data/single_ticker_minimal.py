from typing import Any, Dict, Tuple
import talib
import pandas as pd
from data.preprocess import rolling_z_score
import numpy as np


def compute_features_and_labels(
    df: pd.DataFrame, params: Dict[str, Any] = {}
) -> Tuple[list[str], pd.DataFrame]:
    df = df.copy()
    HORIZON: int = params.get("HORIZON", 5)
    columns = params.get("columns", ["Close", "Open", "High", "Low", "Volume"])
    columns_z = columns.copy()
    # Adding indicators

    roc_fast_t = params.get("roc_fast_t", 5)
    roc_slow_t = params.get("roc_slow_t", 20)
    df[f"roc_slow"] = talib.ROC(df["Close"], timeperiod=roc_fast_t)

    df[f"roc_fast"] = talib.ROC(df["Close"], timeperiod=roc_slow_t)

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
    df["close_shifted"] = df["Close"].shift(-HORIZON)
    df["target_return"] = (df["close_shifted"] - df["Close"]) / df["Close"]
    df["target_scaled"] = df["target_return"] / df["atr"]
    df["forward_return_1d"] = df["Close"].pct_change().shift(-1)
    df["avg_1d_return"] = (
        df["forward_return_1d"].rolling(HORIZON).mean().shift(-HORIZON + 1)
    )

    if HORIZON > 1:
        df["std_1d_return"] = (
            df["forward_return_1d"].rolling(HORIZON).std().shift(-HORIZON + 1)
        )
        df["avg+std_1d_return"] = df["avg_1d_return"] + 1 * df["std_1d_return"]
        df["avg+std_1d_return_z"] = rolling_z_score(df["avg+std_1d_return"])
    df["target_return_over_atr"] = df["target_return"] / df["atr"]
    for i, col in enumerate(columns):
        df[f"{col}_z"] = rolling_z_score(df[col])
        columns_z[i] = f"{col}_z"
    df["target_scaled_z"] = rolling_z_score(df["target_scaled"])
    df["target_return_z"] = rolling_z_score(df["target_return"])
    df["target_return_over_atr_z"] = rolling_z_score(
        df["target_return_over_atr"]
    )
    df["avg_1d_return_z"] = rolling_z_score(df["avg_1d_return"])
    df["target_return_binary"] = (df["target_return_z"] > 0.0).astype(int)
    threshold = params.get("3class_threshold", 0.5)
    df["target_return_3_class"] = pd.cut(
        df["target_return_z"],
        bins=[-np.inf, -threshold, threshold, np.inf],
        labels=[0, 1, 2],
    )
    df["target_return_z_clipped"] = df["target_return_z"].clip(-2.5, 2.5)
    return columns_z, df
