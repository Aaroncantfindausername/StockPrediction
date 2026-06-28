from typing import Tuple, final
import talib
from numpy.typing import NDArray
import pandas as pd

from data.preprocess import load_dataset, preprocess_dataset


def rolling_z_score(series: pd.Series, window=252, min_periods=50) -> pd.Series:
    rolling_mean = series.rolling(window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window, min_periods=min_periods).std()
    z = (series - rolling_mean) / rolling_std
    return z


def get_data() -> Tuple[NDArray, NDArray]:
    ticker: str = "^GSPC"
    HORIZON: int = 5
    df = load_dataset(ticker)
    df = preprocess_dataset(df)
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
    df["rsi_7"] = talib.RSI(df["close"], timeperiod=7)

    df["rsi_14"] = talib.RSI(df["close"], timeperiod=14)

    df["macd_hist"] = talib.MACD(
        df["close"], fastperiod=12, slowperiod=26, signalperiod=9
    )[2]
    df["stochk"], df["stochd"] = talib.STOCH(df["high"], df["low"], df["close"])
    df["adx_14"] = talib.ADX(df["high"], df["low"], df["close"])
    df["bbands_20_upper"], df["bbands_20_middle"], df["bbands_20_lower"] = (
        talib.BBANDS(df["close"], timeperiod=20)
    )
    df["pct_b_20"] = (df["close"] - df["bbands_20_lower"]) / (
        df["bbands_20_upper"] - df["bbands_20_lower"]
    )
    df["atr_14"] = talib.ATR(df["high"], df["low"], df["close"])
    df["atr_14_norm"] = df["atr_14"] / df["close"]
    df["ema_50"] = talib.EMA(df["close"], timeperiod=50)
    df["ema_200"] = talib.EMA(df["close"], timeperiod=200)
    df["dist_ema_50"] = (df["close"] - df["ema_50"]) / df["ema_50"]
    df["dist_ema_200"] = (df["close"] - df["ema_200"]) / df["ema_200"]
    df["obv"] = talib.OBV(df["close"], df["volume"])
    df["obv_roc_5"] = talib.ROC(df["obv"], timeperiod=5)
    df["vol_sma_20"] = talib.SMA(df["volume"], timeperiod=20)
    df["volume_ratio"] = df["volume"] / df["vol_sma_20"]
    # Calculate forward return
    df["future_close_5"] = df["close"].shift(-HORIZON)
    df["target_return"] = (df["future_close_5"] - df["close"]) / df["close"]
    for col in columns:
        df[f"{col}_z"] = rolling_z_score(df[col])
    df.dropna(inplace=True)
    return df[[f"{c}_z" for c in columns]].to_numpy(), df[
        "target_return"
    ].to_numpy()
