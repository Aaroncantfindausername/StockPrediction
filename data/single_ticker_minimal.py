from typing import Tuple
import talib
from numpy.typing import NDArray
import pandas as pd

from data.preprocess import load_dataset, preprocess_dataset


def get_data() -> Tuple[NDArray, NDArray]:
    ticker: str = "^GSPC"
    df = load_dataset(ticker)
    df = preprocess_dataset(df)
    columns: list[str] = [
        "rsi_7",
        "rsi_14",
        "macd_macdhist",
        "stoch_slowk",
        "stoch_slowd",
        "adx_14",
        "pct_b_20",
        "atr_14",
        "ema_50",
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
    add_indicator(df, "bbands", ["close"], timeperiod=20)
    df["pct_b_20"] = (df["close"] - df["bbands_20_lowerband"]) / (
        df["bbands_20_upperband"] - df["bbands_20_lowerband"]
    )
    add_indicator(df, "atr", ["high", "low", "close"])
    df["atr_14_norm"] = df["atr_14"]
    add_indicator(df, "ema", ["close"], timeperiod=50)
    add_indicator(df, "ema", ["close"], timeperiod=200)
    df["price_dist_ema_50"] = (df["close"] - df["ema_50"]) / df["ema_50"]
    df["price_dist_ema_200"] = (df["close"] - df["ema_200"]) / df["ema_200"]
    add_indicator(df, "obv", ["close", "volume"])
    add_indicator(
        df, "roc", columns=["obv"], custom_name="obv_roc_5", timeperiod=5
    )
    add_indicator(
        df, "sma", columns=["volume"], custom_name="vol_sma_20", timeperiod=20
    )
    df["volume_ratio"] = df["volume"] / df["vol_sma_20"]
