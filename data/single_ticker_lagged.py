from typing import Any, Dict, Tuple, final
import talib
from numpy.typing import NDArray
import pandas as pd
import numpy as np
from data import single_ticker_minimal
from data.preprocess import rolling_z_score


def compute_features_and_labels(
    df: pd.DataFrame, params: Dict[str, Any] = {}
) -> Tuple[list[str], pd.DataFrame]:
    lagged_features = [
        "rsi_fast",
        "macd_hist",
        "pct_b",
        "atr_norm",
        "dist_ema_fast",
        "volume_ratio",
    ]
    cols, df = single_ticker_minimal.compute_features_and_labels(df, params)
    # Lag features
    lags = params.get("lags", [1, 2, 3, 5])
    for feat in lagged_features:
        for lag in lags:
            df[f"{feat}_lag{lag}_z"] = df[f"{feat}_z"].shift(lag)
            cols.append(f"{feat}_lag{lag}_z")
    return cols, df
