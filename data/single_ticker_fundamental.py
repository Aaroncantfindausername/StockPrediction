from typing import Any, Dict, Tuple, final
import talib
from numpy.typing import NDArray
import pandas as pd
import numpy as np
from data import single_ticker_minimal
from data.preprocess import rolling_z_score


def compute_fundamental_features(
    df: pd.DataFrame, params: Dict[str, Any] = {}
) -> Tuple[list[str], pd.DataFrame]:
    columns = params.get("columns", ["Close", "Open", "High", "Low", "Volume"])
    df["D/P"] = df["Dividend"] / df["Price"]
    df["E/P"] = df["Earning"] / df["Price"]
