from typing import Any

from numpy.typing import NDArray
import pandas as pd
import talib
import numpy as np
from talib import abstract


def load_dataset(ticker: str) -> pd.DataFrame:
    return pd.read_csv(f"datasets/{ticker}.csv")


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df.dropna(inplace=True)
    df.drop(0, inplace=True)
    df.index = pd.to_datetime(df["Price"])
    df = df.drop("Price", axis=1)
    df = df[["Close", "Open", "Low", "High", "Volume"]].astype(np.float64)
    df.rename(
        columns={col_name: col_name.lower() for col_name in df.columns},
        inplace=True,
    )
    return df


def add_indicator(
    df: pd.DataFrame, indicator_name: str, columns: list[str], *args, **kwargs
) -> None:
    try:
        indicator_func = abstract.Function(indicator_name)
    except Exception as e:
        print(f"Indicator {indicator_name} not found or error: {e}")
        return
    if indicator_name in df.columns:
        print(
            f"Indicator {indicator_name} is already in the dataframe, cannot add again."
        )
        return

    try:
        inputs: list[Any] = [df[col].values for col in columns]
    except KeyError as e:
        print(f"Missing column {e}")
        return
    result = indicator_func(*inputs, *args, **kwargs)
    if isinstance(result, list):
        for out_vals, out_name in zip(result, indicator_func.output_names):
            df[f"{indicator_name}_{out_name}"] = out_vals
    else:
        df[indicator_name] = result
