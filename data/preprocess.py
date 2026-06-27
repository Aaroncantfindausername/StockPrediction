from typing import Any
import yfinance as yf
from numpy.typing import NDArray
import pandas as pd
import talib
import numpy as np
from talib import abstract


def download_datasets(ticker: str) -> None:
    data = yf.download(ticker, start="2000-01-01", end="2025-12-31")
    if data is None:
        raise ValueError("Failed to download dataset")
    data.to_csv(f"datasets/{ticker}.csv")


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
    df: pd.DataFrame,
    indicator_name: str,
    columns: list[str],
    custom_name: str = "",
    *args,
    **kwargs,
) -> None:
    try:
        indicator_func = abstract.Function(indicator_name)
    except Exception as e:
        print(f"Indicator {indicator_name} not found or error: {e}")
        return

    if "timeperiod" in indicator_func.parameters:
        if "timeperiod" in kwargs:
            indicator_name = f"{indicator_name}_{kwargs.get('timeperiod')}"
        else:
            try:
                indicator_name = f"{indicator_name}_{indicator_func.parameters.get('timeperiod')}"
            except Exception as e:
                print(
                    f"Setting timeperiod for {indicator_name} failed error: {e}"
                )
                return
    if custom_name != "":
        indicator_name = custom_name
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
