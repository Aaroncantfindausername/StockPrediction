# %%
from backtesting import Backtest
import yfinance as yf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from stragegy.simple_moving_average import SmaCross
from backtesting.test import GOOG
from data import preprocess
import torch


# %%
def download_datasets(ticker: str) -> None:
    data = yf.download(ticker, start="2000-01-01", end="2025-12-31")
    if data is None:
        raise ValueError("Failed to download dataset")
    data.to_csv(f"datasets/{ticker}.csv")


def load_dataset(ticker: str) -> pd.DataFrame:
    return pd.read_csv(f"datasets/{ticker}.csv")


# %%


# def main():
device = "cuda" if torch.cuda.is_available() else "cpu"
df: pd.DataFrame = load_dataset("^GSPC")
df = preprocess.preprocess_dataset(df)
preprocess.add_indicator(df, "sma", ["close"])

# %%

if __name__ == "__main__":
    pass
