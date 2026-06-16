# %%
import yfinance as yf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# %%
def download_datasets(ticker: str) -> None:
    data = yf.download(ticker, start="2000-01-01", end="2025-12-31")
    if data is None:
        raise ValueError("Failed to download dataset")
    data.to_csv(f"datasets/{ticker}.csv")


def load_dataset(ticker: str) -> pd.DataFrame:
    return pd.read_csv(f"datasets/{ticker}.csv")


# %%


def main():
    df: pd.DataFrame = load_dataset("^GSPC")
    df.dropna(inplace=True)
    df.drop(0, inplace=True)
    df.index = pd.to_datetime(df["Price"])
    df = df.drop("Price", axis=1)
    df = df[["Close", "Open", "Low", "High", "Volume"]].astype(np.float64)
    df.info()
    print(df.head())


# %%

if __name__ == "__main__":
    main()
