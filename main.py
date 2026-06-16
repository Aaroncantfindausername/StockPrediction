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
    data.to_csv("datasets/test.csv")


# %%


def main():
    download_datasets("^GSPC")
    print("Download success")


# %%

if __name__ == "__main__":
    main()
