import pandas as pd
from backtest.backtest import (
    get_common_dates,
    multi_ticker_backtest,
    precompute_tickers,
)


def run():
    df = pd.read_parquet("datasets/dataframe.parquet")
    start, end = get_common_dates(df)
    precompute_tickers("datasets/ETFs.txt", start, end)
    multi_ticker_backtest("datasets/ETFs.txt", start, end, 5)
