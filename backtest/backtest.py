from backtesting import Backtest
from data.preprocess import load_dataset, preprocess_dataset
import pandas as pd

from strategy.simple_nn_regression import SimpleNNRegression


# %%
def backtest(ticker: str = "^GSPC") -> None:
    df = load_dataset(ticker)
    df = preprocess_dataset(df)
    start = df.index[0] + pd.DateOffset(years=5)

    df = df[start:]

    bt = Backtest(df, SimpleNNRegression, cash=100_000, commission=0.0)

    stats = bt.run()
    print(stats)
    bt.plot(
        filename="plots/plot",
        resample=True,
        smooth_equity=True,
        plot_volume=False,
        open_browser=False,
    )
