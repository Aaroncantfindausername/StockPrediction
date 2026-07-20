from backtesting import Backtest
from numpy.typing import NDArray
from data.preprocess import load_dataset, preprocess_dataset
import pandas as pd
import torch
from models.basic_nn import MLP
from strategy.regression import Regression
from strategy.simple_nn_regression import SimpleNNRegression
from data import single_ticker_minimal
import numpy as np


def precompute_outputs() -> None:
    model_path = "weights/model.pth"
    config_path = "weights/config.pth"
    config = torch.load("weights/config.pth")
    model = MLP(config["input_dim"], 1, config)

    # Load dataset
    ticker: str = config["ticker"]
    df = load_dataset(ticker)
    df = preprocess_dataset(df)

    cols, df = single_ticker_minimal.compute_features_and_labels(
        df, {"HORIZON": config["horizon"]}
    )

    df = df.dropna()
    start_date = df.index[0]
    X: NDArray[np.float32] = df[cols].to_numpy()
    state_dict = torch.load("weights/model.pth")
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        outputs = model(torch.tensor(X))
    torch.save(outputs, "backtest/predictions.pth")
    torch.save(start_date, "backtest/start_date.pth")


# %%
def backtest(ticker: str = "^GSPC") -> None:
    df = load_dataset(ticker)
    df = preprocess_dataset(df)

    start = torch.load("backtest/start_date.pth", weights_only=False)

    df = df[start:]

    bt = Backtest(df, Regression, cash=100_000, commission=0.0)

    stats = bt.run()
    print(stats)
    bt.plot(
        filename="plots/plot",
        resample=True,
        smooth_equity=True,
        plot_volume=False,
        open_browser=False,
    )
