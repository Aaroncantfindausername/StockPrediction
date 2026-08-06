import numpy as np
from backtesting import Strategy
from numpy import float32
from numpy.typing import NDArray
import torch


class RankRegression(Strategy):
    predictions_path: str = "backtest/predictions.pth"
    min_buy_threshold: float = 0.7
    close_threshold: float = 0.3

    def init(self) -> None:
        predictions = torch.load(self.predictions_path, weights_only=False)
        self.predictions = self.I(lambda: predictions)

    def next(self) -> None:
        if np.isnan(self.predictions[-1]):
            # Some features still NaN
            return
        if self.predictions[-1] > self.min_buy_threshold:
            self.buy(
                size=min(self.predictions[-1] / 3, 1),
            )
            print("Buying")
        elif self.predictions[-1] < self.close_threshold:
            print(f"Closing positions, predicted rank {self.predictions[-1]}")
            self.position.close()
