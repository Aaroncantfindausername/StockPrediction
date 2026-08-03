import numpy as np
from backtesting import Strategy
from numpy import float32
from numpy.typing import NDArray
import torch


class Regression(Strategy):
    predictions_path: str = "backtest/predictions.pth"
    min_buy_threshold: float = 0.0
    close_threshold: float = -0.2

    def init(self) -> None:
        predictions = torch.load(self.predictions_path, weights_only=False)
        self.predictions = self.I(lambda: predictions)

    def next(self) -> None:
        if np.isnan(self.predictions[-1]):
            # Some features still NaN
            return
        if self.predictions[-1] > self.min_buy_threshold:
            self.buy(
                size=min(self.predictions[-1], 1),
                sl=self.data.Close[-1] * 0.80,
            )
            print(
                f"Attempting buy, predicted returns: {self.predictions[-1]}\n Size = {min(max(self.predictions[-1], 0.05), 1)}"
            )
        elif self.predictions[-1] < self.close_threshold:
            print(
                f"Closing positions, predicted returns {self.predictions[-1]}"
            )
            self.position.close()
