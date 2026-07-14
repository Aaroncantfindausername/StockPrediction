import numpy as np
from backtesting import Strategy
from numpy import float32
from numpy.typing import NDArray
import torch


class Regression(Strategy):
    predictions_path: str = "Strategy/predictions.pth"
    min_buy_threshold: float = 0.002
    close_threshold: float = 0.95

    def init(self) -> None:
        predictions = torch.load(self.predictions_path)
        self.predictions = self.I(lambda: predictions)

    def next(self) -> None:
        if np.isnan(self.predictions[-1]):
            # Some features still NaN
            return
        if self.predictions[-1] > self.min_buy_threshold:
            self.buy(size=min(self.predictions[-1] * 50, 1))
            print(
                f"Attempting buy, predicted returns: {self.predictions[-1]}\n Size = {self.predictions[-1] * 50}"
            )
        elif self.predictions[-1] < self.close_threshold:
            self.position.close()
