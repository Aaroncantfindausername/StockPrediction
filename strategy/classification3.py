import numpy as np
from backtesting import Strategy
from numpy import float32
from numpy.typing import NDArray
import torch


class Classification3(Strategy):
    predictions_path: str = "strategy/predictions.pth"

    def init(self) -> None:
        predictions = torch.load(self.predictions_path, weights_only=False)
        self.predictions = self.I(lambda: predictions)

    def next(self) -> None:
        if np.isnan(self.predictions[-1]):
            # Some features still NaN
            return
        if self.predictions == 2:
            self.buy(size=0.1)
            print("Attempting buy")
        elif self.predictions == 0:
            self.position.close()
