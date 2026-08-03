import numpy as np
from backtesting import Strategy
from numpy import float32
from numpy.typing import NDArray
import torch


class Classification(Strategy):
    predictions_path: str = "backtest/predictions.pth"
    past_ref_len: int = 10

    def init(self) -> None:
        predictions = torch.load(self.predictions_path, weights_only=False)
        self.predictions = self.I(lambda: predictions)

    def next(self) -> None:
        if np.mean(self.predictions[-self.past_ref_len :]) > 0.5:
            self.buy(size=0.1)
            print("Attempting buy size")
        elif np.mean(self.predictions[-self.past_ref_len :]) < 0.3:
            print("Closing positions")
            self.position.close()
