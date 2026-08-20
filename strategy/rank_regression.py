import numpy as np
from backtesting import Strategy
from numpy import float32
from numpy.typing import NDArray
import torch


class RankRegression(Strategy):
    predictions_path: str = "backtest/predictions.pth"
    min_buy_threshold: float = 0.55
    close_threshold: float = 0.45
    past_ref_len: int = 5

    def init(self) -> None:
        predictions = torch.load(self.predictions_path, weights_only=False)
        self.predictions = self.I(lambda: predictions)

    def next(self) -> None:
        if (
            np.mean(self.predictions[-self.past_ref_len :])
            > self.min_buy_threshold
        ):
            self.buy(size=0.3)
        elif self.predictions[-1] < self.close_threshold:
            self.position.close()
