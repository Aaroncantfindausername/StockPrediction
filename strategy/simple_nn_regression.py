from math import tanh
from typing import Callable, Dict, Tuple

from backtesting import Strategy
from numpy import arange, float32, shape
from numpy.typing import NDArray
import torch
from data.single_ticker_minimal import get_features_labels
from models.basic_nn import MLP
import numpy as np


class SimpleNNRegression(Strategy):
    model_path: str = "weights/basic_nn.pth"
    model_config_path: str = "weights/basic_nn_dim.pth"
    min_buy_threshold: float = 0.001
    close_threshold: float = 0.95
    y: NDArray[float32] = np.array([])

    def init(self) -> None:
        self.predictions = self.I(lambda: self.y)

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
