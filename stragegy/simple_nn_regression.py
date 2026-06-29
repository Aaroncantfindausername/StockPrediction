from typing import Callable, Optional, Tuple

from backtesting import Strategy
from numpy import float32
from numpy.typing import NDArray
import torch
from data.single_ticker_minimal import get_features_labels
from models.basic_nn import MLP
import numpy as np


class SimpleNNRegression(Strategy):
    feature_func: Callable[..., Tuple[NDArray, NDArray]] = get_features_labels
    model: Optional[MLP] = None
    hidden_dim: int = 256
    dropout: float = 0.2
    model_path: str = "weights/basic_nn.pth"
    min_buy_threshold: float = 0.0

    def init(self) -> None:
        self.X: NDArray[float32] = self.I(self.feature_func, self.data.df)[0]
        self.model = MLP(self.X.shape[1], self.hidden_dim, out_dim=1)
        state_dict = torch.load(self.model_path)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def next(self) -> None:
        if self.model is None:
            print("Model is not initialised")
            exit(1)
        if np.any(np.isnan(self.X[-1])):
            # Some features still NaN
            return

        predicted_return = self.model(self.X[-1]).squeeze().item()
        if predicted_return > self.min_buy_threshold:
            self.buy(size=0.1)
        elif predicted_return < 0.9:
            self.position.close()
