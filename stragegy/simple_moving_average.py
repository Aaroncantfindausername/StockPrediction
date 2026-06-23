from numpy.typing import (
    NDArray,
)
from typing import cast
from collections.abc import (
    Sequence,
)
import pandas as pd
from backtesting import (
    Strategy,
)
from backtesting.lib import (
    crossover,
)
import numpy as np


def SMA(values: pd.Series, n: int) -> pd.Series:
    """
    Return simple moving average of `values`, at
    each step taking into account `n` previous values.
    """
    return pd.Series(values).rolling(n).mean()


class SmaCross(Strategy):
    # Define the two MA lags as *class variables*
    # for later optimization
    n1: int = 10
    n2: int = 20

    def init(self) -> None:
        # Precompute the two moving averages
        self.sma1: NDArray[np.float64] = self.I(
            SMA,
            self.data.Close,
            self.n1,
        )
        self.sma2: NDArray[np.float64] = self.I(
            SMA,
            self.data.Close,
            self.n2,
        )

    def next(self) -> None:
        # If sma1 crosses above sma2, close any existing
        # short trades, and buy the asset
        if crossover(
            self.sma1,
            self.sma2,
        ):
            self.position.close()
            self.buy(size=0.9)

        # Else, if sma1 crosses below sma2, close any existing
        # long trades, and sell the asset
        elif crossover(
            self.sma2,
            self.sma1,
        ):
            self.position.close()
            self.sell()
