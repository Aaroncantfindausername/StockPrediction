from backtest.backtest import (
    backtest,
    backtest_tickers,
    multi_ticker_backtest,
    precompute_outputs,
    precompute_tickers,
)
from backtest.plot import plot_predictions, plot_predictions_embd
from backtest import multi_ticker_script
from train.train import train
from train import testing
from train import train_multiple_ticker

if __name__ == "__main__":
    # testing.test()
    # train_multiple_ticker.train()
    # multi_ticker_script.walk_forward_validation()
    multi_ticker_script.backtest_full()
