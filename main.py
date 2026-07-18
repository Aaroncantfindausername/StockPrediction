from backtest.backtest import backtest
from train.train import train
from train.train_eval import train_eval
from train import testing
from train import train_multiple_ticker

if __name__ == "__main__":
    # testing.test()
    # train_multiple_ticker.train()
    train()
    # train_eval()
    # backtest("^GSPC")
