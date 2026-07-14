from backtest.backtest import backtest
from train.train import train
from train.train_eval import train_eval
from train import testing

if __name__ == "__main__":
    testing.train()
    # train()
    # train_eval()
    # backtest("^GSPC")
