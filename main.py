from train.train import train
from backtest.backtest import backtest
from train.train_eval import train_predict

if __name__ == "__main__":
    train_predict()
    # backtest("^GSPC")
