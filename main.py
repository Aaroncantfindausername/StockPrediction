from backtest.backtest import backtest, precompute_outputs
from backtest.plot import plot_predictions, plot_predictions_embd
from train.train import train
from train.train_eval import train_eval
from train import testing
from train import train_multiple_ticker
from train import train_classification

if __name__ == "__main__":
    # testing.test()
    train_multiple_ticker.train()
    # train()
    # train_classification.train()
    # train_eval()
    # precompute_outputs()
    # backtest("GOOG")
    # plot_predictions()
    plot_predictions_embd("GOOGL")
