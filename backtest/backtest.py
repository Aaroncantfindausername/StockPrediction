from backtesting import Backtest
from data.preprocess import load_dataset, preprocess_dataset
from stragegy.simple_nn_regression import SimpleNNRegression
from train.pipeline import train_val_split

# %%
ticker: str = "^GSPC"
df = load_dataset(ticker)
df = preprocess_dataset(df)
bt = Backtest(df, SimpleNNRegression, cash=10_000, commission=0.002)
stats = bt.run()
print(stats)
