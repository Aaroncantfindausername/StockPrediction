from typing import Dict

from backtesting import Backtest
from numpy.typing import NDArray
from data.preprocess import load_dataset, preprocess_dataset
from data.single_ticker_minimal import get_features_labels
from models.basic_nn import MLP
from stragegy.simple_nn_regression import SimpleNNRegression
from bokeh.plotting import figure
import torch


# %%
def backtest(ticker: str = "^GSPC") -> None:
    model_path = "weights/basic_nn.pth"
    model_config_path = "weights/basic_nn_dim.pth"
    df = load_dataset(ticker)
    df = preprocess_dataset(df)

    X: NDArray = get_features_labels(df)[0]
    config: Dict = torch.load(model_config_path)
    model = MLP(
        config["input_dim"], config["hidden_dim"], out_dim=config["out_dim"]
    )
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        out: torch.Tensor = model(
            torch.tensor(X, dtype=torch.float32)
        ).squeeze()

    bt = Backtest(df, SimpleNNRegression, cash=100_000, commission=0.0)

    stats = bt.run(y=out)
    print(stats)
    bt.plot(
        filename="plots/plot",
        resample=True,
        smooth_equity=True,
        plot_volume=False,
        open_browser=False,
    )
