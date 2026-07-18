from numpy.typing import NDArray
import torch
import matplotlib.pyplot as plt
import numpy as np
from data import single_ticker_minimal
from data.preprocess import load_dataset, preprocess_dataset
from models.basic_nn import MLP


def plot_predictions() -> None:
    config = torch.load("weights/config.pth")
    model = MLP(config["input_dim"], 1, config)

    # Load dataset
    ticker: str = "^GSPC"
    df = load_dataset(ticker)
    df = preprocess_dataset(df)

    cols, df = single_ticker_minimal.compute_features_and_labels(
        df, {"HORIZON": config["horizon"]}
    )
    df = df.dropna()
    X: NDArray[np.float32] = df[cols].to_numpy()
    state_dict = torch.load("weights/model.pth")
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        outputs = model(torch.tensor(X))
    fig, ax = plt.subplots()
    ax.plot(df.index, outputs, label="Outputs")
    ax.plot(df.index, df[config["target"]], label="Targets")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
