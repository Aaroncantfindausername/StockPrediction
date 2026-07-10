from data.preprocess import load_dataset, preprocess_dataset
from train.pipeline import walk_forward_validation_outer
import torch
import numpy as np


def train_predict() -> None:
    seed = 67
    training_years = 5
    test_years = 1
    HORIZON = 5
    torch.manual_seed(seed)
    np.random.seed(seed)
    ticker = "^GSPC"
    df = load_dataset(ticker)
    df = preprocess_dataset(df)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_dates, predictions = walk_forward_validation_outer(
        df, training_years, test_years, HORIZON, device, n_trials=3
    )
    print(f"Start_date: {start_dates}, n.o predictions: {len(predictions)}")
    torch.save(predictions, "strategy/predictions.pth")
