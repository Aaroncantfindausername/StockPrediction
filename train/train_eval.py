from data.preprocess import load_dataset, preprocess_dataset
from train.pipeline import walk_forward_validation_outer
import torch
import numpy as np


def train_eval() -> None:
    seed = 67
    min_training_years = 10
    max_training_years = 10
    test_years = 1
    HORIZON = 5
    torch.manual_seed(seed)
    np.random.seed(seed)
    ticker = "^GSPC"
    df = load_dataset(ticker)
    df = preprocess_dataset(df)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config, predictions = walk_forward_validation_outer(
        df,
        min_training_years,
        max_training_years,
        test_years,
        HORIZON,
        device,
        n_trials=10,
        trials_timeout=120,
    )
    torch.save(predictions, "strategy/predictions.pth")
    torch.save(config, "strategy/pred_config.pth")
