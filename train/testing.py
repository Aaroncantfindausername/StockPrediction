from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.basic_nn import MLP
import copy
from data.preprocess import load_dataset, preprocess_dataset
from matplotlib import pyplot as plt
from train.pipeline import (
    evaluate,
    train_full,
    train_val_test_split_loader,
)
from data import single_ticker_lagged, single_ticker_minimal, multiple_ticker


def test():
    df = load_dataset("^GSPC")
    df = multiple_ticker.get_feature_target_df(
        ["AMZN", "GOOG", "JNJ", "JPM", "NVDA", "TSLA"]
    )
    print(df.head())
