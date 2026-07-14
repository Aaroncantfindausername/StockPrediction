from math import log
import statistics
from typing import Union
from numpy import float32
from numpy.typing import NDArray
from optuna.trial import Trial
from pandas import DataFrame, DatetimeIndex
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader, TensorDataset
import optuna
from data.single_ticker_minimal import compute_features_and_labels
from models.basic_nn import MLP
from train.pipeline import (
    evaluate,
    train_epoch,
)
from torch import nn, optim
import torch


def objective_full(
    trial: Trial,
    df: DataFrame,
    start_train_date: DatetimeIndex,
    end_train_date: DatetimeIndex,
    device: Union[str, torch.device],
    time_series_splits: int = 3,
    batch_size: int = 64,
    epochs: int = 100,
    horizon: int = 5,
) -> float:

    # Model and training hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 0.1, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 0.1, log=True)
    hidden_dim: int = trial.suggest_categorical(
        "hidden_dim", [32, 64, 128, 256, 512, 1024]
    )

    n_layers = trial.suggest_int("n_layers", 3, int(log(hidden_dim, 2) - 1))
    hidden_dim_decay = trial.suggest_categorical(
        "hidden_dim_decay", [0.5, 0.75, 1]
    )
    activation = trial.suggest_categorical(
        "activation",
        ["relu", "gelu", "hardswish", "leakyrelu"],
    )
    activation_params = []
    if activation == "leakyrelu":
        activation_params.append(
            trial.suggest_float("leakyrelu_alpha", 0.01, 0.03)
        )
    # Dataset hyperparameters
    rsi_fast_t = trial.suggest_float("rsi_fast_t", 3, 14)
    rsi_slow_t = trial.suggest_float("rsi_slow_t", low=rsi_fast_t + 7, high=50)
    macd_fastperiod = trial.suggest_int("macd_fastperiod", 5, 14)
    macd_slowperiod = trial.suggest_int(
        "macd_slowperiod", macd_fastperiod + 5, 40
    )
    macd_signalperiod = trial.suggest_int("macd_signalperiod", 3, 12)

    stoch_fastk = trial.suggest_int("stoch_fastk", 5, 28)
    stoch_slowk = trial.suggest_int("stoch_slowk", 2, 7)
    stoch_slowd = trial.suggest_int("stoch_slowd", 2, 7)
    adx_t = trial.suggest_int("adx_t", 3, 28)
    bbands_t = trial.suggest_int("bbands_t", 5, 30)
    atr_t = trial.suggest_int("atr", 3, 28)

    ema_fast_t = trial.suggest_int("ema_fast_t", 5, 70)
    ema_slow_t = trial.suggest_int("ema_slow_t", ema_fast_t + 70, 300)

    obv_roc_t = trial.suggest_int("obv_roc_t", 3, 10)

    volume_sma_t = trial.suggest_int("volume_sma_t", 3, 10)

    model_params = {
        "dropout_rate": 0.1,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "activation": activation,
        "activation_params": activation_params,
        "hidden_dim_decay": hidden_dim_decay,
    }
    dataset_params = {
        "rsi_fast_t": rsi_fast_t,
        "rsi_slow_t": rsi_slow_t,
        "macd_fastperiod": macd_fastperiod,
        "macd_slowperiod": macd_slowperiod,
        "macd_signalperiod": macd_signalperiod,
        "stoch_fastk": stoch_fastk,
        "stoch_slowk": stoch_slowk,
        "stoch_slowd": stoch_slowd,
        "adx_t": adx_t,
        "bbands_t": bbands_t,
        "atr_t": atr_t,
        "ema_fast_t": ema_fast_t,
        "ema_slow_t": ema_slow_t,
        "obv_roc_t": obv_roc_t,
        "volume_sma_t": volume_sma_t,
    }
    # Generate features and labels with suggested params
    random_seed = trial.number
    g = torch.Generator()
    g.manual_seed(random_seed)
    cols, train_df = compute_features_and_labels(
        df[:end_train_date], dataset_params
    )
    train_df = train_df[start_train_date:].dropna()
    X: NDArray[float32] = train_df[[f"{c}_z" for c in cols]].to_numpy()
    y: NDArray[float32] = train_df["target_return_z"].to_numpy()
    # Mini walk forward validation,  but train all folds epoch by epoch and aggregate evaluation at each
    tscv = TimeSeriesSplit(n_splits=time_series_splits, gap=horizon)
    models = [
        MLP(X.shape[1], 1, model_params).to(device)
        for _ in range(time_series_splits)
    ]
    criterion = nn.HuberLoss()
    optimisers = [
        optim.Adam(m.parameters(), lr=learning_rate, weight_decay=weight_decay)
        for m in models
    ]
    schedulers = [
        optim.lr_scheduler.ReduceLROnPlateau(
            o, mode="min", patience=10, factor=0.5
        )
        for o in optimisers
    ]
    val_loss: list[float] = [0.0 for _ in range(time_series_splits)]
    train_loss: list[float] = [0.0 for _ in range(time_series_splits)]
    for epoch in range(1, epochs + 1):
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            train_dataset = TensorDataset(
                torch.from_numpy(X_train), torch.from_numpy(y_train)
            )
            test_dataset = TensorDataset(
                torch.from_numpy(X_test), torch.from_numpy(y_test)
            )
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, generator=g
            )
            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False
            )
            train_loss[fold] = train_epoch(
                models[fold], train_loader, criterion, optimisers[fold], device
            )
            test_loss = evaluate(models[fold], test_loader, criterion, device)
            schedulers[fold].step(test_loss)
            val_loss[fold] = test_loss
        trial.report(statistics.mean(val_loss), epoch)
        print(
            f"Epoch {epoch}\nTrain loss: {statistics.mean(train_loss):.4f}, Val loss: {statistics.mean(val_loss):.4f}"
        )
        if trial.should_prune():
            raise optuna.TrialPruned()
    print(f"Trial {trial.number} complete")
    return statistics.mean(val_loss)
