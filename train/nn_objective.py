import statistics
from typing import Union
from optuna.trial import Trial
from pandas import DataFrame
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader, TensorDataset
import optuna
from data.single_ticker_minimal import get_features_labels
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
    device: Union[str, torch.device],
    time_series_splits: int = 3,
    batch_size: int = 64,
    epochs: int = 100,
    train_patience: int = 10,
    horizon: int = 5,
) -> float:
    # Copy df to avoid changing df for other trials
    df = df.copy()

    # Model and training hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 0.1, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    n_layers = trial.suggest_categorical("n_layers", range(3, 10))
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 0.1, log=True)
    hidden_dim = [
        trial.suggest_categorical("hidden_dim", [2 ^ i for i in range(3, 7)])
    ]
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
    ema_slow_t = trial.suggest_int("ema_slow_t", ema_fast_t + 50, 300)

    obv_roc_t = trial.suggest_int("obv_roc_t", 3, 10)

    volume_sma_t = trial.suggest_int("volume_sma_t", 3, 10)

    model_params = {
        "dropout_rate": dropout_rate,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "activation": activation,
        "activation_params": activation_params,
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
    X, y = get_features_labels(df, dataset_params)
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
            o, mode="min", patience=5, factor=0.5
        )
        for o in optimisers
    ]
    early_stop = [False for _ in range(time_series_splits)]
    best_test_loss: list[float] = [
        float("inf") for _ in range(time_series_splits)
    ]
    epochs_no_improve = [0 for _ in range(time_series_splits)]
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}")
        if all(early_stop):
            print("All folds early stopped")
            break
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            if early_stop[fold]:
                continue
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
            train_epoch(
                models[fold], train_loader, criterion, optimisers[fold], device
            )
            test_loss = evaluate(models[fold], test_loader, criterion, device)
            schedulers[fold].step(test_loss)
            if test_loss < best_test_loss[fold]:
                epochs_no_improve[fold] = 0
                best_test_loss[fold] = test_loss
            else:
                epochs_no_improve[fold] += 1
                if epochs_no_improve[fold] > train_patience:
                    early_stop[fold] = True
        avg_best_val_loss = statistics.mean(best_test_loss)
        trial.report(avg_best_val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return statistics.mean(best_test_loss)
