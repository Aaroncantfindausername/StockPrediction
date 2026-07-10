# %%
from typing import Any, Tuple, Union
import optuna
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame, DatetimeIndex
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
import copy
from optuna.pruners import MedianPruner

from functools import partial

from data.single_ticker_minimal import get_features_labels
from models.basic_nn import MLP


# %%
# -------------------------------
# 1. Dataset loading & splitting
# -------------------------------
def train_val_test_split_loader(
    X: NDArray,
    y: NDArray,
    batch_size: int = 64,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    # Split into train+val (70%) and test (30%) first
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_ratio, shuffle=False
    )
    # Split train+val into train and validation
    val_size_from_trainval = val_ratio / (1 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size_from_trainval, shuffle=False
    )

    # Convert to tensors and create TensorDatasets
    train_dataset = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(y_val)
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return (
        train_loader,
        val_loader,
        test_loader,
        X_train.shape[1],  # input dimension
    )


def walk_forward_validation_outer(
    df: DataFrame,
    training_years: int,
    test_years: int,
    HORIZON: int,
    device: Union[str, torch.device],
    batch_size: int = 64,
    epochs: int = 100,
    train_patience: int = 5,
    random_seed: int = 67,
    n_trials: int = 100,
    trials_timeout: int = 120,
) -> Tuple[list[DatetimeIndex], list[NDArray]]:
    import train.nn_objective  # Prevent circular import

    all_start_dates: list[DatetimeIndex] = []
    all_predictions: list[NDArray] = []
    MIN_TEST_PERIOD = pd.DateOffset(months=6)
    start_train_date: DatetimeIndex = df.index[0]
    end_train_date: DatetimeIndex = df.index[0] + pd.DateOffset(
        years=training_years, days=-(1 + HORIZON)
    )
    start_test_date: DatetimeIndex = df.index[0] + pd.DateOffset(
        years=training_years
    )  # Offset by HORIZON (purging)
    data_end_date = df.index[-1]
    end_test_date: DatetimeIndex = start_test_date
    while start_test_date + MIN_TEST_PERIOD <= data_end_date:
        end_test_date = min(
            start_test_date + pd.DateOffset(years=test_years, days=-1),
            data_end_date,
        )
        train_data = df[start_train_date:end_train_date]
        test_data = df[start_test_date:end_test_date]
        all_start_dates.append(start_test_date)
        objective = partial(
            train.nn_objective.objective_full,
            horizon=HORIZON,
            df=train_data,
            device=device,
        )
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1)
        sampler = optuna.samplers.TPESampler(seed=random_seed)
        study = optuna.create_study(
            direction="minimize", pruner=pruner, sampler=sampler
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=trials_timeout,
        )
        print(f"Hyperparameters optimised:\n {study.best_params}")
        # Retrain model on all of training data using best hyperparams
        X_trainval, y_trainval = get_features_labels(
            train_data, study.best_params
        )
        X_test: NDArray
        y_test: NDArray
        X_test, y_test = get_features_labels(test_data, study.best_params)
        X_test = X_test.copy()  # Prevent non writable tensor error
        y_test = y_test.copy()
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=0.15, shuffle=False
        )

        train_dataset = TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train)
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_val), torch.from_numpy(y_val)
        )
        test_dataset = TensorDataset(
            torch.from_numpy(X_test), torch.from_numpy(y_test)
        )

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        model = MLP(X_train.shape[1], 1, study.best_params).to(device)
        criterion = torch.nn.HuberLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=study.best_params["learning_rate"],
            weight_decay=study.best_params["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )
        train_and_eval(
            model,
            train_loader,
            val_loader,
            test_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            epochs,
            train_patience,
        )
        model.eval()
        with torch.no_grad():
            predictions = model(torch.tensor(X_test).to(device)).cpu().numpy()
            all_predictions.append(predictions)
        # Shift train window to include test window and shift test window forwards
        start_train_date += pd.DateOffset(years=test_years)
        end_train_date += pd.DateOffset(years=test_years)
        start_test_date += pd.DateOffset(years=test_years)

    # Return predictions start and end date and all predictions
    return (all_start_dates, all_predictions)


# -------------------------------
# 3. Training and evaluation loops
# -------------------------------
def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    criterion,
    optimizer,
    device: str | torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.unsqueeze(1).to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    criterion,
    device: Union[str, torch.device],
) -> float:
    model.eval()
    running_loss: float = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.unsqueeze(1).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    avg_loss: float = running_loss / len(loader.dataset)
    return avg_loss


def train_and_eval(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    test_loader: DataLoader,
    criterion,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    device: str | torch.device,
    epochs: int,
    patience: int,
) -> float:
    if val_loader is None:
        val_loader = copy.deepcopy(test_loader)
    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            # torch.save(model.state_dict(), "weights/tmp/best_model.pth")
        else:
            epochs_no_improve += 1

        if epochs_no_improve > patience:
            break
    model.load_state_dict(best_model_wts)
    return evaluate(model, test_loader, criterion, device)
