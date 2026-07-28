# %%
from typing import Any, Tuple, Union
from multitasking import Dict
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

from data.single_ticker_minimal import compute_features_and_labels
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
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    # Split into train+val (70%) and test (30%) first
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
        train_dataset, batch_size=batch_size, shuffle=shuffle_train
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return (
        train_loader,
        val_loader,
        test_loader,
        X_train.shape[1],  # input dimension
    )


def create_sequential_windows_multiple_tickers(
    df: DataFrame, feature_cols: list[str], target_col: list[str], seq_len: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    windows = []
    ticker_ids = []
    targets = []
    for ticker_id, group in df.groupby("ticker_id"):
        features: NDArray[np.float32] = group[feature_cols].values
        target_vals: NDArray[np.float32] = group[target_col].values
        for i in range(len(features) - seq_len + 1):
            X: NDArray[np.float32] = features[i : i + seq_len]
            y: NDArray[np.float32] = target_vals[i + seq_len - 1]
            windows.append(X)
            targets.append(y)
            ticker_ids.append(ticker_id)
    X_tensor = torch.tensor(np.array(windows))  # (N, seq_len, num_features)
    y_tensor = torch.tensor(np.array(targets))  # (N,)
    ticker_tensor = torch.tensor(np.array(ticker_ids), dtype=torch.long)  # (N,)
    return X_tensor, ticker_tensor, y_tensor


def create_sequential_windows_single_tickers(
    df, feature_cols: list[str], target_col: list[str], seq_len: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    windows = []
    targets = []
    features: NDArray[np.float32] = df[feature_cols].values
    target_vals: NDArray[np.float32] = df[target_col].values
    for i in range(len(features) - seq_len):
        X: NDArray[np.float32] = features[i : i + seq_len]
        y: NDArray[np.float32] = target_vals[i + seq_len]
        windows.append(X)
        targets.append(y)
    X_tensor = torch.tensor(np.array(windows))  # (N, seq_len, num_features)
    y_tensor = torch.tensor(np.array(targets))  # (N,)
    return X_tensor, y_tensor


def walk_forward_validation_outer(
    df: DataFrame,
    min_training_years: int,
    max_training_years: int,
    test_years: int,
    HORIZON: int,
    device: Union[str, torch.device],
    batch_size: int = 64,
    epochs: int = 100,
    train_patience: int = 10,
    random_seed: int = 67,
    n_trials: int = 100,
    trials_timeout: int = 120,
) -> Tuple[Dict[str, Any], list[NDArray]]:
    import train.nn_objective  # Prevent circular import

    all_predictions: list[NDArray] = []
    MIN_TEST_PERIOD = pd.DateOffset(months=6)
    start_train_date: DatetimeIndex = df.index[0]
    end_train_date: DatetimeIndex = df.index[0] + pd.DateOffset(
        years=min_training_years, days=-(1 + HORIZON)
    )
    start_test_date: DatetimeIndex = df.index[0] + pd.DateOffset(
        years=min_training_years
    )  # Offset by HORIZON (purging)

    config = {"start": start_test_date, "step": test_years}
    fold = 0
    data_end_date = df.index[-1]
    end_test_date: DatetimeIndex = start_test_date
    while start_test_date + MIN_TEST_PERIOD <= data_end_date:
        end_test_date = min(
            start_test_date + pd.DateOffset(years=test_years, days=-1),
            data_end_date,
        )
        objective = partial(
            train.nn_objective.objective_full,
            start_train_date=start_train_date,
            end_train_date=end_train_date,
            horizon=HORIZON,
            df=df,
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
        cols, feature_df = compute_features_and_labels(
            df[:end_test_date], study.best_params
        )
        train_df = feature_df[start_train_date:end_train_date].dropna()
        test_df = feature_df[start_test_date:end_test_date]
        X_trainval: NDArray[np.float32] = train_df[
            [f"{c}_z" for c in cols]
        ].to_numpy()
        y_trainval: NDArray[np.float32] = train_df["target_return_z"].to_numpy()
        X_test: NDArray[np.float32] = test_df[
            [f"{c}_z" for c in cols]
        ].to_numpy()
        assert any(test_df.isna()), "test_df contains na values"
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=0.15, shuffle=False
        )
        train_dataset = TensorDataset(
            torch.tensor(X_train), torch.tensor(y_train)
        )
        val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
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
        print(f"Evaluating years {start_test_date} ~ {end_test_date}")
        train_full(
            model,
            train_loader,
            val_loader,
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
            print("Predictions calculated")
        # Shift train window to include test window and shift test window forwards
        fold += 1
        end_train_date += pd.DateOffset(years=test_years)
        start_test_date += pd.DateOffset(years=test_years)
        if min_training_years + fold * test_years > max_training_years:
            start_train_date += pd.DateOffset(years=test_years)
        print(f"Outer fold {fold} complete")

    config["end_date"] = end_test_date
    return (config, all_predictions)


# -------------------------------
# 3. Training and evaluation loops
# -------------------------------
def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    criterion,
    optimizer,
    device: str | torch.device,
    classification: bool = False,
) -> float:
    batch_loss = None
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = (
            targets.unsqueeze(1).to(device)
            if not classification
            else targets.to(device)
        )
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        batch_loss = loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        if classification:
            running_acc += compute_accuracy(outputs, targets) * inputs.size(0)

        # print(f"Batch loss: {batch_loss:.4f}")

    if classification:
        print(f"Train accuracy: {running_acc / len(loader.dataset):.5f}")
    return running_loss / len(loader.dataset)


def train_epoch_with_embedding(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    criterion,
    optimizer,
    device: str | torch.device,
    classification: bool = False,
) -> float:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for inputs, ticker_ids, targets in loader:
        inputs, ticker_ids, targets = (
            inputs.to(device),
            ticker_ids.to(device),
            targets.unsqueeze(1).to(device)
            if not classification
            else targets.to(device),
        )
        optimizer.zero_grad()
        outputs = model(inputs, ticker_ids)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

        if classification:
            running_acc += compute_accuracy(outputs, targets) * inputs.size(0)
    if classification:
        print(f"Train accuracy: {running_acc / len(loader.dataset):.5f}")
    return running_loss / len(loader.dataset)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    criterion,
    device: Union[str, torch.device],
    classification: bool = False,
) -> float:
    model.eval()
    running_loss: float = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = (
                targets.unsqueeze(1).to(device)
                if not classification
                else targets.to(device)
            )
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            if classification:
                running_acc += compute_accuracy(outputs, targets) * inputs.size(
                    0
                )
    if classification:
        print(f"Epoch eval accuracy: {running_acc / len(loader.dataset)}")
    avg_loss: float = running_loss / len(loader.dataset)
    return avg_loss


def evaluate_with_embedding(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    criterion,
    device: Union[str, torch.device],
    classification: bool = False,
) -> float:
    model.eval()
    running_loss: float = 0.0
    running_acc: float = 0.0
    with torch.no_grad():
        for inputs, ticker_ids, targets in loader:
            inputs, ticker_ids, targets = (
                inputs.to(device),
                ticker_ids.to(device),
                targets.unsqueeze(1).to(device)
                if not classification
                else targets.to(device),
            )
            outputs = model(inputs, ticker_ids)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            if classification:
                running_acc += compute_accuracy(outputs, targets) * inputs.size(
                    0
                )
    if classification:
        print(f"Eval accuracy: {running_acc / len(loader.dataset):.5f}")
    avg_loss: float = running_loss / len(loader.dataset)
    return avg_loss


def train_full(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    device: str | torch.device,
    epochs: int,
    patience: int,
    load_best_val_loss: bool = True,
    classification: bool = False,
):
    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, classification
        )
        val_loss = evaluate(
            model, val_loader, criterion, device, classification
        )
        scheduler.step(val_loss)
        # print(f"Epoch {epoch}: LR = {optimizer.param_groups[0]['lr']:.10f}")
        print(
            f"Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss:.10f} , Val loss: {val_loss:.10f}, "
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            # torch.save(model.state_dict(), "weights/tmp/best_model.pth")
        else:
            epochs_no_improve += 1

        if epochs_no_improve > patience:
            print(
                f"Validation loss not improved for more than {patience} epochs, early stop triggered"
            )
            break
    if load_best_val_loss:
        model.load_state_dict(best_model_wts)


def train_full_with_embedding(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    device: str | torch.device,
    epochs: int,
    patience: int,
    load_best_val_loss: bool = True,
    classification: bool = False,
):
    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch_with_embedding(
            model, train_loader, criterion, optimizer, device, classification
        )
        val_loss = evaluate_with_embedding(
            model, val_loader, criterion, device, classification
        )
        scheduler.step(val_loss)
        # print(f"Epoch {epoch}: LR = {optimizer.param_groups[0]['lr']:.10f}")
        print(
            f"Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss:.10f} , Val loss: {val_loss:.10f}, "
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            # torch.save(model.state_dict(), "weights/tmp/best_model.pth")
        else:
            epochs_no_improve += 1

        if epochs_no_improve > patience:
            print(
                f"Validation loss not improved for more than {patience} epochs, early stop triggered"
            )
            break
    if load_best_val_loss:
        model.load_state_dict(best_model_wts)


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total
