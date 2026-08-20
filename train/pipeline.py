# %%
import pickle
from typing import Any, Tuple, Union
from multitasking import Dict
import optuna
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame, DatetimeIndex
from sqlalchemy.sql.base import InPlaceGenerative
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
import copy
from optuna.pruners import MedianPruner
from functools import partial

from data import multiple_ticker
from data.single_ticker_minimal import compute_features_and_labels
from models.mlp import MLP
from train.ticker_embedding_dataset import TickerEmbeddingDataset


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


def walk_forward_validation_outer_embd(
    tickers: list[str],
    start_date: DatetimeIndex,
    end_date: DatetimeIndex,
    min_training_years: int,
    max_training_years: int,
    test_years: int,
    purge_embargo_gap: int,
    features: list[str],
    target_column: str,
    device: Union[str, torch.device],
    embedding_dim: int = 3,
    batch_size: int = 256,
    epochs: int = 100,
    train_patience: int = 30,
    scheduler_patience: int = 10,
    random_seed: int = 67,
    n_trials: int = 100,
    trials_timeout: int = 120,
    save: bool = False,
) -> Tuple[list, Dict[str, Any]]:
    import train.nn_objective
    from backtest.backtest import multi_ticker_backtest, precompute_tickers

    all_results: list[DataFrame] = []
    all_predictions_dict = {t: np.array([]) for t in tickers}
    MIN_TEST_PERIOD = pd.DateOffset(months=6)
    start_train_date: DatetimeIndex = start_date
    end_train_date: DatetimeIndex = start_date + pd.DateOffset(
        years=min_training_years, days=-(1 + purge_embargo_gap)
    )
    start_test_date: DatetimeIndex = start_date + pd.DateOffset(
        years=min_training_years
    )  # Offset by HORIZON (purging)

    fold = 0
    data_end_date = end_date
    end_test_date: DatetimeIndex = start_test_date
    while start_test_date + MIN_TEST_PERIOD <= data_end_date:
        end_test_date = min(
            start_test_date + pd.DateOffset(years=test_years, days=-1),
            data_end_date,
        )
        objective = partial(
            train.nn_objective.objective_full,
            tickers=tickers,
            start_date=start_train_date,
            end_date=end_train_date,
            target_column=target_column,
            device=device,
            features=features,
            embedding_dim=embedding_dim,
        )
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)
        sampler = optuna.samplers.TPESampler(seed=random_seed)
        study = optuna.create_study(
            direction="maximize", pruner=pruner, sampler=sampler
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=trials_timeout,
        )
        print(f"Hyperparameters optimised:\n {study.best_params}")
        # Retrain model on all of training data using best hyperparams
        data_params = study.best_params | {
            "columns": features,
            "cross_sectional_z_score": True,
        }

        df, cols = multiple_ticker.get_feature_target_df(
            tickers,
            data_params,
        )
        ticker_to_id = {t: i for i, t in enumerate(tickers)}
        df["ticker_id"] = df["Ticker"].map(ticker_to_id)
        df_train, df_val = train_test_split(
            df[
                (df.index >= start_train_date) & (df.index <= end_train_date)
            ].sort_index(),
            test_size=0.15,
            shuffle=False,
        )
        assert type(df_train) is pd.DataFrame and type(df_val) is pd.DataFrame
        train_dataset = TickerEmbeddingDataset(
            df_train,
            cols,
            target_column,
        )
        val_dataset = TickerEmbeddingDataset(
            df_val,
            cols,
            target_column,
        )
        train_loader = DataLoader(train_dataset, batch_size, True)
        val_loader = DataLoader(val_dataset, batch_size, False)
        model_params = study.best_params | {
            "hidden_dim": 1024,
            "n_layers": 3,
            "hidden_dim_decay": 0.5,
        }
        model = MLP(
            len(cols),
            1,
            True,
            embedding_dim=embedding_dim,
            num_unique_embeddings=len(tickers),
            params=model_params,
        ).to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=scheduler_patience,
            factor=0.25,
            threshold=0.0,
        )
        print(f"Evaluating years {start_test_date} ~ {end_test_date}")
        train_full_with_embedding(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            epochs,
            train_patience,
            load_best_val_loss=True,
        )

        outputs_dict = precompute_tickers(
            tickers,
            start_test_date,
            end_test_date,
            df=df,
            save=False,
        )
        assert outputs_dict is not None, (
            "dict should be returned when not saving"
        )
        for k, v in outputs_dict.items():
            all_predictions_dict[k] = np.concatenate(
                [all_predictions_dict[k], v], axis=0
            )

        res = multi_ticker_backtest(
            start_test_date,
            end_test_date,
            outputs_dict=outputs_dict,
            n=5,
            plot=False,
            baseline=False,
        )
        all_results.append(res)
        # Shift train window to include test window and shift test window forwards
        fold += 1
        end_train_date += pd.DateOffset(years=test_years)
        start_test_date += pd.DateOffset(years=test_years)
        if min_training_years + fold * test_years > max_training_years:
            start_train_date += pd.DateOffset(years=test_years)
        print(f"Outer fold {fold} complete")
    if save:
        with open("backtest/results/results.pkl", "wb") as f:
            pickle.dump(all_results, f)
        with open("backtest/precomputed_data/data.pkl", "wb") as f:
            pickle.dump(all_predictions_dict, f)
    return (all_results, all_predictions_dict)


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
