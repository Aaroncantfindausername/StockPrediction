# %%
from typing import Any, Tuple

from numpy.typing import NDArray
from pandas import DataFrame
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data import preprocess
from data.preprocess import load_dataset, preprocess_dataset
from models.basic_nn import MLP
import copy


# %%
# -------------------------------
# 1. Dataset loading & splitting
# -------------------------------
def train_val_split(
    X: NDArray,
    y: NDArray,
    batch_size: int = 64,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    # Split into train+val (70%) and test (30%) first
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state
    )
    # Split train+val into train and validation
    val_size_from_trainval = val_ratio / (1 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_size_from_trainval,
        random_state=random_state,
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


# -------------------------------
# 2. Model class template
# -------------------------------


# -------------------------------
# 3. Training and evaluation loops
# -------------------------------
def train_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    running_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device) -> Tuple[float, float]:
    model.eval()
    running_loss: float = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    avg_loss: float = running_loss / len(loader.dataset)
    mae: float = np.mean(
        np.abs(np.array(all_preds) - np.array(all_targets))
    ).item()
    return (avg_loss, mae)


# -------------------------------
# 4. Main training script
# -------------------------------
# %%

# %%
