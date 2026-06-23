# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import copy


# %%
# -------------------------------
# 1. Dataset loading & splitting
# -------------------------------
def load_data(batch_size=64, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Replace this with your actual data loading.
    Here we create a dummy dataset for illustration.
    """
    # Synthetic data: 1000 samples, 20 features, 3 classes
    X = np.random.randn(1000, 20).astype(np.float32)
    y = np.random.randint(0, 3, 1000).astype(np.int64)

    # Split into train+val (70%) and test (30%) first
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state, stratify=y
    )
    # Split train+val into train and validation
    val_size_from_trainval = val_ratio / (1 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_size_from_trainval,
        random_state=random_state,
        stratify=y_trainval,
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
        X_train.shape[1],
        len(np.unique(y)),
    )


# -------------------------------
# 2. Model class template
# -------------------------------
class MLP(nn.Module):
    """
    A simple multi-layer perceptron.
    Replace with your own architecture (e.g., CNN, RNN).
    """

    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------
# 3. Training and evaluation loops
# -------------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    acc = accuracy_score(all_targets, all_preds)
    return running_loss / len(loader.dataset), acc


# -------------------------------
# 4. Main training script
# -------------------------------
# %%
# Hyperparameters
config = {
    "batch_size": 64,
    "lr": 1e-3,
    "epochs": 50,
    "hidden_dim": 128,
    "dropout": 0.2,
    "weight_decay": 1e-5,
    "patience": 10,  # early stopping
    "seed": 42,
}

torch.manual_seed(config["seed"])
np.random.seed(config["seed"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
train_loader, val_loader, test_loader, input_dim, num_classes = load_data(
    batch_size=config["batch_size"]
)

# Model, loss, optimizer
model = MLP(input_dim, config["hidden_dim"], num_classes, config["dropout"]).to(
    device
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=5, factor=0.5
)

# Tracking
best_val_loss = float("inf")
best_model_wts = copy.deepcopy(model.state_dict())
epochs_no_improve = 0

for epoch in range(1, config["epochs"] + 1):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    scheduler.step(val_loss)

    print(
        f"Epoch {epoch:2d}/{config['epochs']} | "
        f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )

    # Checkpointing (best model based on validation loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        epochs_no_improve += 1

    # Early stopping
    if epochs_no_improve >= config["patience"]:
        print("Early stopping triggered.")
        break

# Load best weights for final evaluation
model.load_state_dict(best_model_wts)
test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

# Save final model (optional)
# torch.save(model.state_dict(), "final_model.pth")
# print("Model weights saved to final_model.pth")

# %%
