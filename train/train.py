import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.basic_nn import MLP
from talib import abstract
import copy
from data.preprocess import add_indicator, load_dataset, preprocess_dataset
from train.pipeline import train_epoch, load_data, evaluate
from data import single_ticker_minimal

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

# Load dataset
ticker: str = "^GSPC"
df = load_dataset(ticker)
df = preprocess_dataset(df)
X, y = single_ticker_minimal.get_features_labels(df, ticker)
# %%

# Load data
train_loader, val_loader, test_loader, input_dim = load_data(
    X, y, batch_size=config["batch_size"]
)

# Model, loss, optimizer

model = MLP(input_dim, config["hidden_dim"], 1, config["dropout"]).to(device)
criterion = nn.HuberLoss()
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
test_loss, test_mae = evaluate(model, test_loader, criterion, device)
print(f"\nTest Loss: {test_loss:.4f} | Test MAE: {test_mae:.4f}")
# Save final model (optional)
torch.save(model.state_dict(), "weights/basic_nn.pth")
print("Model weights saved")
if __name__ == "__main__":
    pass
