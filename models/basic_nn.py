# %%
import torch.nn as nn


class MLP(nn.Module):
    """
    A simple multi-layer perceptron.
    Replace with your own architecture (e.g., CNN, RNN).
    """

    def __init__(self, input_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x):
        return self.net(x)
