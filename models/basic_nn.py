# %%
from typing import Any, Dict

import torch.nn as nn


class MLP(nn.Module):
    """
    A simple multi-layer perceptron.
    """

    def __init__(self, input_dim: int, out_dim: int, params: Dict[str, Any]):
        hidden_dims: list[int] = params.get("hidden_dims", [128, 256, 128])
        dropout = params.get("dropout", 0.2)
        layers = params.get("n_layers", 3)
        activation_str = params.get("activation", "relu")
        act_params = params.get("activation_params", [])
        match activation_str:
            case "relu":
                activation_type = nn.ReLU
            case "gelu":
                activation_type = nn.GELU
            case "hardswish":
                activation_type = nn.Hardswish
            case "leakyrelu":
                activation_type = nn.LeakyReLU
            case _:
                raise ValueError(
                    f"ERROR: invalid actiavition: {activation_str}"
                )
        activation_fn = (
            activation_type(*act_params) if act_params else activation_type()
        )
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(activation_fn)
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
