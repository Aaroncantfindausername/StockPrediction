# %%
from typing import Any, Dict
import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    A multi-layer perceptron with an optional embedding for multiple tickers.
    """

    def __init__(
        self,
        feature_dim: int,
        out_dim: int,
        has_embedding: bool = False,
        embedding_dim: int = 0,
        num_unique_embeddings: int = 0,
        params: Dict[str, Any] = {},
    ):
        hidden_dim: int = params.get("hidden_dim", 256)
        hidden_dim_decay: int = params.get("hidden_dim_decay", 0.5)
        dropout = params.get("dropout_rate", 0.2)
        n_layers: int = params.get("n_layers", 3)
        activation_str = params.get("activation", "relu")
        act_params = params.get("activation_params", [])
        self.has_embedding = has_embedding
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
        prev_dim = feature_dim + embedding_dim
        for i in range(n_layers):
            hidden_out_dim = int(hidden_dim * (hidden_dim_decay**i))
            layers.append(nn.Linear(prev_dim, hidden_out_dim))
            layers.append(nn.BatchNorm1d(hidden_out_dim))
            layers.append(activation_fn)
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_out_dim
        layers.append(nn.Linear(prev_dim, out_dim))
        self.net = nn.Sequential(*layers)
        if self.has_embedding:
            self.embedding = nn.Embedding(num_unique_embeddings, embedding_dim)

    def forward(
        self, features: torch.Tensor, ticker_ids: torch.Tensor | None = None
    ):
        if self.has_embedding:
            emb: torch.Tensor = self.embedding(ticker_ids)
            x = torch.concat([features, emb], dim=1)
        else:
            x = features
        return self.net(x)
