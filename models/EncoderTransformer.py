from typing import Any, Dict
import torch
import torch.nn as nn


class EncoderTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        d_model: int,
        encoder_layers: int,
        nhead: int,
        num_tickers: int = 1,
        has_ticker_embeding: bool = False,
        embedding_dim: int = 8,
        ff_hidden_dim: int = 1024,
        dropout: float = 0.0,
        activation_fn: str = "relu",
        max_seq_len: int = 50,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.has_ticker_embedding = has_ticker_embeding
        if has_ticker_embeding:
            self.ticker_embedding = nn.Embedding(num_tickers, embedding_dim)
            self.ticker_embedd_proj = nn.Linear(embedding_dim, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model,
                nhead,
                ff_hidden_dim,
                dropout,
                activation_fn,
                batch_first=True,
                device=device,
            ),
            encoder_layers,
        )
        self.head = nn.Linear(d_model, out_dim)

    def forward(
        self, x: torch.Tensor, ticker_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x : (batch, max_seq_len, num_features);  x has fixed seq len = max_seq_len
        # ticker_ids : (batch,)
        x = self.proj(x)  # (batch, max_seq_len, d_model)
        if self.has_ticker_embedding:
            ticker_embed = self.ticker_embedding(
                ticker_ids
            )  # (batch, embedding_dim)
            ticker_embed = self.ticker_embedd_proj(ticker_embed).unsqueeze(
                1
            )  # (batch, 1, d_model)
            x = x + ticker_embed

        positional_embed = self.pos_embedding(
            torch.arange(0, x.shape[1], device=x.device)
        ).unsqueeze(0)  # (1, seq, d_model)
        x = x + positional_embed
        x = self.encoder(x)  # (batch, max_seq_len, d_model)
        return self.head(x[:, -1, :])
