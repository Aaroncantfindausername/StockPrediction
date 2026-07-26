from typing import Tuple

import torch
from pandas import DataFrame
from torch.utils.data import Dataset


class SequentialEmbeddingDataset(Dataset):
    def __init__(
        self,
        df: DataFrame,
        feature_cols: list[str],
        target_col: str,
        seq_len: int,
    ) -> None:
        self.seq_len = seq_len
        self.features = torch.tensor(df[feature_cols].values)
        self.targets = torch.tensor(df[target_col].values)
        self.ticker_ids = torch.tensor(df["ticker_id"].values, dtype=torch.long)

    def __len__(self):
        return len(self.features - self.seq_len + 1)

    def __getitem__(
        self, index
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.features[index : index + self.seq_len],
            self.ticker_ids[index : index + self.seq_len],
            self.targets[index : index + self.seq_len],
        )
