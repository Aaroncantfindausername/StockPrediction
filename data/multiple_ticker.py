from typing import Any, Dict, Tuple
import pandas as pd
from data.preprocess import load_dataset, preprocess_dataset
from data.single_ticker_minimal import compute_features_and_labels


def get_feature_target_df(
    tickers: list[str], params: Dict[str, Any] = {}
) -> Tuple[pd.DataFrame, list[str]]:
    dfs: list[pd.DataFrame] = []
    cols: list[str] = []
    for t in tickers:
        ticker_df = load_dataset(t)
        ticker_df = preprocess_dataset(ticker_df)
        cs, ticker_df = compute_features_and_labels(ticker_df, params)
        cols = cs
        ticker_df["Ticker"] = [t] * len(ticker_df)
        dfs.append(ticker_df)
    df_long = pd.concat(dfs)
    return df_long, cols
