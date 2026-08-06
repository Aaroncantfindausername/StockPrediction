from typing import Any, Dict, Tuple
import pandas as pd
from data.preprocess import load_dataset, preprocess_dataset, robust_z_score
from data.single_ticker_minimal import compute_features_and_labels
from data import single_ticker_lagged


def get_feature_target_df(
    tickers: list[str], params: Dict[str, Any] = {}
) -> Tuple[pd.DataFrame, list[str]]:

    dfs: list[pd.DataFrame] = []
    cols: list[str] = []
    cols_z: list[str] = []
    for t in tickers:
        ticker_df = load_dataset(t)
        ticker_df = preprocess_dataset(ticker_df)

        if params.get("lag_features", False):
            cs, ticker_df = single_ticker_lagged.compute_features_and_labels(
                ticker_df, params
            )
        else:
            cs, ticker_df = compute_features_and_labels(ticker_df, params)
        cols = cs
        ticker_df["Ticker"] = [t] * len(ticker_df)
        dfs.append(ticker_df)
    df_long = pd.concat(dfs)
    for c in cols:
        cols_z.append(f"{c}_z")
    df_long[cols_z] = df_long.groupby("Date")[cols].transform(robust_z_score)
    return df_long, cols_z
