from typing import Any, Dict, Tuple
import pandas as pd
from data.preprocess import load_dataset, preprocess_dataset, robust_z_score
from data.single_ticker_minimal import compute_features_and_labels
from data import single_ticker_lagged
import numpy as np


def get_feature_target_df(
    tickers: list[str], params: Dict[str, Any] = {}
) -> Tuple[pd.DataFrame, list[str]]:
    start_date = params.get("start_date")
    end_date = params.get("end_date")
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
        if start_date is not None:
            ticker_df = ticker_df[start_date:]
        if end_date is not None:
            ticker_df = ticker_df[:end_date]
        dfs.append(ticker_df)
    df_long = pd.concat(dfs)
    for c in cols:
        cols_z.append(f"{c}_z")
    if params.get("cross_sectional_z_score", False):
        df_long[cols_z] = df_long.groupby("Date")[cols].transform(
            robust_z_score
        )
        df_long["target_return_cross_rank"] = (
            df_long.groupby("Date")["target_return"]
            .rank(pct=True)
            .astype(np.float32)
        )
    return df_long, cols_z
