from enum import unique
from math import log
import statistics
from typing import Union
from numpy import float32
from numpy.typing import NDArray
from optuna.trial import Trial
from pandas import DataFrame, DatetimeIndex
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader, TensorDataset
import optuna
from backtest.backtest import (
    multi_ticker_backtest,
    precompute_outputs,
    precompute_tickers,
)
from data import multiple_ticker
from data.single_ticker_minimal import compute_features_and_labels
from models.mlp import MLP
from train.pipeline import (
    evaluate,
    train_epoch,
    train_full_with_embedding,
)
from torch import nn, optim
import torch

from train.ticker_embedding_dataset import TickerEmbeddingDataset


def objective_full(
    trial: Trial,
    tickers: list[str],
    start_date: DatetimeIndex,
    end_date: DatetimeIndex,
    features: list[str],
    target_column: str,
    device: Union[str, torch.device],
    time_series_splits: int = 3,
    min_train_years: int = 6,
    batch_size: int = 256,
    embedding_dim: int = 3,
    epochs: int = 1000,
    early_stop_patience: int = 30,
    scheduler_patience: int = 5,
    backtest_target: str = "daily_sharpe",
) -> float:

    # Model and training hyperparameters
    dropout_rate = trial.suggest_float("dropout_rate", 0.3, 0.6)
    # Dataset hyperparameters
    horizon = trial.suggest_int("horizon", 5, 20)
    rsi_fast_t = trial.suggest_int("rsi_fast_t", 3, 14)
    rsi_slow_t = trial.suggest_int("rsi_slow_t", low=rsi_fast_t + 7, high=50)
    macd_fastperiod = trial.suggest_int("macd_fastperiod", 5, 14)
    macd_slowperiod = trial.suggest_int(
        "macd_slowperiod", macd_fastperiod + 5, 40
    )
    macd_signalperiod = trial.suggest_int("macd_signalperiod", 3, 12)

    stoch_fastk = trial.suggest_int("stoch_fastk", 5, 28)
    stoch_slowk = trial.suggest_int("stoch_slowk", 2, 7)
    stoch_slowd = trial.suggest_int("stoch_slowd", 2, 7)
    adx_t = trial.suggest_int("adx_t", 3, 28)
    bbands_t = trial.suggest_int("bbands_t", 5, 30)
    atr_t = trial.suggest_int("atr", 3, 28)

    ema_fast_t = trial.suggest_int("ema_fast_t", 5, 50)
    ema_slow_t = trial.suggest_int("ema_slow_t", ema_fast_t * 2, 200)

    sma_fast_t = trial.suggest_int("sma_fast_t", 5, 50)
    sma_slow_t = trial.suggest_int("sma_slow_t", sma_fast_t * 2, 200)
    obv_roc_t = trial.suggest_int("obv_roc_t", 3, 10)

    volume_sma_t = trial.suggest_int("volume_sma_t", 3, 10)

    model_params = {
        "dropout_rate": dropout_rate,
        "hidden_dim": 1024,
        "n_layers": 3,
        "hidden_dim_decay": 0.5,
    }
    dataset_params = {
        "rsi_fast_t": rsi_fast_t,
        "rsi_slow_t": rsi_slow_t,
        "macd_fastperiod": macd_fastperiod,
        "macd_slowperiod": macd_slowperiod,
        "macd_signalperiod": macd_signalperiod,
        "stoch_fastk": stoch_fastk,
        "stoch_slowk": stoch_slowk,
        "stoch_slowd": stoch_slowd,
        "adx_t": adx_t,
        "bbands_t": bbands_t,
        "atr_t": atr_t,
        "ema_fast_t": ema_fast_t,
        "ema_slow_t": ema_slow_t,
        "sma_fast_t": sma_fast_t,
        "sma_slow_t": sma_slow_t,
        "obv_roc_t": obv_roc_t,
        "volume_sma_t": volume_sma_t,
        "HORIZON": horizon,
        "columns": features,
        "cross_sectional_z_score": True,
        "start_date": start_date,
        "end_date": end_date,
    }
    # Generate features and labels with suggested params
    random_seed = trial.number
    g = torch.Generator()
    g.manual_seed(random_seed)

    df, cols = multiple_ticker.get_feature_target_df(
        tickers,
        dataset_params,
    )
    ticker_to_id = {t: i for i, t in enumerate(tickers)}
    df["ticker_id"] = df["Ticker"].map(ticker_to_id)
    num_days = df.index.nunique()
    # Mini walk forward validation,  but train all folds epoch by epoch and aggregate evaluation at each
    tscv = TimeSeriesSplit(n_splits=time_series_splits, gap=horizon)
    unique_dates = df.index.unique()
    fold = 0
    stats = []
    for train_dates_idx, test_dates_idx in tscv.split(unique_dates):
        train_dates = unique_dates[train_dates_idx]
        test_dates = unique_dates[test_dates_idx]
        train_mask = df.index.isin(train_dates)
        test_mask = df.index.isin(test_dates)
        train_dataset = TickerEmbeddingDataset(
            df.loc[train_mask], cols, target_column
        )
        test_dataset = TickerEmbeddingDataset(
            df.loc[test_mask], cols, target_column
        )

        train_loader = DataLoader(train_dataset, batch_size, True)
        test_loader = DataLoader(test_dataset, batch_size, False)
        model = MLP(
            len(cols), 1, True, embedding_dim, len(tickers), model_params
        ).to(device)
        criterion = nn.MSELoss()
        optimiser = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimiser,
            mode="min",
            patience=scheduler_patience,
            factor=0.25,
            threshold=0.0,
        )
        train_full_with_embedding(
            model,
            train_loader,
            test_loader,
            criterion,
            optimiser,
            scheduler,
            device,
            epochs,
            early_stop_patience,
            load_best_val_loss=True,
        )

        model_config = {
            "input_dim": len(cols),
            "classification": False,
            "out_dim": 1,
            "embedding_dim": embedding_dim,
            "num_unique_embeddings": len(tickers),
            "columns": cols,
            "ticker_to_id": ticker_to_id,
        }

        # Generate outputs on test data
        outputs_dict = precompute_tickers(
            tickers,
            test_dates.min(),
            unique_dates.max(),
            save=False,
            df=df,
            model=model,
            config=model_config,
        )

        stat = multi_ticker_backtest(
            test_dates.min(),
            unique_dates.max(),
            outputs_dict=outputs_dict,
            n=5,
            plot=False,
        )
        # Backtest on predictions
        trial.report(stat.loc[backtest_target].iloc[0], fold)
        stats.append(stat.loc[backtest_target].iloc[0])
        print(
            f"Fold {fold} backtest {backtest_target}: {stat.loc[backtest_target].iloc[0]}"
        )
        fold += 1
        if trial.should_prune():
            raise optuna.TrialPruned()
    print(f"Trial {trial.number} complete")
    return statistics.mean(stats)
