import optuna
from optuna.trial import Trial


def objective(trial: Trial):
    # Model and training parameters
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 0.1, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    n_layers = trial.suggest_categorical("n_layers", range(3, 10))

    hidden_dim = [
        trial.suggest_categorical("hidden_dim", [2 ^ i for i in range(3, 7)])
    ]
    activation = trial.suggest_categorical(
        "activation",
        ["relu", "gelu", "hardswish", "leakyrelu"],
    )
    activation_params = []
    if activation == "leakyrelu":
        activation_params.append(
            trial.suggest_float("leakyrelu_alpha", 0.01, 0.03)
        )
    # Dataset parameters
    rsi_fast_t = trial.suggest_float("rsi_fast_t", 3, 14)
    rsi_slow_t = trial.suggest_float("rsi_slow_t", low=rsi_fast_t + 7, high=50)
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

    ema_fast_t = trial.suggest_int("ema_fast_t", 5, 70)
    ema_slow_t = trial.suggest_int("ema_slow_t", ema_fast_t + 50, 300)

    obv_roc_t = trial.suggest_int("obv_roc_t", 3, 10)

    volume_sma_t = trial.suggest_int("volume_sma_t", 3, 10)
    HORIZON: int = trial.suggest_int("HORIZON", 1, 14)
