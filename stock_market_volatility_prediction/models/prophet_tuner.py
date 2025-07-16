import optuna


import numpy as np
import pandas as pd


from math import sqrt
from sklearn.metrics import mean_squared_error
from prophet import Prophet


class ProphetTuner:
    def __init__(self, dates: pd.Series, series: np.ndarray, val_size: int):
        df = pd.DataFrame({"ds": dates, "y": series})
        self.train_df = df.iloc[:-val_size]
        self.val_df = df.iloc[-val_size:]
        self.val_size = val_size

    def _objective(self, trial) -> float:
        cps = trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True)
        spr = trial.suggest_float("seasonality_prior_scale", 0.01, 10, log=True)
        sm = trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"])
        try:
            m = Prophet(
                changepoint_prior_scale=cps,
                seasonality_prior_scale=spr,
                seasonality_mode=sm
            )
            m.fit(self.train_df)
            future = m.make_future_dataframe(periods=self.val_size, freq="D")
            fcst = m.predict(future)
            preds = fcst["yhat"].iloc[-self.val_size:].values
            return sqrt(mean_squared_error(self.val_df["y"].values, preds))
        except Exception:
            return 1e6

    def tune(self, n_trials: int = 30) -> dict:
        study = optuna.create_study(direction="minimize")
        study.optimize(self._objective, n_trials=n_trials)
        bp = study.best_params
        return {
            "changepoint_prior_scale": bp["changepoint_prior_scale"],
            "seasonality_prior_scale": bp["seasonality_prior_scale"],
            "seasonality_mode": bp["seasonality_mode"]
        }
