import optuna


import numpy as np


from math import sqrt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA


class ARIMATuner:
    def __init__(self, series: np.ndarray, val_size: int):
        self.series = series
        self.val_size = val_size
        self.train = series[:-val_size]
        self.val   = series[-val_size:]

    def _objective(self, trial) -> float:
        p = trial.suggest_int("p", 0, 5)
        d = trial.suggest_int("d", 0, 2)
        q = trial.suggest_int("q", 0, 5)
        try:
            m = ARIMA(self.train, order=(p, d, q)).fit()
            preds = m.forecast(steps=self.val_size)
            return sqrt(mean_squared_error(self.val, preds))
        except Exception:
            return 1e6  

    def tune(self, n_trials: int = 50) -> dict:
        study = optuna.create_study(direction="minimize")
        study.optimize(self._objective, n_trials=n_trials)
        return {"p": study.best_params["p"],
                "d": study.best_params["d"],
                "q": study.best_params["q"]}
