import optuna


import numpy as np


from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from stock_market_volatility_prediction.models.lstm_model import LSTMModel


class LSTMTuner:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def _objective(self, trial) -> float:
        params = {
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "l2_reg": trial.suggest_loguniform("l2_reg", 1e-4, 1e-2),
            "lr": trial.suggest_loguniform("lr", 1e-4, 1e-2),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        }
        X_tr, X_val, y_tr, y_val = train_test_split(self.X, self.y, test_size=0.2, shuffle=False)
        model = LSTMModel(
            input_shape=X_tr.shape[1:],
            dropout=params["dropout"],
            l2_reg=params["l2_reg"],
            lr=params["lr"],
        )
        model.train(X_tr, y_tr, X_val, y_val, epochs=50, batch_size=params["batch_size"])
        preds = model.predict(X_val).squeeze()
        return float(sqrt(mean_squared_error(y_val, preds)))

    def tune(self, n_trials: int = 30) -> dict:
        study = optuna.create_study(direction="minimize")
        study.optimize(self._objective, n_trials=n_trials)
        return study.best_params
