import pickle

import pandas as pd

from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from stock_market_volatility_prediction.models.lstm_model import LSTMModel


def train_lstm(Xs, ys, params: dict, save_path: str = "trained_models/lstm.h5"):
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(Xs, ys, test_size=0.2, shuffle=False)
    m = LSTMModel(
        input_shape=X_tr.shape[1:],
        dropout=params["dropout"],
        l2_reg=params["l2_reg"],
        lr=params["lr"]
    )
    m.train(
        X_tr, y_tr,
        X_val, y_val,
        epochs=100,
        batch_size=params["batch_size"],
        patience=10
    )
    m.save(save_path)
    return m

def train_arima(series: pd.Series, params: dict, save_path: str = "trained_models/arima.pkl"):
    m = ARIMA(series, order=(params["p"], params["d"], params["q"])).fit()
    with open(save_path, "wb") as f:
        pickle.dump(m, f)
    return m

def train_prophet(dates: pd.Series, series: pd.Series, params: dict, save_path: str = "trained_models/prophet.pkl"):
    df = pd.DataFrame({"ds": dates, "y": series})
    m  = Prophet(
        changepoint_prior_scale=params["changepoint_prior_scale"],
        seasonality_prior_scale=params["seasonality_prior_scale"],
        seasonality_mode=params["seasonality_mode"]
    )
    m.fit(df)
    with open(save_path, "wb") as f:
        pickle.dump(m, f)
    return m
