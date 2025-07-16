import pickle


import numpy as np
import pandas as pd
import plotly.graph_objects as go


from dash import Input, Output, callback
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from stock_market_volatility_prediction.models.baseline_models import BaselineModels
from stock_market_volatility_prediction.models.lstm_model import LSTMModel
from stock_market_volatility_prediction.utils.config import config



@callback(
    [
        Output("card-actual", "children"),
        Output("card-lstm", "children"),
        Output("card-arima", "children"),
        Output("card-prophet", "children"),
        Output("forecast-graph", "figure"),
    ],
    [
        Input("interval", "n_intervals")
    ]
)
def update_dashboard(n_intervals):
    df = pd.read_csv("data/processed/features.csv", parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    dates  = df["Date"]
    actual = df["Volatility"]

    X = df.drop(columns=["Date", "Volatility"]).values
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    window = 5  
    Xs = np.array([X_scaled[i - window : i] for i in range(window, len(X_scaled))])
    pad = [np.nan] * window

    lstm = LSTMModel.load("trained_models/lstm.h5")
    preds_lstm = lstm.predict(Xs).squeeze()
    preds_lstm_full = np.array(pad + preds_lstm.tolist())

    with open("trained_models/arima.pkl", "rb") as f:
        arima_model = pickle.load(f)
    try:
        preds_arima_full = arima_model.predict(start=0, end=len(actual) - 1)
        preds_arima_full = np.array(preds_arima_full)
    except Exception:
        preds_arima_full = np.full(len(actual), np.nan)

    with open("trained_models/prophet.pkl", "rb") as f:
        prophet_model = pickle.load(f)
    df_prophet = pd.DataFrame({"ds": dates, "y": actual})
    forecast = prophet_model.predict(df_prophet)
    preds_prophet_full = forecast["yhat"].values

    fig = go.Figure([
        go.Scatter(x=dates, y=actual, mode="lines", name="Actual", line=dict(color="black")),
        go.Scatter(x=dates, y=preds_lstm_full, mode="lines", name="LSTM Predicted"),
        go.Scatter(x=dates, y=preds_arima_full, mode="lines", name="ARIMA Predicted"),
        go.Scatter(x=dates, y=preds_prophet_full, mode="lines", name="Prophet Predicted"),
    ])
    fig.update_layout(
        title="Actual vs. Predicted Volatility Over Time",
        xaxis_title="Date",
        yaxis_title="Volatility",
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0)")
    )

    fmt = lambda v: f"{v:.4f}" if not np.isnan(v) else "n/a"
    latest_vals = [
        fmt(actual.iloc[-1]),
        fmt(preds_lstm_full[-1]),
        fmt(preds_arima_full[-1]),
        fmt(preds_prophet_full[-1])
    ]

    return [*latest_vals, fig]
