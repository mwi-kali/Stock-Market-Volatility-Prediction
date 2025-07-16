#!/usr/bin/env python
import os
import json
import click
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from stock_market_volatility_prediction.models.lstm_tuner import LSTMTuner
from stock_market_volatility_prediction.models.arima_tuner import ARIMATuner
from stock_market_volatility_prediction.models.prophet_tuner import ProphetTuner
from stock_market_volatility_prediction.models.training import (
    train_lstm, train_arima, train_prophet
)

PARAMS_PATH = "trained_models/best_params.json"

@click.command()
@click.option(
    "--mode",
    type=click.Choice(["tune", "train"], case_sensitive=False),
    default="tune",
    help="'tune': optimize all models’ hyperparams; 'train': train using tuned params"
)
@click.option(
    "--window",
    default=10,
    show_default=True,
    help="LSTM sliding-window length"
)
def main(mode: str, window: int):
    df = pd.read_csv("data/processed/features.csv", parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    dates = df["Date"]
    vol_series = df["Volatility"].values
    X_raw = df.drop(columns=["Date","Volatility"]).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    Xs, ys = [], []
    for i in range(window, len(X_scaled)):
        Xs.append(X_scaled[i-window:i])
        ys.append(vol_series[i])
        
    Xs = np.array(Xs)
    ys = np.array(ys)

    os.makedirs("trained_models", exist_ok=True)

    if mode.lower() == "tune":
        val_size = max(10, int(len(vol_series)*0.2))
        params = {}

        click.echo("Tuning LSTM...")
        params["lstm"] = LSTMTuner(Xs, ys).tune(n_trials=30)

        click.echo("Tuning ARIMA...")
        params["arima"] = ARIMATuner(vol_series, val_size).tune(n_trials=50)

        click.echo("Tuning Prophet...")
        params["prophet"] = ProphetTuner(dates, vol_series, val_size).tune(n_trials=30)

        with open(PARAMS_PATH, "w") as f:
            json.dump(params, f, indent=2)
        click.echo(f"Saved tuned params → {PARAMS_PATH}")
        return

    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH) as f:
            best = json.load(f)
        click.echo(f"Loaded tuned params from {PARAMS_PATH}")
    else:
        click.echo(f"No tuned params found; using defaults")
        best = {
            "lstm": {"dropout":0.3,"l2_reg":1e-3,"lr":1e-3,"batch_size":32},
            "arima": {"p":1,"d":1,"q":1},
            "prophet": {"changepoint_prior_scale":0.05,"seasonality_prior_scale":1.0,"seasonality_mode":"additive"},
        }

    click.echo("Training LSTM...")
    train_lstm(Xs, ys, best["lstm"], save_path="trained_models/lstm.h5")

    click.echo("Training ARIMA...")
    train_arima(vol_series, best["arima"], save_path="trained_models/arima.pkl")

    click.echo("Training Prophet...")
    train_prophet(dates, vol_series, best["prophet"], save_path="trained_models/prophet.pkl")

    click.echo("All models trained with tuned hyperparameters!")

if __name__ == "__main__":
    main()
