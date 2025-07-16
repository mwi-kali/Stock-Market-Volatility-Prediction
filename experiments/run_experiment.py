import click
import json
import mlflow
import os
import pickle


import numpy as np
import pandas as pd


from datetime import date
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from stock_market_volatility_prediction.ingestion.macro_fetcher import MacroFetcher
from stock_market_volatility_prediction.ingestion.news_fetcher import NewsFetcher
from stock_market_volatility_prediction.ingestion.stock_fetcher import StockFetcher
from stock_market_volatility_prediction.features.feature_engineer import FeatureEngineer
from stock_market_volatility_prediction.features.selector import FeatureSelector
from stock_market_volatility_prediction.models.arima_tuner import ARIMATuner
from stock_market_volatility_prediction.models.lstm_tuner import LSTMTuner
from stock_market_volatility_prediction.models.prophet_tuner import ProphetTuner
from stock_market_volatility_prediction.models.training import train_arima, train_lstm, train_prophet
from stock_market_volatility_prediction.utils.config import config


PARAMS_PATH = "trained_models/best_params.json"


@click.command()
@click.option("--start", default=config.START_DATE, help="Data start date (YYYY-MM-DD)")
@click.option("--window", default=10, type=int, help="LSTM sliding window size")
@click.option("--tune-trials", default=30, type=int, help="Number of Optuna trials per model")
def main(start: str, window: int, tune_trials: int):
    mlflow.set_experiment("VolatilityForecasting")
    with mlflow.start_run():
        end = date.today().isoformat()
        mlflow.log_params({
            "start_date": start,
            "end_date": end,
            "window": window,
            "tune_trials": tune_trials
        })

        stock_df = StockFetcher(config.TICKER, start, end).fetch()
        news_df  = NewsFetcher(config.TICKER_NAME, start, end).fetch_all()
        macro_df = MacroFetcher().fetch_all(start, end)

        os.makedirs("data/raw", exist_ok=True)
        stock_df.to_csv("data/raw/stock.csv", index=False)
        news_df.to_csv("data/raw/news.csv", index=False)
        macro_df.to_csv("data/raw/macro.csv", index=False)
        mlflow.log_artifact("data/raw/stock.csv", artifact_path="raw")
        mlflow.log_artifact("data/raw/news.csv", artifact_path="raw")
        mlflow.log_artifact("data/raw/macro.csv", artifact_path="raw")

        fe = FeatureEngineer()
        tech_df = fe.add_technical(stock_df)
        sent_df = fe.add_sentiment(news_df)
        df = tech_df.merge(sent_df, on="Date", how="left").fillna(0)
        df = df.merge(macro_df, on="Date", how="left").ffill().bfill()

        selector = FeatureSelector()
        selected = selector.select(df, target="Volatility")
        mlflow.log_param("num_features", len(selected))

        features = ["Date", "Volatility"] + selected
        final_df = df[features].dropna().reset_index(drop=True)
        final_df.to_csv("data/processed/features.csv", index=False)
        mlflow.log_artifact("data/processed/features.csv", artifact_path="processed")

        X = final_df.drop(columns=["Date", "Volatility"]).values
        y = final_df["Volatility"].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        Xs, ys = [], []
        for i in range(window, len(X_scaled)):
            Xs.append(X_scaled[i-window:i])
            ys.append(y[i])
        Xs = np.array(Xs)
        ys = np.array(ys)

        val_size = max(10, int(len(y) * 0.2))

        tuned = {}

        click.echo("Tuning LSTM...")
        lstm_params = LSTMTuner(Xs, ys).tune(n_trials=tune_trials)
        tuned["lstm"] = lstm_params
        mlflow.log_params({f"lstm_{k}": v for k, v in lstm_params.items()})

        click.echo("Tuning ARIMA...")
        arima_params = ARIMATuner(y, val_size).tune(n_trials=tune_trials)
        tuned["arima"] = arima_params
        mlflow.log_params({f"arima_{k}": v for k, v in arima_params.items()})

        click.echo("Tuning Prophet...")
        prophet_params = ProphetTuner(final_df["Date"], y, val_size).tune(n_trials=tune_trials)
        tuned["prophet"] = prophet_params
        mlflow.log_params({f"prophet_{k}": v for k, v in prophet_params.items()})

        os.makedirs("trained_models", exist_ok=True)
        with open(PARAMS_PATH, "w") as f:
            json.dump(tuned, f, indent=2)
        mlflow.log_artifact(PARAMS_PATH, artifact_path="trained_models")

        click.echo("Training LSTM on full data...")
        lstm = train_lstm(Xs, ys, tuned["lstm"], save_path="trained_models/lstm.h5")
        mlflow.keras.log_model(lstm.model, artifact_path="trained_models/lstm")

        click.echo("Training ARIMA on full data...")
        arima = train_arima(pd.Series(y), tuned["arima"], save_path="trained_models/arima.pkl")
        mlflow.log_artifact("trained_models/arima.pkl", artifact_path="trained_models")

        click.echo("Training Prophet on full data...")
        prophet = train_prophet(final_df["Date"], pd.Series(y), tuned["prophet"], save_path="trained_models/prophet.pkl")
        mlflow.log_artifact("trained_models/prophet.pkl", artifact_path="trained_models")

        preds_lstm = lstm.predict(Xs).squeeze()
        mse_lstm = mean_squared_error(ys, preds_lstm)
        mlflow.log_metrics({
            "LSTM_MSE": mse_lstm,
            "LSTM_RMSE": np.sqrt(mse_lstm),
            "LSTM_R2": r2_score(ys, preds_lstm)
        })

        preds_arima = arima.forecast(steps=len(y))
        mse_arima = mean_squared_error(y, preds_arima)
        mlflow.log_metric("ARIMA_MSE", mse_arima)

        df_prop = pd.DataFrame({"ds": final_df["Date"], "y": y})
        preds_prophet = prophet.predict(df_prop)["yhat"].values
        mse_prophet = mean_squared_error(y, preds_prophet)
        mlflow.log_metric("Prophet_MSE", mse_prophet)

    click.echo("Experiment complete and logged to MLflow.")

if __name__ == "__main__":
    main()
