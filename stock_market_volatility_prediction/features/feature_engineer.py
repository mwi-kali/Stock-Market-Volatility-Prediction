import pandas as pd


from stock_market_volatility_prediction.utils.config import config
from stock_market_volatility_prediction.utils.logger import get_logger
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from transformers import pipeline


logger = get_logger()


class FeatureEngineer:
    def __init__(self, sentiment_model: str = None):
        self.sentiment_model = sentiment_model or config.SENT_MODEL

    def add_technical(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_values("Date")
        df["MA5"] = df["Close"].rolling(5).mean()
        df["MA10"] = df["Close"].rolling(10).mean()
        df["EMA5"] = df["Close"].ewm(span=5, adjust=False).mean()
        df["Momentum"] = df["Close"] - df["Close"].shift(4)
        bb = BollingerBands(df["Close"], window=20, window_dev=2)
        df["BB_High"] = bb.bollinger_hband()
        df["BB_Low"] = bb.bollinger_lband()
        rsi = RSIIndicator(df["Close"], window=14)
        df["RSI"] = rsi.rsi()
        df.dropna(inplace=True)
        logger.info("Technical indicators added")
        return df

    def add_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Running sentiment with model {self.sentiment_model}")
        pipe = pipeline("sentiment-analysis", model=self.sentiment_model, device=-1)
        def score(text: str) -> float:
            out = pipe(text[:512])[0]
            num = 1.0 if out["label"].upper() == "POSITIVE" else -1.0
            return out["score"] * num

        df = df.copy()
        df["sentiment_score"] = df["Content"].apply(score)
        out = df.groupby("Date")["sentiment_score"].mean().reset_index().rename(columns={"sentiment_score": "avg_sentiment"})
        logger.info("Sentiment aggregated by date")
        return out
