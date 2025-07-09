import pandas as pd
import yfinance as yf


from stock_market_volatility_prediction.utils.logger import get_logger


logger = get_logger()


class StockFetcher:
    def __init__(self, ticker: str, start_date: str, end_date: str):
        self.ticker = ticker
        self.start = start_date
        self.end = end_date

    def fetch(self) -> pd.DataFrame:
        logger.info(f"Fetching stock data for {self.ticker} from {self.start} to {self.end}")
        df = yf.download(self.ticker, start=self.start, end=self.end)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df.reset_index()
        df["Return"] = df["Close"].pct_change()
        df["Volatility"] = df["Return"].rolling(window=5).std()
        df.dropna(inplace=True)
        df["Date"] = df["Date"].dt.date
        logger.info(f"Fetched {len(df)} rows of stock data")
        return df
