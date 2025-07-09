import pandas as pd


from fredapi import Fred
from stock_market_volatility_prediction.utils.config import config
from stock_market_volatility_prediction.utils.logger import get_logger


logger = get_logger()


class MacroFetcher:
    SERIES_IDS = ["DGS10", "DEXUSAL", "CPALTT01USM657N", "CPILFESL", "FEDFUNDS"]

    def __init__(self):
        self.client = Fred(api_key=config.FRED_API_KEY)

    def fetch_series(self, series_id: str, start: str, end: str) -> pd.DataFrame:
        logger.info(f"Fetching FRED series {series_id}")
        s = self.client.get_series(series_id, observation_start=start, observation_end=end)
        df =  s.rename(series_id).reset_index().rename(columns={"index": "Date"})
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    def fetch_all(self, start: str, end: str) -> pd.DataFrame:
        dfs = [self.fetch_series(sid, start, end) for sid in self.SERIES_IDS]
        df = dfs[0]
        for other in dfs[1:]:
            df = df.merge(other, on="Date", how="outer")

        df = df.set_index("Date")
        df = df.interpolate(method="time")
        df = df.reset_index()

        logger.info(f"Fetched macro data. It has {df.shape[0]} rows and {df.columns.tolist()} columns")
        return df
