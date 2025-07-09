import click
import os


import pandas as pd


from datetime import date, timedelta
from stock_market_volatility_prediction.ingestion.macro_fetcher import MacroFetcher
from stock_market_volatility_prediction.ingestion.news_fetcher import NewsFetcher
from stock_market_volatility_prediction.ingestion.stock_fetcher import StockFetcher
from stock_market_volatility_prediction.utils.config import config
from stock_market_volatility_prediction.utils.logger import get_logger


logger = get_logger(__name__)


RAW_DIR = "data/raw"
ARCHIVE_PATH = os.path.join(RAW_DIR, "news_archive.csv")


os.makedirs(RAW_DIR, exist_ok=True)


@click.command()
@click.option("--start", default=config.START_DATE, help="Stock/macro start date (YYYY-MM-DD).")
def main(start: str):
    today = date.today()
    stock_df = StockFetcher(config.TICKER, start, today).fetch()
    stock_df.to_csv(os.path.join(RAW_DIR, "stock.csv"), index=False)
    logger.info(f"Stock data saved ({len(stock_df)} rows).")
    macro_df = MacroFetcher().fetch_all(start, today)
    macro_df.to_csv(os.path.join(RAW_DIR, "macro.csv"), index=False)
    logger.info(f"Macro data saved ({len(macro_df)} rows).")

    
    thirty_days_ago = today - timedelta(days=30)
    news_df = NewsFetcher(
        config.TICKER_NAME,
        thirty_days_ago.isoformat(),
        today.isoformat()
    ).fetch_all()

    news_df["Date"] = pd.to_datetime(news_df["Date"])

    if os.path.exists(ARCHIVE_PATH):
        archive = pd.read_csv(ARCHIVE_PATH, parse_dates=["Date"])
        archive["Date"] = pd.to_datetime(archive["Date"])
    else:
        archive = pd.DataFrame(columns=news_df.columns)

    combined = pd.concat([archive, news_df], ignore_index=True)
    combined.drop_duplicates(subset=["Date", "Content"], inplace=True)
    combined.sort_values("Date", inplace=True)
    combined.to_csv(ARCHIVE_PATH, index=False)

    logger.info(f"News archive updated ({len(combined)} total items).")
    click.echo(
        f"Ingestion complete.\n"
        f" • Stock rows are {len(stock_df)}\n"
        f" • Macro rows are {len(macro_df)}\n"
        f" • News archive rows are {len(combined)}"
    )

if __name__ == "__main__":
    main()
