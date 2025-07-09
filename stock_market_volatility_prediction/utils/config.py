import os


from dotenv import load_dotenv


load_dotenv()


class Config:
    NEWSAPI_KEY: str = os.getenv("NEWSAPI_KEY", "")
    FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")
    TICKER: str = os.getenv("TICKER", "NVDA")
    TICKER_NAME: str = os.getenv("TICKER_NAME", "NVIDIA")
    START_DATE: str = os.getenv("START_DATE", "2024-01-01")
    END_DATE: str = os.getenv("END_DATE", "2025-07-08")
    SENT_MODEL: str = os.getenv("SENT_MODEL", "yiyanghkust/finbert-tone")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

config = Config()
