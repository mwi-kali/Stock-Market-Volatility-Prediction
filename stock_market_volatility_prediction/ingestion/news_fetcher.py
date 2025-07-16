import feedparser
import re
import requests


import pandas as pd


from stock_market_volatility_prediction.utils.config import config
from stock_market_volatility_prediction.utils.logger import get_logger
from typing import List


logger = get_logger()


class NewsFetcher:

    def __init__(self, query: str, start: str, end: str):
        self.query = query
        self.RSS_URLS: List[str] = [
            "https://www.reuters.com/rssFeed/marketsNews",
            "https://www.bloomberg.com/feed/podcast/etf-report.xml",
            "https://feeds.skynews.com/feeds/rss/world.xml",
            "https://feeds.bbci.co.uk/news/world/rss.xml",
            "https://feeds.nbcnews.com/nbcnews/public/news",
            "https://www.cnbc.com/id/100727362/device/rss/rss.html",
            "https://abcnews.go.com/abcnews/internationalheadlines",
            "https://www.aljazeera.com/xml/rss/all.xml",
            "https://www.cbsnews.com/latest/rss/world",
            "https://www.nytimes.com/svc/collections/v1/publish/https://www.nytimes.com/section/world/rss.xml",
            "https://feeds.washingtonpost.com/rss/world",
            "https://globalnews.ca/world/feed/",
            "https://feeds.feedburner.com/time/world",
            "https://feeds.npr.org/1004/rss.xml",
            "https://www.washingtontimes.com/rss/headlines/news/world"
        ]
        self.start = start
        self.end = end

    def fetch_newsapi(self) -> pd.DataFrame:
        logger.info("Fetching news from NewsAPI")
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": self.query,
            "from": self.start,
            "to": self.end,
            "apiKey": config.NEWSAPI_KEY
        }
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        items = [{
            "Date": art["publishedAt"][:10],
            "Content": f"{art.get('title','')} {art.get('description','')}".strip(),
            "Source": art["source"]["name"]
        } for art in resp.json().get("articles", [])]
        df = pd.DataFrame(items)
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        return df

    def fetch_rss(self) -> pd.DataFrame:
        logger.info("Fetching news from RSS feeds")
        records = []

        pattern = re.compile(re.escape(self.query), re.IGNORECASE)

        for url in self.RSS_URLS:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                pub = getattr(entry, "published", None) or getattr(entry, "updated", None)
                if not pub:
                    continue

                title = entry.get("title", "")
                summary = entry.get("summary", "")
                content = f"{title} {summary}".strip()

                if not pattern.search(content):
                    continue

                records.append({
                    "Date": pd.to_datetime(pub).date(),
                    "Content": content,
                    "Source": feed.feed.get("title", url)
                })

        return pd.DataFrame(records)

    def fetch_all(self) -> pd.DataFrame:
        df1 = self.fetch_newsapi()
        df2 = self.fetch_rss()
        df = pd.concat([df1, df2], ignore_index=True).drop_duplicates(subset=["Date", "Content"])
        df.sort_values("Date", inplace=True)
        logger.info(f"Fetched {len(df)} total news items")
        return df.reset_index(drop=True)
