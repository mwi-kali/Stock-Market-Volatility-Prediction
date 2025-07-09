import click
import os


import pandas as pd


from stock_market_volatility_prediction.utils.config import config
from stock_market_volatility_prediction.features.feature_engineer import FeatureEngineer
from stock_market_volatility_prediction.features.selector import FeatureSelector


@click.command()
def main():
    os.makedirs("data/processed", exist_ok=True)
    stock_df = pd.read_csv("data/raw/stock.csv", parse_dates=["Date"])
    news_df = pd.read_csv("data/raw/news_archive.csv",  parse_dates=["Date"])
    stock_df.columns = [col.split()[-1] for col in stock_df.columns]
    num_cols = [c for c in stock_df.columns if c != "Date"]
    stock_df[num_cols] = stock_df[num_cols].apply(pd.to_numeric, errors="coerce")
    stock_df.dropna(subset=["Close"], inplace=True)
    stock_df.reset_index(drop=True, inplace=True)

    fe = FeatureEngineer()
    tech_df = fe.add_technical(stock_df)
    sent_df = fe.add_sentiment(news_df)

    df = tech_df.merge(sent_df, on="Date", how="left").fillna(0)
    selector = FeatureSelector()
    selected_feats = selector.select(df, target="Volatility")
    out_df = df[["Date", "Volatility"] + selected_feats]
    out_df.to_csv("data/processed/features.csv", index=False)

    click.echo("Feature engineering & selection complete.")


if __name__ == "__main__":
    main()
