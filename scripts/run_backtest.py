import click
import os
import pickle


import pandas as pd


from prophet import Prophet
from stock_market_volatility_prediction.models.backtester import Backtester
from stock_market_volatility_prediction.models.baseline_models import BaselineModels


@click.command()
@click.option("--initial", default=100, help="Initial training window size")
@click.option("--step",    default=10,  help="Step size for rolling window")
def main(initial: int, step: int):
    df = pd.read_csv("data/processed/features.csv", parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")
    tester = Backtester(df, target_col="Volatility")

    def arima_factory(series):
        class M:
            def __init__(self, s): self.s = s
            def forecast(self, steps): return BaselineModels.arima(self.s, steps=steps)
        return M(series)

    def ets_factory(series):
        class M:
            def __init__(self, s): self.s = s
            def forecast(self, steps): return BaselineModels.ets(self.s, steps=steps)
        return M(series)

    def garch_factory(series):
        class M:
            def __init__(self, s): self.s = s
            def forecast(self, steps): return BaselineModels.garch(self.s, steps=steps)
        return M(series)

    def prophet_factory(series):
        class M:
            def __init__(self, s):
                df_prop = s.reset_index()        
                df_prop.columns = ["ds", "y"] 
                self.train_df = df_prop

            def forecast(self, steps):
                m = Prophet()
                m.fit(self.train_df)
                future = m.make_future_dataframe(periods=steps, freq="D")
                fc = m.predict(future)
                return fc["yhat"].iloc[-steps:].values
        return M(series)

    experiments = [
        ("ARIMA",   arima_factory),
        ("ETS",     ets_factory),
        ("GARCH",   garch_factory),
        ("Prophet", prophet_factory),
    ]

    all_results = []
    for name, factory in experiments:
        click.echo(f"Backtesting {name}...")
        res = tester.walk_forward(
            model_factory=factory,
            initial_train=initial,
            step=step
        )
        res["model"] = name
        all_results.append(res)

    out = pd.concat(all_results, ignore_index=True)[
        ["window_end", "model", "mse", "mae"]
    ]
    os.makedirs("data/processed", exist_ok=True)
    out.to_csv("data/processed/backtest_all_models.csv", index=False)
    click.echo("Backtest complete. Results saved to data/processed/backtest_all_models.csv")

if __name__ == "__main__":
    main()
