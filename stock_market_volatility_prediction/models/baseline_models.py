import pandas as pd


from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from typing import Sequence


class BaselineModels:
    @staticmethod
    def arima(series: Sequence[float], order=(1,1,1), steps: int = 1) -> pd.Series:
        model = ARIMA(series, order=order).fit()
        return model.forecast(steps)

    @staticmethod
    def ets(series: Sequence[float], seasonal_periods: int = None, trend: str = None, seasonal: str = None, steps: int = 1) -> pd.Series:
        model = ExponentialSmoothing(series, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods).fit()
        return model.forecast(steps)

    @staticmethod
    def garch(series: Sequence[float], p: int = 1, q: int = 1, steps: int = 1):
        model = arch_model(series*100, vol="Garch", p=p, q=q, dist="normal").fit(disp="off")
        f = model.forecast(horizon=steps)
        return (f.mean.iloc[-1]/100).values
