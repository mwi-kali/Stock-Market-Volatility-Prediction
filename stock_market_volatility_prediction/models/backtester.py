import numpy as np
import pandas as pd


from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Callable


class Backtester:

    def __init__(self, df: pd.DataFrame, target_col: str):
        self.df = df
        self.target = target_col

    def walk_forward(self, model_factory: Callable[[pd.Series], object], initial_train: int = 100, step: int = 10) -> pd.DataFrame:
        results = []
        series = self.df[self.target]
        for end in range(initial_train, len(series), step):
            train = series.iloc[:end]
            test = series.iloc[end:end+step]
            model = model_factory(train)
            preds = model.forecast(steps=len(test))
            results.append({
                "window_end": end,
                "mse": mean_squared_error(test, preds),
                "mae": mean_absolute_error(test, preds)
            })
        return pd.DataFrame(results)
