import pandas as pd


from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from stock_market_volatility_prediction.utils.logger import get_logger
from typing import List


logger = get_logger()


class FeatureSelector:
    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold

    def select(self, df: pd.DataFrame, target: str) -> List[str]:
        df = df.dropna().reset_index(drop=True)
        X = df.drop(columns=[target, "Date"])
        y = df[target]
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        imp = permutation_importance(model, X, y, n_repeats=5, random_state=42)
        scores = pd.Series(imp.importances_mean, index=X.columns)
        selected = scores[scores > self.threshold].index.tolist()
        logger.info(f"Selected features: {selected}")
        return selected
