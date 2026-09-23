from pathlib import Path
from typing import Any

import joblib
import pandas as pd


class Predictor:
    """Загружает обученный пайплайн и делает предсказания на списке объектов."""

    def __init__(self, model_path: str | Path):
        artifact = joblib.load(model_path)
        self.pipeline = artifact["pipeline"]
        self.features: list[str] = artifact["features"]

    def predict_proba(self, records: list[dict[str, Any]]) -> list[float]:
        df = pd.DataFrame.from_records(records)
        missing = set(self.features) - set(df.columns)
        if missing:
            raise ValueError(f"Не хватает признаков: {sorted(missing)}")
        return self.pipeline.predict_proba(df[self.features])[:, 1].tolist()
