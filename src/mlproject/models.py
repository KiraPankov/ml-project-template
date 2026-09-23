from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

REGISTRY: dict[str, type[ClassifierMixin]] = {
    "logreg": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "hist_gb": HistGradientBoostingClassifier,
}


def build_model(name: str, params: dict[str, Any], seed: int) -> ClassifierMixin:
    if name not in REGISTRY:
        raise ValueError(f"Неизвестная модель '{name}'. Доступны: {list(REGISTRY)}")
    return REGISTRY[name](random_state=seed, **params)
