from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from mlproject.config import DataConfig
from mlproject.logging_utils import get_logger

logger = get_logger(__name__)


def make_synthetic(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Синтетический датасет, чтобы шаблон запускался «из коробки»."""
    x, y = make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=5,
        weights=[0.9, 0.1],  # дисбаланс классов, как в антифроде и скоринге
        random_state=seed,
    )
    df = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    rng = np.random.default_rng(seed)
    df["segment"] = rng.choice(["a", "b", "c"], size=n_samples)
    df["event_time"] = pd.date_range("2025-01-01", periods=n_samples, freq="h")
    df["target"] = y
    return df


def load_data(cfg: DataConfig, seed: int = 42) -> pd.DataFrame:
    path = Path(cfg.path)
    if path.exists():
        logger.info("Загружаю данные из %s", path)
        return pd.read_csv(path)
    logger.warning("Файл %s не найден, использую синтетические данные", path)
    return make_synthetic(seed=seed)


def split_data(
    df: pd.DataFrame, cfg: DataConfig, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Разбиение по времени, если есть колонка времени, иначе стратифицированное случайное.

    Разбиение по времени имитирует продакшен: модель учится на прошлом и
    предсказывает будущее. Случайное разбиение на таких данных завышает метрики.
    """
    if cfg.time_column and cfg.time_column in df.columns:
        df = df.sort_values(cfg.time_column).reset_index(drop=True)
        cut = int(len(df) * (1 - cfg.test_size))
        logger.info("Разбиение по времени: train=%d, test=%d", cut, len(df) - cut)
        return df.iloc[:cut], df.iloc[cut:]
    logger.info("Стратифицированное случайное разбиение")
    return train_test_split(df, test_size=cfg.test_size, stratify=df[cfg.target], random_state=seed)
