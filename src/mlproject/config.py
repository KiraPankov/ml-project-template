from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    path: str
    target: str
    time_column: str | None = None
    test_size: float = 0.2
    numeric_features: list[str] | None = None
    categorical_features: list[str] | None = None


@dataclass
class ModelConfig:
    name: str = "logreg"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputConfig:
    model_path: str = "models/model.joblib"
    metrics_path: str = "reports/metrics.json"


@dataclass
class TrackingConfig:
    mlflow: bool = False
    experiment_name: str = "mlproject"


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    output: OutputConfig
    tracking: TrackingConfig
    seed: int = 42


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(
        seed=raw.get("seed", 42),
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw.get("model", {})),
        output=OutputConfig(**raw.get("output", {})),
        tracking=TrackingConfig(**raw.get("tracking", {})),
    )
