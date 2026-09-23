"""Обучение модели: python -m mlproject.train --config configs/default.yaml"""

import argparse
import json
import time
from pathlib import Path

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from mlproject.config import Config, load_config
from mlproject.data import load_data, split_data
from mlproject.evaluate import compute_metrics
from mlproject.features import build_preprocessor, infer_feature_columns
from mlproject.logging_utils import get_logger
from mlproject.models import build_model

logger = get_logger(__name__)


def train(cfg: Config) -> dict:
    df = load_data(cfg.data, seed=cfg.seed)
    train_df, test_df = split_data(df, cfg.data, seed=cfg.seed)
    numeric, categorical = infer_feature_columns(df, cfg.data)
    features = numeric + categorical
    x_train, y_train = train_df[features], train_df[cfg.data.target]
    x_test, y_test = test_df[features], test_df[cfg.data.target]

    # Бейзлайн обязателен: без него непонятно, хороша ли модель вообще.
    baseline = DummyClassifier(strategy="prior").fit(x_train, y_train)
    baseline_metrics = compute_metrics(y_test, baseline.predict_proba(x_test)[:, 1])

    pipeline = Pipeline(
        [
            ("preprocess", build_preprocessor(numeric, categorical)),
            ("model", build_model(cfg.model.name, cfg.model.params, cfg.seed)),
        ]
    )
    start = time.perf_counter()
    pipeline.fit(x_train, y_train)
    train_seconds = time.perf_counter() - start

    model_metrics = compute_metrics(y_test, pipeline.predict_proba(x_test)[:, 1])
    results = {
        "model": cfg.model.name,
        "train_seconds": round(train_seconds, 3),
        "baseline": baseline_metrics,
        "test": model_metrics,
        "features": {"numeric": numeric, "categorical": categorical},
    }
    logger.info("Бейзлайн: %s", baseline_metrics)
    logger.info("Модель:   %s", model_metrics)

    model_path = Path(cfg.output.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "features": features}, model_path)

    metrics_path = Path(cfg.output.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Модель сохранена в %s, метрики в %s", model_path, metrics_path)

    if cfg.tracking.mlflow:
        _log_to_mlflow(cfg, results, model_path)
    return results


def _log_to_mlflow(cfg: Config, results: dict, model_path: Path) -> None:
    import mlflow

    mlflow.set_experiment(cfg.tracking.experiment_name)
    with mlflow.start_run():
        mlflow.log_params({"model": cfg.model.name, **cfg.model.params, "seed": cfg.seed})
        mlflow.log_metrics({f"test_{k}": v for k, v in results["test"].items()})
        mlflow.log_metrics({f"baseline_{k}": v for k, v in results["baseline"].items()})
        mlflow.log_artifact(str(model_path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
