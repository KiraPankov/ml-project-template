import pytest

from mlproject.config import Config, DataConfig, ModelConfig, OutputConfig, TrackingConfig


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(
        seed=0,
        data=DataConfig(
            path=str(tmp_path / "missing.csv"),
            target="target",
            time_column="event_time",
            test_size=0.2,
        ),
        model=ModelConfig(name="logreg", params={"max_iter": 500}),
        output=OutputConfig(
            model_path=str(tmp_path / "model.joblib"), metrics_path=str(tmp_path / "metrics.json")
        ),
        tracking=TrackingConfig(mlflow=False),
    )
