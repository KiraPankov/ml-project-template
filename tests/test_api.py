import importlib

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from mlproject.train import train  # noqa: E402


@pytest.fixture
def client(cfg, monkeypatch):
    train(cfg)
    monkeypatch.setenv("MODEL_PATH", cfg.output.model_path)
    import mlproject.api as api

    importlib.reload(api)
    with TestClient(api.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict(client):
    record = {f"f{i}": 0.1 for i in range(8)} | {"segment": "b"}
    response = client.post("/predict", json={"records": [record, record]})
    assert response.status_code == 200
    assert len(response.json()["scores"]) == 2
    assert "X-Latency-Ms" in response.headers


def test_predict_missing_features(client):
    response = client.post("/predict", json={"records": [{"f0": 1.0}]})
    assert response.status_code == 422
