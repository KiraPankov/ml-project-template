import json
from pathlib import Path

from mlproject.predict import Predictor
from mlproject.train import train


def test_train_beats_baseline(cfg):
    results = train(cfg)
    assert Path(cfg.output.model_path).exists()
    assert json.loads(Path(cfg.output.metrics_path).read_text(encoding="utf-8"))
    assert results["test"]["pr_auc"] > results["baseline"]["pr_auc"]


def test_predictor_returns_probabilities(cfg):
    train(cfg)
    predictor = Predictor(cfg.output.model_path)
    record = {f: 0.0 for f in predictor.features}
    record["segment"] = "a"
    scores = predictor.predict_proba([record])
    assert len(scores) == 1 and 0.0 <= scores[0] <= 1.0
