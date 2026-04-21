from __future__ import annotations

import json
from types import SimpleNamespace

from apps.api.sentiment_api.config import settings
from apps.api.sentiment_api.model_service import ModelService
from ml.training.train import build_mlflow_input_example, build_mlflow_output_example


def test_mlflow_signature_examples_match_serving_contract() -> None:
    input_example = build_mlflow_input_example()
    output_example = build_mlflow_output_example(
        {
            "model_version": "local-production",
            "mlflow_run_id": "abc123",
        }
    )

    assert list(input_example.columns) == ["review_text"]
    assert "sentiment" in output_example.columns
    assert "confidence" in output_example.columns
    assert "class_probabilities_json" in output_example.columns
    assert json.loads(output_example.loc[0, "class_probabilities_json"])["positive"] == 0.95


def test_model_service_can_adapt_mlflow_serving_response(monkeypatch) -> None:
    original_mode = settings.model_serving_mode
    monkeypatch.setattr(settings, "model_serving_mode", "mlflow")

    def fake_post(*args, **kwargs):
        return SimpleNamespace(
            json=lambda: {
                "predictions": [
                    {
                        "sentiment": "positive",
                        "confidence": 0.91,
                        "class_probabilities_json": json.dumps(
                            {"negative": 0.02, "neutral": 0.07, "positive": 0.91}
                        ),
                        "explanation_json": json.dumps([{"token": "excellent", "weight": 0.4}]),
                        "model_version": "local-production",
                        "mlflow_run_id": "run-123",
                        "latency_ms": 2.5,
                    }
                ]
            },
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr("apps.api.sentiment_api.model_service.httpx.post", fake_post)
    service = ModelService()

    try:
        result = service.predict("Excellent product quality.")
    finally:
        monkeypatch.setattr(settings, "model_serving_mode", original_mode)

    assert result["sentiment"] == "positive"
    assert result["confidence"] == 0.91
    assert result["class_probabilities"]["positive"] == 0.91
    assert result["explanation"][0]["token"] == "excellent"
    assert result["mlflow_run_id"] == "run-123"
