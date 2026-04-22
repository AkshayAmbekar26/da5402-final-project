from __future__ import annotations

import json

from fastapi.testclient import TestClient

from apps.api.sentiment_api.config import settings
from apps.api.sentiment_api.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_endpoint() -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    payload = response.json()
    assert "model_loaded" in payload
    assert "fallback_mode" in payload


def test_predict_endpoint_returns_contract() -> None:
    response = client.post("/predict", json={"review_text": "Excellent product and fast delivery"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["sentiment"] in {"negative", "neutral", "positive"}
    assert 0 <= payload["confidence"] <= 1
    assert "class_probabilities" in payload
    assert "latency_ms" in payload
    assert "mlflow_run_id" in payload


def test_feedback_endpoint(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(settings, "feedback_path", tmp_path / "feedback.jsonl")
    response = client.post(
        "/feedback",
        json={
            "review_text": "Excellent product and fast delivery",
            "predicted_sentiment": "positive",
            "actual_sentiment": "positive",
            "source": "unit-test",
        },
    )
    assert response.status_code == 200
    assert response.json()["stored"] is True
    stored = json.loads(settings.feedback_path.read_text(encoding="utf-8").splitlines()[-1])
    assert stored["feedback_type"] == "confirmation"
    assert stored["is_correction"] is False
    assert "submitted_at" in stored


def test_metrics_endpoint_exposes_prometheus_text() -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    expected_metrics = [
        "sentiment_api_requests_total",
        "sentiment_active_requests",
        "sentiment_invalid_reviews_total",
        "sentiment_stage_latency_seconds",
        "sentiment_process_resident_memory_bytes",
        "sentiment_process_cpu_percent",
        "sentiment_model_loaded",
        "sentiment_model_fallback_mode",
        "sentiment_model_accepted",
        "sentiment_model_test_macro_f1",
        "sentiment_data_drift_score",
        "sentiment_data_drift_detected",
        "sentiment_pipeline_duration_seconds",
        "sentiment_data_raw_rows",
        "sentiment_data_rejected_ratio",
        "sentiment_batch_pipeline_rows_processed",
        "sentiment_batch_pipeline_quarantined",
        "sentiment_feedback_accuracy_ratio",
    ]
    for metric_name in expected_metrics:
        assert metric_name in response.text


def test_invalid_predict_request_is_counted_in_metrics() -> None:
    response = client.post("/predict", json={"review_text": ""})
    assert response.status_code == 422

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert 'sentiment_invalid_reviews_total{endpoint="/predict",reason="empty_review_text"}' in metrics.text


def test_demo_error_endpoint_is_disabled_by_default() -> None:
    response = client.post("/ops/demo/error", json={})
    assert response.status_code == 404


def test_monitoring_refresh_endpoint_returns_pipeline_summary() -> None:
    response = client.post("/monitoring/refresh")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "refreshed"
    assert "pipeline_summary" in payload


def test_metrics_summary_includes_pipeline_sections() -> None:
    response = client.get("/metrics-summary")
    assert response.status_code == 200
    payload = response.json()
    assert "pipeline_summary" in payload
    assert "pipeline_performance" in payload
    assert "model_comparison" in payload
    assert "model_optimization" in payload
    assert "drift" in payload
    assert "batch_pipeline" in payload
