from __future__ import annotations

from fastapi.testclient import TestClient

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


def test_feedback_endpoint() -> None:
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


def test_metrics_endpoint_exposes_prometheus_text() -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "sentiment_api_requests_total" in response.text


def test_metrics_summary_includes_pipeline_sections() -> None:
    response = client.get("/metrics-summary")
    assert response.status_code == 200
    payload = response.json()
    assert "pipeline_summary" in payload
    assert "pipeline_performance" in payload
    assert "model_comparison" in payload
