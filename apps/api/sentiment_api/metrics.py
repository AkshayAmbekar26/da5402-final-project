from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

REQUEST_COUNT = Counter(
    "sentiment_api_requests_total",
    "Total HTTP requests handled by the API.",
    ["endpoint", "method", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "sentiment_api_request_latency_seconds",
    "HTTP request latency in seconds.",
    ["endpoint", "method"],
)
PREDICTION_COUNT = Counter(
    "sentiment_predictions_total",
    "Total predictions by sentiment class.",
    ["sentiment"],
)
INFERENCE_LATENCY = Histogram(
    "sentiment_model_inference_latency_seconds",
    "Model inference latency in seconds.",
)
ERROR_COUNT = Counter(
    "sentiment_api_errors_total",
    "Total API errors by endpoint.",
    ["endpoint"],
)
MODEL_LOADED = Gauge("sentiment_model_loaded", "Whether the production model is loaded.")
DRIFT_SCORE = Gauge("sentiment_data_drift_score", "Latest batch drift score.")
FEEDBACK_COUNT = Counter(
    "sentiment_feedback_total",
    "Ground-truth feedback submissions.",
    ["actual_sentiment"],
)
GROUND_TRUTH_MATCH_COUNT = Counter(
    "sentiment_feedback_matches_total",
    "Feedback submissions where predicted and actual labels match.",
)

