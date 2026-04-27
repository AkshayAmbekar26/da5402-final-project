from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from prometheus_client import Counter, Gauge, Histogram, Info, Summary

try:
    import psutil
except ImportError:  # pragma: no cover - optional at runtime for local lightweight installs
    psutil = None

_PROCESS = psutil.Process(os.getpid()) if psutil else None

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
ACTIVE_REQUESTS = Gauge(
    "sentiment_active_requests",
    "In-flight HTTP requests currently being handled by the API.",
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
REVIEW_TEXT_LENGTH = Summary(
    "sentiment_review_text_length_chars",
    "Summary of review text length submitted for inference.",
    ["endpoint"],
)
STAGE_LATENCY = Histogram(
    "sentiment_stage_latency_seconds",
    "Latency of named API/model inference stages in seconds.",
    ["stage"],
)
ERROR_COUNT = Counter(
    "sentiment_api_errors_total",
    "Total API errors by endpoint.",
    ["endpoint"],
)
INVALID_REVIEW_COUNT = Counter(
    "sentiment_invalid_reviews_total",
    "Invalid review requests rejected by the API.",
    ["endpoint", "reason"],
)
ALERT_NOTIFICATION_COUNT = Counter(
    "sentiment_alert_notifications_total",
    "AlertManager webhook notifications received by the API demo receiver.",
    ["alertname", "status", "severity"],
)
PROCESS_RESIDENT_MEMORY_BYTES = Gauge(
    "sentiment_process_resident_memory_bytes",
    "Resident memory used by the API process in bytes.",
)
PROCESS_CPU_PERCENT = Gauge(
    "sentiment_process_cpu_percent",
    "CPU percentage used by the API process since the previous scrape/update.",
)
MODEL_LOADED = Gauge("sentiment_model_loaded", "Whether the production model is loaded.")
MODEL_FALLBACK_MODE = Gauge("sentiment_model_fallback_mode", "Whether the API is serving fallback predictions.")
MODEL_ACCEPTED = Gauge("sentiment_model_accepted", "Whether the selected model passed acceptance criteria.")
MODEL_TEST_MACRO_F1 = Gauge("sentiment_model_test_macro_f1", "Latest selected model test macro F1.")
MODEL_CANDIDATE_COUNT = Gauge("sentiment_model_candidate_count", "Number of model candidates compared.")
MODEL_ACCEPTED_CANDIDATE_COUNT = Gauge(
    "sentiment_model_accepted_candidate_count",
    "Number of candidate models that passed acceptance gates.",
)
MODEL_INFO = Info("sentiment_model_info", "Selected model metadata.")
DRIFT_SCORE = Gauge("sentiment_data_drift_score", "Latest batch drift score.")
DRIFT_DETECTED = Gauge("sentiment_data_drift_detected", "Whether batch drift is detected.")
PIPELINE_DURATION_SECONDS = Gauge("sentiment_pipeline_duration_seconds", "Latest full pipeline duration in seconds.")
PIPELINE_STAGE_DURATION_SECONDS = Gauge(
    "sentiment_pipeline_stage_duration_seconds",
    "Latest pipeline stage duration in seconds.",
    ["stage"],
)
PIPELINE_STAGE_THROUGHPUT = Gauge(
    "sentiment_pipeline_stage_throughput_rows_per_second",
    "Latest pipeline stage throughput in rows per second.",
    ["stage"],
)
DATA_RAW_ROWS = Gauge("sentiment_data_raw_rows", "Raw rows ingested by the latest pipeline run.")
DATA_PROCESSED_ROWS = Gauge("sentiment_data_processed_rows", "Rows available after preprocessing.")
DATA_REJECTED_ROWS = Gauge("sentiment_data_rejected_rows", "Rows rejected during preprocessing.")
DATA_REJECTED_RATIO = Gauge("sentiment_data_rejected_ratio", "Rejected rows divided by raw rows.")
BATCH_PIPELINE_ROWS_PROCESSED = Gauge(
    "sentiment_batch_pipeline_rows_processed",
    "Rows processed by the latest incoming review batch pipeline run.",
)
BATCH_PIPELINE_CHUNK_COUNT = Gauge(
    "sentiment_batch_pipeline_chunk_count",
    "Chunks created by the latest incoming review batch pipeline run.",
)
BATCH_PIPELINE_FAILED_CHUNKS = Gauge(
    "sentiment_batch_pipeline_failed_chunks",
    "Failed chunks from the latest incoming review batch pipeline run.",
)
BATCH_PIPELINE_QUARANTINED = Gauge(
    "sentiment_batch_pipeline_quarantined",
    "Whether the latest incoming review batch was quarantined.",
)
FEEDBACK_COUNT = Counter(
    "sentiment_feedback_total",
    "Ground-truth feedback submissions.",
    ["actual_sentiment"],
)
FEEDBACK_CORRECTION_COUNT = Counter(
    "sentiment_feedback_corrections_total",
    "Feedback submissions where the user corrected the predicted label.",
)
GROUND_TRUTH_MATCH_COUNT = Counter(
    "sentiment_feedback_matches_total",
    "Feedback submissions where predicted and actual labels match.",
)
FEEDBACK_ACCURACY_RATIO = Gauge(
    "sentiment_feedback_accuracy_ratio",
    "Feedback match ratio from locally stored ground-truth feedback.",
)
FEEDBACK_OBSERVED_COUNT = Gauge(
    "sentiment_feedback_observed_count",
    "Number of feedback records considered by the latest maintenance evaluation.",
)
FEEDBACK_CORRECTION_COUNT_GAUGE = Gauge(
    "sentiment_feedback_correction_count",
    "Number of correction feedback records considered by the latest maintenance evaluation.",
)
FEEDBACK_RECENT_CORRECTIONS = Gauge(
    "sentiment_feedback_recent_corrections",
    "Number of correction feedback records inside the maintenance time window.",
)
FEEDBACK_MIN_ACCURACY_TARGET = Gauge(
    "sentiment_feedback_min_accuracy_target",
    "Configured minimum acceptable feedback accuracy before retraining is recommended.",
)
FEEDBACK_MIN_CORRECTIONS_TARGET = Gauge(
    "sentiment_feedback_min_corrections_target",
    "Configured minimum number of recent corrections required to recommend retraining.",
)
FEEDBACK_MIN_COUNT_TARGET = Gauge(
    "sentiment_feedback_min_count_target",
    "Configured minimum number of feedback rows before using feedback accuracy as a maintenance signal.",
)
MAINTENANCE_RETRAINING_REQUIRED = Gauge(
    "sentiment_maintenance_retraining_required",
    "Whether the latest maintenance policy evaluation recommends retraining.",
)
MAINTENANCE_COOLDOWN_ACTIVE = Gauge(
    "sentiment_maintenance_cooldown_active",
    "Whether retraining is currently suppressed by the maintenance cooldown window.",
)
MAINTENANCE_REASON_ACTIVE = Gauge(
    "sentiment_maintenance_reason_active",
    "Whether a particular maintenance trigger reason is active in the latest policy evaluation.",
    ["reason"],
)


@contextmanager
def observe_stage(stage: str) -> Iterator[None]:
    timer = STAGE_LATENCY.labels(stage=stage).time()
    with timer:
        yield


def refresh_process_metrics() -> None:
    if _PROCESS is None:
        PROCESS_RESIDENT_MEMORY_BYTES.set(0)
        PROCESS_CPU_PERCENT.set(0)
        return
    PROCESS_RESIDENT_MEMORY_BYTES.set(_PROCESS.memory_info().rss)
    PROCESS_CPU_PERCENT.set(_PROCESS.cpu_percent(interval=None))
