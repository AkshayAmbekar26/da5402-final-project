from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from apps.api.sentiment_api.config import settings
from apps.api.sentiment_api.metrics import (
    BATCH_PIPELINE_CHUNK_COUNT,
    BATCH_PIPELINE_FAILED_CHUNKS,
    BATCH_PIPELINE_QUARANTINED,
    BATCH_PIPELINE_ROWS_PROCESSED,
    DATA_PROCESSED_ROWS,
    DATA_RAW_ROWS,
    DATA_REJECTED_RATIO,
    DATA_REJECTED_ROWS,
    DRIFT_DETECTED,
    DRIFT_SCORE,
    FEEDBACK_ACCURACY_RATIO,
    MODEL_ACCEPTED,
    MODEL_ACCEPTED_CANDIDATE_COUNT,
    MODEL_CANDIDATE_COUNT,
    MODEL_FALLBACK_MODE,
    MODEL_INFO,
    MODEL_LOADED,
    MODEL_TEST_MACRO_F1,
    PIPELINE_DURATION_SECONDS,
    PIPELINE_STAGE_DURATION_SECONDS,
    PIPELINE_STAGE_THROUGHPUT,
)

REPORT_MAP = {
    "ingestion": Path("reports/ingestion_report.json"),
    "validation": Path("reports/data_validation.json"),
    "eda": Path("reports/eda_report.json"),
    "preprocessing": Path("reports/preprocessing_report.json"),
    "feedback_preparation": Path("reports/feedback_preparation_report.json"),
    "feedback_merge": Path("reports/feedback_merge_report.json"),
    "model_comparison": Path("reports/model_comparison.json"),
    "model_optimization": Path("reports/model_optimization_report.json"),
    "evaluation": Path("reports/evaluation.json"),
    "acceptance_gate": Path("reports/acceptance_gate.json"),
    "drift": Path("reports/drift_report.json"),
    "maintenance": Path("reports/maintenance_report.json"),
    "pipeline": Path("reports/pipeline_report.json"),
    "pipeline_performance": Path("reports/pipeline_performance.json"),
    "batch_pipeline": Path("reports/batch_pipeline_report.json"),
}


def read_report(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"status": "not_available"}


def load_reports() -> dict[str, dict[str, Any]]:
    return {name: read_report(path) for name, path in REPORT_MAP.items()}


def feedback_accuracy_ratio() -> float:
    """Compute a best-effort feedback accuracy from locally stored ground-truth labels."""
    path = settings.feedback_path
    if not path.exists():
        return 0.0
    total = 0
    matched = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        total += 1
        matched += int(row.get("predicted_sentiment") == row.get("actual_sentiment"))
    return matched / total if total else 0.0


def refresh_report_metrics(model_info: dict[str, Any]) -> dict[str, Any]:
    """Project JSON lifecycle reports into Prometheus gauges and frontend summary data."""
    reports = load_reports()
    ingestion = reports["ingestion"]
    preprocessing = reports["preprocessing"]
    comparison = reports["model_comparison"]
    evaluation = reports["evaluation"]
    drift = reports["drift"]
    pipeline = reports["pipeline"]
    performance = reports["pipeline_performance"]
    batch_pipeline = reports["batch_pipeline"]

    MODEL_LOADED.set(1 if model_info.get("model_loaded") else 0)
    MODEL_FALLBACK_MODE.set(1 if model_info.get("fallback_mode") else 0)
    metadata = model_info.get("metadata", {})
    MODEL_INFO.info(
        {
            "model_name": str(metadata.get("model_name", "unknown")),
            "model_version": str(metadata.get("model_version", "unknown")),
            "mlflow_run_id": str(metadata.get("mlflow_run_id", "unknown")),
            "git_commit": str(metadata.get("git_commit", "unknown")),
        }
    )

    accepted = bool(evaluation.get("accepted", False))
    MODEL_ACCEPTED.set(1 if accepted else 0)
    MODEL_TEST_MACRO_F1.set(float(evaluation.get("metrics", {}).get("macro_f1", 0.0) or 0.0))
    MODEL_CANDIDATE_COUNT.set(len(comparison.get("candidates", []) or []))
    MODEL_ACCEPTED_CANDIDATE_COUNT.set(len(comparison.get("accepted_candidates", []) or []))

    drift_score = float(drift.get("drift_score", 0.0) or 0.0)
    DRIFT_SCORE.set(drift_score)
    DRIFT_DETECTED.set(1 if drift.get("drift_detected", False) else 0)

    raw_rows = int(ingestion.get("rows", 0) or 0)
    processed_rows = int(preprocessing.get("final_rows", 0) or 0)
    rejected_rows = int(preprocessing.get("rejected_rows", 0) or 0)
    DATA_RAW_ROWS.set(raw_rows)
    DATA_PROCESSED_ROWS.set(processed_rows)
    DATA_REJECTED_ROWS.set(rejected_rows)
    DATA_REJECTED_RATIO.set(rejected_rows / raw_rows if raw_rows else 0.0)

    BATCH_PIPELINE_ROWS_PROCESSED.set(float(batch_pipeline.get("rows_processed", 0) or 0))
    chunk_count = batch_pipeline.get("chunk_count")
    if chunk_count is None:
        # Older reports only stored completed/failed counts; keep the dashboard backward-compatible.
        chunk_count = int(batch_pipeline.get("completed_chunks", 0) or 0) + int(batch_pipeline.get("failed_chunks", 0) or 0)
    BATCH_PIPELINE_CHUNK_COUNT.set(float(chunk_count or 0))
    BATCH_PIPELINE_FAILED_CHUNKS.set(float(batch_pipeline.get("failed_chunks", 0) or 0))
    BATCH_PIPELINE_QUARANTINED.set(1 if batch_pipeline.get("status") == "quarantined" else 0)

    duration = float(performance.get("total_duration_seconds", 0.0) or 0.0)
    PIPELINE_DURATION_SECONDS.set(duration)
    for stage_name, stage in (performance.get("stages", {}) or {}).items():
        PIPELINE_STAGE_DURATION_SECONDS.labels(stage=stage_name).set(float(stage.get("duration_seconds", 0.0) or 0.0))
        if "throughput_rows_per_second" in stage:
            PIPELINE_STAGE_THROUGHPUT.labels(stage=stage_name).set(
                float(stage.get("throughput_rows_per_second", 0.0) or 0.0)
            )

    FEEDBACK_ACCURACY_RATIO.set(feedback_accuracy_ratio())
    return {
        **reports,
        "pipeline_summary": pipeline.get("summary", {}),
    }
