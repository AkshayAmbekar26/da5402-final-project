from __future__ import annotations

from time import perf_counter

from ml.common import REPORTS, read_json, utc_now, write_json
from ml.monitoring.performance import PERFORMANCE_PATH, record_stage_performance

REPORT_FILES = [
    "ingestion_report.json",
    "data_validation.json",
    "eda_report.json",
    "preprocessing_report.json",
    "feature_baseline_report.json",
    "training_metrics.json",
    "model_comparison.json",
    "evaluation.json",
    "drift_report.json",
    "pipeline_performance.json",
]


def publish_pipeline_report() -> dict[str, object]:
    stage_start = perf_counter()
    reports = {}
    for filename in REPORT_FILES:
        path = REPORTS / filename
        if path.exists():
            reports[filename] = read_json(path)
    ingestion = reports.get("ingestion_report.json", {})
    eda = reports.get("eda_report.json", {})
    preprocessing = reports.get("preprocessing_report.json", {})
    comparison = reports.get("model_comparison.json", {})
    evaluation = reports.get("evaluation.json", {})
    drift = reports.get("drift_report.json", {})
    performance = reports.get("pipeline_performance.json", {})
    payload = {
        "stage": "publish_pipeline_report",
        "status": "success",
        "generated_at": utc_now(),
        "summary": {
            "dataset_name": ingestion.get("dataset_name"),
            "raw_rows": ingestion.get("rows"),
            "processed_rows": preprocessing.get("final_rows"),
            "rejected_rows": preprocessing.get("rejected_rows"),
            "eda_rows": eda.get("rows"),
            "selected_model": comparison.get("selected_candidate"),
            "selected_mlflow_run_id": comparison.get("selected_mlflow_run_id"),
            "candidate_count": len(comparison.get("candidates", [])),
            "test_macro_f1": evaluation.get("metrics", {}).get("macro_f1"),
            "accepted": evaluation.get("accepted"),
            "drift_detected": drift.get("drift_detected"),
            "drift_score": drift.get("drift_score"),
            "total_duration_seconds": performance.get("total_duration_seconds"),
        },
        "stage_statuses": {
            filename.replace(".json", ""): report.get("status", "unknown")
            for filename, report in reports.items()
            if isinstance(report, dict)
        },
        "reports": reports,
    }
    write_json(REPORTS / "pipeline_report.json", payload)
    record_stage_performance(
        "publish_pipeline_report",
        perf_counter() - stage_start,
        extra={"report_count": len(reports)},
    )
    if PERFORMANCE_PATH.exists():
        payload["reports"]["pipeline_performance.json"] = read_json(PERFORMANCE_PATH)
        payload["summary"]["total_duration_seconds"] = payload["reports"]["pipeline_performance.json"].get(
            "total_duration_seconds"
        )
        write_json(REPORTS / "pipeline_report.json", payload)
    return payload


if __name__ == "__main__":
    publish_pipeline_report()
