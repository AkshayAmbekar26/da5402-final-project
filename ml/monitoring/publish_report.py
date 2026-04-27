from __future__ import annotations

from time import perf_counter

from ml.common import path_for, read_json, utc_now, write_json
from ml.monitoring.performance import PERFORMANCE_PATH, record_stage_performance

REPORT_ARTIFACTS = [
    "ingestion_report",
    "data_validation_report",
    "eda_report",
    "preprocessing_report",
    "feedback_preparation_report",
    "feedback_merge_report",
    "feature_baseline_report",
    "training_metrics",
    "model_comparison",
    "model_optimization_report",
    "evaluation_report",
    "acceptance_gate",
    "drift_report",
    "pipeline_performance",
]


def publish_pipeline_report() -> dict[str, object]:
    stage_start = perf_counter()
    reports = {}
    for artifact_name in REPORT_ARTIFACTS:
        path = path_for(artifact_name)
        if path.exists():
            reports[path.name] = read_json(path)
    ingestion = reports.get(path_for("ingestion_report").name, {})
    eda = reports.get(path_for("eda_report").name, {})
    preprocessing = reports.get(path_for("preprocessing_report").name, {})
    feedback_merge = reports.get(path_for("feedback_merge_report").name, {})
    comparison = reports.get(path_for("model_comparison").name, {})
    evaluation = reports.get(path_for("evaluation_report").name, {})
    drift = reports.get(path_for("drift_report").name, {})
    performance = reports.get(path_for("pipeline_performance").name, {})
    payload = {
        "stage": "publish_pipeline_report",
        "status": "success",
        "generated_at": utc_now(),
        "summary": {
            "dataset_name": ingestion.get("dataset_name"),
            "raw_rows": ingestion.get("rows"),
            "processed_rows": preprocessing.get("final_rows"),
            "feedback_rows_used": feedback_merge.get("feedback_rows_used"),
            "augmented_train_rows": feedback_merge.get("augmented_train_rows"),
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
    write_json(path_for("pipeline_report"), payload)
    record_stage_performance(
        "publish_pipeline_report",
        perf_counter() - stage_start,
        extra={"report_count": len(reports)},
    )
    if PERFORMANCE_PATH.exists():
        payload["reports"][PERFORMANCE_PATH.name] = read_json(PERFORMANCE_PATH)
        payload["summary"]["total_duration_seconds"] = payload["reports"][PERFORMANCE_PATH.name].get(
            "total_duration_seconds"
        )
        write_json(path_for("pipeline_report"), payload)
    return payload


if __name__ == "__main__":
    publish_pipeline_report()
