from __future__ import annotations

from ml.common import REPORTS, read_json, utc_now, write_json

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
]


def publish_pipeline_report() -> dict[str, object]:
    reports = {}
    for filename in REPORT_FILES:
        path = REPORTS / filename
        if path.exists():
            reports[filename] = read_json(path)
    payload = {
        "stage": "publish_pipeline_report",
        "status": "success",
        "generated_at": utc_now(),
        "reports": reports,
    }
    write_json(REPORTS / "pipeline_report.json", payload)
    return payload


if __name__ == "__main__":
    publish_pipeline_report()
