from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.operators.bash import BashOperator
from airflow.operators.python import ShortCircuitOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from airflow import DAG

PROJECT_DIR = Path(os.getenv("PROJECT_DIR", Path.cwd()))
sys.path.insert(0, str(PROJECT_DIR))

from ml.common import path_for  # noqa: E402
from ml.monitoring.maintenance import evaluate_retraining_policy  # noqa: E402

PYTHON = os.getenv("PYTHON", "python")


def _schedule_interval(name: str, default: str) -> str | None:
    value = os.getenv(name, default).strip()
    return None if value.lower() in {"", "none", "manual", "off"} else value


def should_trigger_retraining() -> bool:
    report = evaluate_retraining_policy(
        drift_report_path=path_for("drift_report"),
        feedback_path=path_for("feedback_log"),
        output_path=path_for("maintenance_report"),
        drift_threshold=float(os.getenv("SENTIMENT_RETRAIN_DRIFT_THRESHOLD", "0.25")),
        min_feedback_count=int(os.getenv("SENTIMENT_RETRAIN_MIN_FEEDBACK_COUNT", "10")),
        min_feedback_accuracy=float(os.getenv("SENTIMENT_RETRAIN_MIN_FEEDBACK_ACCURACY", "0.8")),
        min_correction_count=int(os.getenv("SENTIMENT_RETRAIN_MIN_CORRECTIONS", "10")),
        correction_window_hours=float(os.getenv("SENTIMENT_RETRAIN_CORRECTION_WINDOW_HOURS", "72")),
        cooldown_hours=float(os.getenv("SENTIMENT_RETRAIN_COOLDOWN_HOURS", "6")),
    )
    return bool(report["should_retrain"])


default_args = {
    "owner": "mlops-course-project",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


with DAG(
    dag_id="sentiment_monitoring_maintenance",
    description="Scheduled maintenance DAG that checks drift/feedback and triggers retraining when needed.",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=_schedule_interval("SENTIMENT_MAINTENANCE_SCHEDULE", "0 * * * *"),
    catchup=False,
    max_active_runs=1,
    tags=["sentiment", "mlops", "monitoring", "retraining"],
) as dag:
    refresh_drift_report = BashOperator(
        task_id="refresh_drift_report",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} -m ml.monitoring.drift",
    )

    evaluate_policy = ShortCircuitOperator(
        task_id="evaluate_retraining_policy",
        python_callable=should_trigger_retraining,
    )

    trigger_training_pipeline = TriggerDagRunOperator(
        task_id="trigger_training_pipeline_if_needed",
        trigger_dag_id="sentiment_training_pipeline",
        conf={
            "triggered_by": "sentiment_monitoring_maintenance",
            "reason": "drift_feedback_or_correction_threshold",
        },
        wait_for_completion=False,
        reset_dag_run=False,
    )

    refresh_drift_report >> evaluate_policy >> trigger_training_pipeline
