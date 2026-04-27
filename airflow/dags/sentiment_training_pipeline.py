from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow.operators.bash import BashOperator

from airflow import DAG

PROJECT_DIR = Path(os.getenv("PROJECT_DIR", Path.cwd()))
PYTHON = os.getenv("PYTHON", "python")


def _schedule_interval(name: str, default: str) -> str | None:
    value = os.getenv(name, default).strip()
    return None if value.lower() in {"", "none", "manual", "off"} else value


default_args = {
    "owner": "mlops-course-project",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


with DAG(
    dag_id="sentiment_training_pipeline",
    description="End-to-end product review sentiment MLOps pipeline.",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=_schedule_interval("SENTIMENT_TRAINING_SCHEDULE", "0 2 * * 0"),
    catchup=False,
    max_active_runs=1,
    tags=["sentiment", "mlops", "dvc", "mlflow"],
) as dag:
    ingest_data = BashOperator(
        task_id="ingest_data",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} -m ml.data_ingestion.ingest",
    )

    validate_raw_data = BashOperator(
        task_id="validate_raw_data",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} -m ml.validation.validate_data",
    )

    run_eda = BashOperator(
        task_id="run_eda",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} -m ml.eda.analyze",
    )

    preprocess_data = BashOperator(
        task_id="preprocess_data",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} -m ml.preprocessing.preprocess",
    )

    prepare_feedback_corrections = BashOperator(
        task_id="prepare_feedback_corrections",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} -m ml.monitoring.prepare_feedback",
    )

    merge_feedback_corrections = BashOperator(
        task_id="merge_feedback_corrections",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} -m ml.preprocessing.merge_feedback",
    )

    generate_features = BashOperator(
        task_id="generate_features",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} -m ml.features.compute_baseline",
    )

    compute_drift_baseline = BashOperator(
        task_id="compute_drift_baseline",
        bash_command=(
            f"cd {PROJECT_DIR} && {PYTHON} -c "
            "'from ml.common import path_for; raise SystemExit(0 if path_for(\"feature_baseline\").exists() else 1)'"
        ),
    )

    train_and_compare_models = BashOperator(
        task_id="train_and_compare_models",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} -m ml.training.train",
    )

    evaluate_model = BashOperator(
        task_id="evaluate_model",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} -m ml.evaluation.evaluate",
    )

    register_model_if_accepted = BashOperator(
        task_id="register_model_if_accepted",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} -m ml.evaluation.check_acceptance",
    )

    run_batch_drift_check = BashOperator(
        task_id="run_batch_drift_check",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} -m ml.monitoring.drift",
    )

    publish_pipeline_report = BashOperator(
        task_id="publish_pipeline_report",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} -m ml.monitoring.publish_report",
    )

    (
        ingest_data
        >> validate_raw_data
        >> run_eda
        >> preprocess_data
        >> prepare_feedback_corrections
        >> merge_feedback_corrections
        >> generate_features
        >> compute_drift_baseline
        >> train_and_compare_models
        >> evaluate_model
        >> register_model_if_accepted
        >> run_batch_drift_check
        >> publish_pipeline_report
    )
