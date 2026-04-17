from __future__ import annotations

from datetime import datetime, timedelta

from airflow.operators.bash import BashOperator

from airflow import DAG

PROJECT_DIR = "/opt/airflow/project"
PYTHON = "python"


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
    schedule_interval=None,
    catchup=False,
    tags=["sentiment", "mlops", "dvc", "mlflow"],
) as dag:
    ingest_data = BashOperator(
        task_id="ingest_data",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} -m ml.data_ingestion.ingest --config configs/data_config.json",
    )

    validate_raw_data = BashOperator(
        task_id="validate_raw_data",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} -m ml.validation.validate_data --config configs/data_config.json",
    )

    run_eda = BashOperator(
        task_id="run_eda",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} -m ml.eda.analyze --config configs/data_config.json",
    )

    preprocess_data = BashOperator(
        task_id="preprocess_data",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} -m ml.preprocessing.preprocess --config configs/data_config.json",
    )

    generate_features = BashOperator(
        task_id="generate_features",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} -m ml.features.compute_baseline",
    )

    compute_drift_baseline = BashOperator(
        task_id="compute_drift_baseline",
        bash_command=f"cd {PROJECT_DIR} && test -f data/baselines/feature_baseline.json",
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} -m ml.training.train",
    )

    evaluate_model = BashOperator(
        task_id="evaluate_model",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} -m ml.evaluation.evaluate",
    )

    register_model_if_accepted = BashOperator(
        task_id="register_model_if_accepted",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} -c \"import json; r=json.load(open('reports/evaluation.json')); assert r['accepted'], r\"",
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
        >> generate_features
        >> compute_drift_baseline
        >> train_model
        >> evaluate_model
        >> register_model_if_accepted
        >> run_batch_drift_check
        >> publish_pipeline_report
    )
