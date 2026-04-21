from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _dag_schedule(dag: object) -> object:
    return getattr(dag, "schedule", getattr(dag, "schedule_interval", None))


@pytest.mark.skipif(importlib.util.find_spec("airflow.models") is None, reason="Airflow is not installed locally")
def test_airflow_dags_import_without_errors(monkeypatch, tmp_path: Path) -> None:
    from airflow.models.dagbag import DagBag

    monkeypatch.setenv("AIRFLOW_HOME", str(tmp_path / "airflow-home"))
    monkeypatch.setenv("AIRFLOW__CORE__LOAD_EXAMPLES", "false")
    monkeypatch.setenv("PROJECT_DIR", str(Path(__file__).resolve().parents[1]))

    dag_bag = DagBag(
        dag_folder=str(Path(__file__).resolve().parents[1] / "airflow" / "dags"),
        include_examples=False,
    )

    assert dag_bag.import_errors == {}
    assert "sentiment_training_pipeline" in dag_bag.dags
    assert "sentiment_batch_ingestion_pipeline" in dag_bag.dags
    assert "sentiment_monitoring_maintenance" in dag_bag.dags
    assert _dag_schedule(dag_bag.dags["sentiment_training_pipeline"]) is not None
    assert _dag_schedule(dag_bag.dags["sentiment_monitoring_maintenance"]) is not None
