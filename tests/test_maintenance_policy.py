from __future__ import annotations

import json
from pathlib import Path

from ml.monitoring.maintenance import evaluate_retraining_policy, feedback_summary


def test_feedback_summary_computes_accuracy(tmp_path: Path) -> None:
    feedback_path = tmp_path / "feedback.jsonl"
    feedback_path.write_text(
        "\n".join(
            [
                json.dumps({"predicted_sentiment": "positive", "actual_sentiment": "positive"}),
                json.dumps({"predicted_sentiment": "negative", "actual_sentiment": "positive"}),
            ]
        ),
        encoding="utf-8",
    )

    summary = feedback_summary(feedback_path)

    assert summary["count"] == 2
    assert summary["matches"] == 1
    assert summary["accuracy"] == 0.5


def test_retraining_policy_triggers_on_drift(tmp_path: Path) -> None:
    drift_path = tmp_path / "drift_report.json"
    feedback_path = tmp_path / "feedback.jsonl"
    output_path = tmp_path / "maintenance_report.json"
    drift_path.write_text(json.dumps({"drift_score": 0.42, "drift_detected": True}), encoding="utf-8")

    report = evaluate_retraining_policy(
        drift_report_path=drift_path,
        feedback_path=feedback_path,
        output_path=output_path,
        drift_threshold=0.25,
        min_feedback_count=10,
        min_feedback_accuracy=0.8,
    )

    assert report["should_retrain"] is True
    assert report["reasons"] == ["data_drift"]
    assert output_path.exists()


def test_retraining_policy_triggers_on_feedback_degradation(tmp_path: Path) -> None:
    drift_path = tmp_path / "drift_report.json"
    feedback_path = tmp_path / "feedback.jsonl"
    output_path = tmp_path / "maintenance_report.json"
    drift_path.write_text(json.dumps({"drift_score": 0.01, "drift_detected": False}), encoding="utf-8")
    feedback_path.write_text(
        "\n".join(
            [
                json.dumps({"predicted_sentiment": "positive", "actual_sentiment": "negative"}),
                json.dumps({"predicted_sentiment": "neutral", "actual_sentiment": "neutral"}),
            ]
        ),
        encoding="utf-8",
    )

    report = evaluate_retraining_policy(
        drift_report_path=drift_path,
        feedback_path=feedback_path,
        output_path=output_path,
        drift_threshold=0.25,
        min_feedback_count=2,
        min_feedback_accuracy=0.8,
    )

    assert report["should_retrain"] is True
    assert report["reasons"] == ["feedback_accuracy_drop"]


def test_retraining_policy_continues_monitoring_when_thresholds_pass(tmp_path: Path) -> None:
    drift_path = tmp_path / "drift_report.json"
    feedback_path = tmp_path / "feedback.jsonl"
    output_path = tmp_path / "maintenance_report.json"
    drift_path.write_text(json.dumps({"drift_score": 0.02, "drift_detected": False}), encoding="utf-8")
    feedback_path.write_text(
        "\n".join(
            [
                json.dumps({"predicted_sentiment": "positive", "actual_sentiment": "positive"}),
                json.dumps({"predicted_sentiment": "neutral", "actual_sentiment": "neutral"}),
            ]
        ),
        encoding="utf-8",
    )

    report = evaluate_retraining_policy(
        drift_report_path=drift_path,
        feedback_path=feedback_path,
        output_path=output_path,
        drift_threshold=0.25,
        min_feedback_count=2,
        min_feedback_accuracy=0.8,
    )

    assert report["should_retrain"] is False
    assert report["action"] == "continue_monitoring"


def test_retraining_policy_respects_cooldown_after_recent_trigger(tmp_path: Path) -> None:
    drift_path = tmp_path / "drift_report.json"
    output_path = tmp_path / "maintenance_report.json"
    drift_path.write_text(json.dumps({"drift_score": 0.42, "drift_detected": True}), encoding="utf-8")

    first = evaluate_retraining_policy(
        drift_report_path=drift_path,
        output_path=output_path,
        drift_threshold=0.25,
        cooldown_hours=6,
    )
    second = evaluate_retraining_policy(
        drift_report_path=drift_path,
        output_path=output_path,
        drift_threshold=0.25,
        cooldown_hours=6,
    )

    assert first["should_retrain"] is True
    assert second["should_retrain"] is False
    assert second["action"] == "cooldown_active"
