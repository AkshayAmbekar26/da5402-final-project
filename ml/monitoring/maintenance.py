from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ml.common import FEEDBACK, REPORTS, write_json


def feedback_summary(feedback_path: Path) -> dict[str, float | int | None]:
    total = 0
    matched = 0
    if not feedback_path.exists():
        return {"count": 0, "matches": 0, "accuracy": None}

    for line in feedback_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        total += 1
        matched += int(row.get("predicted_sentiment") == row.get("actual_sentiment"))

    return {
        "count": total,
        "matches": matched,
        "accuracy": matched / total if total else None,
    }


def read_drift_report(drift_report_path: Path) -> dict[str, Any]:
    if not drift_report_path.exists():
        return {"drift_score": 0.0, "drift_detected": False, "status": "not_available"}
    return json.loads(drift_report_path.read_text(encoding="utf-8"))


def previous_trigger_time(report_path: Path) -> datetime | None:
    if not report_path.exists():
        return None
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    value = payload.get("last_triggered_at")
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def evaluate_retraining_policy(
    *,
    drift_report_path: Path = REPORTS / "drift_report.json",
    feedback_path: Path = FEEDBACK / "feedback.jsonl",
    output_path: Path = REPORTS / "maintenance_report.json",
    drift_threshold: float | None = None,
    min_feedback_count: int | None = None,
    min_feedback_accuracy: float | None = None,
    cooldown_hours: float | None = None,
) -> dict[str, Any]:
    """Decide whether the maintenance DAG should trigger a retraining run."""
    drift_threshold = float(drift_threshold if drift_threshold is not None else os.getenv("SENTIMENT_RETRAIN_DRIFT_THRESHOLD", "0.25"))
    min_feedback_count = int(
        min_feedback_count if min_feedback_count is not None else os.getenv("SENTIMENT_RETRAIN_MIN_FEEDBACK_COUNT", "10")
    )
    min_feedback_accuracy = float(
        min_feedback_accuracy
        if min_feedback_accuracy is not None
        else os.getenv("SENTIMENT_RETRAIN_MIN_FEEDBACK_ACCURACY", "0.8")
    )
    cooldown_hours = float(
        cooldown_hours
        if cooldown_hours is not None
        else os.getenv("SENTIMENT_RETRAIN_COOLDOWN_HOURS", "6")
    )

    drift = read_drift_report(drift_report_path)
    feedback = feedback_summary(feedback_path)
    checked_at = datetime.now(timezone.utc)
    drift_score = float(drift.get("drift_score", 0.0) or 0.0)
    drift_detected = bool(drift.get("drift_detected", False)) or drift_score > drift_threshold
    feedback_accuracy = feedback["accuracy"]
    enough_feedback = int(feedback["count"]) >= min_feedback_count
    feedback_degraded = (
        enough_feedback
        and feedback_accuracy is not None
        and float(feedback_accuracy) < min_feedback_accuracy
    )

    reasons: list[str] = []
    if drift_detected:
        reasons.append("data_drift")
    if feedback_degraded:
        reasons.append("feedback_accuracy_drop")

    last_triggered_at = previous_trigger_time(output_path)
    cooldown_until = (
        last_triggered_at + timedelta(hours=cooldown_hours)
        if last_triggered_at is not None and cooldown_hours > 0
        else None
    )
    cooldown_active = bool(reasons and cooldown_until and checked_at < cooldown_until)
    should_retrain = bool(reasons) and not cooldown_active
    if should_retrain:
        last_triggered_at = checked_at

    report = {
        "stage": "maintenance_policy",
        "status": "success",
        "checked_at": checked_at.isoformat(),
        "should_retrain": should_retrain,
        "reasons": reasons,
        "drift": {
            "score": drift_score,
            "detected": drift_detected,
            "threshold": drift_threshold,
            "report_path": str(drift_report_path),
        },
        "feedback": {
            "count": feedback["count"],
            "matches": feedback["matches"],
            "accuracy": feedback_accuracy,
            "min_count": min_feedback_count,
            "min_accuracy": min_feedback_accuracy,
            "degraded": feedback_degraded,
            "path": str(feedback_path),
        },
        "cooldown": {
            "hours": cooldown_hours,
            "active": cooldown_active,
            "last_triggered_at": last_triggered_at.isoformat() if last_triggered_at else None,
            "cooldown_until": cooldown_until.isoformat() if cooldown_until else None,
        },
        "action": (
            "trigger_retraining"
            if should_retrain
            else "cooldown_active"
            if cooldown_active
            else "continue_monitoring"
        ),
        "last_triggered_at": last_triggered_at.isoformat() if last_triggered_at else None,
    }
    write_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retraining policy from drift and feedback evidence.")
    parser.add_argument("--drift-report", type=Path, default=REPORTS / "drift_report.json")
    parser.add_argument("--feedback", type=Path, default=FEEDBACK / "feedback.jsonl")
    parser.add_argument("--output", type=Path, default=REPORTS / "maintenance_report.json")
    parser.add_argument("--drift-threshold", type=float, default=None)
    parser.add_argument("--min-feedback-count", type=int, default=None)
    parser.add_argument("--min-feedback-accuracy", type=float, default=None)
    parser.add_argument("--cooldown-hours", type=float, default=None)
    args = parser.parse_args()
    report = evaluate_retraining_policy(
        drift_report_path=args.drift_report,
        feedback_path=args.feedback,
        output_path=args.output,
        drift_threshold=args.drift_threshold,
        min_feedback_count=args.min_feedback_count,
        min_feedback_accuracy=args.min_feedback_accuracy,
        cooldown_hours=args.cooldown_hours,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
