from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ml.common import FEEDBACK, REPORTS, write_json


def parse_feedback_time(row: dict[str, Any]) -> datetime | None:
    value = row.get("submitted_at") or row.get("timestamp") or row.get("created_at")
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def feedback_summary(
    feedback_path: Path,
    *,
    correction_window_hours: float | None = None,
    now: datetime | None = None,
) -> dict[str, float | int | None]:
    total = 0
    matched = 0
    corrections = 0
    recent_corrections = 0
    checked_at = now or datetime.now(timezone.utc)
    cutoff = (
        checked_at - timedelta(hours=correction_window_hours)
        if correction_window_hours is not None and correction_window_hours > 0
        else None
    )
    if not feedback_path.exists():
        return {"count": 0, "matches": 0, "corrections": 0, "recent_corrections": 0, "accuracy": None}

    for line in feedback_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        total += 1
        is_match = row.get("predicted_sentiment") == row.get("actual_sentiment")
        matched += int(is_match)
        if not is_match:
            corrections += 1
            submitted_at = parse_feedback_time(row)
            if cutoff is None or submitted_at is None or submitted_at >= cutoff:
                recent_corrections += 1

    return {
        "count": total,
        "matches": matched,
        "corrections": corrections,
        "recent_corrections": recent_corrections,
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
    min_correction_count: int | None = None,
    correction_window_hours: float | None = None,
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
    min_correction_count = int(
        min_correction_count
        if min_correction_count is not None
        else os.getenv("SENTIMENT_RETRAIN_MIN_CORRECTIONS", "10")
    )
    correction_window_hours = float(
        correction_window_hours
        if correction_window_hours is not None
        else os.getenv("SENTIMENT_RETRAIN_CORRECTION_WINDOW_HOURS", "72")
    )
    cooldown_hours = float(
        cooldown_hours
        if cooldown_hours is not None
        else os.getenv("SENTIMENT_RETRAIN_COOLDOWN_HOURS", "6")
    )

    drift = read_drift_report(drift_report_path)
    checked_at = datetime.now(timezone.utc)
    feedback = feedback_summary(
        feedback_path,
        correction_window_hours=correction_window_hours,
        now=checked_at,
    )
    drift_score = float(drift.get("drift_score", 0.0) or 0.0)
    drift_detected = bool(drift.get("drift_detected", False)) or drift_score > drift_threshold
    feedback_accuracy = feedback["accuracy"]
    enough_feedback = int(feedback["count"]) >= min_feedback_count
    feedback_degraded = (
        enough_feedback
        and feedback_accuracy is not None
        and float(feedback_accuracy) < min_feedback_accuracy
    )
    enough_corrections = int(feedback["recent_corrections"]) >= min_correction_count

    reasons: list[str] = []
    if drift_detected:
        reasons.append("data_drift")
    if feedback_degraded:
        reasons.append("feedback_accuracy_drop")
    if enough_corrections:
        reasons.append("feedback_correction_threshold")

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
            "corrections": feedback["corrections"],
            "recent_corrections": feedback["recent_corrections"],
            "accuracy": feedback_accuracy,
            "min_count": min_feedback_count,
            "min_accuracy": min_feedback_accuracy,
            "min_corrections": min_correction_count,
            "correction_window_hours": correction_window_hours,
            "degraded": feedback_degraded,
            "correction_threshold_reached": enough_corrections,
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
    parser.add_argument("--min-corrections", type=int, default=None)
    parser.add_argument("--correction-window-hours", type=float, default=None)
    parser.add_argument("--cooldown-hours", type=float, default=None)
    args = parser.parse_args()
    report = evaluate_retraining_policy(
        drift_report_path=args.drift_report,
        feedback_path=args.feedback,
        output_path=args.output,
        drift_threshold=args.drift_threshold,
        min_feedback_count=args.min_feedback_count,
        min_feedback_accuracy=args.min_feedback_accuracy,
        min_correction_count=args.min_corrections,
        correction_window_hours=args.correction_window_hours,
        cooldown_hours=args.cooldown_hours,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
