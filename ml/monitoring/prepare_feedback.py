from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ml.common import (
    DATA_INTERIM,
    FEEDBACK,
    REPORTS,
    SENTIMENT_LABELS,
    ensure_dirs,
    read_params,
    write_json,
)
from ml.monitoring.performance import timed_stage

OUTPUT_COLUMNS = [
    "feedback_id",
    "review_text",
    "sentiment",
    "predicted_sentiment",
    "actual_sentiment",
    "source",
    "submitted_at",
    "feedback_type",
]
UNKNOWN_SUBMITTED_AT = "1970-01-01T00:00:00+00:00"


def normalize_text(value: object) -> str:
    return " ".join(str(value).strip().split())


def parse_feedback_time(row: dict[str, Any]) -> str:
    value = row.get("submitted_at") or row.get("timestamp") or row.get("created_at")
    if not value:
        return UNKNOWN_SUBMITTED_AT
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return UNKNOWN_SUBMITTED_AT
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).isoformat()


def iter_feedback_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            rows.append({"_invalid_reason": "invalid_json", "_line_number": line_number})
            continue
        if isinstance(payload, dict):
            payload["_line_number"] = line_number
            rows.append(payload)
        else:
            rows.append({"_invalid_reason": "not_object", "_line_number": line_number})
    return rows


def validate_feedback_row(row: dict[str, Any], *, min_text_length: int, max_text_length: int) -> tuple[dict[str, Any] | None, str | None]:
    if row.get("_invalid_reason"):
        return None, str(row["_invalid_reason"])

    review_text = normalize_text(row.get("review_text", ""))
    predicted = str(row.get("predicted_sentiment", "")).strip().lower()
    actual = str(row.get("actual_sentiment", "")).strip().lower()

    if not review_text:
        return None, "empty_review_text"
    if len(review_text) < min_text_length:
        return None, "below_min_text_length"
    if len(review_text) > max_text_length:
        return None, "above_max_text_length"
    if predicted not in SENTIMENT_LABELS:
        return None, "invalid_predicted_sentiment"
    if actual not in SENTIMENT_LABELS:
        return None, "invalid_actual_sentiment"
    if predicted == actual:
        return None, "not_a_correction"

    submitted_at = parse_feedback_time(row)
    line_number = int(row.get("_line_number", 0) or 0)
    return (
        {
            "feedback_id": f"feedback-{line_number}",
            "review_text": review_text,
            "sentiment": actual,
            "predicted_sentiment": predicted,
            "actual_sentiment": actual,
            "source": str(row.get("source", "feedback")),
            "submitted_at": submitted_at,
            "feedback_type": "correction",
        },
        None,
    )


def prepare_feedback(
    feedback_path: Path = FEEDBACK / "feedback.jsonl",
    output_path: Path = DATA_INTERIM / "validated_feedback.csv",
    report_path: Path = REPORTS / "feedback_preparation_report.json",
) -> dict[str, object]:
    """Validate user corrections and publish a deterministic retraining-ready CSV."""
    record_performance = report_path == REPORTS / "feedback_preparation_report.json"
    with timed_stage("prepare_feedback_corrections", enabled=record_performance) as perf:
        ensure_dirs()
        params = read_params("feedback")
        data_params = read_params("data")
        min_text_length = int(params.get("min_text_length", data_params.get("min_text_length", 20)))
        max_text_length = int(params.get("max_text_length", data_params.get("max_text_length", 3000)))
        max_feedback_rows = int(os.getenv("SENTIMENT_RETRAIN_MAX_FEEDBACK_ROWS", params.get("max_feedback_rows", 1000)))

        raw_rows = iter_feedback_rows(feedback_path)
        valid_rows: list[dict[str, Any]] = []
        invalid_reasons: dict[str, int] = {}
        for row in raw_rows:
            validated, reason = validate_feedback_row(
                row,
                min_text_length=min_text_length,
                max_text_length=max_text_length,
            )
            if validated is None:
                invalid_reasons[str(reason)] = invalid_reasons.get(str(reason), 0) + 1
                continue
            valid_rows.append(validated)

        df = pd.DataFrame(valid_rows, columns=OUTPUT_COLUMNS)
        duplicate_rows = 0
        if not df.empty:
            df["_dedupe_key"] = df["review_text"].str.lower() + "|" + df["actual_sentiment"]
            before = len(df)
            df = df.sort_values("submitted_at", ascending=False).drop_duplicates("_dedupe_key", keep="first")
            duplicate_rows = before - len(df)
            df = df.drop(columns=["_dedupe_key"]).head(max_feedback_rows)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        perf["rows_processed"] = int(len(raw_rows))
        perf["extra"] = {"valid_corrections": int(len(df)), "invalid_rows": int(sum(invalid_reasons.values()))}
        report = {
            "stage": "prepare_feedback_corrections",
            "status": "success",
            "feedback_path": str(feedback_path),
            "output_path": str(output_path),
            "raw_feedback_rows": int(len(raw_rows)),
            "valid_correction_rows": int(len(df)),
            "invalid_rows": int(sum(invalid_reasons.values())),
            "invalid_reasons": invalid_reasons,
            "duplicate_rows_removed": int(duplicate_rows),
            "max_feedback_rows": max_feedback_rows,
            "min_text_length": min_text_length,
            "max_text_length": max_text_length,
        }
        write_json(report_path, report)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate feedback corrections for retraining.")
    parser.add_argument("--feedback", type=Path, default=FEEDBACK / "feedback.jsonl")
    parser.add_argument("--output", type=Path, default=DATA_INTERIM / "validated_feedback.csv")
    parser.add_argument("--report", type=Path, default=REPORTS / "feedback_preparation_report.json")
    args = parser.parse_args()
    report = prepare_feedback(args.feedback, args.output, args.report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
