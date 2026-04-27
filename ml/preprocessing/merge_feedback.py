from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from ml.common import ensure_dirs, path_for, utc_now, write_json
from ml.monitoring.performance import timed_stage

SENTIMENT_RATING = {"negative": 1, "neutral": 3, "positive": 5}


def review_id(text: str, sentiment: str) -> str:
    digest = hashlib.sha256(f"{text}|{sentiment}".encode()).hexdigest()[:16]
    return f"feedback-{digest}"


def feedback_to_training_rows(feedback_df: pd.DataFrame) -> pd.DataFrame:
    if feedback_df.empty:
        return pd.DataFrame(columns=["review_id", "review_text", "rating", "sentiment", "source", "ingested_at"])

    rows = feedback_df.copy()
    rows["sentiment"] = rows["actual_sentiment"].astype(str)
    rows["rating"] = rows["sentiment"].map(SENTIMENT_RATING).fillna(3).astype(int)
    rows["review_id"] = [
        review_id(str(text), str(sentiment))
        for text, sentiment in zip(rows["review_text"], rows["sentiment"], strict=False)
    ]
    rows["source"] = "user_feedback_correction"
    rows["ingested_at"] = rows.get("submitted_at", utc_now())
    return rows[["review_id", "review_text", "rating", "sentiment", "source", "ingested_at"]]


def merge_feedback(
    train_path: Path | None = None,
    feedback_path: Path | None = None,
    output_path: Path | None = None,
    report_path: Path | None = None,
) -> dict[str, object]:
    """Append validated correction feedback to the training split without changing validation/test sets."""
    train_path = train_path or path_for("train")
    feedback_path = feedback_path or path_for("validated_feedback")
    output_path = output_path or path_for("train_augmented")
    report_path = report_path or path_for("feedback_merge_report")
    record_performance = report_path == path_for("feedback_merge_report")
    with timed_stage("merge_feedback_corrections", enabled=record_performance) as perf:
        ensure_dirs()
        train_df = pd.read_csv(train_path)
        feedback_df = pd.read_csv(feedback_path) if feedback_path.exists() else pd.DataFrame()
        correction_df = feedback_to_training_rows(feedback_df)

        before_merge_rows = len(train_df)
        combined = pd.concat([train_df, correction_df], ignore_index=True, sort=False)
        combined["_dedupe_key"] = combined["review_text"].astype(str).str.lower() + "|" + combined["sentiment"].astype(str)
        before_dedupe = len(combined)
        combined = combined.drop_duplicates("_dedupe_key", keep="last").drop(columns=["_dedupe_key"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_path, index=False)

        feedback_rows_used = int(len(combined) - before_merge_rows)
        perf["rows_processed"] = int(len(combined))
        perf["extra"] = {"feedback_rows_used": feedback_rows_used}
        report = {
            "stage": "merge_feedback_corrections",
            "status": "success",
            "train_path": str(train_path),
            "feedback_path": str(feedback_path),
            "output_path": str(output_path),
            "base_train_rows": int(before_merge_rows),
            "validated_feedback_rows": int(len(correction_df)),
            "duplicates_removed": int(before_dedupe - len(combined)),
            "feedback_rows_used": feedback_rows_used,
            "augmented_train_rows": int(len(combined)),
            "class_distribution": {
                str(key): int(value)
                for key, value in combined["sentiment"].value_counts().sort_index().to_dict().items()
            },
        }
        write_json(report_path, report)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge validated feedback corrections into the training split.")
    parser.add_argument("--train", type=Path, default=path_for("train"))
    parser.add_argument("--feedback", type=Path, default=path_for("validated_feedback"))
    parser.add_argument("--output", type=Path, default=path_for("train_augmented"))
    parser.add_argument("--report", type=Path, default=path_for("feedback_merge_report"))
    args = parser.parse_args()
    report = merge_feedback(args.train, args.feedback, args.output, args.report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
