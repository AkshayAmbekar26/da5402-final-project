from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ml.common import DATA_RAW, REPORTS, SENTIMENT_LABELS, ensure_dirs, write_json

REQUIRED_COLUMNS = ["review_id", "review_text", "rating", "sentiment", "source", "ingested_at"]


def validate_dataframe(df: pd.DataFrame) -> dict[str, object]:
    errors: list[str] = []
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
        return {"status": "failed", "errors": errors}

    null_counts = df[REQUIRED_COLUMNS].isnull().sum().to_dict()
    duplicate_review_ids = int(df["review_id"].duplicated().sum())
    invalid_ratings = int((~df["rating"].between(1, 5)).sum())
    invalid_sentiments = sorted(set(df["sentiment"]) - set(SENTIMENT_LABELS))
    empty_reviews = int((df["review_text"].astype(str).str.strip().str.len() == 0).sum())
    class_distribution = df["sentiment"].value_counts().to_dict()

    if any(count > 0 for count in null_counts.values()):
        errors.append(f"Null values found: {null_counts}")
    if duplicate_review_ids:
        errors.append(f"Duplicate review_id values found: {duplicate_review_ids}")
    if invalid_ratings:
        errors.append(f"Invalid rating values found: {invalid_ratings}")
    if invalid_sentiments:
        errors.append(f"Invalid sentiments found: {invalid_sentiments}")
    if empty_reviews:
        errors.append(f"Empty review_text values found: {empty_reviews}")

    return {
        "status": "success" if not errors else "failed",
        "errors": errors,
        "rows": int(len(df)),
        "null_counts": {str(key): int(value) for key, value in null_counts.items()},
        "duplicate_review_ids": duplicate_review_ids,
        "invalid_ratings": invalid_ratings,
        "invalid_sentiments": invalid_sentiments,
        "empty_reviews": empty_reviews,
        "class_distribution": {str(key): int(value) for key, value in class_distribution.items()},
    }


def validate_data(input_path: Path = DATA_RAW / "reviews.csv") -> dict[str, object]:
    ensure_dirs()
    df = pd.read_csv(input_path)
    report = validate_dataframe(df)
    report["stage"] = "validate_raw_data"
    report["input_path"] = str(input_path)
    write_json(REPORTS / "data_validation.json", report)
    if report["status"] != "success":
        raise ValueError(f"Data validation failed: {report['errors']}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate raw review data.")
    parser.add_argument("--input", type=Path, default=DATA_RAW / "reviews.csv")
    args = parser.parse_args()
    validate_data(args.input)


if __name__ == "__main__":
    main()

