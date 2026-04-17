from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ml.common import DATA_RAW, REPORTS, ROOT, SENTIMENT_LABELS, ensure_dirs, read_json, write_json

REQUIRED_COLUMNS = ["review_id", "review_text", "rating", "sentiment", "source", "ingested_at"]


def validate_dataframe(
    df: pd.DataFrame,
    min_text_length: int = 20,
    max_text_length: int = 3000,
    imbalance_warning_ratio: float = 3.0,
) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
        return {"status": "failed", "errors": errors, "warnings": warnings}

    null_counts = df[REQUIRED_COLUMNS].isnull().sum().to_dict()
    duplicate_review_ids = int(df["review_id"].duplicated().sum())
    duplicate_review_text = int(df["review_text"].astype(str).str.lower().duplicated().sum())
    invalid_ratings = int((~df["rating"].between(1, 5)).sum())
    invalid_sentiments = sorted(set(df["sentiment"]) - set(SENTIMENT_LABELS))
    text_lengths = df["review_text"].astype(str).str.strip().str.len()
    empty_reviews = int((text_lengths == 0).sum())
    short_reviews = int((text_lengths < min_text_length).sum())
    long_reviews = int((text_lengths > max_text_length).sum())
    class_distribution = df["sentiment"].value_counts().to_dict()
    rating_distribution = df["rating"].value_counts().sort_index().to_dict()
    duplicate_label_conflicts = int(
        df.assign(_normalized_text=df["review_text"].astype(str).str.lower().str.strip())
        .groupby("_normalized_text")["sentiment"]
        .nunique()
        .gt(1)
        .sum()
    )

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
    missing_classes = sorted(set(SENTIMENT_LABELS) - set(class_distribution))
    if missing_classes:
        errors.append(f"No examples found for sentiment classes: {missing_classes}")
    if duplicate_review_text:
        warnings.append(f"Duplicate review_text values found: {duplicate_review_text}")
    if short_reviews:
        warnings.append(f"Reviews shorter than {min_text_length} characters found: {short_reviews}")
    if long_reviews:
        warnings.append(f"Reviews longer than {max_text_length} characters found: {long_reviews}")
    if duplicate_label_conflicts:
        warnings.append(f"Same review text appears with multiple sentiment labels: {duplicate_label_conflicts}")
    if class_distribution:
        min_class = min(class_distribution.values())
        max_class = max(class_distribution.values())
        imbalance_ratio = max_class / max(1, min_class)
        if imbalance_ratio > imbalance_warning_ratio:
            warnings.append(f"Class imbalance ratio {imbalance_ratio:.2f} exceeds {imbalance_warning_ratio:.2f}")
    else:
        imbalance_ratio = 0.0

    return {
        "status": "success" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "rows": int(len(df)),
        "null_counts": {str(key): int(value) for key, value in null_counts.items()},
        "duplicate_review_ids": duplicate_review_ids,
        "duplicate_review_text": duplicate_review_text,
        "invalid_ratings": invalid_ratings,
        "invalid_sentiments": invalid_sentiments,
        "empty_reviews": empty_reviews,
        "short_reviews": short_reviews,
        "long_reviews": long_reviews,
        "duplicate_label_conflicts": duplicate_label_conflicts,
        "class_imbalance_ratio": float(imbalance_ratio),
        "class_distribution": {str(key): int(value) for key, value in class_distribution.items()},
        "rating_distribution": {str(key): int(value) for key, value in rating_distribution.items()},
    }


def validate_data(
    input_path: Path = DATA_RAW / "reviews.csv",
    config_path: Path = ROOT / "configs" / "data_config.json",
) -> dict[str, object]:
    ensure_dirs()
    config = read_json(config_path) if config_path.exists() else {}
    df = pd.read_csv(input_path)
    report = validate_dataframe(
        df,
        min_text_length=int(config.get("min_text_length", 20)),
        max_text_length=int(config.get("max_text_length", 3000)),
    )
    report["stage"] = "validate_raw_data"
    report["input_path"] = str(input_path)
    report["config_path"] = str(config_path)
    write_json(REPORTS / "data_validation.json", report)
    if report["status"] != "success":
        raise ValueError(f"Data validation failed: {report['errors']}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate raw review data.")
    parser.add_argument("--input", type=Path, default=DATA_RAW / "reviews.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "data_config.json")
    args = parser.parse_args()
    validate_data(args.input, args.config)


if __name__ == "__main__":
    main()
