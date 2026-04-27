from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.common import (
    RANDOM_SEED,
    ensure_dirs,
    path_for,
    read_params,
    write_json,
)
from ml.monitoring.performance import timed_stage


def clean_text(value: object) -> str:
    """Normalize whitespace without changing wording; sentiment signal should remain intact."""
    return " ".join(str(value).strip().split())


def reject_rows(df: pd.DataFrame, mask: pd.Series, reason: str) -> pd.DataFrame:
    """Keep rejected rows auditable instead of silently dropping them."""
    rejected = df.loc[mask].copy()
    rejected["rejection_reason"] = reason
    return rejected


def preprocess(
    input_path: Path | None = None,
    processed_dir: Path | None = None,
    rejected_output: Path | None = None,
    report_output_path: Path | None = None,
) -> dict[str, object]:
    """Clean reviews, write rejected-row evidence, and create deterministic stratified splits."""
    input_path = input_path or path_for("raw_reviews")
    processed_dir = processed_dir or path_for("train").parent
    rejected_output = rejected_output or path_for("rejected_reviews")
    report_output_path = report_output_path or path_for("preprocessing_report")
    record_performance = report_output_path == path_for("preprocessing_report")
    with timed_stage("preprocess_data", enabled=record_performance) as perf:
        ensure_dirs()
        config = read_params("data")
        min_text_length = int(config.get("min_text_length", 20))
        max_text_length = int(config.get("max_text_length", 3000))
        validation_size = float(config.get("validation_size", 0.15))
        test_size = float(config.get("test_size", 0.15))
        random_seed = int(config.get("random_seed", RANDOM_SEED))

        df = pd.read_csv(input_path)
        input_rows = len(df)
        df["review_text"] = df["review_text"].map(clean_text)
        df["_normalized_text"] = df["review_text"].str.lower()
        rejected_parts: list[pd.DataFrame] = []

        empty_mask = df["review_text"].str.len() == 0
        rejected_parts.append(reject_rows(df, empty_mask, "empty_review_text"))
        df = df.loc[~empty_mask].copy()

        too_short_mask = df["review_text"].str.len() < min_text_length
        rejected_parts.append(reject_rows(df, too_short_mask, "below_min_text_length"))
        df = df.loc[~too_short_mask].copy()

        too_long_mask = df["review_text"].str.len() > max_text_length
        rejected_parts.append(reject_rows(df, too_long_mask, "above_max_text_length"))
        df = df.loc[~too_long_mask].copy()

        duplicate_mask = df["_normalized_text"].duplicated(keep="first")
        rejected_parts.append(reject_rows(df, duplicate_mask, "duplicate_review_text"))
        df = df.loc[~duplicate_mask].drop(columns=["_normalized_text"]).copy()

        rejected_df = pd.concat(rejected_parts, ignore_index=True)
        rejected_output.parent.mkdir(parents=True, exist_ok=True)
        rejected_df.drop(columns=["_normalized_text"], errors="ignore").to_csv(rejected_output, index=False)

        holdout_size = validation_size + test_size
        train_df, holdout_df = train_test_split(
            df,
            test_size=holdout_size,
            random_state=random_seed,
            stratify=df["sentiment"],
        )
        valid_df, test_df = train_test_split(
            holdout_df,
            test_size=test_size / holdout_size,
            random_state=random_seed,
            stratify=holdout_df["sentiment"],
        )

        outputs = {
            "train": processed_dir / path_for("train").name,
            "validation": processed_dir / path_for("validation").name,
            "test": processed_dir / path_for("test").name,
        }
        processed_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(outputs["train"], index=False)
        valid_df.to_csv(outputs["validation"], index=False)
        test_df.to_csv(outputs["test"], index=False)

        perf["rows_processed"] = int(input_rows)
        perf["extra"] = {"output_rows": int(len(df)), "rejected_rows": int(len(rejected_df))}
        report = {
            "stage": "preprocess_data",
            "status": "success",
            "input_rows": int(input_rows),
            "rows_removed_empty": int(len(rejected_parts[0])),
            "rows_removed_too_short": int(len(rejected_parts[1])),
            "rows_removed_too_long": int(len(rejected_parts[2])),
            "rows_removed_duplicates": int(len(rejected_parts[3])),
            "rejected_rows": int(len(rejected_df)),
            "final_rows": int(len(df)),
            "splits": {
                "train": int(len(train_df)),
                "validation": int(len(valid_df)),
                "test": int(len(test_df)),
            },
            "split_class_distribution": {
                "train": {str(key): int(value) for key, value in train_df["sentiment"].value_counts().to_dict().items()},
                "validation": {
                    str(key): int(value) for key, value in valid_df["sentiment"].value_counts().to_dict().items()
                },
                "test": {str(key): int(value) for key, value in test_df["sentiment"].value_counts().to_dict().items()},
            },
            "split_rating_distribution": {
                "train": {
                    str(key): int(value) for key, value in train_df["rating"].value_counts().sort_index().to_dict().items()
                },
                "validation": {
                    str(key): int(value) for key, value in valid_df["rating"].value_counts().sort_index().to_dict().items()
                },
                "test": {
                    str(key): int(value) for key, value in test_df["rating"].value_counts().sort_index().to_dict().items()
                },
            },
            "random_seed": random_seed,
            "rejected_output": str(rejected_output),
            "outputs": {name: str(path) for name, path in outputs.items()},
        }
        write_json(report_output_path, report)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and split review data.")
    parser.add_argument("--input", type=Path, default=path_for("raw_reviews"))
    args = parser.parse_args()
    preprocess(args.input)


if __name__ == "__main__":
    main()
