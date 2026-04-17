from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.common import (
    DATA_INTERIM,
    DATA_PROCESSED,
    DATA_RAW,
    RANDOM_SEED,
    REPORTS,
    ROOT,
    ensure_dirs,
    read_json,
    write_json,
)
from ml.monitoring.performance import timed_stage


def clean_text(value: object) -> str:
    return " ".join(str(value).strip().split())


def reject_rows(df: pd.DataFrame, mask: pd.Series, reason: str) -> pd.DataFrame:
    rejected = df.loc[mask].copy()
    rejected["rejection_reason"] = reason
    return rejected


def preprocess(
    input_path: Path = DATA_RAW / "reviews.csv",
    config_path: Path = ROOT / "configs" / "data_config.json",
    processed_dir: Path = DATA_PROCESSED,
    rejected_output: Path = DATA_INTERIM / "rejected_reviews.csv",
    report_output_path: Path = REPORTS / "preprocessing_report.json",
) -> dict[str, object]:
    with timed_stage("preprocess_data") as perf:
        ensure_dirs()
        config = read_json(config_path) if config_path.exists() else {}
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
            "train": processed_dir / "train.csv",
            "validation": processed_dir / "validation.csv",
            "test": processed_dir / "test.csv",
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
    parser.add_argument("--input", type=Path, default=DATA_RAW / "reviews.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "data_config.json")
    args = parser.parse_args()
    preprocess(args.input, args.config)


if __name__ == "__main__":
    main()
