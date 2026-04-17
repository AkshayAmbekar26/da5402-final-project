from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.common import DATA_PROCESSED, DATA_RAW, RANDOM_SEED, REPORTS, ensure_dirs, write_json


def clean_text(value: object) -> str:
    return " ".join(str(value).strip().split())


def preprocess(input_path: Path = DATA_RAW / "reviews.csv") -> dict[str, object]:
    ensure_dirs()
    df = pd.read_csv(input_path)
    df["review_text"] = df["review_text"].map(clean_text)
    df = df[df["review_text"].str.len() > 0].copy()

    train_df, holdout_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=df["sentiment"],
    )
    valid_df, test_df = train_test_split(
        holdout_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=holdout_df["sentiment"],
    )

    outputs = {
        "train": DATA_PROCESSED / "train.csv",
        "validation": DATA_PROCESSED / "validation.csv",
        "test": DATA_PROCESSED / "test.csv",
    }
    train_df.to_csv(outputs["train"], index=False)
    valid_df.to_csv(outputs["validation"], index=False)
    test_df.to_csv(outputs["test"], index=False)

    report = {
        "stage": "preprocess_data",
        "status": "success",
        "input_rows": int(len(df)),
        "splits": {
            "train": int(len(train_df)),
            "validation": int(len(valid_df)),
            "test": int(len(test_df)),
        },
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    write_json(REPORTS / "preprocessing_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and split review data.")
    parser.add_argument("--input", type=Path, default=DATA_RAW / "reviews.csv")
    args = parser.parse_args()
    preprocess(args.input)


if __name__ == "__main__":
    main()

