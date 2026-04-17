from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from ml.common import DATA_BASELINES, DATA_PROCESSED, REPORTS, ensure_dirs, write_json


def compute_baseline(
    input_path: Path = DATA_PROCESSED / "train.csv",
    baseline_output_path: Path = DATA_BASELINES / "feature_baseline.json",
    report_output_path: Path = REPORTS / "feature_baseline_report.json",
) -> dict[str, object]:
    ensure_dirs()
    df = pd.read_csv(input_path)
    text_lengths = df["review_text"].astype(str).str.len()
    vectorizer = TfidfVectorizer(max_features=50, ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(df["review_text"].astype(str))
    feature_means = matrix.mean(axis=0).A1
    feature_names = vectorizer.get_feature_names_out()

    baseline = {
        "stage": "compute_drift_baseline",
        "rows": int(len(df)),
        "text_length": {
            "mean": float(text_lengths.mean()),
            "variance": float(text_lengths.var()),
            "min": int(text_lengths.min()),
            "max": int(text_lengths.max()),
        },
        "sentiment_distribution": {
            str(key): int(value) for key, value in df["sentiment"].value_counts().to_dict().items()
        },
        "rating_distribution": {
            str(key): int(value) for key, value in df["rating"].value_counts().to_dict().items()
        },
        "tfidf_feature_means": {
            str(name): float(value) for name, value in zip(feature_names, feature_means, strict=False)
        },
    }
    write_json(baseline_output_path, baseline)
    write_json(
        report_output_path,
        {
            "stage": "generate_features",
            "status": "success",
            "baseline_path": str(baseline_output_path),
            "tracked_features": len(baseline["tfidf_feature_means"]),
        },
    )
    return baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute baseline statistics for drift detection.")
    parser.add_argument("--input", type=Path, default=DATA_PROCESSED / "train.csv")
    args = parser.parse_args()
    compute_baseline(args.input)


if __name__ == "__main__":
    main()
