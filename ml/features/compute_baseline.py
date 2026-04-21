from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from ml.common import DATA_BASELINES, DATA_PROCESSED, REPORTS, ensure_dirs, write_json
from ml.monitoring.performance import record_stage_performance


def compute_baseline(
    input_path: Path = DATA_PROCESSED / "train.csv",
    baseline_output_path: Path = DATA_BASELINES / "feature_baseline.json",
    report_output_path: Path = REPORTS / "feature_baseline_report.json",
) -> dict[str, object]:
    start = perf_counter()
    ensure_dirs()
    df = pd.read_csv(input_path)
    text_lengths = df["review_text"].astype(str).str.len()
    vectorizer = TfidfVectorizer(max_features=50, ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(df["review_text"].astype(str))
    feature_means = matrix.mean(axis=0).A1
    feature_variances = matrix.power(2).mean(axis=0).A1 - feature_means**2
    feature_names = vectorizer.get_feature_names_out()
    count_vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), max_features=100)
    count_matrix = count_vectorizer.fit_transform(df["review_text"].astype(str))
    term_counts = count_matrix.sum(axis=0).A1
    count_features = count_vectorizer.get_feature_names_out()
    top_terms = sorted(zip(count_features, term_counts, strict=False), key=lambda item: item[1], reverse=True)[:25]
    sentiment_counts = df["sentiment"].value_counts().to_dict()
    rating_counts = df["rating"].value_counts().sort_index().to_dict()
    total_rows = max(1, len(df))

    baseline = {
        "stage": "compute_drift_baseline",
        "rows": int(len(df)),
        "text_length": {
            "mean": float(text_lengths.mean()),
            "variance": float(text_lengths.var()),
            "min": int(text_lengths.min()),
            "max": int(text_lengths.max()),
            "median": float(text_lengths.median()),
            "p95": float(text_lengths.quantile(0.95)),
            "p99": float(text_lengths.quantile(0.99)),
        },
        "sentiment_distribution": {
            str(key): int(value) for key, value in sentiment_counts.items()
        },
        "sentiment_priors": {
            str(key): float(value / total_rows) for key, value in sentiment_counts.items()
        },
        "rating_distribution": {
            str(key): int(value) for key, value in rating_counts.items()
        },
        "rating_priors": {
            str(key): float(value / total_rows) for key, value in rating_counts.items()
        },
        "vocabulary_size": int(len(count_vectorizer.vocabulary_)),
        "top_terms": [{"term": str(term), "count": int(count)} for term, count in top_terms],
        "tfidf_feature_means": {
            str(name): float(value) for name, value in zip(feature_names, feature_means, strict=False)
        },
        "tfidf_feature_variances": {
            str(name): float(max(value, 0.0))
            for name, value in zip(feature_names, feature_variances, strict=False)
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
            "tracked_feature_variances": len(baseline["tfidf_feature_variances"]),
        },
    )
    if baseline_output_path == DATA_BASELINES / "feature_baseline.json":
        record_stage_performance("compute_drift_baseline", perf_counter() - start, rows_processed=len(df))
    return baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute baseline statistics for drift detection.")
    parser.add_argument("--input", type=Path, default=DATA_PROCESSED / "train.csv")
    args = parser.parse_args()
    compute_baseline(args.input)


if __name__ == "__main__":
    main()
