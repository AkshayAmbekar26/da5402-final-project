from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from ml.common import path_for, read_json, write_json
from ml.monitoring.performance import record_stage_performance


def distribution_delta(reference: dict[str, int], current: dict[str, int]) -> float:
    """Return total-variation distance between two discrete distributions."""
    keys = set(reference) | set(current)
    ref_total = max(1, sum(reference.values()))
    cur_total = max(1, sum(current.values()))
    return float(
        sum(abs(reference.get(key, 0) / ref_total - current.get(key, 0) / cur_total) for key in keys) / 2
    )


def feature_stat_delta(reference: dict[str, float], current: dict[str, float]) -> float:
    """Average normalized absolute movement across tracked numeric feature statistics."""
    keys = set(reference) & set(current)
    if not keys:
        return 0.0
    deltas = [
        abs(float(current[key]) - float(reference[key])) / max(abs(float(reference[key])), 0.001)
        for key in keys
    ]
    return float(sum(deltas) / len(deltas))


def detect_drift(
    baseline_path: Path | None = None,
    current_path: Path | None = None,
) -> dict[str, object]:
    """Compare current data with the stored training baseline and publish a drift report."""
    stage_start = perf_counter()
    baseline_path = baseline_path or path_for("feature_baseline")
    current_path = current_path or path_for("test")
    baseline = read_json(baseline_path)
    current_df = pd.read_csv(current_path)
    current_lengths = current_df["review_text"].astype(str).str.len()
    current_sentiments = {
        str(key): int(value) for key, value in current_df["sentiment"].value_counts().to_dict().items()
    }
    feature_names = sorted(str(name) for name in baseline.get("tfidf_feature_means", {}))
    current_feature_means: dict[str, float] = {}
    current_feature_variances: dict[str, float] = {}
    if feature_names:
        vectorizer = TfidfVectorizer(vocabulary=feature_names, stop_words="english")
        matrix = vectorizer.fit_transform(current_df["review_text"].astype(str))
        means = matrix.mean(axis=0).A1
        variances = matrix.power(2).mean(axis=0).A1 - means**2
        current_feature_means = {
            str(name): float(value) for name, value in zip(feature_names, means, strict=False)
        }
        current_feature_variances = {
            str(name): float(max(value, 0.0))
            for name, value in zip(feature_names, variances, strict=False)
        }

    baseline_mean = float(baseline["text_length"]["mean"])
    current_mean = float(current_lengths.mean())
    length_delta_ratio = abs(current_mean - baseline_mean) / max(1.0, baseline_mean)
    sentiment_delta = distribution_delta(baseline["sentiment_distribution"], current_sentiments)
    feature_mean_delta = feature_stat_delta(baseline.get("tfidf_feature_means", {}), current_feature_means)
    feature_variance_delta = feature_stat_delta(
        baseline.get("tfidf_feature_variances", {}),
        current_feature_variances,
    )
    drift_score = max(length_delta_ratio, sentiment_delta, feature_mean_delta, feature_variance_delta)

    report = {
        "stage": "run_batch_drift_check",
        "status": "success",
        "baseline_path": str(baseline_path),
        "current_path": str(current_path),
        "text_length_mean_baseline": baseline_mean,
        "text_length_mean_current": current_mean,
        "length_delta_ratio": float(length_delta_ratio),
        "sentiment_distribution_delta": float(sentiment_delta),
        "tfidf_feature_mean_delta": float(feature_mean_delta),
        "tfidf_feature_variance_delta": float(feature_variance_delta),
        "current_tfidf_feature_means": current_feature_means,
        "current_tfidf_feature_variances": current_feature_variances,
        "drift_score": float(drift_score),
        "drift_detected": bool(drift_score > 0.25),
        "threshold": 0.25,
    }
    write_json(path_for("drift_report"), report)
    record_stage_performance(
        "run_batch_drift_check",
        perf_counter() - stage_start,
        rows_processed=len(current_df),
        extra={"drift_detected": report["drift_detected"], "drift_score": report["drift_score"]},
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch drift detection.")
    parser.add_argument("--baseline", type=Path, default=path_for("feature_baseline"))
    parser.add_argument("--current", type=Path, default=path_for("test"))
    args = parser.parse_args()
    detect_drift(args.baseline, args.current)


if __name__ == "__main__":
    main()
