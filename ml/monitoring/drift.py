from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ml.common import DATA_BASELINES, DATA_PROCESSED, REPORTS, read_json, write_json


def distribution_delta(reference: dict[str, int], current: dict[str, int]) -> float:
    keys = set(reference) | set(current)
    ref_total = max(1, sum(reference.values()))
    cur_total = max(1, sum(current.values()))
    return float(
        sum(abs(reference.get(key, 0) / ref_total - current.get(key, 0) / cur_total) for key in keys) / 2
    )


def detect_drift(
    baseline_path: Path = DATA_BASELINES / "feature_baseline.json",
    current_path: Path = DATA_PROCESSED / "test.csv",
) -> dict[str, object]:
    baseline = read_json(baseline_path)
    current_df = pd.read_csv(current_path)
    current_lengths = current_df["review_text"].astype(str).str.len()
    current_sentiments = {
        str(key): int(value) for key, value in current_df["sentiment"].value_counts().to_dict().items()
    }

    baseline_mean = float(baseline["text_length"]["mean"])
    current_mean = float(current_lengths.mean())
    length_delta_ratio = abs(current_mean - baseline_mean) / max(1.0, baseline_mean)
    sentiment_delta = distribution_delta(baseline["sentiment_distribution"], current_sentiments)
    drift_score = max(length_delta_ratio, sentiment_delta)

    report = {
        "stage": "run_batch_drift_check",
        "status": "success",
        "baseline_path": str(baseline_path),
        "current_path": str(current_path),
        "text_length_mean_baseline": baseline_mean,
        "text_length_mean_current": current_mean,
        "length_delta_ratio": float(length_delta_ratio),
        "sentiment_distribution_delta": float(sentiment_delta),
        "drift_score": float(drift_score),
        "drift_detected": bool(drift_score > 0.25),
        "threshold": 0.25,
    }
    write_json(REPORTS / "drift_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch drift detection.")
    parser.add_argument("--baseline", type=Path, default=DATA_BASELINES / "feature_baseline.json")
    parser.add_argument("--current", type=Path, default=DATA_PROCESSED / "test.csv")
    args = parser.parse_args()
    detect_drift(args.baseline, args.current)


if __name__ == "__main__":
    main()

