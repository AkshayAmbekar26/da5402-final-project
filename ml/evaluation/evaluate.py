from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd

from ml.common import DATA_PROCESSED, MODELS, REPORTS, ensure_dirs, write_json
from ml.monitoring.performance import record_stage_performance
from ml.training.train import evaluate_predictions


def evaluate(
    model_path: Path = MODELS / "sentiment_model.joblib",
    test_path: Path = DATA_PROCESSED / "test.csv",
) -> dict[str, object]:
    stage_start = perf_counter()
    ensure_dirs()
    model = joblib.load(model_path)
    test_df = pd.read_csv(test_path)
    start = perf_counter()
    predictions = model.predict(test_df["review_text"])
    elapsed_ms = (perf_counter() - start) * 1000
    metrics = evaluate_predictions(test_df["sentiment"], predictions)
    report = {
        "stage": "evaluate_model",
        "status": "success",
        "model_path": str(model_path),
        "test_path": str(test_path),
        "metrics": metrics,
        "total_latency_ms": float(elapsed_ms),
        "latency_ms_per_review": float(elapsed_ms / max(1, len(test_df))),
        "accepted": bool(metrics["macro_f1"] >= 0.75),
        "acceptance_threshold_macro_f1": 0.75,
    }
    write_json(REPORTS / "evaluation.json", report)
    record_stage_performance(
        "evaluate_selected_model",
        perf_counter() - stage_start,
        rows_processed=len(test_df),
        extra={"accepted": report["accepted"], "macro_f1": metrics["macro_f1"]},
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained sentiment model.")
    parser.add_argument("--model", type=Path, default=MODELS / "sentiment_model.joblib")
    parser.add_argument("--test", type=Path, default=DATA_PROCESSED / "test.csv")
    args = parser.parse_args()
    evaluate(args.model, args.test)


if __name__ == "__main__":
    main()
