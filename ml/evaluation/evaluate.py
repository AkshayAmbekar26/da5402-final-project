from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd

from ml.common import (
    SENTIMENT_LABELS,
    ensure_dirs,
    path_for,
    read_json,
    read_params,
    write_json,
)
from ml.monitoring.performance import record_stage_performance
from ml.training.train import evaluate_predictions


def write_confusion_matrix_plot(metrics: dict[str, object], output_path: Path | None = None) -> None:
    output_path = output_path or path_for("confusion_matrix")
    matrix = metrics["confusion_matrix"]
    rows = []
    cell_index = 0
    for actual_index, actual_label in enumerate(SENTIMENT_LABELS):
        for predicted_index, predicted_label in enumerate(SENTIMENT_LABELS):
            cell_index += 1
            rows.append(
                {
                    "cell_index": cell_index,
                    "actual": actual_label,
                    "predicted": predicted_label,
                    "count": int(matrix[actual_index][predicted_index]),
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def evaluate(
    model_path: Path | None = None,
    test_path: Path | None = None,
) -> dict[str, object]:
    stage_start = perf_counter()
    ensure_dirs()
    model_path = model_path or path_for("sentiment_model")
    test_path = test_path or path_for("test")
    model = joblib.load(model_path)
    test_df = pd.read_csv(test_path)
    start = perf_counter()
    predictions = model.predict(test_df["review_text"])
    elapsed_ms = (perf_counter() - start) * 1000
    metrics = evaluate_predictions(test_df["sentiment"], predictions)
    training_params = read_params("training")
    acceptance_threshold = float(training_params.get("acceptance_test_macro_f1", 0.75))
    acceptance_latency_ms = float(training_params.get("acceptance_latency_ms", 200.0))
    latency_ms_per_review = float(elapsed_ms / max(1, len(test_df)))
    report = {
        "stage": "evaluate_model",
        "status": "success",
        "model_path": str(model_path),
        "test_path": str(test_path),
        "metrics": metrics,
        "total_latency_ms": float(elapsed_ms),
        "latency_ms_per_review": latency_ms_per_review,
        "accepted": bool(metrics["macro_f1"] >= acceptance_threshold and latency_ms_per_review < acceptance_latency_ms),
        "acceptance_threshold_macro_f1": acceptance_threshold,
        "acceptance_latency_ms": acceptance_latency_ms,
    }
    write_json(path_for("evaluation_report"), report)
    selected_model_name = "unknown"
    metadata_path = path_for("model_metadata")
    if metadata_path.exists():
        selected_model_name = str(read_json(metadata_path).get("model_name", "unknown"))
    write_json(
        path_for("final_metrics"),
        {
            "selected_model": selected_model_name,
            "test_accuracy": float(metrics["accuracy"]),
            "test_macro_precision": float(metrics["macro_precision"]),
            "test_macro_recall": float(metrics["macro_recall"]),
            "test_macro_f1": float(metrics["macro_f1"]),
            "latency_ms_per_review": float(report["latency_ms_per_review"]),
            "accepted": bool(report["accepted"]),
        },
    )
    write_confusion_matrix_plot(metrics)
    record_stage_performance(
        "evaluate_selected_model",
        perf_counter() - stage_start,
        rows_processed=len(test_df),
        extra={"accepted": report["accepted"], "macro_f1": metrics["macro_f1"]},
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained sentiment model.")
    parser.add_argument("--model", type=Path, default=path_for("sentiment_model"))
    parser.add_argument("--test", type=Path, default=path_for("test"))
    args = parser.parse_args()
    evaluate(args.model, args.test)


if __name__ == "__main__":
    main()
