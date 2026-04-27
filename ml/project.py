from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

from ml.common import path_for, read_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the sentiment lifecycle as an MLflow Project.")
    parser.add_argument("--tracking-uri", type=str, default="file:./mlruns")
    parser.add_argument("--experiment-name", type=str, default="product-review-sentiment")
    parser.add_argument("--registered-model-name", type=str, default="ProductReviewSentimentModel")
    parser.add_argument("--max-rows-total", type=int, default=None)
    parser.add_argument("--seed-rows-per-class", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--acceptance-test-macro-f1", type=float, default=None)
    parser.add_argument("--acceptance-latency-ms", type=float, default=None)
    return parser.parse_args()


def load_params() -> dict[str, Any]:
    return read_params()


def apply_overrides(params: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updated = dict(params)
    updated.setdefault("data", {})
    updated.setdefault("training", {})
    data_params = dict(updated["data"])
    training_params = dict(updated["training"])

    if args.max_rows_total is not None:
        data_params["max_rows_total"] = args.max_rows_total
    if args.seed_rows_per_class is not None:
        data_params["seed_rows_per_class"] = args.seed_rows_per_class
    if args.random_seed is not None:
        data_params["random_seed"] = args.random_seed
        training_params["random_seed"] = args.random_seed
    if args.acceptance_test_macro_f1 is not None:
        training_params["acceptance_test_macro_f1"] = args.acceptance_test_macro_f1
    if args.acceptance_latency_ms is not None:
        training_params["acceptance_latency_ms"] = args.acceptance_latency_ms

    updated["data"] = data_params
    updated["training"] = training_params
    return updated


def run_lifecycle() -> None:
    from ml.data_ingestion.ingest import ingest
    from ml.eda.analyze import analyze
    from ml.evaluation.evaluate import evaluate
    from ml.features.compute_baseline import compute_baseline
    from ml.monitoring.drift import detect_drift
    from ml.monitoring.prepare_feedback import prepare_feedback
    from ml.monitoring.publish_report import publish_pipeline_report
    from ml.preprocessing.merge_feedback import merge_feedback
    from ml.preprocessing.preprocess import preprocess
    from ml.training.train import train
    from ml.validation.validate_data import validate_data

    ingest()
    validate_data()
    analyze()
    preprocess()
    prepare_feedback()
    merge_feedback()
    compute_baseline(input_path=path_for("train_augmented"))
    train(train_path=path_for("train_augmented"))
    evaluate()
    detect_drift()
    publish_pipeline_report()


def main() -> None:
    args = parse_args()
    os.environ["MLFLOW_TRACKING_URI"] = args.tracking_uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = args.experiment_name
    os.environ["MLFLOW_REGISTERED_MODEL_NAME"] = args.registered_model_name

    params = apply_overrides(load_params(), args)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
        yaml.safe_dump(params, handle, sort_keys=False)
        temp_params_path = handle.name

    previous_params_path = os.environ.get("PARAMS_PATH")
    os.environ["PARAMS_PATH"] = temp_params_path
    try:
        run_lifecycle()
    finally:
        if previous_params_path is None:
            os.environ.pop("PARAMS_PATH", None)
        else:
            os.environ["PARAMS_PATH"] = previous_params_path
        Path(temp_params_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
