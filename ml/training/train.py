from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from ml.common import (
    DATA_PROCESSED,
    MODELS,
    RANDOM_SEED,
    REPORTS,
    SENTIMENT_LABELS,
    dvc_data_version,
    ensure_dirs,
    git_commit_hash,
    read_json,
    utc_now,
    write_json,
)


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "class_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=SENTIMENT_LABELS).tolist(),
        "labels": SENTIMENT_LABELS,
    }


def extract_feature_importance(model: Pipeline, top_k: int = 20) -> list[dict[str, object]]:
    vectorizer: TfidfVectorizer = model.named_steps["tfidf"]
    classifier: LogisticRegression = model.named_steps["classifier"]
    features = vectorizer.get_feature_names_out()
    importance: list[dict[str, object]] = []
    for class_index, label in enumerate(classifier.classes_):
        coefs = classifier.coef_[class_index]
        top_indices = np.argsort(coefs)[-top_k:][::-1]
        importance.append(
            {
                "class": str(label),
                "tokens": [
                    {"token": str(features[idx]), "weight": float(coefs[idx])}
                    for idx in top_indices
                ],
            }
        )
    return importance


def train(
    train_path: Path = DATA_PROCESSED / "train.csv",
    validation_path: Path = DATA_PROCESSED / "validation.csv",
    test_path: Path = DATA_PROCESSED / "test.csv",
) -> dict[str, object]:
    ensure_dirs()
    train_df = pd.read_csv(train_path)
    validation_df = pd.read_csv(validation_path)
    test_df = pd.read_csv(test_path)

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=80000,
                    ngram_range=(1, 3),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    C=3.0,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "product-review-sentiment"))

    with mlflow.start_run(run_name="tfidf-logistic-regression") as run:
        model.fit(train_df["review_text"], train_df["sentiment"])

        validation_pred = model.predict(validation_df["review_text"])
        test_pred = model.predict(test_df["review_text"])
        validation_metrics = evaluate_predictions(validation_df["sentiment"], validation_pred)
        test_metrics = evaluate_predictions(test_df["sentiment"], test_pred)

        start = perf_counter()
        _ = model.predict(test_df["review_text"].head(min(10, len(test_df))))
        latency_ms = (perf_counter() - start) * 1000 / max(1, min(10, len(test_df)))

        model_path = MODELS / "sentiment_model.joblib"
        metadata_path = MODELS / "model_metadata.json"
        importance_path = MODELS / "feature_importance.json"
        evaluation_path = REPORTS / "training_metrics.json"

        joblib.dump(model, model_path)
        feature_importance = extract_feature_importance(model)
        write_json(importance_path, {"feature_importance": feature_importance})

        metadata = {
            "model_name": "tfidf-logistic-regression",
            "model_version": "local-production",
            "mlflow_run_id": run.info.run_id,
            "git_commit": git_commit_hash(),
            "data_version": dvc_data_version(),
            "trained_at": utc_now(),
            "model_path": str(model_path),
            "labels": SENTIMENT_LABELS,
            "latency_ms_p50_estimate": float(latency_ms),
        }
        write_json(metadata_path, metadata)

        result = {
            "stage": "train_model",
            "status": "success",
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
            "feature_importance_path": str(importance_path),
            "validation": validation_metrics,
            "test": test_metrics,
            "latency_ms_per_review": float(latency_ms),
            "metadata": metadata,
        }
        write_json(evaluation_path, result)

        mlflow.log_params(
            {
                "model_type": "tfidf_logistic_regression",
                "max_features": 80000,
                "ngram_range": "1,3",
                "min_df": 1,
                "sublinear_tf": True,
                "C": 3.0,
                "class_weight": "balanced",
                "random_seed": RANDOM_SEED,
            }
        )
        mlflow.log_metrics(
            {
                "validation_macro_f1": validation_metrics["macro_f1"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_accuracy": test_metrics["accuracy"],
                "latency_ms_per_review": float(latency_ms),
            }
        )
        mlflow.set_tags(
            {
                "git_commit": metadata["git_commit"],
                "dvc_data_version": json.dumps(metadata["data_version"])[:500],
                "stage": "training",
            }
        )
        mlflow.log_artifact(str(evaluation_path))
        mlflow.log_artifact(str(importance_path))
        for artifact_name in [
            "ingestion_report.json",
            "data_validation.json",
            "eda_report.json",
            "eda_report.md",
            "preprocessing_report.json",
            "feature_baseline_report.json",
        ]:
            artifact_path = REPORTS / artifact_name
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path), artifact_path="data_quality")
        figures_dir = REPORTS / "figures"
        if figures_dir.exists():
            for figure_path in figures_dir.glob("*.png"):
                mlflow.log_artifact(str(figure_path), artifact_path="eda_figures")
        ingestion_report_path = REPORTS / "ingestion_report.json"
        preprocessing_report_path = REPORTS / "preprocessing_report.json"
        if ingestion_report_path.exists():
            ingestion_report = read_json(ingestion_report_path)
            mlflow.log_params(
                {
                    "dataset_name": ingestion_report.get("dataset_name", "unknown"),
                    "dataset_rows": ingestion_report.get("rows", 0),
                    "dataset_fallback_used": ingestion_report.get("fallback_used", False),
                }
            )
        if preprocessing_report_path.exists():
            preprocessing_report = read_json(preprocessing_report_path)
            mlflow.log_params(
                {
                    "train_rows": preprocessing_report.get("splits", {}).get("train", 0),
                    "validation_rows": preprocessing_report.get("splits", {}).get("validation", 0),
                    "test_rows": preprocessing_report.get("splits", {}).get("test", 0),
                }
            )
        mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="ProductReviewSentimentModel")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sentiment classification model.")
    parser.add_argument("--train", type=Path, default=DATA_PROCESSED / "train.csv")
    parser.add_argument("--validation", type=Path, default=DATA_PROCESSED / "validation.csv")
    parser.add_argument("--test", type=Path, default=DATA_PROCESSED / "test.csv")
    args = parser.parse_args()
    train(args.train, args.validation, args.test)


if __name__ == "__main__":
    main()
