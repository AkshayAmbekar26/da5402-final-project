from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
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
from ml.monitoring.performance import record_stage_performance

ACCEPTANCE_TEST_MACRO_F1 = 0.75
ACCEPTANCE_LATENCY_MS = 200.0


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    model_type: str
    max_features: int
    ngram_range: tuple[int, int]
    min_df: int
    sublinear_tf: bool
    classifier: str
    classifier_params: dict[str, object]


def candidate_specs() -> list[CandidateSpec]:
    return [
        CandidateSpec(
            name="tfidf_logistic_baseline",
            model_type="tfidf_logistic_regression",
            max_features=2500,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=False,
            classifier="LogisticRegression",
            classifier_params={"C": 1.0, "max_iter": 1000, "class_weight": "balanced"},
        ),
        CandidateSpec(
            name="tfidf_logistic_regularized",
            model_type="tfidf_logistic_regression",
            max_features=30000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            classifier="LogisticRegression",
            classifier_params={"C": 1.5, "max_iter": 2000, "class_weight": "balanced"},
        ),
        CandidateSpec(
            name="tfidf_logistic_tuned",
            model_type="tfidf_logistic_regression",
            max_features=80000,
            ngram_range=(1, 3),
            min_df=1,
            sublinear_tf=True,
            classifier="LogisticRegression",
            classifier_params={"C": 3.0, "max_iter": 2500, "class_weight": "balanced"},
        ),
        CandidateSpec(
            name="tfidf_sgd_log_loss",
            model_type="tfidf_sgd_classifier",
            max_features=80000,
            ngram_range=(1, 3),
            min_df=2,
            sublinear_tf=True,
            classifier="SGDClassifier",
            classifier_params={
                "alpha": 1e-5,
                "loss": "log_loss",
                "max_iter": 2000,
                "class_weight": "balanced",
            },
        ),
    ]


def build_candidate_model(spec: CandidateSpec) -> Pipeline:
    vectorizer = TfidfVectorizer(
        max_features=spec.max_features,
        ngram_range=spec.ngram_range,
        min_df=spec.min_df,
        sublinear_tf=spec.sublinear_tf,
    )
    if spec.classifier == "LogisticRegression":
        classifier = LogisticRegression(random_state=RANDOM_SEED, **spec.classifier_params)
    elif spec.classifier == "SGDClassifier":
        classifier = SGDClassifier(random_state=RANDOM_SEED, **spec.classifier_params)
    else:
        raise ValueError(f"Unsupported classifier: {spec.classifier}")
    return Pipeline(steps=[("tfidf", vectorizer), ("classifier", classifier)])


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
    classifier = model.named_steps["classifier"]
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


def latency_benchmark_ms(model: Pipeline, texts: pd.Series, sample_size: int = 50) -> float:
    sample = texts.head(min(sample_size, len(texts)))
    start = perf_counter()
    _ = model.predict(sample)
    return (perf_counter() - start) * 1000 / max(1, len(sample))


def model_size_bytes(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def log_data_artifacts() -> None:
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


def dataset_params() -> dict[str, object]:
    params: dict[str, object] = {}
    ingestion_report_path = REPORTS / "ingestion_report.json"
    preprocessing_report_path = REPORTS / "preprocessing_report.json"
    if ingestion_report_path.exists():
        ingestion_report = read_json(ingestion_report_path)
        params.update(
            {
                "dataset_name": ingestion_report.get("dataset_name", "unknown"),
                "dataset_rows": ingestion_report.get("rows", 0),
                "dataset_fallback_used": ingestion_report.get("fallback_used", False),
            }
        )
    if preprocessing_report_path.exists():
        preprocessing_report = read_json(preprocessing_report_path)
        params.update(
            {
                "train_rows": preprocessing_report.get("splits", {}).get("train", 0),
                "validation_rows": preprocessing_report.get("splits", {}).get("validation", 0),
                "test_rows": preprocessing_report.get("splits", {}).get("test", 0),
            }
        )
    return params


def candidate_params(spec: CandidateSpec) -> dict[str, object]:
    params = {
        "candidate_name": spec.name,
        "model_type": spec.model_type,
        "max_features": spec.max_features,
        "ngram_range": f"{spec.ngram_range[0]},{spec.ngram_range[1]}",
        "min_df": spec.min_df,
        "sublinear_tf": spec.sublinear_tf,
        "classifier": spec.classifier,
        "random_seed": RANDOM_SEED,
    }
    params.update({f"classifier_{key}": value for key, value in spec.classifier_params.items()})
    params.update(dataset_params())
    return params


def candidate_metrics(prefix: str, metrics: dict[str, object]) -> dict[str, float]:
    return {
        f"{prefix}_accuracy": float(metrics["accuracy"]),
        f"{prefix}_macro_precision": float(metrics["macro_precision"]),
        f"{prefix}_macro_recall": float(metrics["macro_recall"]),
        f"{prefix}_macro_f1": float(metrics["macro_f1"]),
    }


def run_candidate(
    spec: CandidateSpec,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    git_commit: str,
    data_version: str,
) -> dict[str, object]:
    model = build_candidate_model(spec)
    with mlflow.start_run(run_name=spec.name) as run:
        model.fit(train_df["review_text"], train_df["sentiment"])
        validation_pred = model.predict(validation_df["review_text"])
        test_pred = model.predict(test_df["review_text"])
        validation_metrics = evaluate_predictions(validation_df["sentiment"], validation_pred)
        test_metrics = evaluate_predictions(test_df["sentiment"], test_pred)
        latency_ms = latency_benchmark_ms(model, test_df["review_text"])

        params = candidate_params(spec)
        mlflow.log_params(params)
        mlflow.log_metrics(
            {
                **candidate_metrics("validation", validation_metrics),
                **candidate_metrics("test", test_metrics),
                "latency_ms_per_review": float(latency_ms),
            }
        )
        mlflow.set_tags(
            {
                "git_commit": git_commit,
                "dvc_data_version": json.dumps(data_version)[:500],
                "stage": "candidate_training",
                "candidate_name": spec.name,
            }
        )
        log_data_artifacts()

    return {
        "candidate_name": spec.name,
        "mlflow_run_id": run.info.run_id,
        "params": params,
        "validation": validation_metrics,
        "test": test_metrics,
        "latency_ms_per_review": float(latency_ms),
        "passes_acceptance": bool(
            test_metrics["macro_f1"] >= ACCEPTANCE_TEST_MACRO_F1
            and latency_ms < ACCEPTANCE_LATENCY_MS
        ),
        "model": model,
    }


def select_best_candidate(candidates: list[dict[str, object]]) -> dict[str, object]:
    passing = [candidate for candidate in candidates if candidate["passes_acceptance"]]
    eligible = passing if passing else candidates
    return sorted(
        eligible,
        key=lambda candidate: (
            candidate["validation"]["macro_f1"],
            candidate["test"]["macro_f1"],
            -candidate["latency_ms_per_review"],
        ),
        reverse=True,
    )[0]


def serializable_candidate(candidate: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in candidate.items() if key != "model"}


def write_model_comparison(
    candidates: list[dict[str, object]],
    selected: dict[str, object],
    output_json: Path = REPORTS / "model_comparison.json",
    output_md: Path = REPORTS / "model_comparison.md",
) -> None:
    payload = {
        "stage": "model_selection",
        "status": "success",
        "selection_rule": (
            "Choose the candidate with highest validation macro F1 among models where "
            f"test macro F1 >= {ACCEPTANCE_TEST_MACRO_F1} and latency < {ACCEPTANCE_LATENCY_MS} ms. "
            "If no model passes, choose the highest validation macro F1 and mark acceptance false."
        ),
        "accepted_candidates": [
            candidate["candidate_name"] for candidate in candidates if candidate["passes_acceptance"]
        ],
        "selected_candidate": selected["candidate_name"],
        "selected_mlflow_run_id": selected["mlflow_run_id"],
        "candidates": [serializable_candidate(candidate) for candidate in candidates],
    }
    write_json(output_json, payload)

    lines = [
        "# Model Comparison Report",
        "",
        payload["selection_rule"],
        "",
        "| Candidate | Validation Macro F1 | Test Macro F1 | Test Accuracy | Latency ms/review | Accepted | MLflow Run |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for candidate in sorted(candidates, key=lambda item: item["validation"]["macro_f1"], reverse=True):
        lines.append(
            "| {name} | {val:.4f} | {test:.4f} | {acc:.4f} | {lat:.3f} | {accepted} | `{run}` |".format(
                name=candidate["candidate_name"],
                val=candidate["validation"]["macro_f1"],
                test=candidate["test"]["macro_f1"],
                acc=candidate["test"]["accuracy"],
                lat=candidate["latency_ms_per_review"],
                accepted="yes" if candidate["passes_acceptance"] else "no",
                run=candidate["mlflow_run_id"],
            )
        )
    lines.extend(["", f"Selected candidate: `{selected['candidate_name']}`", ""])
    output_md.write_text("\n".join(lines), encoding="utf-8")


def train(
    train_path: Path = DATA_PROCESSED / "train.csv",
    validation_path: Path = DATA_PROCESSED / "validation.csv",
    test_path: Path = DATA_PROCESSED / "test.csv",
) -> dict[str, object]:
    stage_start = perf_counter()
    ensure_dirs()
    train_df = pd.read_csv(train_path)
    validation_df = pd.read_csv(validation_path)
    test_df = pd.read_csv(test_path)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "product-review-sentiment"))

    git_commit = git_commit_hash()
    data_version = dvc_data_version()
    candidates = [
        run_candidate(spec, train_df, validation_df, test_df, git_commit, data_version)
        for spec in candidate_specs()
    ]
    selected = select_best_candidate(candidates)
    selected_model: Pipeline = selected["model"]

    model_path = MODELS / "sentiment_model.joblib"
    metadata_path = MODELS / "model_metadata.json"
    importance_path = MODELS / "feature_importance.json"
    training_metrics_path = REPORTS / "training_metrics.json"

    joblib.dump(selected_model, model_path)
    feature_importance = extract_feature_importance(selected_model)
    write_json(importance_path, {"feature_importance": feature_importance})
    model_size = model_size_bytes(model_path)

    metadata = {
        "model_name": selected["candidate_name"],
        "model_version": "local-production",
        "mlflow_run_id": selected["mlflow_run_id"],
        "git_commit": git_commit,
        "data_version": data_version,
        "trained_at": utc_now(),
        "model_path": str(model_path),
        "labels": SENTIMENT_LABELS,
        "latency_ms_p50_estimate": float(selected["latency_ms_per_review"]),
        "model_size_bytes": model_size,
        "selection_rule": "highest validation macro F1 among candidates passing test macro F1 and latency gates",
    }
    write_json(metadata_path, metadata)

    result = {
        "stage": "train_model",
        "status": "success",
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "feature_importance_path": str(importance_path),
        "model_comparison_path": str(REPORTS / "model_comparison.json"),
        "selected_candidate": serializable_candidate(selected),
        "validation": selected["validation"],
        "test": selected["test"],
        "latency_ms_per_review": float(selected["latency_ms_per_review"]),
        "model_size_bytes": model_size,
        "metadata": metadata,
        "candidates": [serializable_candidate(candidate) for candidate in candidates],
    }
    write_json(training_metrics_path, result)
    write_model_comparison(candidates, selected)

    with mlflow.start_run(run_id=str(selected["mlflow_run_id"])):
        mlflow.set_tags({"stage": "selected_model", "selected_for_deployment": "true"})
        mlflow.log_metrics({"model_size_bytes": float(model_size)})
        mlflow.log_artifact(str(training_metrics_path))
        mlflow.log_artifact(str(importance_path))
        mlflow.log_artifact(str(REPORTS / "model_comparison.json"))
        mlflow.log_artifact(str(REPORTS / "model_comparison.md"))
        mlflow.sklearn.log_model(
            selected_model,
            artifact_path="model",
            registered_model_name="ProductReviewSentimentModel",
        )

    record_stage_performance(
        "train_and_compare_models",
        perf_counter() - stage_start,
        rows_processed=len(train_df) + len(validation_df) + len(test_df),
        extra={
            "candidate_models": len(candidates),
            "selected_candidate": str(selected["candidate_name"]),
            "accepted_candidates": len([candidate for candidate in candidates if candidate["passes_acceptance"]]),
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and select sentiment classification model.")
    parser.add_argument("--train", type=Path, default=DATA_PROCESSED / "train.csv")
    parser.add_argument("--validation", type=Path, default=DATA_PROCESSED / "validation.csv")
    parser.add_argument("--test", type=Path, default=DATA_PROCESSED / "test.csv")
    args = parser.parse_args()
    train(args.train, args.validation, args.test)


if __name__ == "__main__":
    main()
