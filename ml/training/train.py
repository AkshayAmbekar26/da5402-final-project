from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import joblib
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from ml.common import (
    RANDOM_SEED,
    SENTIMENT_LABELS,
    dir_for,
    dvc_data_version,
    ensure_dirs,
    git_commit_hash,
    path_for,
    read_json,
    read_params,
    utc_now,
    write_json,
)
from ml.monitoring.performance import record_stage_performance
from ml.serving.pyfunc_model import SentimentPyfuncModel

ACCEPTANCE_TEST_MACRO_F1 = 0.75
ACCEPTANCE_LATENCY_MS = 200.0
DEFAULT_LATENCY_SAMPLE_SIZE = 50


@dataclass(frozen=True)
class CandidateSpec:
    """Typed view of one params.yaml model candidate."""

    name: str
    model_type: str
    vectorizer: str
    max_features: int
    ngram_range: tuple[int, int]
    min_df: int
    sublinear_tf: bool
    classifier: str
    classifier_params: dict[str, object]


DEFAULT_CANDIDATES: list[dict[str, object]] = [
    {
        "name": "tfidf_logistic_baseline",
        "model_type": "tfidf_logistic_regression",
        "vectorizer": "tfidf",
        "max_features": 2500,
        "ngram_range": [1, 2],
        "min_df": 2,
        "sublinear_tf": False,
        "classifier": "LogisticRegression",
        "classifier_params": {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"},
    },
    {
        "name": "tfidf_logistic_regularized",
        "model_type": "tfidf_logistic_regression",
        "vectorizer": "tfidf",
        "max_features": 30000,
        "ngram_range": [1, 2],
        "min_df": 2,
        "sublinear_tf": True,
        "classifier": "LogisticRegression",
        "classifier_params": {"C": 1.5, "max_iter": 2000, "class_weight": "balanced"},
    },
    {
        "name": "tfidf_logistic_tuned",
        "model_type": "tfidf_logistic_regression",
        "vectorizer": "tfidf",
        "max_features": 80000,
        "ngram_range": [1, 3],
        "min_df": 1,
        "sublinear_tf": True,
        "classifier": "LogisticRegression",
        "classifier_params": {"C": 3.0, "max_iter": 2500, "class_weight": "balanced"},
    },
    {
        "name": "tfidf_sgd_log_loss",
        "model_type": "tfidf_sgd_classifier",
        "vectorizer": "tfidf",
        "max_features": 80000,
        "ngram_range": [1, 3],
        "min_df": 2,
        "sublinear_tf": True,
        "classifier": "SGDClassifier",
        "classifier_params": {
            "alpha": 1e-5,
            "loss": "log_loss",
            "max_iter": 2000,
            "class_weight": "balanced",
        },
    },
    {
        "name": "count_naive_bayes",
        "model_type": "count_multinomial_naive_bayes",
        "vectorizer": "count",
        "max_features": 50000,
        "ngram_range": [1, 2],
        "min_df": 2,
        "sublinear_tf": False,
        "classifier": "MultinomialNB",
        "classifier_params": {"alpha": 0.5},
    },
]


def training_config() -> dict[str, object]:
    config: dict[str, object] = {
        "random_seed": RANDOM_SEED,
        "acceptance_test_macro_f1": ACCEPTANCE_TEST_MACRO_F1,
        "acceptance_latency_ms": ACCEPTANCE_LATENCY_MS,
        "latency_sample_size": DEFAULT_LATENCY_SAMPLE_SIZE,
        "candidates": DEFAULT_CANDIDATES,
    }
    params = read_params("training")
    config.update({key: value for key, value in params.items() if key != "candidates"})
    if params.get("candidates"):
        config["candidates"] = params["candidates"]
    return config


def candidate_specs(config: dict[str, object] | None = None) -> list[CandidateSpec]:
    """Validate params.yaml candidate definitions before any expensive training starts."""
    config = config or training_config()
    specs: list[CandidateSpec] = []
    raw_candidates = config.get("candidates", DEFAULT_CANDIDATES)
    if not isinstance(raw_candidates, list):
        raise ValueError("training.candidates must be a list in params.yaml")
    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, dict):
            raise ValueError("Each training candidate must be a mapping in params.yaml")
        ngram_range = raw_candidate.get("ngram_range", [1, 2])
        if not isinstance(ngram_range, list | tuple) or len(ngram_range) != 2:
            raise ValueError(f"Invalid ngram_range for candidate {raw_candidate.get('name')}")
        classifier_params = raw_candidate.get("classifier_params", {})
        specs.append(
            CandidateSpec(
                name=str(raw_candidate["name"]),
                model_type=str(raw_candidate["model_type"]),
                vectorizer=str(raw_candidate.get("vectorizer", "tfidf")),
                max_features=int(raw_candidate["max_features"]),
                ngram_range=(int(ngram_range[0]), int(ngram_range[1])),
                min_df=int(raw_candidate.get("min_df", 1)),
                sublinear_tf=bool(raw_candidate.get("sublinear_tf", False)),
                classifier=str(raw_candidate["classifier"]),
                classifier_params=dict(classifier_params) if isinstance(classifier_params, dict) else {},
            )
        )
    return specs


def build_candidate_model(spec: CandidateSpec, random_seed: int) -> Pipeline:
    """Build a sklearn pipeline from a candidate spec without fitting it yet."""
    if spec.vectorizer == "tfidf":
        vectorizer = TfidfVectorizer(
            max_features=spec.max_features,
            ngram_range=spec.ngram_range,
            min_df=spec.min_df,
            sublinear_tf=spec.sublinear_tf,
        )
    elif spec.vectorizer == "count":
        vectorizer = CountVectorizer(
            max_features=spec.max_features,
            ngram_range=spec.ngram_range,
            min_df=spec.min_df,
        )
    else:
        raise ValueError(f"Unsupported vectorizer: {spec.vectorizer}")

    if spec.classifier == "LogisticRegression":
        classifier = LogisticRegression(random_state=random_seed, **spec.classifier_params)
    elif spec.classifier == "SGDClassifier":
        classifier = SGDClassifier(random_state=random_seed, **spec.classifier_params)
    elif spec.classifier == "MultinomialNB":
        classifier = MultinomialNB(**spec.classifier_params)
    else:
        raise ValueError(f"Unsupported classifier: {spec.classifier}")
    return Pipeline(steps=[("features", vectorizer), ("classifier", classifier)])


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


def classifier_weight_matrix(classifier: object) -> np.ndarray:
    if hasattr(classifier, "coef_"):
        return classifier.coef_
    if hasattr(classifier, "feature_log_prob_"):
        return classifier.feature_log_prob_
    raise ValueError(f"Classifier does not expose feature weights: {type(classifier).__name__}")


def extract_feature_importance(model: Pipeline, top_k: int = 20) -> list[dict[str, object]]:
    """Extract top class-specific tokens for explainability and demo artifacts."""
    vectorizer = model.named_steps["features"]
    classifier = model.named_steps["classifier"]
    features = vectorizer.get_feature_names_out()
    weights = classifier_weight_matrix(classifier)
    importance: list[dict[str, object]] = []
    for class_index, label in enumerate(classifier.classes_):
        coefs = weights[class_index]
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


def latency_benchmark_ms(model: Pipeline, texts: pd.Series, sample_size: int = DEFAULT_LATENCY_SAMPLE_SIZE) -> float:
    sample = texts.head(min(sample_size, len(texts)))
    start = perf_counter()
    _ = model.predict(sample)
    return (perf_counter() - start) * 1000 / max(1, len(sample))


def model_size_bytes(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def build_mlflow_input_example() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "review_text": "Excellent quality and fast delivery. I would buy this product again.",
            }
        ]
    )


def build_mlflow_output_example(metadata: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sentiment": "positive",
                "confidence": 0.95,
                "class_probabilities_json": json.dumps(
                    {"negative": 0.02, "neutral": 0.03, "positive": 0.95},
                    sort_keys=True,
                ),
                "explanation_json": json.dumps(
                    [{"token": "excellent", "weight": 0.42}],
                    sort_keys=True,
                ),
                "model_version": str(metadata.get("model_version", "unknown")),
                "mlflow_run_id": str(metadata.get("mlflow_run_id", "unknown")),
                "latency_ms": 1.0,
            }
        ]
    )


def mlflow_model_requirements() -> list[str]:
    packages = ["mlflow", "pandas", "numpy", "scikit-learn", "joblib"]
    return [f"{package}=={version(package)}" for package in packages]


def log_and_export_pyfunc_model(
    model_path: Path,
    metadata_path: Path,
    feature_importance_path: Path,
    metadata: dict[str, object],
    registered_model_name: str,
) -> Path:
    """Log the selected model to MLflow and export a local pyfunc serving artifact."""
    input_example = build_mlflow_input_example()
    output_example = build_mlflow_output_example(metadata)
    signature = infer_signature(input_example, output_example)
    artifacts = {
        "model": str(model_path),
        "metadata": str(metadata_path),
        "feature_importance": str(feature_importance_path),
    }
    pyfunc_model = SentimentPyfuncModel()
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=pyfunc_model,
        artifacts=artifacts,
        code_paths=["ml"],
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        pip_requirements=mlflow_model_requirements(),
    )

    local_model_dir = path_for("mlflow_model_dir")
    if local_model_dir.exists():
        shutil.rmtree(local_model_dir)
    mlflow.pyfunc.save_model(
        path=str(local_model_dir),
        python_model=SentimentPyfuncModel(),
        artifacts=artifacts,
        code_paths=["ml"],
        signature=signature,
        input_example=input_example,
        pip_requirements=mlflow_model_requirements(),
    )
    return local_model_dir


def log_data_artifacts() -> None:
    for artifact_name in [
        "ingestion_report",
        "data_validation_report",
        "eda_report",
        "eda_markdown",
        "preprocessing_report",
        "feature_baseline_report",
    ]:
        artifact_path = path_for(artifact_name)
        if artifact_path.exists():
            mlflow.log_artifact(str(artifact_path), artifact_path="data_quality")
    figures_dir = dir_for("report_figures")
    if figures_dir.exists():
        for figure_path in figures_dir.glob("*.png"):
            mlflow.log_artifact(str(figure_path), artifact_path="eda_figures")


def dataset_params() -> dict[str, object]:
    params: dict[str, object] = {}
    ingestion_report_path = path_for("ingestion_report")
    preprocessing_report_path = path_for("preprocessing_report")
    feedback_preparation_path = path_for("feedback_preparation_report")
    feedback_merge_path = path_for("feedback_merge_report")
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
    if feedback_preparation_path.exists():
        feedback_preparation = read_json(feedback_preparation_path)
        params.update(
            {
                "feedback_raw_rows": feedback_preparation.get("raw_feedback_rows", 0),
                "feedback_valid_correction_rows": feedback_preparation.get("valid_correction_rows", 0),
            }
        )
    if feedback_merge_path.exists():
        feedback_merge = read_json(feedback_merge_path)
        params.update(
            {
                "feedback_rows_used": feedback_merge.get("feedback_rows_used", 0),
                "augmented_train_rows": feedback_merge.get("augmented_train_rows", 0),
            }
        )
    return params


def candidate_params(spec: CandidateSpec, random_seed: int) -> dict[str, object]:
    params = {
        "candidate_name": spec.name,
        "model_type": spec.model_type,
        "vectorizer": spec.vectorizer,
        "max_features": spec.max_features,
        "ngram_range": f"{spec.ngram_range[0]},{spec.ngram_range[1]}",
        "min_df": spec.min_df,
        "sublinear_tf": spec.sublinear_tf,
        "classifier": spec.classifier,
        "random_seed": random_seed,
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
    random_seed: int,
    acceptance_test_macro_f1: float,
    acceptance_latency_ms: float,
    latency_sample_size: int,
) -> dict[str, object]:
    """Train one candidate model and log its reproducibility evidence to MLflow."""
    model = build_candidate_model(spec, random_seed=random_seed)
    with mlflow.start_run(run_name=spec.name) as run:
        model.fit(train_df["review_text"], train_df["sentiment"])
        validation_pred = model.predict(validation_df["review_text"])
        test_pred = model.predict(test_df["review_text"])
        validation_metrics = evaluate_predictions(validation_df["sentiment"], validation_pred)
        test_metrics = evaluate_predictions(test_df["sentiment"], test_pred)
        latency_ms = latency_benchmark_ms(model, test_df["review_text"], sample_size=latency_sample_size)

        params = candidate_params(spec, random_seed=random_seed)
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
            test_metrics["macro_f1"] >= acceptance_test_macro_f1
            and latency_ms < acceptance_latency_ms
        ),
        "model": model,
    }


def select_best_candidate(candidates: list[dict[str, object]]) -> dict[str, object]:
    """Promote the best accepted model, falling back to the best candidate for inspection."""
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
    acceptance_test_macro_f1: float,
    acceptance_latency_ms: float,
    output_json: Path | None = None,
    output_md: Path | None = None,
    output_plot_csv: Path | None = None,
) -> None:
    """Write both machine-readable and human-readable candidate comparison artifacts."""
    output_json = output_json or path_for("model_comparison")
    output_md = output_md or path_for("model_comparison_markdown")
    output_plot_csv = output_plot_csv or path_for("model_comparison_plot")
    payload = {
        "stage": "model_selection",
        "status": "success",
        "selection_rule": (
            "Choose the candidate with highest validation macro F1 among models where "
            f"test macro F1 >= {acceptance_test_macro_f1} and latency < {acceptance_latency_ms} ms. "
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
    pd.DataFrame(
        [
            {
                "candidate_rank": index + 1,
                "candidate_name": str(candidate["candidate_name"]),
                "validation_macro_f1": float(candidate["validation"]["macro_f1"]),
                "test_macro_f1": float(candidate["test"]["macro_f1"]),
                "test_accuracy": float(candidate["test"]["accuracy"]),
                "latency_ms_per_review": float(candidate["latency_ms_per_review"]),
                "accepted": bool(candidate["passes_acceptance"]),
                "selected": candidate["candidate_name"] == selected["candidate_name"],
            }
            for index, candidate in enumerate(sorted(candidates, key=lambda item: str(item["candidate_name"])))
        ]
    ).to_csv(output_plot_csv, index=False)


def write_model_optimization_report(
    selected: dict[str, object],
    candidates: list[dict[str, object]],
    metadata: dict[str, object],
    acceptance_latency_ms: float,
    output_path: Path | None = None,
) -> dict[str, object]:
    output_path = output_path or path_for("model_optimization_report")
    selected_latency = float(selected["latency_ms_per_review"])
    model_size_bytes = int(metadata.get("model_size_bytes", 0) or 0)
    candidate_latencies = {
        str(candidate["candidate_name"]): float(candidate["latency_ms_per_review"])
        for candidate in candidates
    }
    fastest_candidate = min(candidate_latencies, key=candidate_latencies.get)
    report: dict[str, object] = {
        "stage": "model_optimization",
        "status": "success",
        "selected_model": str(selected["candidate_name"]),
        "model_family": "classical_sparse_text_classifier",
        "optimization_goal": "CPU-only local inference under the 200 ms business latency target.",
        "resource_constraints": {
            "cloud_required": False,
            "gpu_required": False,
            "serving_target": "local Docker Compose / on-prem CPU",
        },
        "chosen_strategy": [
            "Use TF-IDF sparse text features instead of transformer embeddings for low memory and fast CPU inference.",
            "Compare multiple lightweight sklearn candidates before promotion.",
            "Select only candidates passing macro F1 and latency gates.",
            "Persist a small joblib model artifact and MLflow pyfunc serving artifact.",
        ],
        "quantization_or_pruning": {
            "applied": False,
            "reason": (
                "The final model is a sparse linear sklearn pipeline, not a neural network. "
                "Quantization/pruning would add complexity without meaningful benefit for this artifact."
            ),
        },
        "latency": {
            "selected_latency_ms_per_review": selected_latency,
            "target_latency_ms_per_review": float(acceptance_latency_ms),
            "passes_target": bool(selected_latency < acceptance_latency_ms),
            "target_headroom_ratio": float(acceptance_latency_ms / max(selected_latency, 0.001)),
            "candidate_latency_ms_per_review": candidate_latencies,
            "fastest_candidate": fastest_candidate,
        },
        "model_size": {
            "bytes": model_size_bytes,
            "megabytes": float(model_size_bytes / (1024 * 1024)),
        },
        "quality_tradeoff": {
            "selected_test_macro_f1": float(selected["test"]["macro_f1"]),
            "selection_rule": str(metadata.get("selection_rule", "")),
        },
    }
    write_json(output_path, report)
    return report


def train(
    train_path: Path | None = None,
    validation_path: Path | None = None,
    test_path: Path | None = None,
) -> dict[str, object]:
    """Train all configured candidates, select one, and publish local-production artifacts."""
    stage_start = perf_counter()
    ensure_dirs()
    if train_path is None:
        augmented_path = path_for("train_augmented")
        train_path = augmented_path if augmented_path.exists() else path_for("train")
    validation_path = validation_path or path_for("validation")
    test_path = test_path or path_for("test")
    train_df = pd.read_csv(train_path)
    validation_df = pd.read_csv(validation_path)
    test_df = pd.read_csv(test_path)
    config = training_config()
    random_seed = int(config["random_seed"])
    acceptance_test_macro_f1 = float(config["acceptance_test_macro_f1"])
    acceptance_latency_ms = float(config["acceptance_latency_ms"])
    latency_sample_size = int(config["latency_sample_size"])

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "product-review-sentiment"))

    git_commit = git_commit_hash()
    data_version = dvc_data_version()
    candidates = [
        run_candidate(
            spec,
            train_df,
            validation_df,
            test_df,
            git_commit,
            data_version,
            random_seed=random_seed,
            acceptance_test_macro_f1=acceptance_test_macro_f1,
            acceptance_latency_ms=acceptance_latency_ms,
            latency_sample_size=latency_sample_size,
        )
        for spec in candidate_specs(config)
    ]
    selected = select_best_candidate(candidates)
    selected_model: Pipeline = selected["model"]

    model_path = path_for("sentiment_model")
    metadata_path = path_for("model_metadata")
    importance_path = path_for("feature_importance")
    training_metrics_path = path_for("training_metrics")
    optimization_report_path = path_for("model_optimization_report")
    registered_model_name = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "ProductReviewSentimentModel")

    joblib.dump(selected_model, model_path)
    feature_importance = extract_feature_importance(selected_model)
    write_json(importance_path, {"feature_importance": feature_importance})
    model_size = model_size_bytes(model_path)
    feedback_merge_path = path_for("feedback_merge_report")
    feedback_preparation_path = path_for("feedback_preparation_report")
    feedback_merge_report = read_json(feedback_merge_path) if feedback_merge_path.exists() else {}
    feedback_preparation_report = (
        read_json(feedback_preparation_path)
        if feedback_preparation_path.exists()
        else {}
    )

    metadata = {
        "model_name": selected["candidate_name"],
        "model_version": "local-production",
        "mlflow_run_id": selected["mlflow_run_id"],
        "git_commit": git_commit,
        "data_version": data_version,
        "trained_at": utc_now(),
        "model_path": str(model_path),
        "training_data_path": str(train_path),
        "mlflow_model_uri": f"runs:/{selected['mlflow_run_id']}/model",
        "mlflow_registered_model_name": registered_model_name,
        "mlflow_serving_artifact_path": str(path_for("mlflow_model_dir")),
        "labels": SENTIMENT_LABELS,
        "latency_ms_p50_estimate": float(selected["latency_ms_per_review"]),
        "model_size_bytes": model_size,
        "selection_rule": "highest validation macro F1 among candidates passing test macro F1 and latency gates",
        "feedback_rows_used": int(feedback_merge_report.get("feedback_rows_used", 0) or 0),
        "feedback_valid_correction_rows": int(feedback_preparation_report.get("valid_correction_rows", 0) or 0),
    }
    write_json(metadata_path, metadata)

    result = {
        "stage": "train_model",
        "status": "success",
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "feature_importance_path": str(importance_path),
        "model_comparison_path": str(path_for("model_comparison")),
        "selected_candidate": serializable_candidate(selected),
        "validation": selected["validation"],
        "test": selected["test"],
        "latency_ms_per_review": float(selected["latency_ms_per_review"]),
        "model_size_bytes": model_size,
        "training_data_path": str(train_path),
        "feedback": {
            "preparation": feedback_preparation_report,
            "merge": feedback_merge_report,
        },
        "metadata": metadata,
        "candidates": [serializable_candidate(candidate) for candidate in candidates],
    }
    write_json(training_metrics_path, result)
    write_model_comparison(candidates, selected, acceptance_test_macro_f1, acceptance_latency_ms)
    optimization_report = write_model_optimization_report(
        selected,
        candidates,
        metadata,
        acceptance_latency_ms,
        optimization_report_path,
    )
    result["model_optimization_report_path"] = str(optimization_report_path)
    result["resource_optimization"] = optimization_report
    write_json(training_metrics_path, result)

    with mlflow.start_run(run_id=str(selected["mlflow_run_id"])):
        mlflow.set_tags({"stage": "selected_model", "selected_for_deployment": "true"})
        mlflow.log_params(
            {
                "training_data_path": str(train_path),
                "feedback_rows_used": int(feedback_merge_report.get("feedback_rows_used", 0) or 0),
                "feedback_valid_correction_rows": int(feedback_preparation_report.get("valid_correction_rows", 0) or 0),
            }
        )
        mlflow.log_metrics({"model_size_bytes": float(model_size)})
        mlflow.log_artifact(str(training_metrics_path))
        mlflow.log_artifact(str(importance_path))
        mlflow.log_artifact(str(optimization_report_path))
        mlflow.log_artifact(str(path_for("model_comparison")))
        mlflow.log_artifact(str(path_for("model_comparison_markdown")))
        if feedback_preparation_path.exists():
            mlflow.log_artifact(str(feedback_preparation_path), artifact_path="feedback")
        if feedback_merge_path.exists():
            mlflow.log_artifact(str(feedback_merge_path), artifact_path="feedback")
        local_mlflow_model_dir = log_and_export_pyfunc_model(
            model_path=model_path,
            metadata_path=metadata_path,
            feature_importance_path=importance_path,
            metadata=metadata,
            registered_model_name=registered_model_name,
        )
        mlflow.log_param("mlflow_serving_artifact_path", str(local_mlflow_model_dir))

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
    parser.add_argument("--train", type=Path, default=None)
    parser.add_argument("--validation", type=Path, default=path_for("validation"))
    parser.add_argument("--test", type=Path, default=path_for("test"))
    args = parser.parse_args()
    train(args.train, args.validation, args.test)


if __name__ == "__main__":
    main()
