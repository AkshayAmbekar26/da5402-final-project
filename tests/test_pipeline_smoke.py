from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.data_ingestion.ingest import build_seed_dataset
from ml.features.compute_baseline import compute_baseline
from ml.monitoring.drift import distribution_delta
from ml.monitoring.prepare_feedback import prepare_feedback
from ml.preprocessing.merge_feedback import merge_feedback
from ml.preprocessing.preprocess import clean_text
from ml.training.train import write_model_optimization_report


def test_seed_dataset_is_balanced_and_uses_required_columns() -> None:
    df = build_seed_dataset(rows_per_class=2)
    assert set(df.columns) == {"review_id", "review_text", "rating", "sentiment", "source", "ingested_at"}
    assert df["sentiment"].value_counts().to_dict() == {"negative": 2, "neutral": 2, "positive": 2}


def test_clean_text_normalizes_whitespace() -> None:
    assert clean_text("  product   works\nwell  ") == "product works well"


def test_distribution_delta_zero_for_identical_distribution() -> None:
    assert distribution_delta({"positive": 2, "negative": 1}, {"positive": 2, "negative": 1}) == 0


def test_compute_baseline_writes_expected_statistics(tmp_path: Path) -> None:
    input_path = tmp_path / "train.csv"
    baseline_path = tmp_path / "feature_baseline.json"
    report_path = tmp_path / "feature_baseline_report.json"
    pd.DataFrame(
        [
            {"review_text": "excellent useful product", "sentiment": "positive", "rating": 5},
            {"review_text": "bad broken product", "sentiment": "negative", "rating": 1},
            {"review_text": "average product", "sentiment": "neutral", "rating": 3},
        ]
    ).to_csv(input_path, index=False)
    baseline = compute_baseline(input_path, baseline_path, report_path)
    assert baseline["rows"] == 3
    assert "text_length" in baseline
    assert "tfidf_feature_means" in baseline
    assert "tfidf_feature_variances" in baseline
    assert baseline["tfidf_feature_means"].keys() == baseline["tfidf_feature_variances"].keys()
    assert baseline_path.exists()
    assert report_path.exists()


def test_model_optimization_report_records_local_resource_strategy(tmp_path: Path) -> None:
    selected = {
        "candidate_name": "tfidf_logistic_tuned",
        "latency_ms_per_review": 0.08,
        "test": {"macro_f1": 0.77},
    }
    candidates = [
        selected,
        {"candidate_name": "count_naive_bayes", "latency_ms_per_review": 0.04},
    ]
    metadata = {
        "model_size_bytes": 5_242_880,
        "selection_rule": "highest validation macro F1 among accepted candidates",
    }
    report = write_model_optimization_report(
        selected,
        candidates,
        metadata,
        acceptance_latency_ms=200.0,
        output_path=tmp_path / "model_optimization_report.json",
    )
    assert report["resource_constraints"]["cloud_required"] is False
    assert report["resource_constraints"]["gpu_required"] is False
    assert report["quantization_or_pruning"]["applied"] is False
    assert report["latency"]["passes_target"] is True


def test_feedback_corrections_are_validated_and_merged_into_training_data(tmp_path: Path) -> None:
    feedback_path = tmp_path / "feedback.jsonl"
    validated_path = tmp_path / "validated_feedback.csv"
    feedback_report_path = tmp_path / "feedback_preparation_report.json"
    train_path = tmp_path / "train.csv"
    augmented_path = tmp_path / "train_augmented.csv"
    merge_report_path = tmp_path / "feedback_merge_report.json"

    feedback_path.write_text(
        "\n".join(
            [
                '{"review_text":"This product broke after one day and support ignored me","predicted_sentiment":"positive","actual_sentiment":"negative","source":"unit"}',
                '{"review_text":"This product is excellent","predicted_sentiment":"positive","actual_sentiment":"positive","source":"unit"}',
            ]
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "review_id": "base-1",
                "review_text": "Excellent useful product with fast delivery",
                "rating": 5,
                "sentiment": "positive",
                "source": "unit",
                "ingested_at": "2026-01-01T00:00:00+00:00",
            }
        ]
    ).to_csv(train_path, index=False)

    feedback_report = prepare_feedback(feedback_path, validated_path, feedback_report_path)
    merge_report = merge_feedback(train_path, validated_path, augmented_path, merge_report_path)
    augmented = pd.read_csv(augmented_path)

    assert feedback_report["valid_correction_rows"] == 1
    assert feedback_report["invalid_reasons"]["not_a_correction"] == 1
    assert merge_report["feedback_rows_used"] == 1
    assert len(augmented) == 2
    assert "user_feedback_correction" in set(augmented["source"])
