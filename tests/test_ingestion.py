from __future__ import annotations

import pandas as pd

from ml.common import SENTIMENT_LABELS
from ml.data_ingestion.ingest import (
    build_seed_dataset,
    canonicalize_huggingface_row,
    label_to_rating,
    load_cached_public_reviews,
)


def test_label_to_rating_maps_zero_based_huggingface_label() -> None:
    assert label_to_rating(0) == 1
    assert label_to_rating(2) == 3
    assert label_to_rating(4) == 5


def test_canonicalize_huggingface_row_uses_project_schema() -> None:
    record = canonicalize_huggingface_row(
        {
            "id": "en_0000001",
            "text": "Excellent product and quick delivery.",
            "label": 4,
            "label_text": "5",
        }
    )
    assert record["review_id"] == "en_0000001"
    assert record["review_text"] == "Excellent product and quick delivery."
    assert record["rating"] == 5
    assert record["sentiment"] == "positive"
    assert record["source"] == "SetFit/amazon_reviews_multi_en"


def test_seed_dataset_rows_are_unique_after_normalization() -> None:
    df = build_seed_dataset(rows_per_class=25)
    normalized_text_count = df["review_text"].str.lower().str.strip().nunique()
    assert normalized_text_count == len(df)


def test_cached_public_reviews_are_reused_when_source_matches(monkeypatch, tmp_path) -> None:
    data_raw = tmp_path / "raw"
    data_raw.mkdir()
    monkeypatch.setattr("ml.data_ingestion.ingest.DATA_RAW", data_raw)
    pd.DataFrame(
        [
            {
                "review_id": "en_1",
                "review_text": "A realistic public review.",
                "rating": 5,
                "sentiment": SENTIMENT_LABELS[-1],
                "source": "SetFit/amazon_reviews_multi_en",
                "ingested_at": "2026-04-20T00:00:00+00:00",
            }
        ]
    ).to_csv(data_raw / "reviews.csv", index=False)

    cached = load_cached_public_reviews({"dataset_name": "SetFit/amazon_reviews_multi_en"})

    assert cached is not None
    df, metadata = cached
    assert len(df) == 1
    assert metadata["cache_path"].endswith("reviews.csv")
