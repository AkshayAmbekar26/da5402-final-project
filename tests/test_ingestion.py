from __future__ import annotations

from ml.data_ingestion.ingest import canonicalize_huggingface_row, label_to_rating


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

