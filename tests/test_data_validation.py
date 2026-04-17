from __future__ import annotations

import pandas as pd

from ml.common import rating_to_sentiment
from ml.validation.validate_data import validate_dataframe


def test_rating_to_sentiment_mapping() -> None:
    assert rating_to_sentiment(1) == "negative"
    assert rating_to_sentiment(2) == "negative"
    assert rating_to_sentiment(3) == "neutral"
    assert rating_to_sentiment(4) == "positive"
    assert rating_to_sentiment(5) == "positive"


def test_validate_dataframe_accepts_valid_schema() -> None:
    df = pd.DataFrame(
        [
            {
                "review_id": "r1",
                "review_text": "Great product with fast delivery",
                "rating": 5,
                "sentiment": "positive",
                "source": "unit",
                "ingested_at": "2026-04-17T00:00:00Z",
            },
            {
                "review_id": "r2",
                "review_text": "Average product with acceptable delivery",
                "rating": 3,
                "sentiment": "neutral",
                "source": "unit",
                "ingested_at": "2026-04-17T00:00:00Z",
            },
            {
                "review_id": "r3",
                "review_text": "Poor product with damaged packaging",
                "rating": 1,
                "sentiment": "negative",
                "source": "unit",
                "ingested_at": "2026-04-17T00:00:00Z",
            },
        ]
    )
    report = validate_dataframe(df)
    assert report["status"] == "success"
    assert report["rows"] == 3


def test_validate_dataframe_rejects_bad_rating() -> None:
    df = pd.DataFrame(
        [
            {
                "review_id": "r1",
                "review_text": "Great product",
                "rating": 9,
                "sentiment": "positive",
                "source": "unit",
                "ingested_at": "2026-04-17T00:00:00Z",
            }
        ]
    )
    report = validate_dataframe(df)
    assert report["status"] == "failed"
    assert report["invalid_ratings"] == 1


def test_validate_dataframe_warns_about_duplicate_text() -> None:
    df = pd.DataFrame(
        [
            {
                "review_id": "r1",
                "review_text": "Great product with fast delivery",
                "rating": 5,
                "sentiment": "positive",
                "source": "unit",
                "ingested_at": "2026-04-17T00:00:00Z",
            },
            {
                "review_id": "r2",
                "review_text": "Great product with fast delivery",
                "rating": 5,
                "sentiment": "positive",
                "source": "unit",
                "ingested_at": "2026-04-17T00:00:00Z",
            },
            {
                "review_id": "r3",
                "review_text": "Average product and acceptable delivery",
                "rating": 3,
                "sentiment": "neutral",
                "source": "unit",
                "ingested_at": "2026-04-17T00:00:00Z",
            },
            {
                "review_id": "r4",
                "review_text": "Poor product and damaged packaging",
                "rating": 1,
                "sentiment": "negative",
                "source": "unit",
                "ingested_at": "2026-04-17T00:00:00Z",
            },
        ]
    )
    report = validate_dataframe(df)
    assert report["status"] == "success"
    assert report["duplicate_review_text"] == 1
    assert report["warnings"]
