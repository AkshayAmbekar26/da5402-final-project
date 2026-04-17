from __future__ import annotations

import argparse
from itertools import cycle
from pathlib import Path
from typing import Any

import pandas as pd

from ml.common import (
    DATA_RAW,
    REPORTS,
    ROOT,
    ensure_dirs,
    rating_to_sentiment,
    read_json,
    utc_now,
    write_json,
)

POSITIVE_REVIEWS = [
    "Excellent quality and the delivery was faster than expected.",
    "The battery lasts all day and the design feels premium.",
    "Great value for money, I would happily buy this again.",
    "Customer support solved my issue quickly and politely.",
    "The packaging was neat and the product works perfectly.",
    "Very comfortable to use and the build quality is impressive.",
    "The item matched the description and exceeded my expectations.",
    "Setup was simple and performance has been reliable.",
    "I love the clean finish and the product feels durable.",
    "Fast shipping, useful features, and a smooth experience overall.",
]

NEUTRAL_REVIEWS = [
    "The product is okay, neither amazing nor disappointing.",
    "It works as described but there is nothing special about it.",
    "Average quality for the price and delivery took a few days.",
    "The design is fine, although I expected slightly better materials.",
    "Useful for basic tasks, but advanced users may want more.",
    "Packaging was acceptable and the item arrived without damage.",
    "The product does the job, but I am not especially impressed.",
    "Some features are good while others feel unfinished.",
    "It is a reasonable purchase if expectations are moderate.",
    "The color and size are accurate, but performance is just average.",
]

NEGATIVE_REVIEWS = [
    "The product stopped working after two days and support was slow.",
    "Poor quality materials and the item arrived scratched.",
    "Very disappointing purchase, not worth the money.",
    "The delivery was late and the package was damaged.",
    "Battery life is terrible and the device heats up quickly.",
    "The description was misleading and the product feels cheap.",
    "I received the wrong item and replacement took too long.",
    "It broke during normal use and the warranty process was painful.",
    "The app crashes often and the setup instructions are confusing.",
    "Low performance, bad packaging, and overall frustrating experience.",
]


def build_seed_dataset(rows_per_class: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    templates = [
        (cycle(NEGATIVE_REVIEWS), 1, "negative"),
        (cycle(NEUTRAL_REVIEWS), 3, "neutral"),
        (cycle(POSITIVE_REVIEWS), 5, "positive"),
    ]
    for review_iter, rating, sentiment in templates:
        for idx in range(rows_per_class):
            rows.append(
                {
                    "review_id": f"{sentiment}-{idx:04d}",
                    "review_text": next(review_iter),
                    "rating": rating,
                    "sentiment": sentiment,
                    "source": "local_seed_ecommerce_reviews",
                    "ingested_at": utc_now(),
                }
            )
    return pd.DataFrame(rows).sample(frac=1.0, random_state=42).reset_index(drop=True)


def label_to_rating(label: object) -> int:
    return int(label) + 1


def canonicalize_huggingface_row(row: dict[str, Any]) -> dict[str, object]:
    rating = label_to_rating(row["label"])
    review_text = str(row["text"]).strip()
    return {
        "review_id": str(row["id"]),
        "review_text": review_text,
        "rating": rating,
        "sentiment": rating_to_sentiment(rating),
        "source": "SetFit/amazon_reviews_multi_en",
        "ingested_at": utc_now(),
    }


def load_huggingface_reviews(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, object]]:
    from datasets import load_dataset

    dataset_name = str(config["dataset_name"])
    split = str(config.get("dataset_split", "train"))
    max_rows_total = int(config.get("max_rows_total", 9000))
    allowed_ratings = {int(rating) for rating in config.get("allowed_ratings", [1, 3, 5])}
    target_per_sentiment = max(1, max_rows_total // 3)
    min_text_length = int(config.get("min_text_length", 20))
    max_text_length = int(config.get("max_text_length", 3000))

    dataset = load_dataset(dataset_name, split=split, streaming=True)
    buckets: dict[str, list[dict[str, object]]] = {"negative": [], "neutral": [], "positive": []}
    source_rows_seen = 0
    filtered_rows = 0

    for row in dataset:
        source_rows_seen += 1
        try:
            record = canonicalize_huggingface_row(row)
        except (KeyError, TypeError, ValueError):
            filtered_rows += 1
            continue

        text_length = len(str(record["review_text"]))
        rating = int(record["rating"])
        sentiment = str(record["sentiment"])
        if rating not in allowed_ratings or text_length < min_text_length or text_length > max_text_length:
            filtered_rows += 1
            continue
        if len(buckets[sentiment]) < target_per_sentiment:
            buckets[sentiment].append(record)
        if all(len(records) >= target_per_sentiment for records in buckets.values()):
            break

    rows = [record for records in buckets.values() for record in records]
    if not rows:
        raise RuntimeError(f"No usable rows were loaded from {dataset_name}.")

    df = pd.DataFrame(rows).sample(frac=1.0, random_state=int(config.get("random_seed", 42))).reset_index(drop=True)
    metadata = {
        "dataset_name": dataset_name,
        "dataset_split": split,
        "source_rows_seen": source_rows_seen,
        "rows_filtered_during_ingestion": filtered_rows,
        "target_rows_per_sentiment": target_per_sentiment,
        "allowed_ratings": sorted(allowed_ratings),
        "source_homepage": config.get("source_homepage"),
    }
    return df, metadata


def load_data_config(config_path: Path) -> dict[str, Any]:
    if config_path.exists():
        return read_json(config_path)
    return {
        "dataset_name": "local_seed_ecommerce_reviews",
        "max_rows_total": 900,
        "fallback_to_seed_data": True,
        "seed_rows_per_class": 300,
        "random_seed": 42,
    }


def ingest(config_path: Path = ROOT / "configs" / "data_config.json") -> pd.DataFrame:
    ensure_dirs()
    config = load_data_config(config_path)
    fallback_used = False
    ingestion_error = None
    try:
        df, source_metadata = load_huggingface_reviews(config)
    except Exception as exc:
        if not bool(config.get("fallback_to_seed_data", True)):
            raise
        fallback_used = True
        ingestion_error = str(exc)
        df = build_seed_dataset(rows_per_class=int(config.get("seed_rows_per_class", 300)))
        source_metadata = {
            "dataset_name": "local_seed_ecommerce_reviews",
            "dataset_split": "generated",
            "source_rows_seen": int(len(df)),
            "rows_filtered_during_ingestion": 0,
            "source_homepage": "local fallback",
        }

    df["sentiment"] = df["rating"].map(rating_to_sentiment)
    output_path = DATA_RAW / "reviews.csv"
    df.to_csv(output_path, index=False)
    write_json(
        REPORTS / "ingestion_report.json",
        {
            "stage": "ingest_data",
            "status": "success",
            "dataset_name": source_metadata["dataset_name"],
            "dataset_split": source_metadata["dataset_split"],
            "source_homepage": source_metadata.get("source_homepage"),
            "source_rows_seen": int(source_metadata["source_rows_seen"]),
            "rows_after_schema_mapping": int(len(df)),
            "rows_after_quality_filters": int(len(df)),
            "rows": int(len(df)),
            "class_distribution": {
                str(key): int(value) for key, value in df["sentiment"].value_counts().to_dict().items()
            },
            "rating_distribution": {
                str(key): int(value) for key, value in df["rating"].value_counts().sort_index().to_dict().items()
            },
            "fallback_used": fallback_used,
            "fallback_error": ingestion_error,
            "output_path": str(output_path),
            "source": source_metadata["dataset_name"],
            "ingested_at": utc_now(),
        },
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest e-commerce review data.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "data_config.json")
    args = parser.parse_args()
    ingest(config_path=args.config)


if __name__ == "__main__":
    main()
