from __future__ import annotations

import argparse
from itertools import cycle

import pandas as pd

from ml.common import DATA_RAW, REPORTS, ensure_dirs, rating_to_sentiment, utc_now, write_json

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


def ingest(rows_per_class: int = 30) -> pd.DataFrame:
    ensure_dirs()
    df = build_seed_dataset(rows_per_class=rows_per_class)
    df["sentiment"] = df["rating"].map(rating_to_sentiment)
    output_path = DATA_RAW / "reviews.csv"
    df.to_csv(output_path, index=False)
    write_json(
        REPORTS / "ingestion_report.json",
        {
            "stage": "ingest_data",
            "status": "success",
            "rows": int(len(df)),
            "output_path": str(output_path),
            "source": "local seed dataset compatible with public Amazon review schema",
            "ingested_at": utc_now(),
        },
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest e-commerce review data.")
    parser.add_argument("--rows-per-class", type=int, default=30)
    args = parser.parse_args()
    ingest(rows_per_class=args.rows_per_class)


if __name__ == "__main__":
    main()

