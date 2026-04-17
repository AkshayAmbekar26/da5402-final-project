from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.preprocessing.preprocess import preprocess


def test_preprocess_writes_splits_and_rejected_audit(tmp_path: Path) -> None:
    rows = []
    for idx in range(6):
        rows.append(
            {
                "review_id": f"neg-{idx}",
                "review_text": f"Poor damaged product with slow support number {idx}",
                "rating": 1,
                "sentiment": "negative",
                "source": "unit",
                "ingested_at": "now",
            }
        )
        rows.append(
            {
                "review_id": f"neu-{idx}",
                "review_text": f"Average product that works as expected number {idx}",
                "rating": 3,
                "sentiment": "neutral",
                "source": "unit",
                "ingested_at": "now",
            }
        )
        rows.append(
            {
                "review_id": f"pos-{idx}",
                "review_text": f"Excellent reliable product with fast delivery number {idx}",
                "rating": 5,
                "sentiment": "positive",
                "source": "unit",
                "ingested_at": "now",
            }
        )
    rows.append(
        {
            "review_id": "short",
            "review_text": "bad",
            "rating": 1,
            "sentiment": "negative",
            "source": "unit",
            "ingested_at": "now",
        }
    )
    input_path = tmp_path / "reviews.csv"
    config_path = tmp_path / "data_config.json"
    processed_dir = tmp_path / "processed"
    rejected_path = tmp_path / "rejected.csv"
    report_path = tmp_path / "preprocessing_report.json"
    pd.DataFrame(rows).to_csv(input_path, index=False)
    config_path.write_text(
        '{"min_text_length":20,"max_text_length":3000,"validation_size":0.15,"test_size":0.15,"random_seed":42}',
        encoding="utf-8",
    )

    report = preprocess(input_path, config_path, processed_dir, rejected_path, report_path)

    assert report["rows_removed_too_short"] == 1
    assert report["splits"]["train"] > report["splits"]["validation"]
    assert (processed_dir / "train.csv").exists()
    assert (processed_dir / "validation.csv").exists()
    assert (processed_dir / "test.csv").exists()
    assert rejected_path.exists()
    assert report_path.exists()

