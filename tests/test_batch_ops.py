from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ml.orchestration.batch_ops import (
    BatchPaths,
    archive_batch,
    connect_ops_db,
    finalize_batch,
    initialize_ops_db,
    latest_batch_summary,
    normalize_incoming_dataframe,
    prepare_batch_file,
    process_chunk,
    quarantine_batch,
)


def _paths(tmp_path: Path) -> BatchPaths:
    return BatchPaths(
        incoming_dir=tmp_path / "incoming",
        archive_dir=tmp_path / "archive",
        quarantine_dir=tmp_path / "quarantine",
        interim_dir=tmp_path / "interim" / "batches",
        ops_db_path=tmp_path / "ops" / "pipeline.db",
        report_path=tmp_path / "reports" / "batch_pipeline_report.json",
    )


def test_normalize_incoming_dataframe_maps_rating_to_sentiment() -> None:
    df = pd.DataFrame(
        [
            {"review_text": "Excellent product and fast delivery", "rating": 5},
            {"review_text": "Terrible product and broken packaging", "rating": 1},
        ]
    )

    normalized, metadata = normalize_incoming_dataframe(df)

    assert normalized["sentiment"].tolist() == ["positive", "negative"]
    assert metadata["valid_rows"] == 2
    assert metadata["sentiment_distribution"] == {"negative": 1, "positive": 1}


def test_normalize_incoming_dataframe_rejects_missing_review_text() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        normalize_incoming_dataframe(pd.DataFrame([{"rating": 5}]))


def test_prepare_process_finalize_and_archive_batch(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.incoming_dir.mkdir(parents=True)
    batch_path = paths.incoming_dir / "reviews.csv"
    pd.DataFrame(
        [
            {"review_id": "r1", "review_text": "Excellent product and fast delivery", "rating": 5},
            {"review_id": "r2", "review_text": "Average product for the price", "rating": 3},
            {"review_id": "r3", "review_text": "Poor product and damaged packaging", "rating": 1},
        ]
    ).to_csv(batch_path, index=False)

    prepared = prepare_batch_file(batch_path, chunk_size=2, paths=paths)
    assert prepared["metadata"]["valid_rows"] == 3
    assert len(prepared["chunks"]) == 2

    results = [process_chunk(chunk, paths=paths) for chunk in prepared["chunks"]]
    summary = finalize_batch(prepared, results, paths=paths)
    archived_path = archive_batch(prepared, paths=paths)

    assert summary["rows_processed"] == 3
    assert summary["completed_chunks"] == 2
    assert Path(archived_path).exists()
    assert not batch_path.exists()

    latest = latest_batch_summary(paths)
    assert latest["status"] == "completed"
    assert latest["valid_rows"] == 3


def test_quarantine_batch_records_event(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.incoming_dir.mkdir(parents=True)
    bad_path = paths.incoming_dir / "bad.csv"
    bad_path.write_text("rating\n5\n", encoding="utf-8")

    quarantined_path = quarantine_batch(bad_path, "missing review_text", paths=paths)

    assert Path(quarantined_path).exists()
    assert not bad_path.exists()
    with connect_ops_db(paths.ops_db_path) as connection:
        initialize_ops_db(connection)
        event = connection.execute("SELECT event_type, severity FROM batch_events").fetchone()
    assert dict(event) == {"event_type": "batch_quarantined", "severity": "error"}
