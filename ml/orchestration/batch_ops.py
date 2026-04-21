from __future__ import annotations

import hashlib
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.common import ROOT, SENTIMENT_LABELS, rating_to_sentiment, utc_now, write_json

DATA_INCOMING = ROOT / "data" / "incoming"
DATA_ARCHIVE = ROOT / "data" / "archive"
DATA_QUARANTINE = ROOT / "data" / "quarantine"
DATA_BATCH_INTERIM = ROOT / "data" / "interim" / "batches"
DATA_OPS = ROOT / "data" / "ops"
BATCH_REPORT_PATH = ROOT / "reports" / "batch_pipeline_report.json"
OPS_DB_PATH = DATA_OPS / "sentiment_pipeline_ops.db"

REQUIRED_COLUMNS = {"review_text"}
OPTIONAL_COLUMNS = {"review_id", "rating", "sentiment", "source", "ingested_at"}


@dataclass(frozen=True)
class BatchPaths:
    """Filesystem and database locations used by the Airflow batch pipeline."""

    incoming_dir: Path = DATA_INCOMING
    archive_dir: Path = DATA_ARCHIVE
    quarantine_dir: Path = DATA_QUARANTINE
    interim_dir: Path = DATA_BATCH_INTERIM
    ops_db_path: Path = OPS_DB_PATH
    report_path: Path = BATCH_REPORT_PATH


DEFAULT_BATCH_PATHS = BatchPaths()


def ensure_batch_dirs(paths: BatchPaths = DEFAULT_BATCH_PATHS) -> None:
    for directory in [
        paths.incoming_dir,
        paths.archive_dir,
        paths.quarantine_dir,
        paths.interim_dir,
        paths.ops_db_path.parent,
        paths.report_path.parent,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def compute_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_oldest_csv(incoming_dir: Path = DATA_INCOMING) -> Path:
    candidates = sorted(incoming_dir.glob("*.csv"), key=lambda path: (path.stat().st_mtime, path.name))
    if not candidates:
        raise FileNotFoundError(f"No incoming review batch files were found in {incoming_dir}.")
    return candidates[0]


def connect_ops_db(db_path: Path = OPS_DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


def initialize_ops_db(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS batch_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_name TEXT NOT NULL,
            source_path TEXT NOT NULL,
            file_sha256 TEXT NOT NULL,
            status TEXT NOT NULL,
            total_rows INTEGER NOT NULL DEFAULT 0,
            valid_rows INTEGER NOT NULL DEFAULT 0,
            invalid_rows INTEGER NOT NULL DEFAULT 0,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            duplicate_review_ids INTEGER NOT NULL DEFAULT 0,
            sentiment_distribution_json TEXT NOT NULL DEFAULT '{}',
            failure_reason TEXT,
            created_at TEXT NOT NULL,
            completed_at TEXT,
            archived_path TEXT,
            UNIQUE(source_name, file_sha256)
        );

        CREATE TABLE IF NOT EXISTS batch_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_path TEXT NOT NULL,
            status TEXT NOT NULL,
            row_count INTEGER NOT NULL DEFAULT 0,
            sentiment_distribution_json TEXT NOT NULL DEFAULT '{}',
            processed_at TEXT,
            UNIQUE(batch_id, chunk_index),
            FOREIGN KEY(batch_id) REFERENCES batch_files(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS batch_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id INTEGER,
            event_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(batch_id) REFERENCES batch_files(id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS pipeline_state (
            state_key TEXT PRIMARY KEY,
            state_value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    connection.commit()


def normalize_incoming_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate and normalize an incoming batch before it can enter the pipeline."""
    original_columns = set(df.columns)
    missing_columns = sorted(REQUIRED_COLUMNS - original_columns)
    if missing_columns:
        raise ValueError(f"Incoming batch is missing required columns: {missing_columns}")

    normalized = df.copy()
    normalized["review_text"] = normalized["review_text"].astype(str).str.strip()
    empty_text = normalized["review_text"].eq("")
    if empty_text.any():
        raise ValueError(f"Incoming batch contains empty review_text values: {int(empty_text.sum())}")

    if "review_id" not in normalized.columns:
        normalized["review_id"] = [f"incoming-{index:06d}" for index in range(1, len(normalized) + 1)]
    normalized["review_id"] = normalized["review_id"].astype(str).str.strip()
    missing_ids = normalized["review_id"].eq("")
    if missing_ids.any():
        raise ValueError(f"Incoming batch contains empty review_id values: {int(missing_ids.sum())}")

    duplicate_review_ids = int(normalized["review_id"].duplicated().sum())

    if "rating" in normalized.columns:
        normalized["rating"] = pd.to_numeric(normalized["rating"], errors="coerce")
        invalid_rating = normalized["rating"].isna() | ~normalized["rating"].between(1, 5)
        if invalid_rating.any():
            raise ValueError(f"Incoming batch contains invalid rating values: {int(invalid_rating.sum())}")
        normalized["rating"] = normalized["rating"].astype(int)

    if "sentiment" in normalized.columns:
        normalized["sentiment"] = normalized["sentiment"].astype(str).str.strip().str.lower()
        invalid_sentiment = ~normalized["sentiment"].isin(SENTIMENT_LABELS)
        if invalid_sentiment.any():
            raise ValueError(f"Incoming batch contains invalid sentiment values: {int(invalid_sentiment.sum())}")
    elif "rating" in normalized.columns:
        normalized["sentiment"] = normalized["rating"].map(rating_to_sentiment)
    else:
        normalized["sentiment"] = "unknown"

    if "source" not in normalized.columns:
        normalized["source"] = "incoming_review_batch"
    if "ingested_at" not in normalized.columns:
        normalized["ingested_at"] = utc_now()

    allowed_columns = ["review_id", "review_text", "rating", "sentiment", "source", "ingested_at"]
    for column in allowed_columns:
        if column not in normalized.columns:
            normalized[column] = None

    normalized = normalized[allowed_columns]
    metadata = {
        "total_rows": int(len(df)),
        "valid_rows": int(len(normalized)),
        "invalid_rows": 0,
        "duplicate_review_ids": duplicate_review_ids,
        "sentiment_distribution": {
            str(key): int(value) for key, value in normalized["sentiment"].value_counts().sort_index().to_dict().items()
        },
    }
    return normalized, metadata


def start_batch_record(
    connection: sqlite3.Connection,
    *,
    source_name: str,
    source_path: str,
    file_sha256: str,
    metadata: dict[str, Any],
    chunk_count: int,
) -> int:
    """Create or reset the operational DB record for a source file and its chunks."""
    existing = connection.execute(
        "SELECT id FROM batch_files WHERE source_name = ? AND file_sha256 = ?",
        (source_name, file_sha256),
    ).fetchone()
    sentiment_distribution = metadata.get("sentiment_distribution", {})
    import json

    if existing:
        batch_id = int(existing["id"])
        connection.execute(
            """
            UPDATE batch_files
            SET source_path = ?,
                status = 'RUNNING',
                total_rows = ?,
                valid_rows = ?,
                invalid_rows = ?,
                chunk_count = ?,
                duplicate_review_ids = ?,
                sentiment_distribution_json = ?,
                failure_reason = NULL,
                completed_at = NULL,
                archived_path = NULL
            WHERE id = ?
            """,
            (
                source_path,
                int(metadata["total_rows"]),
                int(metadata["valid_rows"]),
                int(metadata["invalid_rows"]),
                chunk_count,
                int(metadata["duplicate_review_ids"]),
                json.dumps(sentiment_distribution, sort_keys=True),
                batch_id,
            ),
        )
        connection.execute("DELETE FROM batch_chunks WHERE batch_id = ?", (batch_id,))
        connection.commit()
        return batch_id

    cursor = connection.execute(
        """
        INSERT INTO batch_files (
            source_name,
            source_path,
            file_sha256,
            status,
            total_rows,
            valid_rows,
            invalid_rows,
            chunk_count,
            duplicate_review_ids,
            sentiment_distribution_json,
            created_at
        ) VALUES (?, ?, ?, 'RUNNING', ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            source_name,
            source_path,
            file_sha256,
            int(metadata["total_rows"]),
            int(metadata["valid_rows"]),
            int(metadata["invalid_rows"]),
            chunk_count,
            int(metadata["duplicate_review_ids"]),
            json.dumps(sentiment_distribution, sort_keys=True),
            utc_now(),
        ),
    )
    connection.commit()
    return int(cursor.lastrowid)


def record_batch_event(
    connection: sqlite3.Connection,
    *,
    event_type: str,
    severity: str,
    message: str,
    batch_id: int | None = None,
) -> None:
    """Persist an operational event so batch failures are visible after task retries."""
    connection.execute(
        """
        INSERT INTO batch_events (batch_id, event_type, severity, message, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (batch_id, event_type, severity, message, utc_now()),
    )
    connection.commit()


def prepare_batch_file(
    source_path: Path,
    *,
    chunk_size: int = 500,
    paths: BatchPaths = DEFAULT_BATCH_PATHS,
) -> dict[str, Any]:
    ensure_batch_dirs(paths)
    with connect_ops_db(paths.ops_db_path) as connection:
        initialize_ops_db(connection)

    df = pd.read_csv(source_path)
    normalized, metadata = normalize_incoming_dataframe(df)
    file_sha256 = compute_file_sha256(source_path)
    chunk_size = max(1, int(chunk_size))
    chunk_dir = paths.interim_dir / f"{source_path.stem}-{file_sha256[:12]}"
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[dict[str, Any]] = []
    for chunk_index, start in enumerate(range(0, len(normalized), chunk_size), start=1):
        chunk_df = normalized.iloc[start : start + chunk_size]
        chunk_path = chunk_dir / f"chunk_{chunk_index:03d}.csv"
        chunk_df.to_csv(chunk_path, index=False)
        chunks.append({"chunk_index": chunk_index, "chunk_path": str(chunk_path), "row_count": int(len(chunk_df))})

    with connect_ops_db(paths.ops_db_path) as connection:
        initialize_ops_db(connection)
        batch_id = start_batch_record(
            connection,
            source_name=source_path.name,
            source_path=str(source_path),
            file_sha256=file_sha256,
            metadata=metadata,
            chunk_count=len(chunks),
        )
        record_batch_event(
            connection,
            batch_id=batch_id,
            event_type="batch_prepared",
            severity="info",
            message=f"Prepared {source_path.name} into {len(chunks)} chunks.",
        )
        for chunk in chunks:
            connection.execute(
                """
                INSERT OR REPLACE INTO batch_chunks (
                    batch_id,
                    chunk_index,
                    chunk_path,
                    status,
                    row_count,
                    sentiment_distribution_json
                ) VALUES (?, ?, ?, 'PENDING', ?, '{}')
                """,
                (batch_id, chunk["chunk_index"], chunk["chunk_path"], chunk["row_count"]),
            )
        connection.commit()

    payload = {
        "batch_id": batch_id,
        "source_name": source_path.name,
        "source_path": str(source_path),
        "file_sha256": file_sha256,
        "metadata": metadata,
        "chunks": [{**chunk, "batch_id": batch_id} for chunk in chunks],
    }
    write_batch_report({"status": "running", **payload}, paths)
    return payload


def process_chunk(
    target: dict[str, Any],
    *,
    paths: BatchPaths = DEFAULT_BATCH_PATHS,
) -> dict[str, Any]:
    chunk_path = Path(str(target["chunk_path"]))
    chunk_df = pd.read_csv(chunk_path)
    sentiment_distribution = {
        str(key): int(value) for key, value in chunk_df["sentiment"].value_counts().sort_index().to_dict().items()
    }
    row_count = int(len(chunk_df))
    import json

    with connect_ops_db(paths.ops_db_path) as connection:
        initialize_ops_db(connection)
        connection.execute(
            """
            UPDATE batch_chunks
            SET status = 'COMPLETED',
                row_count = ?,
                sentiment_distribution_json = ?,
                processed_at = ?
            WHERE batch_id = ? AND chunk_index = ?
            """,
            (
                row_count,
                json.dumps(sentiment_distribution, sort_keys=True),
                utc_now(),
                int(target["batch_id"]),
                int(target["chunk_index"]),
            ),
        )
        connection.commit()

    return {
        "batch_id": int(target["batch_id"]),
        "chunk_index": int(target["chunk_index"]),
        "row_count": row_count,
        "sentiment_distribution": sentiment_distribution,
        "status": "completed",
    }


def finalize_batch(
    prepared_batch: dict[str, Any],
    chunk_results: list[dict[str, Any]],
    *,
    paths: BatchPaths = DEFAULT_BATCH_PATHS,
) -> dict[str, Any]:
    batch_id = int(prepared_batch["batch_id"])
    rows_processed = sum(int(result.get("row_count", 0)) for result in chunk_results)
    completed_chunks = sum(1 for result in chunk_results if result.get("status") == "completed")
    failed_chunks = len(chunk_results) - completed_chunks
    status = "COMPLETED" if failed_chunks == 0 else "COMPLETED_WITH_ERRORS"
    with connect_ops_db(paths.ops_db_path) as connection:
        initialize_ops_db(connection)
        connection.execute(
            """
            UPDATE batch_files
            SET status = ?, completed_at = ?
            WHERE id = ?
            """,
            (status, utc_now(), batch_id),
        )
        record_batch_event(
            connection,
            batch_id=batch_id,
            event_type="batch_finalized",
            severity="info" if failed_chunks == 0 else "warning",
            message=f"Batch finalized with {completed_chunks} completed chunks and {failed_chunks} failed chunks.",
        )

    payload = {
        "status": status.lower(),
        "batch_id": batch_id,
        "source_name": prepared_batch["source_name"],
        "source_path": prepared_batch["source_path"],
        "file_sha256": prepared_batch["file_sha256"],
        "rows_processed": rows_processed,
        "completed_chunks": completed_chunks,
        "failed_chunks": failed_chunks,
        "metadata": prepared_batch["metadata"],
        "completed_at": utc_now(),
    }
    write_batch_report(payload, paths)
    return payload


def archive_batch(
    prepared_batch: dict[str, Any],
    *,
    paths: BatchPaths = DEFAULT_BATCH_PATHS,
) -> str:
    ensure_batch_dirs(paths)
    source_path = Path(str(prepared_batch["source_path"]))
    archived_path = paths.archive_dir / f"{utc_now().replace(':', '').replace('+', 'Z')}_{source_path.name}"
    if source_path.exists():
        shutil.move(str(source_path), str(archived_path))
    with connect_ops_db(paths.ops_db_path) as connection:
        initialize_ops_db(connection)
        connection.execute(
            "UPDATE batch_files SET archived_path = ? WHERE id = ?",
            (str(archived_path), int(prepared_batch["batch_id"])),
        )
        record_batch_event(
            connection,
            batch_id=int(prepared_batch["batch_id"]),
            event_type="batch_archived",
            severity="info",
            message=f"Archived processed batch to {archived_path}.",
        )
    return str(archived_path)


def quarantine_batch(source_path: Path, reason: str, *, paths: BatchPaths = DEFAULT_BATCH_PATHS) -> str:
    ensure_batch_dirs(paths)
    quarantined_path = paths.quarantine_dir / f"{utc_now().replace(':', '').replace('+', 'Z')}_{source_path.name}"
    if source_path.exists():
        shutil.move(str(source_path), str(quarantined_path))
    with connect_ops_db(paths.ops_db_path) as connection:
        initialize_ops_db(connection)
        record_batch_event(
            connection,
            event_type="batch_quarantined",
            severity="error",
            message=f"Quarantined {source_path.name}: {reason}",
        )
    write_batch_report(
        {
            "status": "quarantined",
            "source_name": source_path.name,
            "source_path": str(source_path),
            "quarantined_path": str(quarantined_path),
            "failure_reason": reason,
            "completed_at": utc_now(),
        },
        paths,
    )
    return str(quarantined_path)


def write_batch_report(payload: dict[str, Any], paths: BatchPaths = DEFAULT_BATCH_PATHS) -> None:
    write_json(
        paths.report_path,
        {
            "stage": "sentiment_batch_ingestion_pipeline",
            "generated_at": utc_now(),
            **payload,
        },
    )


def latest_batch_summary(paths: BatchPaths = DEFAULT_BATCH_PATHS) -> dict[str, Any]:
    if not paths.ops_db_path.exists():
        return {"status": "not_available"}
    with connect_ops_db(paths.ops_db_path) as connection:
        initialize_ops_db(connection)
        batch = connection.execute(
            """
            SELECT *
            FROM batch_files
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
        if batch is None:
            return {"status": "not_available"}
        events = connection.execute(
            """
            SELECT event_type, severity, message, created_at
            FROM batch_events
            WHERE batch_id = ? OR batch_id IS NULL
            ORDER BY id DESC
            LIMIT 10
            """,
            (int(batch["id"]),),
        ).fetchall()
    return {
        "status": str(batch["status"]).lower(),
        "batch_id": int(batch["id"]),
        "source_name": batch["source_name"],
        "total_rows": int(batch["total_rows"]),
        "valid_rows": int(batch["valid_rows"]),
        "invalid_rows": int(batch["invalid_rows"]),
        "chunk_count": int(batch["chunk_count"]),
        "duplicate_review_ids": int(batch["duplicate_review_ids"]),
        "completed_at": batch["completed_at"],
        "archived_path": batch["archived_path"],
        "recent_events": [dict(event) for event in events],
    }
