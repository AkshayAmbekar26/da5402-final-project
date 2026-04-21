from __future__ import annotations

import logging
import os
import smtplib
import sys
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from airflow.decorators import task
from airflow.exceptions import AirflowException
from airflow.hooks.base import BaseHook
from airflow.sensors.filesystem import FileSensor

from airflow import DAG

PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/opt/airflow/project"))
sys.path.insert(0, str(PROJECT_DIR))

from ml.orchestration.batch_ops import (  # noqa: E402
    BatchPaths,
    archive_batch,
    connect_ops_db,
    ensure_batch_dirs,
    finalize_batch,
    initialize_ops_db,
    prepare_batch_file,
    process_chunk,
    quarantine_batch,
    select_oldest_csv,
)

LOGGER = logging.getLogger(__name__)

INCOMING_DIR = PROJECT_DIR / "data" / "incoming"
ARCHIVE_DIR = PROJECT_DIR / "data" / "archive"
QUARANTINE_DIR = PROJECT_DIR / "data" / "quarantine"
INTERIM_DIR = PROJECT_DIR / "data" / "interim" / "batches"
OPS_DB_PATH = PROJECT_DIR / "data" / "ops" / "sentiment_pipeline_ops.db"
REPORT_PATH = PROJECT_DIR / "reports" / "batch_pipeline_report.json"
PATHS = BatchPaths(
    incoming_dir=INCOMING_DIR,
    archive_dir=ARCHIVE_DIR,
    quarantine_dir=QUARANTINE_DIR,
    interim_dir=INTERIM_DIR,
    ops_db_path=OPS_DB_PATH,
    report_path=REPORT_PATH,
)


def _get_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _schedule_interval(name: str, default: str) -> str | None:
    value = os.getenv(name, default).strip()
    return None if value.lower() in {"", "none", "manual", "off"} else value


def _alert_recipients() -> tuple[str, ...]:
    raw = os.getenv("SENTIMENT_PIPELINE_ALERT_EMAILS") or os.getenv("ALERT_EMAIL_TO", "")
    return tuple(email.strip() for email in raw.split(",") if email.strip())


def _smtp_config() -> dict[str, object]:
    try:
        connection = BaseHook.get_connection(os.getenv("SENTIMENT_PIPELINE_SMTP_CONN_ID", "smtp_default"))
        return {
            "host": connection.host,
            "port": int(connection.port or 587),
            "username": connection.login,
            "password": connection.password,
            "from_email": connection.extra_dejson.get("from_email") or connection.login,
            "starttls": str(connection.extra_dejson.get("disable_tls", "false")).lower() != "true",
        }
    except Exception:
        smarthost = os.getenv("ALERT_SMTP_SMARTHOST", "")
        host, _, port = smarthost.partition(":")
        return {
            "host": host,
            "port": int(port or 587),
            "username": os.getenv("ALERT_SMTP_AUTH_USERNAME"),
            "password": os.getenv("ALERT_SMTP_AUTH_PASSWORD"),
            "from_email": os.getenv("ALERT_SMTP_FROM") or os.getenv("ALERT_SMTP_AUTH_USERNAME"),
            "starttls": True,
        }


def send_pipeline_email(subject: str, html_body: str) -> None:
    recipients = _alert_recipients()
    if not recipients:
        LOGGER.info("No pipeline alert recipients configured; skipping email: %s", subject)
        return

    config = _smtp_config()
    if not config["host"] or not config["username"] or not config["password"] or not config["from_email"]:
        LOGGER.warning("SMTP settings are incomplete; skipping email: %s", subject)
        return

    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = str(config["from_email"])
    message["To"] = ", ".join(recipients)
    message.attach(MIMEText(html_body, "html", "utf-8"))

    with smtplib.SMTP(str(config["host"]), int(config["port"]), timeout=30) as smtp:
        if config["starttls"]:
            smtp.starttls()
        smtp.login(str(config["username"]), str(config["password"]))
        smtp.sendmail(str(config["from_email"]), list(recipients), message.as_string())


def send_dry_pipeline_alert(context: dict) -> None:
    subject = "Sentiment Batch Pipeline Alert | No Incoming CSV"
    html_body = (
        "<html><body>"
        "<h2>No Incoming Review Batch Detected</h2>"
        f"<p>The FileSensor timed out while watching <code>{INCOMING_DIR}</code>.</p>"
        f"<p>DAG: <strong>{context.get('dag').dag_id if context.get('dag') else 'unknown'}</strong></p>"
        "<p>This usually means the upstream batch feed did not deliver a new review CSV.</p>"
        "</body></html>"
    )
    send_pipeline_email(subject, html_body)


def send_batch_failure_alert(source_name: str, reason: str, quarantined_path: str) -> None:
    subject = f"Sentiment Batch Pipeline Alert | Quarantined {source_name}"
    html_body = (
        "<html><body>"
        "<h2>Malformed Review Batch Quarantined</h2>"
        f"<p><strong>Source file:</strong> {source_name}</p>"
        f"<p><strong>Reason:</strong> {reason}</p>"
        f"<p><strong>Quarantine path:</strong> <code>{quarantined_path}</code></p>"
        "<p>The training pipeline was protected from bad input data.</p>"
        "</body></html>"
    )
    send_pipeline_email(subject, html_body)


def send_summary_email(summary: dict[str, object]) -> None:
    subject = f"Sentiment Batch Pipeline Summary | {summary['source_name']}"
    html_body = (
        "<html><body>"
        "<h2>Review Batch Processed</h2>"
        f"<p><strong>Source file:</strong> {summary['source_name']}</p>"
        f"<p><strong>Status:</strong> {summary['status']}</p>"
        f"<p><strong>Rows processed:</strong> {summary['rows_processed']}</p>"
        f"<p><strong>Completed chunks:</strong> {summary['completed_chunks']}</p>"
        f"<p><strong>Failed chunks:</strong> {summary['failed_chunks']}</p>"
        "</body></html>"
    )
    send_pipeline_email(subject, html_body)


default_args = {
    "owner": "mlops-course-project",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(seconds=_get_int("SENTIMENT_BATCH_RETRY_DELAY_SECONDS", 30)),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(seconds=_get_int("SENTIMENT_BATCH_MAX_RETRY_DELAY_SECONDS", 300)),
}


with DAG(
    dag_id="sentiment_batch_ingestion_pipeline",
    description="Operational incoming-review batch pipeline with sensors, pools, retries, and alerts.",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=_schedule_interval("SENTIMENT_BATCH_SCHEDULE", "*/15 * * * *"),
    catchup=False,
    max_active_runs=1,
    tags=["sentiment", "mlops", "airflow", "batch-monitoring"],
) as dag:
    wait_for_review_batch = FileSensor(
        task_id="wait_for_review_batch",
        filepath="*.csv",
        fs_conn_id=os.getenv("SENTIMENT_BATCH_FS_CONN_ID", "fs_incoming_reviews"),
        poke_interval=_get_int("SENTIMENT_BATCH_SENSOR_POKE_INTERVAL", 60),
        timeout=_get_int("SENTIMENT_BATCH_DRY_TIMEOUT_SECONDS", 12 * 60 * 60),
        mode="reschedule",
        on_failure_callback=send_dry_pipeline_alert,
    )

    @task(task_id="ensure_batch_runtime_ready")
    def ensure_batch_runtime_ready() -> None:
        ensure_batch_dirs(PATHS)
        with connect_ops_db(PATHS.ops_db_path) as connection:
            initialize_ops_db(connection)

    @task(task_id="select_next_review_batch")
    def select_next_review_batch() -> str:
        return str(select_oldest_csv(PATHS.incoming_dir))

    @task(task_id="prepare_review_batch", retries=0)
    def prepare_review_batch(source_path: str) -> dict[str, object]:
        """Validate one incoming CSV; malformed files are quarantined once, not retried."""
        path = Path(source_path)
        try:
            return prepare_batch_file(
                path,
                chunk_size=_get_int("SENTIMENT_BATCH_CHUNK_SIZE", 500),
                paths=PATHS,
            )
        except Exception as exc:
            reason = str(exc)
            quarantined_path = quarantine_batch(path, reason, paths=PATHS)
            send_batch_failure_alert(path.name, reason, quarantined_path)
            raise AirflowException(f"Incoming review batch was quarantined: {reason}") from exc

    @task(task_id="build_review_chunk_targets")
    def build_review_chunk_targets(prepared_batch: dict[str, object]) -> list[dict[str, object]]:
        return list(prepared_batch["chunks"])

    @task(
        task_id="process_review_chunk",
        retries=_get_int("SENTIMENT_BATCH_CHUNK_RETRIES", 3),
        retry_delay=timedelta(seconds=_get_int("SENTIMENT_BATCH_RETRY_DELAY_SECONDS", 30)),
        retry_exponential_backoff=True,
        max_retry_delay=timedelta(seconds=_get_int("SENTIMENT_BATCH_MAX_RETRY_DELAY_SECONDS", 300)),
        max_active_tis_per_dag=_get_int("SENTIMENT_BATCH_MAX_ACTIVE_CHUNKS", 3),
        pool=os.getenv("SENTIMENT_BATCH_POOL", "sentiment_batch_pool"),
    )
    def process_review_chunk(target: dict[str, object]) -> dict[str, object]:
        """Process chunks under pool/concurrency limits so large batches stay controlled."""
        return process_chunk(target, paths=PATHS)

    @task(task_id="finalize_review_batch")
    def finalize_review_batch(
        prepared_batch: dict[str, object],
        chunk_results: list[dict[str, object]],
    ) -> dict[str, object]:
        return finalize_batch(prepared_batch, chunk_results, paths=PATHS)

    @task(task_id="maybe_send_batch_summary")
    def maybe_send_batch_summary(summary: dict[str, object]) -> dict[str, object]:
        """Send email only when the batch is large enough or has failures worth attention."""
        threshold = _get_int("SENTIMENT_BATCH_SUMMARY_ROW_THRESHOLD", 1000)
        if int(summary["rows_processed"]) >= threshold or int(summary["failed_chunks"]) > 0:
            send_summary_email(summary)
            return {**summary, "summary_email_sent": True}
        return {**summary, "summary_email_sent": False}

    @task(task_id="archive_review_batch")
    def archive_review_batch(prepared_batch: dict[str, object]) -> str:
        return archive_batch(prepared_batch, paths=PATHS)

    runtime_ready = ensure_batch_runtime_ready()
    selected_batch = select_next_review_batch()
    prepared_batch = prepare_review_batch(selected_batch)
    chunk_targets = build_review_chunk_targets(prepared_batch)
    chunk_results = process_review_chunk.expand(target=chunk_targets)
    batch_summary = finalize_review_batch(prepared_batch, chunk_results)
    summary_email = maybe_send_batch_summary(batch_summary)
    archived_batch = archive_review_batch(prepared_batch)

    wait_for_review_batch >> runtime_ready >> selected_batch >> prepared_batch
    prepared_batch >> chunk_targets >> chunk_results >> batch_summary
    batch_summary >> summary_email
    batch_summary >> archived_batch
