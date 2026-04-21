#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/opt/airflow/project}"
AIRFLOW_ADMIN_USER="${AIRFLOW_ADMIN_USER:-admin}"
AIRFLOW_ADMIN_PASSWORD="${AIRFLOW_ADMIN_PASSWORD:-admin}"
AIRFLOW_ADMIN_EMAIL="${AIRFLOW_ADMIN_EMAIL:-admin@example.com}"
SENTIMENT_BATCH_POOL="${SENTIMENT_BATCH_POOL:-sentiment_batch_pool}"
SENTIMENT_BATCH_POOL_SLOTS="${SENTIMENT_BATCH_POOL_SLOTS:-3}"
SENTIMENT_TRAINING_POOL="${SENTIMENT_TRAINING_POOL:-sentiment_training_pool}"
SENTIMENT_TRAINING_POOL_SLOTS="${SENTIMENT_TRAINING_POOL_SLOTS:-1}"
FS_CONN_ID="${SENTIMENT_BATCH_FS_CONN_ID:-fs_incoming_reviews}"
INCOMING_DIR="${SENTIMENT_BATCH_INCOMING_DIR:-$PROJECT_DIR/data/incoming}"

mkdir -p \
  "$PROJECT_DIR/data/incoming" \
  "$PROJECT_DIR/data/archive" \
  "$PROJECT_DIR/data/quarantine" \
  "$PROJECT_DIR/data/interim/batches" \
  "$PROJECT_DIR/data/ops" \
  "$PROJECT_DIR/reports"

airflow db migrate

AUTH_MANAGER="$(airflow config get-value core auth_manager 2>/dev/null || true)"
if [[ "$AUTH_MANAGER" == *"simple_auth_manager"* ]]; then
  echo "Airflow SimpleAuthManager is active; demo login is configured through AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_USERS."
else
  airflow users create \
    --username "$AIRFLOW_ADMIN_USER" \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email "$AIRFLOW_ADMIN_EMAIL" \
    --password "$AIRFLOW_ADMIN_PASSWORD" || true
fi

airflow pools set "$SENTIMENT_BATCH_POOL" "$SENTIMENT_BATCH_POOL_SLOTS" "Throttle incoming review batch chunk processing"
airflow pools set "$SENTIMENT_TRAINING_POOL" "$SENTIMENT_TRAINING_POOL_SLOTS" "Throttle local model training work"

python - <<'PY'
from __future__ import annotations

import os

from airflow.settings import Session
from sqlalchemy import text

conn_ids = [
    os.getenv("SENTIMENT_BATCH_FS_CONN_ID", "fs_incoming_reviews"),
    os.getenv("SENTIMENT_PIPELINE_SMTP_CONN_ID", "smtp_default"),
]

with Session() as session:
    for conn_id in conn_ids:
        session.execute(text("DELETE FROM connection WHERE conn_id = :conn_id"), {"conn_id": conn_id})
    session.commit()
PY

airflow connections add "$FS_CONN_ID" \
  --conn-type fs \
  --conn-extra "{\"path\":\"$INCOMING_DIR\"}"

python - <<'PY'
from __future__ import annotations

import json
import os
import subprocess

smarthost = os.getenv("ALERT_SMTP_SMARTHOST", "")
username = os.getenv("ALERT_SMTP_AUTH_USERNAME", "")
password = os.getenv("ALERT_SMTP_AUTH_PASSWORD", "")
from_email = os.getenv("ALERT_SMTP_FROM", username)
conn_id = os.getenv("SENTIMENT_PIPELINE_SMTP_CONN_ID", "smtp_default")

if not (smarthost and username and password and from_email):
    print("Skipping smtp_default bootstrap because SMTP alert env vars are incomplete.")
    raise SystemExit(0)

host, sep, port = smarthost.partition(":")
if not sep:
    port = "587"

subprocess.run(
    [
        "airflow",
        "connections",
        "add",
        conn_id,
        "--conn-type",
        "smtp",
        "--conn-host",
        host,
        "--conn-port",
        port,
        "--conn-login",
        username,
        "--conn-password",
        password,
        "--conn-extra",
        json.dumps({"from_email": from_email, "disable_tls": False, "timeout": 30}),
    ],
    check=True,
)
print(f"Bootstrapped Airflow SMTP connection: {conn_id}")
PY
