from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_BASELINES = ROOT / "data" / "baselines"
DATA_INCOMING = ROOT / "data" / "incoming"
DATA_ARCHIVE = ROOT / "data" / "archive"
DATA_QUARANTINE = ROOT / "data" / "quarantine"
DATA_OPS = ROOT / "data" / "ops"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FEEDBACK = ROOT / "feedback"

SENTIMENT_LABELS = ["negative", "neutral", "positive"]
RANDOM_SEED = 42


def ensure_dirs() -> None:
    for directory in [
        DATA_RAW,
        DATA_INTERIM,
        DATA_PROCESSED,
        DATA_BASELINES,
        DATA_INCOMING,
        DATA_ARCHIVE,
        DATA_QUARANTINE,
        DATA_OPS,
        MODELS,
        REPORTS,
        FEEDBACK,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_params(section: str | None = None) -> dict[str, Any]:
    params_path = Path(os.getenv("PARAMS_PATH", ROOT / "params.yaml"))
    if not params_path.exists():
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read params.yaml") from exc
    payload = yaml.safe_load(params_path.read_text(encoding="utf-8")) or {}
    if section is None:
        return dict(payload)
    section_payload = payload.get(section, {})
    return dict(section_payload) if isinstance(section_payload, dict) else {}


def rating_to_sentiment(rating: int) -> str:
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return os.getenv("GIT_COMMIT", "unavailable")


def dvc_data_version() -> str:
    lock_path = ROOT / "dvc.lock"
    try:
        return (
            subprocess.check_output(
                ["dvc", "status", "--json"],
                cwd=ROOT,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            or "{}"
        )
    except Exception:
        if lock_path.exists():
            return f"dvc-lock-sha256:{hashlib.sha256(lock_path.read_bytes()).hexdigest()}"
        return "dvc-unavailable"
