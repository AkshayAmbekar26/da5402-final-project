from __future__ import annotations

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
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FEEDBACK = ROOT / "feedback"

SENTIMENT_LABELS = ["negative", "neutral", "positive"]
RANDOM_SEED = 42


def ensure_dirs() -> None:
    for directory in [DATA_RAW, DATA_INTERIM, DATA_PROCESSED, DATA_BASELINES, MODELS, REPORTS, FEEDBACK]:
        directory.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
        return "dvc-unavailable"
