from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
RANDOM_SEED = 42


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


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _path_section(kind: str) -> dict[str, str]:
    paths = read_params("paths")
    section = paths.get(kind, {})
    return dict(section) if isinstance(section, dict) else {}


def dir_for(name: str) -> Path:
    directories = _path_section("directories")
    if name not in directories:
        raise KeyError(f"paths.directories.{name} is not configured in params.yaml")
    return resolve_project_path(directories[name])


def path_for(name: str) -> Path:
    artifacts = _path_section("artifacts")
    if name not in artifacts:
        raise KeyError(f"paths.artifacts.{name} is not configured in params.yaml")
    return resolve_project_path(artifacts[name])


def configured_artifact_paths() -> dict[str, Path]:
    return {name: resolve_project_path(value) for name, value in _path_section("artifacts").items()}


def ensure_dirs() -> None:
    directories = _path_section("directories")
    for configured_path in directories.values():
        resolve_project_path(configured_path).mkdir(parents=True, exist_ok=True)
    for configured_path in _path_section("artifacts").values():
        resolve_project_path(configured_path).parent.mkdir(parents=True, exist_ok=True)


# Backward-compatible directory handles; new code should prefer dir_for/path_for.
CONFIGS = dir_for("configs")
DATA_RAW = dir_for("data_raw")
DATA_INTERIM = dir_for("data_interim")
DATA_PROCESSED = dir_for("data_processed")
DATA_BASELINES = dir_for("data_baselines")
DATA_INCOMING = dir_for("data_incoming")
DATA_ARCHIVE = dir_for("data_archive")
DATA_QUARANTINE = dir_for("data_quarantine")
DATA_BATCH_INTERIM = dir_for("data_batch_interim")
DATA_OPS = dir_for("data_ops")
MODELS = dir_for("models")
REPORTS = dir_for("reports")
REPORT_FIGURES = dir_for("report_figures")
REPORT_PERFORMANCE = dir_for("report_performance")
FEEDBACK = dir_for("feedback")


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
