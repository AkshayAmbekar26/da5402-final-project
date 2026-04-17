from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from ml.common import REPORTS, read_json, utc_now, write_json

PERFORMANCE_PATH = REPORTS / "pipeline_performance.json"
PERFORMANCE_DIR = REPORTS / "performance"


def load_performance_report() -> dict[str, Any]:
    stages = {}
    if PERFORMANCE_DIR.exists():
        for path in sorted(PERFORMANCE_DIR.glob("*.json")):
            stages[path.stem] = read_json(path)
    report = {
        "stage": "pipeline_performance",
        "status": "success",
        "generated_at": utc_now(),
        "total_duration_seconds": float(
            sum(stage.get("duration_seconds", 0.0) for stage in stages.values())
        ),
        "stages": stages,
    }
    return report


def record_stage_performance(
    stage_name: str,
    duration_seconds: float,
    rows_processed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = load_performance_report()
    stage_payload: dict[str, Any] = {
        "duration_seconds": float(duration_seconds),
        "completed_at": utc_now(),
    }
    if rows_processed is not None:
        stage_payload["rows_processed"] = int(rows_processed)
        stage_payload["throughput_rows_per_second"] = float(rows_processed / max(duration_seconds, 1e-9))
    if extra:
        stage_payload.update(extra)
    PERFORMANCE_DIR.mkdir(parents=True, exist_ok=True)
    write_json(PERFORMANCE_DIR / f"{stage_name}.json", stage_payload)
    report = load_performance_report()
    write_json(PERFORMANCE_PATH, report)
    return report


@contextmanager
def timed_stage(
    stage_name: str,
    rows_processed: int | None = None,
    extra: dict[str, Any] | None = None,
    enabled: bool = True,
) -> Iterator[dict[str, Any]]:
    context: dict[str, Any] = {}
    start = perf_counter()
    try:
        yield context
    finally:
        if enabled:
            final_rows = context.get("rows_processed", rows_processed)
            final_extra = dict(extra or {})
            final_extra.update(context.get("extra", {}))
            record_stage_performance(stage_name, perf_counter() - start, final_rows, final_extra)
