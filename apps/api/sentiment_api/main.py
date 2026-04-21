from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from apps.api.sentiment_api.config import settings
from apps.api.sentiment_api.logging_config import configure_logging
from apps.api.sentiment_api.metrics import (
    ACTIVE_REQUESTS,
    ALERT_NOTIFICATION_COUNT,
    ERROR_COUNT,
    FEEDBACK_COUNT,
    GROUND_TRUTH_MATCH_COUNT,
    INFERENCE_LATENCY,
    INVALID_REVIEW_COUNT,
    PREDICTION_COUNT,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    REVIEW_TEXT_LENGTH,
    observe_stage,
    refresh_process_metrics,
)
from apps.api.sentiment_api.model_service import ModelService
from apps.api.sentiment_api.report_metrics import refresh_report_metrics
from apps.api.sentiment_api.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
)

configure_logging()
logger = logging.getLogger(__name__)
model_service = ModelService()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Prime report-backed metrics once so Grafana has lifecycle values before traffic starts."""
    refresh_report_metrics(model_service.info())
    yield


app = FastAPI(
    title=settings.api_name,
    version="0.1.0",
    description="FastAPI inference service for e-commerce product review sentiment analysis.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_metrics_middleware(request: Request, call_next):
    """Record one consistent audit/metrics envelope around every API request."""
    start = perf_counter()
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    status_code = 500
    active = ACTIVE_REQUESTS.labels(endpoint=request.url.path, method=request.method)
    active.inc()
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        ERROR_COUNT.labels(endpoint=request.url.path).inc()
        logger.exception("Unhandled request error", extra={"request_id": request_id, "endpoint": request.url.path})
        raise
    finally:
        latency = perf_counter() - start
        active.dec()
        REQUEST_COUNT.labels(
            endpoint=request.url.path,
            method=request.method,
            status_code=str(status_code),
        ).inc()
        REQUEST_LATENCY.labels(endpoint=request.url.path, method=request.method).observe(latency)
        refresh_process_metrics()
        logger.info(
            "request_completed",
            extra={
                "request_id": request_id,
                "endpoint": request.url.path,
                "status_code": status_code,
                "latency_ms": round(latency * 1000, 3),
            },
        )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service=settings.api_name)


@app.get("/ready", response_model=ReadyResponse)
def ready() -> ReadyResponse:
    """Readiness is strict by default: fallback predictions do not count as production-ready."""
    is_ready = model_service.loaded or (settings.allow_fallback_ready and model_service.fallback_mode)
    return ReadyResponse(
        ready=is_ready,
        model_loaded=model_service.loaded,
        fallback_mode=model_service.fallback_mode,
        model_path=str(settings.model_path),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    start = perf_counter()
    with observe_stage("predict_request_recording"):
        REVIEW_TEXT_LENGTH.labels(endpoint="/predict").observe(len(payload.review_text))
    with observe_stage("predict_total_model_service"):
        result = model_service.predict(payload.review_text)
    INFERENCE_LATENCY.observe(perf_counter() - start)
    PREDICTION_COUNT.labels(sentiment=result["sentiment"]).inc()
    return PredictResponse(**result)


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(payload: FeedbackRequest) -> FeedbackResponse:
    settings.feedback_path.parent.mkdir(parents=True, exist_ok=True)
    with settings.feedback_path.open("a", encoding="utf-8") as handle:
        handle.write(payload.model_dump_json() + "\n")
    FEEDBACK_COUNT.labels(actual_sentiment=payload.actual_sentiment).inc()
    if payload.predicted_sentiment == payload.actual_sentiment:
        GROUND_TRUTH_MATCH_COUNT.inc()
    return FeedbackResponse(status="accepted", stored=True)


@app.get("/model/info")
def model_info() -> dict[str, object]:
    return model_service.info()


@app.get("/metrics-summary")
def metrics_summary() -> dict[str, object]:
    """Return a UI-friendly snapshot assembled from model state and latest pipeline reports."""
    reports = refresh_report_metrics(model_service.info())
    summary: dict[str, object] = {
        "api": {
            "healthy": True,
            "ready": model_service.loaded or (settings.allow_fallback_ready and model_service.fallback_mode),
        },
        "model": model_service.info(),
        "links": {
            "airflow": settings.airflow_url,
            "mlflow": settings.mlflow_url,
            "prometheus": settings.prometheus_url,
            "grafana": settings.grafana_url,
            "alertmanager": settings.alertmanager_url,
        },
    }
    summary.update(reports)
    return summary


@app.post("/ops/alerts")
async def alertmanager_webhook(request: Request) -> dict[str, object]:
    payload = await request.json()
    alerts = payload.get("alerts", []) if isinstance(payload, dict) else []
    for alert in alerts:
        labels = alert.get("labels", {}) if isinstance(alert, dict) else {}
        ALERT_NOTIFICATION_COUNT.labels(
            alertname=str(labels.get("alertname", "unknown")),
            status=str(alert.get("status", "unknown")),
            severity=str(labels.get("severity", "unknown")),
        ).inc()
    logger.info("alertmanager_webhook_received", extra={"status_code": 202, "endpoint": "/ops/alerts"})
    return {"status": "accepted", "alerts_received": len(alerts)}


@app.post("/ops/demo/error")
def demo_error() -> None:
    if not settings.enable_demo_ops_endpoints:
        raise HTTPException(status_code=404, detail="Not found")
    ERROR_COUNT.labels(endpoint="/ops/demo/error").inc()
    raise HTTPException(status_code=500, detail="Intentional monitoring demo error")


@app.post("/monitoring/refresh")
def monitoring_refresh() -> dict[str, object]:
    reports = refresh_report_metrics(model_service.info())
    return {"status": "refreshed", "pipeline_summary": reports.get("pipeline_summary", {})}


@app.get("/metrics")
def metrics() -> Response:
    refresh_process_metrics()
    refresh_report_metrics(model_service.info())
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.exception_handler(ValueError)
async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Turn Pydantic failures into labeled monitoring events for invalid user input."""
    reason = "validation_error"
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", []))
        error_type = str(error.get("type", ""))
        if "review_text" in location and error_type.endswith("missing"):
            reason = "missing_review_text"
            break
        if "review_text" in location and "too_short" in error_type:
            reason = "empty_review_text"
            break
        if "review_text" in location and "too_long" in error_type:
            reason = "review_text_too_long"
            break
    INVALID_REVIEW_COUNT.labels(endpoint=request.url.path, reason=reason).inc()
    return JSONResponse(status_code=422, content={"detail": exc.errors()})
