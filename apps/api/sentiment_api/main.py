from __future__ import annotations

import json
import logging
import uuid
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from apps.api.sentiment_api.config import settings
from apps.api.sentiment_api.logging_config import configure_logging
from apps.api.sentiment_api.metrics import (
    DRIFT_SCORE,
    ERROR_COUNT,
    FEEDBACK_COUNT,
    GROUND_TRUTH_MATCH_COUNT,
    INFERENCE_LATENCY,
    MODEL_LOADED,
    PREDICTION_COUNT,
    REQUEST_COUNT,
    REQUEST_LATENCY,
)
from apps.api.sentiment_api.model_service import ModelService
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
MODEL_LOADED.set(1 if model_service.loaded else 0)

app = FastAPI(
    title=settings.api_name,
    version="0.1.0",
    description="FastAPI inference service for e-commerce product review sentiment analysis.",
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
    start = perf_counter()
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    status_code = 500
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
        REQUEST_COUNT.labels(
            endpoint=request.url.path,
            method=request.method,
            status_code=str(status_code),
        ).inc()
        REQUEST_LATENCY.labels(endpoint=request.url.path, method=request.method).observe(latency)
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
    return ReadyResponse(
        ready=True,
        model_loaded=model_service.loaded,
        fallback_mode=model_service.fallback_mode,
        model_path=str(settings.model_path),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    start = perf_counter()
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
    summary: dict[str, object] = {
        "api": {"healthy": True, "ready": True},
        "model": model_service.info(),
        "links": {
            "airflow": "http://localhost:8080",
            "mlflow": "http://localhost:5000",
            "prometheus": "http://localhost:9090",
            "grafana": "http://localhost:3000",
        },
    }
    report_map = {
        "ingestion": "reports/ingestion_report.json",
        "validation": "reports/data_validation.json",
        "eda": "reports/eda_report.json",
        "preprocessing": "reports/preprocessing_report.json",
        "model_comparison": "reports/model_comparison.json",
        "evaluation": "reports/evaluation.json",
        "drift": "reports/drift_report.json",
        "pipeline": "reports/pipeline_report.json",
        "pipeline_performance": "reports/pipeline_performance.json",
    }
    for key, path in report_map.items():
        try:
            with open(path, encoding="utf-8") as handle:
                summary[key] = json.load(handle)
                if key == "drift":
                    DRIFT_SCORE.set(float(summary[key].get("drift_score", 0)))
        except FileNotFoundError:
            summary[key] = {"status": "not_available"}
    summary["pipeline_summary"] = summary.get("pipeline", {}).get("summary", {})
    return summary


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.exception_handler(ValueError)
async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})
