# Low-Level Design

## Scope

This document defines the API contracts, module responsibilities, validation rules, error handling, and operational interfaces for the Product Review Sentiment Analyzer.

## API Base

Local default:

```text
http://localhost:8000
```

Frontend configuration:

```text
VITE_API_BASE_URL=http://localhost:8000
```

## Public API Definitions

### `GET /health`

Purpose: process-level health check.

Response:

```json
{
  "status": "ok",
  "service": "Product Review Sentiment API"
}
```

Expected status: `200 OK`.

### `GET /ready`

Purpose: readiness check for orchestration. The API is ready only when a trained model is loaded, unless fallback readiness is explicitly enabled through `ALLOW_FALLBACK_READY=true`.

Response:

```json
{
  "ready": true,
  "model_loaded": true,
  "fallback_mode": false,
  "model_path": "models/sentiment_model.joblib"
}
```

Expected status: `200 OK`.

### `POST /predict`

Purpose: classify one product review.

Request:

```json
{
  "review_text": "The product quality is excellent and delivery was fast."
}
```

Validation:

| Field | Type | Rule |
| --- | --- | --- |
| `review_text` | string | required |
| `review_text` | string | minimum length `1` |
| `review_text` | string | maximum length `5000` |

Response:

```json
{
  "sentiment": "positive",
  "confidence": 0.93,
  "class_probabilities": {
    "negative": 0.02,
    "neutral": 0.05,
    "positive": 0.93
  },
  "explanation": [
    {"token": "excellent", "weight": 0.41},
    {"token": "fast", "weight": 0.22}
  ],
  "model_version": "local-production",
  "mlflow_run_id": "61f2ee995e7d4084a210a3513d83eec8",
  "latency_ms": 37.4
}
```

Errors:

| Condition | Expected response |
| --- | --- |
| Missing `review_text` | `422` with validation detail and Prometheus invalid-review counter |
| Empty text | `422` with validation detail |
| Text longer than `5000` characters | `422` with validation detail |
| Model service value error | `400` with structured detail |
| Unhandled exception | logged and counted in Prometheus |

### `POST /feedback`

Purpose: store ground-truth labels when they become available.

Request:

```json
{
  "review_text": "The product quality is excellent and delivery was fast.",
  "predicted_sentiment": "positive",
  "actual_sentiment": "positive",
  "source": "demo"
}
```

Validation:

| Field | Type | Rule |
| --- | --- | --- |
| `review_text` | string | required, `1` to `5000` characters |
| `predicted_sentiment` | enum | `negative`, `neutral`, or `positive` |
| `actual_sentiment` | enum | `negative`, `neutral`, or `positive` |
| `source` | string | optional, defaults to `demo` |

Response:

```json
{
  "status": "accepted",
  "stored": true
}
```

Storage: appends one JSON object per line to the configured feedback path. Prometheus counters track submitted labels and matches between predicted and actual labels.

### `GET /model/info`

Purpose: return current model loading state and metadata.

Response includes:

- `model_loaded`
- `fallback_mode`
- `serving_mode`
- `model_path`
- `mlflow_serving_url`
- `metadata.model_name`
- `metadata.model_version`
- `metadata.mlflow_run_id`
- `metadata.git_commit`
- `metadata.data_version`
- `metadata.trained_at`

### `GET /metrics-summary`

Purpose: frontend-friendly MLOps summary. The React MLOps screen uses this endpoint for status tiles, status-aware pipeline visualization, recent events, data quality, model metadata, drift, and links.

Response sections include:

- `api`
- `model`
- `links`
- `ingestion`
- `validation`
- `eda`
- `preprocessing`
- `model_comparison`
- `evaluation`
- `acceptance_gate`
- `drift`
- `pipeline`
- `pipeline_summary`
- `pipeline_performance`
- `batch_pipeline`

### `POST /monitoring/refresh`

Purpose: refresh report-backed Prometheus gauges before opening Grafana or Prometheus.

Response:

```json
{
  "status": "refreshed",
  "pipeline_summary": {
    "total_duration_seconds": 44.6244435019762,
    "selected_model": "tfidf_logistic_tuned",
    "test_macro_f1": 0.7736922040873718
  }
}
```

### `GET /metrics`

Purpose: expose Prometheus text-format metrics.

Behavior:

- Refreshes process metrics.
- Refreshes report-backed lifecycle gauges.
- Returns `CONTENT_TYPE_LATEST`.

### `POST /ops/alerts`

Purpose: receive AlertManager webhook notifications and count them through Prometheus.

Response:

```json
{
  "status": "accepted",
  "alerts_received": 1
}
```

### `POST /ops/demo/error`

Purpose: optional monitoring demo endpoint for intentionally generating a `500` error. It is available only when `ENABLE_DEMO_OPS_ENDPOINTS=true`.

## Frontend Screens

| Screen | Purpose | API dependencies |
| --- | --- | --- |
| Analyzer | Paste review, run prediction, view confidence/explanation, submit feedback | `/predict`, `/feedback` |
| MLOps | Show health, model metrics, pipeline status, recent events, tool links | `/metrics-summary`, `/monitoring/refresh` |
| Guide panel | Non-technical user manual inside the UI | none |
| Product tour | Interactive onboarding and UI explanation | none |

## ML Module Responsibilities

| Module | Responsibility |
| --- | --- |
| `ml.data_ingestion` | Download/import Hugging Face data or generate offline seed fallback |
| `ml.validation` | Validate schema, nulls, duplicates, rating range, sentiment labels, class distribution |
| `ml.eda` | Generate EDA JSON, Markdown, and chart artifacts |
| `ml.preprocessing` | Normalize text, reject bad rows, create fixed train/validation/test splits |
| `ml.features` | Compute drift baseline statistics |
| `ml.training` | Train candidates, log MLflow runs, compare models, save selected artifact |
| `ml.evaluation` | Evaluate selected model and enforce acceptance gate |
| `ml.monitoring` | Drift calculation, performance reports, dashboard summary publishing |
| `ml.orchestration` | Batch pipeline utilities for Airflow incoming-file workflow |
| `apps.api.sentiment_api` | API serving, metrics, logging, feedback, monitoring refresh |

## Data Contracts

Canonical review schema:

| Column | Type | Meaning |
| --- | --- | --- |
| `review_id` | string | stable review identifier |
| `review_text` | string | customer review body |
| `rating` | integer | original star rating |
| `sentiment` | string | `negative`, `neutral`, `positive` |
| `source` | string | dataset/source label |
| `ingested_at` | datetime string | ingestion timestamp |

Sentiment mapping:

- Ratings `1-2` -> `negative`
- Rating `3` -> `neutral`
- Ratings `4-5` -> `positive`

The current default training subset uses ratings `1`, `3`, and `5` to avoid ambiguous borderline labels.

## Logging And Exception Handling

- API middleware creates a request ID when one is not provided.
- Every request records endpoint, status code, and latency.
- Unhandled API exceptions increment `sentiment_api_errors_total`.
- Pydantic validation errors increment `sentiment_invalid_reviews_total` with a reason label.
- Pipeline stages write JSON reports with `status`, counts, warnings, errors, and timing where applicable.
- Airflow task failures are visible in the DAG UI and operational email hooks exist for missing or malformed batch files.

## Configuration

Important environment variables:

| Variable | Purpose |
| --- | --- |
| `MODEL_PATH` | model artifact path |
| `MODEL_METADATA_PATH` | model metadata JSON |
| `FEEDBACK_PATH` | feedback JSONL path |
| `CORS_ORIGINS` | allowed browser origins |
| `ALLOW_FALLBACK_READY` | whether fallback serving counts as ready |
| `MLFLOW_TRACKING_URI` | MLflow server or local file tracking path |
| `VITE_API_BASE_URL` | frontend API base URL |

## Current Implementation Evidence

- API tests pass through `pytest`.
- Frontend production build passes through `npm run build`.
- `/ready` reports trained model loaded and fallback disabled.
- `/predict` returns sentiment, confidence, explanation, latency, model version, and MLflow run ID.
- `/metrics-summary` provides real data for the MLOps dashboard.

