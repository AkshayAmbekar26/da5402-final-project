# Low-Level Design

## API Definitions

### `GET /health`

Response:

```json
{
  "status": "ok",
  "service": "Product Review Sentiment API"
}
```

### `GET /ready`

Response:

```json
{
  "ready": true,
  "model_loaded": true,
  "fallback_mode": false,
  "model_path": "models/sentiment_model.joblib"
}
```

### `POST /predict`

Request:

```json
{
  "review_text": "The product quality is excellent and delivery was fast."
}
```

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
    {"token": "excellent", "weight": 0.41}
  ],
  "model_version": "local-production",
  "mlflow_run_id": "abc123",
  "latency_ms": 37.4
}
```

Validation:

- `review_text` is required.
- Length must be from `1` to `5000` characters.
- Invalid payloads return structured `4xx` responses.

### `POST /feedback`

Request:

```json
{
  "review_text": "The product quality is excellent and delivery was fast.",
  "predicted_sentiment": "positive",
  "actual_sentiment": "positive",
  "source": "demo"
}
```

Response:

```json
{
  "status": "accepted",
  "stored": true
}
```

### `GET /model/info`

Returns model loading state, metadata, version, MLflow run ID, Git commit hash, and artifact path.

### `GET /metrics-summary`

Returns a frontend-friendly summary of API health, model state, latest evaluation report, latest drift report, and MLOps tool links.

### `GET /metrics`

Returns Prometheus text-format metrics.

## Module Responsibilities

- `ml.data_ingestion`: create or import raw review dataset.
- `ml.validation`: enforce schema, null, duplicate, range, and class checks.
- `ml.preprocessing`: clean text and create reproducible data splits.
- `ml.features`: calculate baseline statistics for drift detection.
- `ml.training`: train model, log MLflow run, save metadata and artifacts.
- `ml.evaluation`: evaluate trained model against acceptance threshold.
- `ml.monitoring`: calculate drift and publish pipeline reports.
- `apps.api.sentiment_api`: serve predictions, feedback, health checks, metadata, logs, and metrics.

## Error Handling

- Validation failures raise explicit exceptions and write failure reports.
- API request validation is handled by Pydantic.
- Unhandled API exceptions are logged with request ID, endpoint, status, and latency.
- If no model artifact exists, the API starts in fallback mode so the UI remains demonstrable while `/ready` reports that the trained model is not loaded.

