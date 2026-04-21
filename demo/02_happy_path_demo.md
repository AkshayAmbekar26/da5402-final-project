# Demo 02: Complete Happy Path Walkthrough

## Goal

Use this script after the architecture walkthrough. This demo shows the successful path of the system from user prediction to monitoring evidence. It covers:

1. Frontend prediction
2. Feedback loop
3. MLOps dashboard
4. Airflow training pipeline
5. DVC reproducibility
6. MLflow experiment tracking
7. Model acceptance gate
8. Prometheus metrics
9. Grafana monitoring dashboard

This is the positive scenario: everything is healthy, the model is loaded, the pipeline has succeeded, metrics are visible, and the app is usable by a non-technical user.

## Pre-Flight Checks

Run these before recording:

```bash
docker compose --profile mlflow-serving up -d
docker compose ps
curl -s http://localhost:8000/ready | jq
curl -s http://localhost:8000/metrics-summary | jq '{api, ingestion, model_run: .model.metadata.mlflow_run_id, pipeline_status: .pipeline.status}'
```

Expected state:

- API is healthy.
- API is ready.
- `model_loaded` is `true`.
- `fallback_mode` is `false`.
- Dataset is `SetFit/amazon_reviews_multi_en`.
- Pipeline status is `success`.

Useful current evidence command:

```bash
jq '{dataset: .dataset_name, rows, fallback_used, cache_used}' reports/ingestion_report.json
jq '{selected: .selected_candidate.candidate_name, test_macro_f1: .test.macro_f1, latency: .latency_ms_per_review, mlflow_run: .metadata.mlflow_run_id}' reports/training_metrics.json
```

## Browser Tabs To Open

- Frontend: `http://localhost:5173`
- API docs: `http://localhost:8000/docs`
- Airflow: `http://localhost:8080`
- MLflow: `http://localhost:5001`
- Prometheus: `http://localhost:9091`
- Grafana: `http://localhost:3001`

Airflow login:

- Username: `admin`
- Password: `admin`

Grafana login:

- Username: `admin`
- Password: `admin`

## Scene 1: Show The User-Facing Prediction Flow

### What To Open

Open:

```text
http://localhost:5173
```

Go to the main analyzer screen.

### What To Do

Paste this positive review:

```text
The product quality is excellent and delivery was fast. The packaging was neat and the item feels very durable.
```

Click the analyze button.

### What To Show

Point at:

- Predicted sentiment
- Confidence score
- Class probability breakdown
- Influential words
- Request latency
- Model version
- MLflow run ID

### What To Say

> This is the main product experience. A non-technical user only has to paste a review and click analyze. The frontend sends the text to the backend using `POST /predict`. The backend validates the input, runs the loaded production model, computes class probabilities, extracts influential tokens from the trained vectorizer and classifier, measures request latency, and returns the result.

> The important MLOps detail is that the response includes the model version and MLflow run ID. That means a prediction is not just an output; it is traceable back to the experiment and model artifact that produced it.

### Optional Quick API Proof

Run:

```bash
curl -s -X POST http://localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"review_text":"The product quality is excellent and delivery was fast."}' | jq
```

What to say:

> This shows the same backend contract directly. The frontend is not using hidden backend internals; it is using this REST API response.

## Scene 2: Show Positive, Neutral, And Negative Examples

### What To Do

Run three quick examples in the frontend.

Positive:

```text
Amazing product. It works perfectly, looks premium, and arrived earlier than expected.
```

Neutral:

```text
The product is okay. It works as described, but there is nothing very special about it.
```

Negative:

```text
The item stopped working after two days and support was very slow. I am disappointed.
```

### What To Say

> The model supports three sentiment classes: positive, neutral, and negative. The UI is intentionally simple, but it still exposes useful model evidence: confidence, probability distribution, explanation tokens, latency, and model metadata. This makes the system more understandable during a demo and more auditable in production.

## Scene 3: Submit Feedback

### What To Do

After one prediction, use the feedback controls and select the actual sentiment.

### What To Show

Show the success message after feedback submission.

Optional command:

```bash
tail -n 3 feedback/feedback.jsonl
```

### What To Say

> This is the feedback loop. In real systems, ground-truth labels usually arrive after prediction. Here, the user can submit the actual sentiment. The backend stores the feedback in JSONL format with the review text, predicted sentiment, actual sentiment, source, and timestamp.

> Later, the maintenance logic uses this feedback to calculate real-world accuracy. If enough feedback arrives and accuracy drops below the threshold, the maintenance DAG can trigger retraining.

## Scene 4: Show The MLOps Dashboard In The Frontend

### What To Open

In the frontend, switch to the MLOps dashboard screen.

### What To Show

Point at:

- API health
- Model readiness
- Macro F1
- Drift status
- Pipeline lifecycle timeline
- Recent pipeline events
- Data quality
- Model metadata
- Tool links for Airflow, MLflow, Prometheus, Grafana, AlertManager

### What To Say

> This screen is meant to connect the product demo with the MLOps implementation. A non-technical evaluator can see whether the API is healthy, whether the model is ready, what the latest model metrics are, whether drift is detected, and whether the pipeline succeeded.

> The dashboard reads from `GET /metrics-summary`. The API gathers information from lifecycle report files, model metadata, drift reports, feedback logs, and monitoring state. This gives one simple operational view while still allowing deeper inspection in Airflow, MLflow, Prometheus, and Grafana.

### Optional Refresh

Click refresh in the dashboard or run:

```bash
curl -s -X POST http://localhost:8000/monitoring/refresh | jq
```

What to say:

> This refresh endpoint reloads report-backed metrics into the API so Prometheus and the frontend see the latest lifecycle state.

## Scene 5: Show Airflow Training Pipeline

### What To Open

Open:

```text
http://localhost:8080
```

Go to DAG:

```text
sentiment_training_pipeline
```

### What To Show

Show the successful DAG run and graph/grid view.

Tasks to point at:

- `ingest_data`
- `validate_raw_data`
- `run_eda`
- `preprocess_data`
- `generate_features`
- `compute_drift_baseline`
- `train_and_compare_models`
- `evaluate_model`
- `register_model_if_accepted`
- `run_batch_drift_check`
- `publish_pipeline_report`

### What To Say

> Airflow is the operational orchestration layer. It gives a visible pipeline console with task states, logs, retries, and run history. This DAG runs the end-to-end ML lifecycle. It starts by ingesting the Amazon review dataset, validates the schema and quality, runs EDA, preprocesses and splits the data, generates drift baselines, trains candidate models, evaluates the selected model, checks the acceptance gate, runs drift detection, and publishes the pipeline report.

> This is different from simply running a notebook. Each step is automated, repeatable, logged, and visible in Airflow.

### Optional Terminal Evidence

Run:

```bash
docker compose exec -T airflow-webserver airflow dags list-runs \
  -d sentiment_training_pipeline --no-backfill -o table | head -n 8
```

What to say:

> This shows the DAG run history from the command line. For the demo, the key evidence is that the latest run is successful.

## Scene 6: Show DVC Reproducibility

### What To Run

In terminal:

```bash
dvc dag
dvc status
dvc metrics show
```

Optional:

```bash
dvc plots show
```

### What To Say

> DVC is used for reproducibility and versioned lifecycle execution. The DVC DAG mirrors the ML lifecycle: ingestion, validation, EDA, preprocessing, feature baseline creation, training, evaluation, acceptance, drift, and report publishing.

> The important point is that parameters live in `params.yaml`, stages live in `dvc.yaml`, and outputs are tracked through DVC. If I change a parameter, DVC knows which downstream stages are affected. If I need to reproduce the pipeline, I can run `dvc repro`. If I need to restore an older state, I can use Git plus `dvc checkout`.

### What To Show In Files

Open:

- `dvc.yaml`
- `params.yaml`
- `dvc.lock`

What to say:

> `dvc.yaml` defines the staged pipeline. `params.yaml` defines reproducible configuration like dataset size, split ratios, validation rules, training candidates, and acceptance thresholds. `dvc.lock` records the exact dependency and output hashes from the last reproduction.

## Scene 7: Show MLflow Experiment Tracking

### What To Open

Open:

```text
http://localhost:5001
```

Open experiment:

```text
product-review-sentiment
```

### What To Show

Point at:

- Multiple candidate runs
- Parameters
- Metrics
- Artifacts
- Model registry entry `ProductReviewSentimentModel`

### What To Say

> MLflow tracks the model development process. Each candidate model gets its own MLflow run. The project logs hyperparameters, dataset information, validation metrics, test metrics, latency, feature importance, confusion matrix, reports, Git commit hash, and DVC data version.

> This is important because every experiment is traceable. If a model is used in production, I can identify which MLflow run produced it, what code version was used, what data version was used, and what metrics it achieved.

### Current Model Evidence

Run:

```bash
jq '{model_name, mlflow_run_id, git_commit, data_version, trained_at}' models/model_metadata.json
```

What to say:

> This metadata is what connects runtime inference back to experiment tracking and source control.

## Scene 8: Show Model Acceptance Gate

### What To Open Or Run

Run:

```bash
jq '{accepted, macro_f1, min_macro_f1, latency_ms_per_review, max_latency_ms, reason}' reports/acceptance_gate.json
```

If the shape is different, show the full file:

```bash
jq . reports/acceptance_gate.json
```

Also show:

```bash
jq '{selected: .selected_candidate.candidate_name, test_macro_f1: .test.macro_f1, latency: .latency_ms_per_review}' reports/training_metrics.json
```

### What To Say

> The model is not promoted just because training completed. It must pass acceptance criteria. The current project uses two main gates: macro F1 must be at least `0.75`, and inference latency must be below `200 ms`. Macro F1 is used because this is a three-class sentiment problem, and we do not want the model to perform well only on the majority class.

> If the selected model fails the gate, the pipeline marks the model as not accepted and promotion is blocked. This is a basic but important MLOps control because it prevents silent deployment of a poor model.

## Scene 9: Show Prometheus Metrics

### What To Open

Open:

```text
http://localhost:9091
```

### Queries To Run

Use the Prometheus expression box.

API readiness:

```promql
sentiment_model_loaded
```

Prediction counts:

```promql
sentiment_predictions_total
```

API latency:

```promql
histogram_quantile(0.95, sum(rate(sentiment_api_request_latency_seconds_bucket[5m])) by (le))
```

Inference latency:

```promql
histogram_quantile(0.95, sum(rate(sentiment_model_inference_latency_seconds_bucket[5m])) by (le))
```

Drift:

```promql
sentiment_data_drift_score
```

Feedback accuracy:

```promql
sentiment_feedback_accuracy_ratio
```

Infrastructure:

```promql
node_load1
```

### What To Say

> Prometheus is scraping live metrics from the API and node exporter. The API metrics include request counts, request latency, prediction distribution, inference latency, model loaded state, drift score, feedback accuracy, and pipeline metrics. Node exporter adds host-level metrics like CPU, memory, disk, and load. This lets us correlate model or API behavior with infrastructure conditions.

> For example, if latency increases, I can check whether the API is slow, whether inference is slow, or whether the host machine is under CPU or memory pressure.

## Scene 10: Show Grafana Dashboard

### What To Open

Open:

```text
http://localhost:3001
```

Go to the provisioned dashboard:

```text
Product Review Sentiment MLOps
```

### What To Show

Point at:

- API health and latency panels
- Request throughput
- Prediction distribution
- Model loaded state
- Model acceptance / macro F1
- Drift score
- Feedback accuracy
- Data quality counts
- Pipeline duration and stage throughput
- Host CPU, memory, disk, and load
- Alert panels

### What To Say

> Grafana gives the operational view. Instead of reading raw Prometheus queries one by one, the dashboard visualizes application health, model health, data quality, drift, feedback, pipeline performance, and infrastructure metrics together.

> This is useful because production ML systems fail in different ways. Sometimes the model is unhealthy, sometimes the input data distribution changes, sometimes the API is slow, and sometimes the machine itself is under pressure. The dashboard is designed to make those signals visible in one place.

## Scene 11: Close The Happy Path

### What To Say

> This completes the happy path. We started with a user submitting a product review, received a sentiment prediction with confidence and explanation, submitted feedback, checked the MLOps dashboard, verified the Airflow pipeline, showed DVC reproducibility, inspected MLflow experiments, confirmed the acceptance gate, and finally verified live monitoring in Prometheus and Grafana.

> The important point is that the prediction is supported by a complete lifecycle: versioned data, reproducible training, tracked experiments, accepted model promotion, containerized serving, live metrics, and operational dashboards.

## Quick Recovery If Something Looks Stale During Recording

If the frontend dashboard looks stale:

```bash
curl -s -X POST http://localhost:8000/monitoring/refresh | jq
```

If the API is not ready:

```bash
docker compose --profile mlflow-serving restart api
curl -s http://localhost:8000/ready | jq
```

If Airflow does not show the latest run:

```bash
docker compose exec -T airflow-webserver airflow dags list-runs \
  -d sentiment_training_pipeline --no-backfill -o table | head -n 8
```

If Grafana does not show new prediction metrics, submit one more prediction in the frontend and wait for the next Prometheus scrape.
