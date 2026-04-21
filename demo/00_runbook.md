# Demo 00: Project Startup Runbook

## Purpose

Use this runbook before recording the demo. It lists the exact commands needed to start the complete project, verify that all services are healthy, open the right browser tabs, and recover if something looks stale.

All commands should be run from the project root:

```bash
cd /Users/akshayambekar/Code/da5402-mlops-assignments/da5402-final-project
```

## Terminal Plan

Use three terminals:

- Terminal 1: Start and verify Docker Compose services.
- Terminal 2: Keep logs available while recording.
- Terminal 3: Run quick demo evidence commands.

You do not need to run separate frontend or backend dev servers. The demo uses Docker Compose containers.

## Terminal 1: Start The Complete Project

### 1. Go To Project Root

```bash
cd /Users/akshayambekar/Code/da5402-mlops-assignments/da5402-final-project
```

### 2. Start All Services

```bash
docker compose --profile mlflow-serving up -d
```

What this starts:

- `frontend`: React/Vite app served by Nginx
- `api`: FastAPI inference and operations API
- `mlflow`: experiment tracking and model registry UI
- `mlflow-model-server`: optional MLflow pyfunc model serving endpoint
- `airflow-webserver`: Airflow UI
- `airflow-scheduler`: Airflow scheduler
- `airflow-dag-processor`: Airflow 3 DAG parser/processor
- `airflow-init`: one-time Airflow bootstrap container
- `postgres`: Airflow metadata database
- `prometheus`: metrics scraping and alert rules
- `grafana`: dashboards
- `alertmanager`: alert routing and silencing
- `node-exporter`: host/system metrics exporter

### 3. Check Container Status

```bash
docker compose ps
```

Expected:

- Long-running services should show `Up`.
- Health-checked services should eventually show `healthy`.
- `airflow-init` may show `Exited` after successful bootstrap. That is normal.

Important healthy services:

- `api`
- `frontend`
- `mlflow`
- `mlflow-model-server`
- `airflow-webserver`
- `airflow-scheduler`
- `airflow-dag-processor`
- `postgres`
- `prometheus`
- `grafana`

Airflow 3 demo login:

```text
admin / admin
```

This is pinned through `airflow/config/simple_auth_manager_passwords.json`. Airflow 3 treats `AIRFLOW_SIMPLE_AUTH_MANAGER_USERS=admin:admin` as `username:role`, so the password file is required to make the password deterministic for screenshots and live demos.

Airflow 3 task execution also needs the scheduler workers to call the API server through Docker networking. That is why `AIRFLOW_EXECUTION_API_SERVER_URL` is set to `http://airflow-webserver:8080/execution/`.

### 4. Run Full Docker Smoke Test

```bash
RUN_MLFLOW_SERVING_SMOKE=true ./scripts/docker_smoke.sh
```

This verifies:

- Frontend loads.
- API `/health` works.
- API `/ready` confirms model is loaded.
- API `/predict` returns a prediction.
- API `/metrics` exposes Prometheus metrics.
- API `/monitoring/refresh` works.
- MLflow UI health endpoint works.
- MLflow model server `/ping` works.
- MLflow model server prediction works.
- Airflow health endpoint works.
- Airflow 3 health is checked through `/api/v2/monitor/health`.
- Prometheus is ready.
- Grafana health endpoint works.
- AlertManager is ready.
- Node exporter exposes system metrics.

Expected final line:

```text
Docker smoke test passed.
```

### 5. Check MLflow UI And Experiment State

Open:

```text
http://localhost:5001
```

Expected:

- MLflow UI loads without `403 Forbidden`.
- Experiment `product-review-sentiment` is visible.
- Registered model `ProductReviewSentimentModel` is visible.

CLI check:

```bash
curl -s http://localhost:5001/ajax-api/2.0/mlflow/experiments/search \
  -H 'content-type: application/json' \
  -d '{"max_results":10}' | jq '{experiments: [.experiments[] | {id: .experiment_id, name}]}'

curl -s 'http://localhost:5001/ajax-api/2.0/mlflow/registered-models/search?max_results=10' \
  | jq '{models: [.registered_models[] | {name, latest_versions: [.latest_versions[]? | {version, current_stage, run_id}]}]}'
```

Why this is required:

- MLflow 3.x has security middleware that validates the browser `Host` header.
- The Compose command explicitly allows `localhost:5001`, `127.0.0.1:5001`, Docker service hostnames, and localhost variants.
- The backend store is configured as `sqlite:////mlflow/mlflow.db`, which stores MLflow experiments in the mounted Docker volume instead of inside the disposable container filesystem.
- This means stopping and starting containers with `docker compose down` and `docker compose up -d` should preserve MLflow experiments, model registry entries, and run metadata.

### 6. Run Monitoring Status Check

```bash
./scripts/monitoring_status.sh
```

This verifies:

- API metrics endpoint is reachable.
- Prometheus is ready.
- Grafana is ready.
- AlertManager is ready.
- Node exporter is reachable.
- Prometheus targets are up.

Expected final line:

```text
Monitoring status check passed.
```

### 7. Optional: Repopulate MLflow If It Is Empty

You should not usually need this. Use it only if MLflow opens but the `product-review-sentiment` experiment is missing, for example after deleting Docker volumes or rebuilding from a clean machine.

Run one training pass through the Airflow container environment:

```bash
docker compose exec -T airflow-webserver sh -c 'cd /opt/airflow/project && python -m ml.training.train'
```

Then reload the serving containers so the API and MLflow model server use the latest artifact:

```bash
docker compose --profile mlflow-serving restart api mlflow-model-server
```

Verify:

```bash
curl -s http://localhost:8000/ready | jq
curl -s http://localhost:8000/model/info | jq '.metadata | {model_name, mlflow_run_id, trained_at}'
curl -s 'http://localhost:5001/ajax-api/2.0/mlflow/registered-models/search?max_results=10' | jq
```

Why this is required:

- MLflow UI shows experiment runs from the MLflow backend database, not directly from `models/model_metadata.json`.
- If the MLflow backend database is empty, the application can still serve the local model artifact, but the MLflow UI will not show the experiment history.
- Running `ml.training.train` logs candidate runs, artifacts, and the registered model back into MLflow.

## Terminal 2: Keep Logs Ready

Use this terminal only if you want to show live logs or debug during recording.

```bash
cd /Users/akshayambekar/Code/da5402-mlops-assignments/da5402-final-project
docker compose logs -f --tail=100 api airflow-scheduler prometheus alertmanager
```

What to mention if showing logs:

- API logs are structured JSON logs with request ID, endpoint, status, and latency.
- Airflow scheduler logs show operational pipeline scheduling.
- Prometheus and AlertManager logs show monitoring and alerting services are active.

To stop following logs, press:

```text
Ctrl + C
```

This only stops log viewing. It does not stop the project.

## Terminal 3: Demo Evidence Commands

Use this terminal during the walkthrough when you want quick proof.

### API Readiness

```bash
curl -s http://localhost:8000/ready | jq
```

Expected:

- `ready: true`
- `model_loaded: true`
- `fallback_mode: false`

### Prediction API

```bash
curl -s -X POST http://localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"review_text":"The product quality is excellent and delivery was fast."}' | jq
```

What this proves:

- FastAPI is serving the model.
- Inference returns sentiment, confidence, probabilities, explanation, model version, MLflow run ID, and latency.

### MLOps Summary

```bash
curl -s http://localhost:8000/metrics-summary | jq '{api, ingestion, model_run: .model.metadata.mlflow_run_id, pipeline_status: .pipeline.status, maintenance: .maintenance.action}'
```

What this proves:

- API health and readiness are visible.
- Dataset and pipeline reports are exposed to the frontend MLOps dashboard.
- Latest model run ID is visible.
- Maintenance state is visible.

### Dataset Evidence

```bash
jq '{dataset_name, rows, fallback_used, cache_used, class_distribution, rating_distribution}' reports/ingestion_report.json
```

Expected:

- Dataset: `SetFit/amazon_reviews_multi_en`
- Rows: `15000`
- `fallback_used: false`

### Model Evidence

```bash
jq '{model_name, mlflow_run_id, git_commit, data_version, trained_at}' models/model_metadata.json
```

What this proves:

- The deployed model is traceable to an MLflow run ID.
- Git commit and DVC data version are stored for reproducibility.

### Training Metrics

```bash
jq '{selected: .selected_candidate.candidate_name, test_macro_f1: .test.macro_f1, latency: .latency_ms_per_review}' reports/training_metrics.json
```

Expected current model:

- Selected model: `tfidf_logistic_tuned`
- Test macro F1: around `0.7737`
- Evaluation latency: well below `200 ms`

### Acceptance Gate

```bash
jq '{accepted, macro_f1, min_macro_f1, latency_ms_per_review, max_latency_ms, reason}' reports/acceptance_gate.json
```

What this proves:

- Model promotion is gated by quality and latency.
- Macro F1 must be at least `0.75`.
- Latency must be below `200 ms`.

### Maintenance Report

```bash
jq '{action, should_retrain, reasons, drift, feedback, cooldown}' reports/maintenance_report.json
```

What this proves:

- Drift and feedback are checked.
- Retraining policy is automated.
- Cooldown prevents repeated retraining triggers for the same unresolved condition.

### DVC Evidence

```bash
dvc dag
dvc status
dvc metrics show
```

What this proves:

- The ML workflow is represented as a DVC DAG.
- DVC knows whether artifacts are up to date.
- Metrics are tracked outside ad hoc notebooks.

### Airflow Run Evidence

```bash
docker compose exec -T airflow-webserver airflow dags list-runs \
  -d sentiment_training_pipeline --no-backfill -o table | head -n 8
```

What this proves:

- The training pipeline has visible Airflow run history.
- The latest successful run can be shown from CLI or Airflow UI.

## Browser Tabs For Recording

Open these after Terminal 1 checks pass.

| Component | URL | What To Show |
| --- | --- | --- |
| Frontend | `http://localhost:5173` | Sentiment analyzer and MLOps dashboard |
| API docs | `http://localhost:8000/docs` | FastAPI endpoint contract |
| Airflow | `http://localhost:8080` | DAGs, pipeline runs, task logs |
| MLflow | `http://localhost:5001` | Experiment runs, metrics, artifacts, model registry |
| Prometheus | `http://localhost:9091` | Live metric queries and targets |
| Grafana | `http://localhost:3001` | Product Review Sentiment MLOps dashboard |
| AlertManager | `http://localhost:19093` | Alerts, grouping, and silences |

Airflow login:

- Username: `admin`
- Password: `admin`

Grafana login:

- Username: `admin`
- Password: `admin`

## Prometheus Queries For Demo

Use these in the Prometheus UI.

Model loaded:

```promql
sentiment_model_loaded
```

Prediction counts:

```promql
sentiment_predictions_total
```

P95 API latency:

```promql
histogram_quantile(0.95, sum(rate(sentiment_api_request_latency_seconds_bucket[5m])) by (le))
```

P95 model inference latency:

```promql
histogram_quantile(0.95, sum(rate(sentiment_model_inference_latency_seconds_bucket[5m])) by (le))
```

Drift score:

```promql
sentiment_data_drift_score
```

Feedback accuracy:

```promql
sentiment_feedback_accuracy_ratio
```

System load:

```promql
node_load1
```

## If Something Looks Stale

### MLflow Shows `403 Forbidden`

Check whether the MLflow service allows browser hostnames:

```bash
docker compose config mlflow | rg -A2 allowed-hosts
```

The allowed hosts should include:

```text
localhost:5001
127.0.0.1:5001
mlflow
mlflow:5000
```

If you recently changed `.env` or `docker-compose.yml`, recreate only MLflow:

```bash
docker compose --profile mlflow-serving up -d --no-deps --force-recreate mlflow
curl -i http://localhost:5001/ | head
```

Expected:

```text
HTTP/1.1 200 OK
```

Why this happens:

- MLflow 3.x rejects requests whose `Host` header is not in `--allowed-hosts`.
- Health checks can still pass because they run inside the container against `localhost:5000`.
- The browser uses `localhost:5001`, so that exact host and port must be allowed.

### MLflow Opens But Experiments Are Missing

First check the experiment list:

```bash
curl -s http://localhost:5001/ajax-api/2.0/mlflow/experiments/search \
  -H 'content-type: application/json' \
  -d '{"max_results":10}' | jq
```

If `product-review-sentiment` is missing, repopulate MLflow:

```bash
docker compose exec -T airflow-webserver sh -c 'cd /opt/airflow/project && python -m ml.training.train'
docker compose --profile mlflow-serving restart api mlflow-model-server
```

Why this happens:

- Experiments live in the MLflow backend database.
- The project now stores that database in the persisted `/mlflow/mlflow.db` Docker volume.
- If volumes were deleted, MLflow starts clean and must be repopulated by running training again.

### Refresh Report-Backed API Metrics

```bash
curl -s -X POST http://localhost:8000/monitoring/refresh | jq
```

### Restart API To Reload Latest Model Artifact

```bash
docker compose --profile mlflow-serving restart api
curl -s http://localhost:8000/ready | jq
```

### Restart MLflow Model Server To Reload Latest Pyfunc Artifact

```bash
docker compose --profile mlflow-serving restart mlflow-model-server
curl -s http://localhost:5002/ping
```

### Check Recent API Logs

```bash
docker compose logs --tail=100 api
```

### Check Airflow DAG Runs

```bash
docker compose exec -T airflow-webserver airflow dags list-runs \
  -d sentiment_training_pipeline --no-backfill -o table | head -n 8
```

## Optional: Trigger A Fresh Training Pipeline

Only do this if you have time during preparation. It can take around a minute or more depending on the machine.

```bash
docker compose exec -T airflow-webserver airflow dags trigger sentiment_training_pipeline \
  --conf '{"triggered_by":"manual_demo_prep","reason":"fresh_demo_run"}'
```

Watch status:

```bash
docker compose exec -T airflow-webserver airflow dags list-runs \
  -d sentiment_training_pipeline --no-backfill -o table | head -n 8
```

After the run succeeds, reload serving:

```bash
docker compose --profile mlflow-serving restart api mlflow-model-server
```

## Shutdown After Recording

Use this when you are completely done.

```bash
docker compose --profile mlflow-serving down
```

If you want to remove orphaned old containers too:

```bash
docker compose --profile mlflow-serving down --remove-orphans
```

Do not use volume deletion before submission unless you intentionally want to erase local MLflow, Grafana, Prometheus, and Airflow state.
