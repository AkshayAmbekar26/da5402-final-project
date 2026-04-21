# MLOps Report

## Summary

This project implements a local end-to-end MLOps application for product review sentiment analysis. The system demonstrates data versioning, reproducible pipelines, experiment tracking, model acceptance gates, containerized deployment, monitoring, alerting, feedback collection, and rollback readiness.

The final submitted model is intentionally lightweight: TF-IDF plus Logistic Regression. This supports the course constraints because it is fast, explainable, reproducible, and practical on local hardware.

## Dataset And Data Engineering

Primary dataset:

```text
SetFit/amazon_reviews_multi_en
```

Canonical schema:

| Column | Meaning |
| --- | --- |
| `review_id` | stable review identifier |
| `review_text` | review text |
| `rating` | source star rating |
| `sentiment` | mapped label |
| `source` | dataset source |
| `ingested_at` | ingestion timestamp |

Sentiment mapping:

- Ratings `1-2` -> `negative`
- Rating `3` -> `neutral`
- Ratings `4-5` -> `positive`

The default dataset subset uses ratings `1`, `3`, and `5` to reduce ambiguity.

Data engineering stages:

1. Ingest reviews from Hugging Face or local seed fallback.
2. Validate schema, nulls, rating range, duplicates, class distribution, empty reviews, and text length.
3. Run EDA and generate JSON, Markdown, and chart artifacts.
4. Normalize text and remove invalid rows.
5. Create deterministic train/validation/test splits.
6. Compute drift baselines.

Current data evidence:

| Metric | Value |
| --- | --- |
| Raw rows | `15000` |
| Processed rows | `14987` |
| Rejected rows | `13` |
| Classes | `negative`, `neutral`, `positive` |
| Class balance | `5000` raw rows per class |
| Fallback used | `false` |

## DVC Reproducibility

DVC defines the lifecycle DAG:

```text
ingest -> validate -> eda -> preprocess -> featurize -> train -> evaluate -> accept -> drift -> publish
```

DVC tracks:

- raw dataset
- rejected rows
- processed splits
- feature/drift baselines
- model artifacts
- MLflow serving artifact
- model metadata
- feature importance
- metrics and reports
- DVC-native plot sources

Parameterized experiments are controlled through `params.yaml`, including data size, text length rules, split sizes, acceptance thresholds, latency benchmark size, and model candidates.

Useful commands:

```bash
dvc dag
dvc status
dvc repro
dvc metrics show
dvc plots show
dvc exp run -S training.acceptance_test_macro_f1=0.78
```

Older states can be recovered with:

```bash
git checkout <commit>
dvc checkout
dvc repro
```

The project uses a local DVC remote named `local_artifacts` at `dvc_remote/` so the demo remains cloud-free while still showing artifact push/pull concepts.

## Experiment Tracking With MLflow

MLflow logs one run per candidate model. Logged information includes:

- candidate name
- model type
- vectorizer configuration
- classifier hyperparameters
- train/validation/test row counts
- dataset source and fallback flag
- validation metrics
- test metrics
- class-wise F1 scores
- confusion matrix
- latency benchmark
- model size
- feature importance artifact
- Git commit hash
- DVC data version
- model artifact
- selected model metadata

Current model evidence:

| Item | Value |
| --- | --- |
| Selected model | `tfidf_logistic_tuned` |
| Selected MLflow run ID | `61f2ee995e7d4084a210a3513d83eec8` |
| Registered model name | `ProductReviewSentimentModel` |
| Test macro F1 | `0.7737` |
| Test accuracy | `0.7741` |
| Latency per review | `0.0467 ms` |
| Candidate models | `5` |
| Accepted candidates | `5` |

## Model Acceptance Gate

Promotion rule:

1. Candidate must satisfy test macro F1 `>= 0.75`.
2. Candidate must satisfy latency `< 200 ms`.
3. Among passing candidates, select the highest validation macro F1.
4. Write `reports/acceptance_gate.json`.
5. Fail the DVC pipeline if the selected model does not satisfy the gate.

This avoids silently promoting a poor model and gives a clear pass/fail artifact for CI-style reproducibility.

## Airflow Orchestration

Airflow DAGs:

| DAG | Purpose |
| --- | --- |
| `sentiment_training_pipeline` | Runs the main ML lifecycle on a weekly configurable schedule: ingest, validate, preprocess, baseline, train, evaluate, accept, drift, publish |
| `sentiment_monitoring_maintenance` | Runs hourly configurable maintenance checks, refreshes drift, evaluates drift/feedback thresholds, and triggers retraining when needed |
| `sentiment_batch_ingestion_pipeline` | Watches incoming CSV files, chunks batches, controls concurrency, persists operational state, archives/quarantines files, and sends alerts |

Operational behavior:

- retry logic
- exponential backoff
- task-level logs
- DAG run history
- worker pool controls for chunked batch processing
- malformed input quarantine
- missing-file alert hooks
- summary email hooks
- periodic scheduled retraining through `SENTIMENT_TRAINING_SCHEDULE`
- drift/feedback-triggered retraining through `sentiment_monitoring_maintenance`

Airflow gives the professor a visible pipeline management console, while DVC gives reproducible artifact lineage.

## Deployment And Packaging

Docker Compose services:

- `frontend`
- `api`
- `mlflow`
- `airflow-webserver`
- `airflow-scheduler`
- `airflow-dag-processor`
- `postgres`
- `prometheus`
- `alertmanager`
- `node-exporter`
- `grafana`

Packaging evidence:

- Separate Dockerfiles for frontend, API, and Airflow.
- Frontend built with `npm ci` and `package-lock.json`.
- API installs the Python package and pinned dependencies.
- Airflow image includes the project package for DAG execution.
- Compose file defines health checks and readiness-aware service dependencies.
- `.dockerignore` prevents local caches, raw data, logs, feedback, and build output from bloating images.
- `make docker-smoke` verifies the full stack.
- GitHub Actions validates the local Docker Compose deployment by building images, starting the stack, running smoke checks, uploading logs on failure, and tearing down the stack.

## API Serving

FastAPI exposes:

- `/health`
- `/ready`
- `/predict`
- `/feedback`
- `/model/info`
- `/metrics-summary`
- `/monitoring/refresh`
- `/metrics`
- `/ops/alerts`

The React frontend uses only REST calls, preserving loose coupling between UI and inference engine.

## Monitoring And Alerting

Prometheus metrics include:

- request count by endpoint/status
- active requests
- request latency histogram
- invalid review count
- prediction count by sentiment
- inference latency
- model loaded state
- fallback mode
- model acceptance status
- macro F1
- candidate counts
- feedback count
- feedback match count
- feedback accuracy ratio
- drift score
- drift-detected status
- raw, processed, and rejected rows
- rejected-row ratio
- pipeline duration
- stage duration
- stage throughput
- AlertManager notification counts
- CPU, memory, disk, filesystem, and load through node_exporter

Grafana dashboards show:

- API health and latency
- request throughput
- error rate
- prediction distribution
- model readiness
- model acceptance
- macro F1
- data quality
- drift
- pipeline speed and throughput
- feedback accuracy
- infrastructure CPU, memory, disk, and load
- firing alerts and AlertManager activity

Alert rules cover:

- API down
- 5xx error rate above 5%
- model not loaded
- fallback mode
- failed model acceptance
- drift
- latency above 200 ms
- long pipeline duration
- rejected-row ratio above 5%
- node exporter down
- high CPU, memory, or disk usage
- AlertManager down

## Feedback Loop

The frontend allows users to submit actual sentiment after a prediction. Feedback is stored as JSONL and used to compute feedback count and match ratio. This is the ground-truth hook required for monitoring real-world performance decay.

## Maintenance And Retraining

Retraining is automated in two ways:

- `sentiment_training_pipeline` runs on the configurable `SENTIMENT_TRAINING_SCHEDULE` schedule. The default is weekly.
- `sentiment_monitoring_maintenance` runs on `SENTIMENT_MAINTENANCE_SCHEDULE`. The default is hourly. It refreshes drift, reads feedback accuracy, writes `reports/maintenance_report.json`, and triggers `sentiment_training_pipeline` if drift exceeds `SENTIMENT_RETRAIN_DRIFT_THRESHOLD` or if feedback accuracy falls below `SENTIMENT_RETRAIN_MIN_FEEDBACK_ACCURACY` after enough feedback has arrived.
- `SENTIMENT_RETRAIN_COOLDOWN_HOURS` prevents repeated retraining triggers from the same unresolved drift or feedback condition.

The default maintenance thresholds are:

- drift score above `0.25`
- at least `10` feedback labels
- feedback accuracy below `0.8`
- retraining cooldown of `6` hours

## Rollback

Rollback approach:

1. Identify previous acceptable MLflow run, Git commit, or DVC-tracked model artifact.
2. Run `make rollback ROLLBACK_ARGS="--git-rev <commit-or-tag>"` to fetch old artifacts into `models/rollback/` without changing the current worktree.
3. Review the generated `.env.rollback` file.
4. Run `make rollback-restart ROLLBACK_ARGS="--git-rev <commit-or-tag>"` to restart the API with the rollback artifact.
5. Verify `/ready`, `/predict`, `/metrics-summary`, and Grafana panels.

For artifact-based rollback, use:

```bash
make rollback-restart ROLLBACK_ARGS="--model-path models/rollback/<version>/sentiment_model.joblib"
```

## Current Limitations

- TLS and authentication are not enabled in the local demo.
- Sensitive production data is not used.
- Transformer-based models are future work because the baseline already satisfies latency and reproducibility goals.
