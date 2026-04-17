# MLOps Report

## Version Control

- Git tracks source code, configs, tests, documentation, and infrastructure definitions.
- DVC tracks generated data, model artifacts, and lifecycle stages.
- `.gitignore` prevents raw data, model binaries, secrets, and local runtime state from being committed directly.

## Data Engineering

The pipeline ingests `SetFit/amazon_reviews_multi_en` into a standard schema, validates it, runs EDA, cleans it, and creates deterministic train/validation/test splits. The Airflow DAG gives a visual orchestration console and DVC gives reproducible local execution.

EDA produces JSON, Markdown, and chart artifacts covering row counts, missing values, duplicate text, class distribution, rating distribution, text-length statistics, top tokens, and dataset limitation notes.

## Experiment Tracking

MLflow logs:

- One run per candidate model
- Model type and hyperparameters
- Validation and test metrics
- Macro F1, accuracy, and latency benchmark
- Model size for the selected model
- Feature importance artifact
- Evaluation report
- Model comparison report
- Git commit hash
- DVC data state
- Registered model name: `ProductReviewSentimentModel`

Model selection uses a documented promotion rule: choose the candidate with the highest validation macro F1 among models where test macro F1 is at least `0.75` and latency is below `200 ms`. If no candidate passes, the report records that no model passed and selects the highest validation macro F1 candidate for inspection rather than silently promoting a weak model.

## Reproducibility

Reproducibility is anchored by:

- Git commit hash in model metadata
- MLflow run ID in model metadata
- DVC pipeline and artifact state
- Fixed random seed
- `MLproject`
- Docker images and Docker Compose

## Monitoring

FastAPI exposes Prometheus metrics for:

- Request count
- Request latency
- Error count
- Prediction count by sentiment
- Model inference latency
- Model loaded status
- Feedback count
- Feedback match count
- Drift score

Grafana visualizes these metrics and Prometheus alert rules detect:

- API down
- Error rate above 5%
- Model not loaded
- Drift score above threshold
- P95 latency above 200 ms

## Deployment And Rollback

Deployment uses Docker Compose with separate frontend and backend services. Model rollback is handled by keeping old MLflow versions and changing the configured model artifact or registry stage before restarting the API.

## Current Incomplete Items

- The local seed dataset remains as a fallback so the demo is offline-safe if Hugging Face access fails.
- A transformer model is intentionally optional because the local baseline gives lower latency and easier explainability.
- Authentication and TLS are documented as production requirements but not enabled for the local course demo.
