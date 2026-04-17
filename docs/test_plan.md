# Test Plan

## Strategy

Testing covers data quality, ML pipeline utilities, API contracts, monitoring output, and frontend build readiness. The goal is to show that the application meets the professor's acceptance criteria and can be reproduced locally.

## Test Cases

| ID | Area | Scenario | Expected Result |
| --- | --- | --- | --- |
| T01 | Data | Rating-to-sentiment mapping | Ratings 1-2 negative, 3 neutral, 4-5 positive |
| T02 | Data | Valid raw schema | Validation passes |
| T03 | Data | Invalid rating | Validation fails with clear error |
| T04 | Pipeline | Seed data generation | Balanced classes and required columns |
| T05 | Ingestion | Hugging Face schema mapping | Source row maps to canonical project schema |
| T06 | EDA | EDA report generation | JSON, Markdown, and chart artifacts are generated |
| T07 | Pipeline | Text cleaning | Whitespace normalized |
| T08 | Pipeline | Baseline statistics | Text, label, rating, and TF-IDF stats are generated |
| T09 | Preprocessing | Rejected-row audit | Filtered rows are written with rejection reasons |
| T10 | Monitoring | Distribution drift utility | Identical distributions produce zero delta |
| T11 | API | `/health` | Returns HTTP 200 and service status |
| T12 | API | `/ready` | Returns model loaded/fallback state |
| T13 | API | `/predict` | Returns sentiment, confidence, probabilities, explanation, model metadata, and latency |
| T14 | API | `/feedback` | Stores ground-truth feedback |
| T15 | API | `/metrics` | Exposes Prometheus metrics |
| T16 | Frontend | React build | Production bundle builds successfully |
| T17 | Docker | Compose config | Docker Compose validates service graph |
| T18 | DVC | `dvc repro` | Reproduces lifecycle pipeline |

## Acceptance Criteria

- `pytest` passes.
- `npm run build` passes for the frontend.
- `docker compose config` validates.
- `dvc repro` produces raw data, EDA artifacts, processed splits, baselines, model artifacts, evaluation, drift report, and pipeline report.
- `/predict` responds under `200 ms` for the baseline model on typical local hardware.
- MLflow contains the training run and model artifacts.
- Airflow has a successful DAG run for `sentiment_training_pipeline`.
- Grafana dashboard shows live Prometheus metrics.

## Test Report Template

| Date | Command | Total | Passed | Failed | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| Fill during demo prep | `pytest` |  |  |  |  |
| Fill during demo prep | `npm run build` |  |  |  |  |
| Fill during demo prep | `dvc repro` |  |  |  |  |
| Fill during demo prep | `docker compose config` |  |  |  |  |
