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
| T05 | Pipeline | Text cleaning | Whitespace normalized |
| T06 | Pipeline | Baseline statistics | Text, label, rating, and TF-IDF stats are generated |
| T07 | Monitoring | Distribution drift utility | Identical distributions produce zero delta |
| T08 | API | `/health` | Returns HTTP 200 and service status |
| T09 | API | `/ready` | Returns model loaded/fallback state |
| T10 | API | `/predict` | Returns sentiment, confidence, probabilities, explanation, model metadata, and latency |
| T11 | API | `/feedback` | Stores ground-truth feedback |
| T12 | API | `/metrics` | Exposes Prometheus metrics |
| T13 | Frontend | React build | Production bundle builds successfully |
| T14 | Docker | Compose config | Docker Compose validates service graph |
| T15 | DVC | `dvc repro` | Reproduces lifecycle pipeline |

## Acceptance Criteria

- `pytest` passes.
- `npm run build` passes for the frontend.
- `docker compose config` validates.
- `dvc repro` produces data, baselines, model artifacts, evaluation, drift report, and pipeline report.
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

