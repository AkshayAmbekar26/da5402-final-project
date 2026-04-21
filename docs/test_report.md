# Test Report

Date: 2026-04-19

## Automated Checks

| Command | Result | Notes |
| --- | --- | --- |
| `ruff check .` | Passed | Python linting clean |
| `pytest` | Passed | 32 passed, 1 skipped, 1 dependency warning |
| `dvc repro` | Passed | Real dataset pipeline reproduced through publish stage and acceptance gate |
| `dvc status` | Passed | Data and pipelines up to date |
| `dvc push` | Passed | Versioned data/model artifacts pushed to the local DVC remote |
| `dvc metrics show --json` | Passed | Metrics include accepted model, F1, latency, data rows, drift, and pipeline duration |
| `npm audit --json` | Passed | 0 frontend vulnerabilities |
| `npm run build` | Passed | React/Vite production build generated after UI redesign |
| `docker compose config` | Passed | Compose service graph validates |
| `bash -n scripts/docker_smoke.sh` | Passed | Docker smoke script syntax validates |
| `docker compose build api frontend airflow-init airflow-webserver airflow-scheduler airflow-dag-processor` | Passed | API, frontend, and Airflow images build successfully |
| Airflow DAG import check in Docker image | Passed | `sentiment_training_pipeline` and `sentiment_batch_ingestion_pipeline` import with zero errors |
| Prometheus config validation | Passed | Scrape config, recording rules, and 21 alert rules validate |
| AlertManager config validation | Passed | Local alert routing config validates |
| `git diff --check` | Passed | No whitespace errors |

## Live Endpoint Checks

| Check | Result | Evidence |
| --- | --- | --- |
| Live `/ready` request | Passed | Trained model loaded, fallback disabled |
| Live `/predict` request | Passed | Returned sentiment using trained local-production model |
| Live `/metrics-summary` request | Passed | Returned trained model, dataset, pipeline, and monitoring evidence |
| Live `/monitoring/refresh` request | Passed | Refreshed pipeline summary and monitoring gauges |
| Live `/metrics` request | Passed | Exported model, API, drift, feedback, data-quality, and pipeline metrics |

## Checks Not Re-run In This Pass

| Check | Status | Reason |
| --- | --- | --- |
| `make docker-smoke` | Not run in this documentation pass | Requires full Docker Compose stack running |
| Manual browser responsiveness check | User-owned final check | User requested to perform browser check manually |

## Current ML And Pipeline Results

| Item | Value |
| --- | --- |
| Dataset | `SetFit/amazon_reviews_multi_en` |
| Fallback used | `false` |
| Raw rows | `15000` |
| Processed rows | `14987` |
| Rejected rows | `13` |
| Candidate models | `5` |
| Accepted candidates | `5` |
| Selected model | `tfidf_logistic_tuned` |
| Test accuracy | `0.7741` |
| Test macro F1 | `0.7737` |
| Test macro precision | `0.7733` |
| Test macro recall | `0.7741` |
| Latency per review | `0.0467 ms` |
| Acceptance gate | Passed |
| Drift detected | `false` |
| Drift score | `0.0386` |
| Pipeline duration | `44.6 s` |
| Timed lifecycle stages | `9` |
| Selected MLflow run ID | `61f2ee995e7d4084a210a3513d83eec8` |

## Acceptance Status

- Unit and API tests passed.
- DVC lifecycle is reproducible.
- Dataset ingestion used the real public dataset with no fallback.
- EDA generated JSON, Markdown, and chart artifacts.
- Model macro F1 is above the `0.75` acceptance threshold.
- Model latency is below the `200 ms` target.
- MLflow logged five candidate model runs.
- Model comparison selected `tfidf_logistic_tuned` by the documented acceptance and validation-F1 rule.
- Pipeline performance report records duration and throughput stage by stage.
- Frontend builds successfully.
- Frontend analyzer includes validation, examples, product tour, guide panel, confidence visualization, explanation tokens, model metadata, and feedback submission.
- Frontend MLOps screen shows API/model status, status-aware pipeline stages, recent pipeline events, model acceptance, data quality, drift, duration, throughput, metadata, and tool links.
- Docker Compose validates.
- API, frontend, and Airflow Docker images build successfully.
- Airflow DAGs import successfully inside the built Airflow image.
- Prometheus/Grafana/AlertManager configs are present and validated.
