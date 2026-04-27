# Test Report

Date: 2026-04-27

## Automated Checks

| Command | Result | Notes |
| --- | --- | --- |
| `ruff check .` | Passed | Latest lint run completed without violations |
| `pytest --disable-warnings` | Passed | `41 passed, 1 skipped` in `3.27s` |
| `docker compose --profile mlflow-serving config` | Passed | Compose service graph validates cleanly |
| `bash -n scripts/docker_smoke.sh` | Passed | Smoke script syntax is valid |
| `npm run build` | Passed | Vite production build completed successfully after the latest frontend/documentation polish |

## Live Endpoint Checks

| Check | Result | Evidence |
| --- | --- | --- |
| Live `/ready` request | Passed | API reported trained model loaded and fallback disabled in the latest clean stack run |
| Live `/predict` request | Passed | Positive sample prediction returned successfully with sub-200 ms latency |
| Live `/metrics-summary` request | Passed | Returned lifecycle, model, drift, and monitoring summary data for the frontend |
| Live `/monitoring/refresh` request | Passed | Report-backed monitoring gauges refreshed successfully |
| Live `/metrics` request | Passed | Prometheus endpoint exposed application, model, drift, and pipeline metrics |

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
| Test accuracy | `0.7741218319253002` |
| Test macro F1 | `0.7736922040873718` |
| Test macro precision | `0.7733483772327219` |
| Test macro recall | `0.7741171932947634` |
| Latency per review | `0.028952127167523958 ms` |
| Acceptance gate | Passed |
| Drift detected | `false` |
| Drift score | `0.09261533721669622` |
| Pipeline duration | `24.14291595902614 s` |
| Timed lifecycle stages | `11` |
| Selected MLflow run ID | `5addd516eb57451bab36369d640deac9` |

## Acceptance Status

- Unit and API tests passed.
- DVC lifecycle code and configuration remain in place for reproducible training and evaluation.
- Dataset ingestion used the real public dataset with no fallback.
- EDA generated JSON, Markdown, and chart artifacts.
- Model macro F1 is above the `0.75` acceptance threshold.
- Model latency is below the `200 ms` target.
- MLflow logged five candidate model runs.
- Model comparison selected `tfidf_logistic_tuned` by the documented acceptance and validation-F1 rule.
- Pipeline performance report records duration and throughput stage by stage.
- Frontend build passes on the latest UI iteration, including the in-app manual link.
- Frontend analyzer includes validation, examples, product tour, guide panel, confidence visualization, explanation tokens, model metadata, and feedback submission.
- Frontend MLOps screen shows API/model status, status-aware pipeline stages, recent pipeline events, model acceptance, data quality, drift, duration, throughput, metadata, and tool links.
- Docker Compose validates.
- Prometheus/Grafana/AlertManager configs are present and validated in the running stack.
