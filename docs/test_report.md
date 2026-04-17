# Test Report

Date: 2026-04-17

| Command | Result | Notes |
| --- | --- | --- |
| `ruff check .` | Passed | Python linting clean |
| `pytest` | Passed | 20 passed, 1 dependency deprecation warning |
| `dvc repro` | Passed | Real dataset pipeline reproduced through publish stage |
| `dvc status` | Passed | Data and pipelines up to date |
| `npm audit --json` | Passed | 0 frontend vulnerabilities after Vite upgrade |
| `npm run build` | Passed | React/Vite production build generated |
| `docker compose config` | Passed | Compose service graph validates |
| Live `/ready` request | Passed | Trained model loaded, fallback disabled |
| Live `/predict` request | Passed | Returned sentiment using trained local-production model |

## Acceptance Status

- Unit and API tests passed.
- DVC lifecycle is reproducible.
- Dataset ingestion used `SetFit/amazon_reviews_multi_en` with no fallback.
- EDA generated JSON, Markdown, and chart artifacts.
- Model macro F1 is `0.7737`, above the `0.75` acceptance threshold.
- MLflow logged four candidate model runs.
- Model comparison selected `tfidf_logistic_tuned` by validation macro F1 among accepted candidates.
- Pipeline performance report includes 9 timed stages with total duration around 34.5 seconds.
- API latency is below the `200 ms` target for the baseline model.
- Frontend builds successfully.
- Docker Compose validates.
- Prometheus/Grafana configs are present for live monitoring.
