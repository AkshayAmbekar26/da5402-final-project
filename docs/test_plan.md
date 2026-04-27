# Test Plan

## Objective

The test plan proves that the application is usable, reproducible, deployable, observable, and aligned with the course acceptance criteria. Tests cover data quality, preprocessing, training utilities, API contracts, monitoring, frontend build, Docker packaging, DVC reproducibility, and Airflow DAG importability.

## Test Strategy

| Test layer | Purpose | Tools |
| --- | --- | --- |
| Unit tests | Validate pure functions and module behavior | `pytest` |
| API contract tests | Verify endpoint schemas and error behavior | `pytest`, FastAPI test client |
| Pipeline tests | Verify data validation, preprocessing, drift, acceptance gate | `pytest`, DVC |
| Frontend build test | Verify React app compiles | `npm run build` |
| Static checks | Verify Python style and syntax | `ruff`, `bash -n` |
| Docker checks | Validate Compose and image builds | `docker compose config`, `docker compose build` |
| Monitoring checks | Validate Prometheus and AlertManager config | `promtool`, `amtool` or containerized validation |
| Smoke tests | Verify demo services respond | `scripts/docker_smoke.sh` |

## Test Cases

| ID | Area | Scenario | Expected Result |
| --- | --- | --- | --- |
| T01 | Data | Rating-to-sentiment mapping | Ratings 1-2 negative, 3 neutral, 4-5 positive |
| T02 | Data | Seed fallback generation | Balanced classes, unique normalized reviews, required columns |
| T03 | Ingestion | Hugging Face schema mapping | Source row maps to canonical schema |
| T04 | Validation | Valid raw data | Validation report has `status=success` |
| T05 | Validation | Invalid rating | Validation fails with clear error |
| T06 | Validation | Duplicate text | Warning is recorded, not silently ignored |
| T07 | EDA | Report generation | JSON, Markdown, and chart artifacts are generated |
| T08 | Preprocessing | Text cleaning | Whitespace normalized and bad rows audited |
| T09 | Preprocessing | Fixed split | Train/validation/test splits are deterministic |
| T10 | Features | Drift baseline | Text length, class distribution, rating distribution, TF-IDF stats saved |
| T11 | Training | Candidate comparison | Multiple candidates are logged and compared |
| T12 | Training | Model metadata | Git commit, DVC data version, MLflow run ID saved |
| T13 | Evaluation | Acceptance gate pass | Accepted model writes `reports/acceptance_gate.json` |
| T14 | Evaluation | Acceptance gate fail | Gate exits nonzero when F1/latency fail |
| T15 | Monitoring | Drift calculation | Stable data reports no drift above threshold |
| T16 | API | `/health` | Returns HTTP 200 and service name |
| T17 | API | `/ready` | Reports model loaded and fallback state |
| T18 | API | `/predict` | Returns sentiment, probabilities, explanation, model metadata, latency |
| T19 | API | invalid prediction request | Returns structured `422` and increments invalid-review metric |
| T20 | API | `/feedback` | Stores feedback and updates feedback metrics |
| T21 | API | `/model/info` | Returns run ID, model version, Git commit, DVC version |
| T22 | API | `/metrics` | Exposes Prometheus text-format metrics |
| T23 | API | `/metrics-summary` | Returns frontend-friendly pipeline/model/drift summary |
| T24 | API | `/monitoring/refresh` | Refreshes report-backed gauges |
| T25 | Frontend | Analyzer build | Production bundle builds |
| T26 | Frontend | Analyzer UX | Empty/long/API-error states are visible |
| T27 | Frontend | MLOps screen | Pipeline stages, recent events, metrics, tool links render from summary |
| T28 | DVC | `dvc repro` | Lifecycle reproduces through acceptance, drift, and publish |
| T29 | DVC | `dvc status` | Pipeline reports up to date |
| T30 | DVC | `dvc metrics show` | Current metrics are visible |
| T31 | Docker | Compose config | Service graph validates |
| T32 | Docker | Image builds | API, frontend, and Airflow images build |
| T33 | Airflow | DAG import | Training and batch DAGs import with zero errors |
| T34 | Monitoring | Prometheus rules | Scrape config, recording rules, and alerts validate |
| T35 | Monitoring | AlertManager config | Routing and receiver config validates |
| T36 | End-to-end | Docker smoke test | Frontend, API, MLflow, Airflow, Prometheus, Grafana respond |

## Acceptance Criteria

The project is accepted for demo when:

- `pytest` passes.
- `ruff check .` passes.
- `npm run build` passes.
- `docker compose config` passes.
- `dvc repro` succeeds from ingestion through publish.
- `dvc status` reports the pipeline up to date.
- The selected model has test macro F1 `>= 0.75`.
- The selected model latency is `< 200 ms` for typical single-review inference.
- MLflow contains candidate model runs and the selected run ID.
- The API `/ready` reports trained model loaded.
- The frontend analyzer can produce predictions and submit feedback.
- The frontend MLOps screen shows status-aware pipeline stages and recent events.
- Airflow DAGs import and can be shown in the Airflow UI.
- Prometheus and AlertManager configs validate.
- Grafana dashboards are provisioned.
