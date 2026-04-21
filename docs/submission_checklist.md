# Submission Checklist And Rubric Evidence

This checklist maps the professor's evaluation criteria to concrete project evidence.

## Demonstration

### Web Application Frontend UI/UX

| Criterion | Status | Evidence |
| --- | --- | --- |
| UX intuitive for problem statement | Covered | Analyzer screen follows paste review -> analyze -> read sentiment |
| Easy for non-technical user | Covered | Example reviews, product tour, guide panel, simple result cards |
| Foolproof UX | Covered | Empty input, long input, loading, API error, disabled action states |
| Free of UI errors | Covered | `npm run build` passes |
| Colors, shapes, look-and-feel | Covered | Modern SentinelAI UI in `apps/frontend/src/styles.css` |
| Responsive UI | Covered | Breakpoints for mobile/tablet/desktop |
| User manual | Covered | In-app Guide panel and `docs/user_manual.md` |

### ML Pipeline Visualization

| Criterion | Status | Evidence |
| --- | --- | --- |
| Separate UI screen for ML pipeline | Covered | Frontend MLOps tab |
| Data ingestion/engineering pipeline visible | Covered | Ingest, Validate, EDA, Preprocess, Baseline stages |
| Tool UI orchestration | Covered | Links to Airflow, MLflow, Prometheus, Grafana, AlertManager |
| Pipeline management console | Covered | Airflow DAG UI plus frontend MLOps command center |
| Errors, failures, successful runs | Covered | Status-aware pipeline stages and Recent Pipeline Events panel |
| Speed and throughput | Covered | `reports/pipeline_performance.json` and frontend stage throughput |

## Software Engineering

### Design Principle

| Criterion | Status | Evidence |
| --- | --- | --- |
| Design document | Covered | `docs/hld.md`, `docs/lld.md` |
| OO or functional design | Covered | Functional ML modules, service-oriented API/frontend |
| LLD endpoint definitions | Covered | `docs/lld.md` |
| Architecture diagram | Covered | `docs/architecture.md` |
| HLD diagram | Covered | `docs/hld.md` |
| Loose frontend/backend coupling | Covered | React calls REST endpoints only |

### Implementation

| Criterion | Status | Evidence |
| --- | --- | --- |
| Standardized Python style | Covered | `ruff check .` passes |
| Logging | Covered | API middleware structured logs, pipeline reports |
| Exception handling | Covered | Pydantic validation handlers, `ValueError` handler, Airflow quarantine logic |
| API follows design doc | Covered | FastAPI routes match `docs/lld.md` |
| Unit tests | Covered | `tests/` and `pytest` |

### Testing

| Criterion | Status | Evidence |
| --- | --- | --- |
| Test plan | Covered | `docs/test_plan.md` |
| Test cases | Covered | `docs/test_plan.md` |
| Test report | Covered | `docs/test_report.md` |
| Acceptance criteria | Covered | `docs/test_plan.md` |
| Criteria met | Covered | `docs/test_report.md` |

## MLOps Implementation

### Data Engineering

| Criterion | Status | Evidence |
| --- | --- | --- |
| Ingestion/transformation pipeline | Covered | DVC stages and Airflow DAG |
| Uses Airflow or Spark | Covered | Airflow |
| Throughput and speed | Covered | `reports/pipeline_performance.json` |
| Pipeline console | Covered | Airflow plus frontend MLOps dashboard |

### Source Control And CI

| Criterion | Status | Evidence |
| --- | --- | --- |
| DVC CI-style pipeline | Covered | `dvc.yaml` |
| DVC DAG | Covered | `dvc dag` |
| Git/DVC versioning | Covered | Git, DVC local remote, `dvc.lock` |
| Parameterized experiments | Covered | `params.yaml` |

### Experiment Tracking

| Criterion | Status | Evidence |
| --- | --- | --- |
| Experiments tracked | Covered | MLflow candidate runs |
| Metrics and params tracked | Covered | Macro F1, accuracy, precision, recall, latency, hyperparameters |
| Artifacts tracked | Covered | Model artifact, feature importance, confusion matrix, reports |
| Beyond autolog | Covered | Manual logging of Git commit, DVC data version, model metadata, reports |

### Exporter Instrumentation And Visualization

| Criterion | Status | Evidence |
| --- | --- | --- |
| Prometheus instrumentation | Covered | FastAPI `/metrics` |
| Information points monitored | Covered | API, model, data quality, drift, pipeline, feedback, infrastructure |
| Components monitored | Covered | API and host metrics through node_exporter; tool readiness through scrape/health checks |
| Grafana visualization | Covered | Provisioned dashboards |
| Maintenance automation | Covered | Scheduled Airflow maintenance DAG triggers retraining on drift or feedback degradation, with cooldown control |

### Software Packaging

| Criterion | Status | Evidence |
| --- | --- | --- |
| MLflow model APIification | Covered | MLflow model artifact and local serving metadata |
| MLproject | Covered | `MLproject`, `conda.yaml` |
| FastAPI APIs | Covered | `apps/api/sentiment_api/main.py` |
| Dockerized backend/frontend | Covered | `apps/api/Dockerfile`, `apps/frontend/Dockerfile` |
| Docker Compose separate services | Covered | `docker-compose.yml` |

## Documentation Requirements

| Requirement | File |
| --- | --- |
| Architecture diagram and explanation | `docs/architecture.md` |
| High-level design | `docs/hld.md` |
| Low-level design and endpoints | `docs/lld.md` |
| Test plan and test cases | `docs/test_plan.md` |
| Test report | `docs/test_report.md` |
| User manual | `docs/user_manual.md` |
| MLOps report | `docs/mlops_report.md` |
| Viva notes | `docs/viva_notes.md` |
| Demo script | `docs/demo_script.md` |
| Monitoring details | `docs/monitoring.md` |
| Docker deployment | `docs/docker_deployment.md` |
| DVC experiments | `docs/dvc_experiments.md` |
| Dataset/data card | `docs/data_card.md` |
