# Product Review Sentiment Analyzer with MLOps

This project is a local, end-to-end AI application for classifying e-commerce product reviews as `positive`, `neutral`, or `negative`. It uses `SetFit/amazon_reviews_multi_en` as the primary public dataset, with a local seed fallback for offline demos. It is intentionally built around MLOps evidence: DVC pipelines, Airflow orchestration, MLflow tracking, FastAPI serving, React UI, Prometheus/Grafana monitoring, Docker Compose packaging, CI, and documentation.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m ml.data_ingestion.ingest
python -m ml.validation.validate_data
python -m ml.eda.analyze
python -m ml.preprocessing.preprocess
python -m ml.features.compute_baseline
python -m ml.training.train
python -m ml.evaluation.evaluate
uvicorn apps.api.sentiment_api.main:app --reload
```

Frontend:

```bash
cd apps/frontend
npm install
npm run dev
```

Docker stack:

```bash
cp .env.example .env
make docker-up
make docker-smoke
```

`docker-compose.yml` also contains safe defaults, so the stack can start without a local `.env` file. Use `.env` only when you want to override paths, credentials, or URLs.

## Main URLs

- Frontend: <http://localhost:5173>
- API: <http://localhost:8000>
- API docs: <http://localhost:8000/docs>
- MLflow: <http://localhost:5001>
- Airflow: <http://localhost:8080>
- Prometheus: <http://localhost:9091>
- Grafana: <http://localhost:3001>
- AlertManager: <http://localhost:19093>
- Node exporter metrics: <http://localhost:19100/metrics>

Default local credentials:

- Airflow: `admin` / `admin`
- Grafana: `admin` / `admin`

## MLOps Evidence

- `dvc.yaml` defines a reproducible lifecycle DAG.
- `params.yaml` controls DVC parameterized experiments for dataset size, split logic, model candidates, acceptance thresholds, and latency benchmarks.
- `dvc remote default local_artifacts` stores data/model artifacts in the local `dvc_remote/` directory for offline reproducibility demos.
- `airflow/dags/sentiment_training_pipeline.py` exposes the training lifecycle in Airflow.
- `airflow/dags/sentiment_batch_pipeline.py` adds incoming-file sensing, chunked processing, pools, quarantine handling, and pipeline alerts.
- `docs/data_card.md` documents dataset source, label mapping, limitations, and preprocessing.
- `docs/dvc_experiments.md` documents DVC reproducibility, `dvc repro`, `dvc checkout`, metrics, plots, and example parameter sweeps.
- `MLproject` supports reproducible MLflow project runs.
- `apps/api` exposes health, readiness, prediction, feedback, metrics, model metadata, and monitoring refresh APIs.
- `infra/prometheus`, `infra/alertmanager`, and `infra/grafana` provide monitoring, alert routing, silencing, and dashboards.
- `docs/docker_deployment.md` explains the Compose stack, health checks, credentials, and smoke test.
- `docs/monitoring.md` explains Prometheus metrics, Grafana panels, alert rules, and demo steps.
- `docs/` contains the architecture, design, testing, deployment, monitoring, and user-facing documentation used for submission.

## Docker Demo Commands

```bash
make compose-config
make docker-build
make docker-up
make docker-smoke
make docker-down
```

The smoke test checks the frontend, FastAPI health/readiness, prediction, Prometheus metrics, monitoring refresh, MLflow, Airflow, Prometheus readiness, and Grafana health.

## DVC Experiment Commands

```bash
dvc dag
dvc status
dvc repro
dvc metrics show
dvc plots show
dvc exp run -S training.acceptance_test_macro_f1=0.78
```

## Documentation Index

| Document | Purpose |
| --- | --- |
| [Architecture](docs/architecture.md) | System architecture, block explanations, data flow, and deployment view |
| [High-Level Design](docs/hld.md) | Design goals, service boundaries, and technology rationale |
| [Low-Level Design](docs/lld.md) | API definitions, schemas, module responsibilities, and error handling |
| [Test Plan](docs/test_plan.md) | Test strategy, test cases, and acceptance criteria |
| [Test Report](docs/test_report.md) | Current verification evidence and latest observed results |
| [User Manual](docs/user_manual.md) | Non-technical guide for using the application |
| [Data Card](docs/data_card.md) | Dataset source, label mapping, assumptions, and limitations |
| [Docker Deployment Guide](docs/docker_deployment.md) | Compose services, health checks, and deployment troubleshooting |
| [DVC Experiments Guide](docs/dvc_experiments.md) | Reproducibility workflow, parameterized experiments, and metrics/plots usage |
| [Monitoring Guide](docs/monitoring.md) | Prometheus, Grafana, AlertManager, and monitoring demo notes |
