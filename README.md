# Product Review Sentiment Analyzer with MLOps

This project is a local, end-to-end AI application for classifying e-commerce product reviews as `positive`, `neutral`, or `negative`. It is intentionally built around MLOps evidence: DVC pipelines, Airflow orchestration, MLflow tracking, FastAPI serving, React UI, Prometheus/Grafana monitoring, Docker Compose packaging, CI, and documentation.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m ml.data_ingestion.ingest
python -m ml.validation.validate_data
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
docker compose up --build
```

## Main URLs

- Frontend: <http://localhost:5173>
- API: <http://localhost:8000>
- API docs: <http://localhost:8000/docs>
- MLflow: <http://localhost:5000>
- Airflow: <http://localhost:8080>
- Prometheus: <http://localhost:9090>
- Grafana: <http://localhost:3000>

## MLOps Evidence

- `dvc.yaml` defines a reproducible lifecycle DAG.
- `airflow/dags/sentiment_training_pipeline.py` exposes the same lifecycle in Airflow.
- `MLproject` supports reproducible MLflow project runs.
- `apps/api` exposes health, readiness, prediction, feedback, metrics, and model metadata APIs.
- `infra/prometheus` and `infra/grafana` provide monitoring and alerting.
- `docs/` contains HLD, LLD, architecture, test plan, user manual, MLOps report, and viva notes.

