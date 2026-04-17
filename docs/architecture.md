# Architecture

## System Overview

The application is split into independent services so the frontend and backend model inference engine are loosely coupled through REST APIs only.

```mermaid
flowchart LR
  User["Non-technical user"] --> Frontend["React/Vite frontend"]
  Frontend -->|REST /predict /feedback /model/info| API["FastAPI inference API"]
  API --> Model["Local model artifact"]
  API --> Feedback["Feedback log"]
  API --> Metrics["Prometheus /metrics"]
  Prometheus["Prometheus"] --> Grafana["Grafana dashboards and alerts"]
  Airflow["Airflow DAG"] --> Pipeline["ML pipeline package"]
  DVC["DVC DAG"] --> Pipeline
  Pipeline --> Data["DVC-versioned data"]
  Pipeline --> MLflow["MLflow experiments and registry"]
  Pipeline --> Model
```

## Blocks

- **Frontend:** Non-technical UI for review analysis and an MLOps dashboard.
- **FastAPI API:** Owns prediction, feedback, health, readiness, model metadata, and Prometheus metrics.
- **ML package:** Owns ingestion, validation, preprocessing, baseline statistics, training, evaluation, drift detection, and report publishing.
- **Airflow:** Provides visual orchestration, retries, status, and run history.
- **DVC:** Provides reproducible data/model pipeline DAG and artifact versioning.
- **MLflow:** Tracks parameters, metrics, artifacts, Git commit hash, DVC state, and registered model versions.
- **Prometheus/Grafana:** Monitors API health, latency, error rates, model loading, prediction distribution, feedback, and drift.

## Deployment View

```mermaid
flowchart TB
  subgraph DockerCompose["Docker Compose network"]
    FE["frontend: nginx + static React"]
    API["api: FastAPI + model artifact"]
    MLF["mlflow: tracking server"]
    PG["postgres: Airflow metadata"]
    AW["airflow-webserver"]
    AS["airflow-scheduler"]
    PR["prometheus"]
    GF["grafana"]
  end
  FE --> API
  AS --> API
  AS --> MLF
  AW --> PG
  AS --> PG
  PR --> API
  GF --> PR
```

## Data Flow

```mermaid
flowchart LR
  Raw["Raw reviews"] --> Validate["Schema and quality validation"]
  Validate --> EDA["EDA reports and charts"]
  EDA --> Split["Train/validation/test split"]
  Split --> Baseline["Drift baseline statistics"]
  Split --> Train["TF-IDF + Logistic Regression"]
  Train --> Evaluate["Metrics and acceptance gate"]
  Evaluate --> Registry["MLflow model registry"]
  Registry --> Serve["FastAPI model serving"]
  Serve --> Feedback["Ground-truth feedback loop"]
```

## Security Notes

The submitted project is local-only and contains no private customer data. Secrets are kept out of Git through `.env` and `.env.example`. Production-like deployment would require TLS, authenticated service endpoints, encrypted artifact storage, and stricter access control around Airflow, MLflow, and Grafana.
