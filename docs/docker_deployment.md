# Docker Deployment Guide

## Services

Docker Compose runs the local demo stack:

| Service | Port | Purpose |
| --- | ---: | --- |
| `frontend` | 5173 | React/Vite UI served by Nginx |
| `api` | 8000 | FastAPI inference and operational APIs |
| `mlflow` | 5001 | Experiment tracking and model registry UI |
| `airflow-webserver` | 8080 | Pipeline orchestration UI |
| `airflow-scheduler` | internal | Airflow DAG scheduler |
| `airflow-dag-processor` | internal | Airflow 3 DAG parsing and import validation |
| `postgres` | internal | Airflow metadata database |
| `prometheus` | 9091 | Metrics scraping and alert rules |
| `alertmanager` | 19093 | Alert routing, notification receiver status, and silences |
| `node-exporter` | 19100 | Local infrastructure CPU, memory, disk, and load metrics |
| `grafana` | 3001 | Monitoring dashboards |

## Start And Verify

```bash
make compose-config
make docker-up
make docker-smoke
```

`make docker-smoke` checks the frontend, API health/readiness, prediction, Prometheus metrics, monitoring refresh, MLflow, Airflow, Prometheus readiness, Grafana health, AlertManager readiness, and node exporter metrics.

To include the separate MLflow model server in the deployment smoke test:

```bash
docker compose --profile mlflow-serving up -d --build
RUN_MLFLOW_SERVING_SMOKE=true make docker-smoke
```

The API uses local model serving by default for the fastest demo path. The optional `mlflow-serving` profile proves that the same exported MLflow PyFunc model can also be hosted behind a dedicated model-server container.

## Automated Deployment Validation

GitHub Actions performs local deployment validation without using a cloud runtime:

1. Run Python linting, tests, and `dvc repro`.
2. Upload generated model/report artifacts for the deployment job.
3. Build the Docker Compose services.
4. Start the stack with the `mlflow-serving` profile.
5. Run `scripts/docker_smoke.sh` against the live frontend, API, MLflow, MLflow model server, Airflow, Prometheus, Grafana, AlertManager, and node exporter.
6. Upload Compose logs on failure and tear the stack down.

This is the project CD equivalent: the deployment target is a reproducible local Docker Compose stack instead of cloud production.

## Credentials

| Tool | Username | Password |
| --- | --- | --- |
| Airflow | `admin` | `admin` |
| Grafana | `admin` | `admin` |

Airflow 3 uses `SimpleAuthManager`; `admin:admin` in `AIRFLOW_SIMPLE_AUTH_MANAGER_USERS` means username `admin` with role `admin`. The deterministic demo password is stored in `airflow/config/simple_auth_manager_passwords.json` and mounted through the project volume.

## Environment Overrides

The Compose file has safe defaults and can start without `.env`. Copy `.env.example` to `.env` when you want to override URLs, credentials, or model paths.

Important variables:

- `MODEL_PATH`
- `MODEL_METADATA_PATH`
- `FEEDBACK_PATH`
- `CORS_ORIGINS`
- `MLFLOW_TRACKING_URI`
- `GRAFANA_ADMIN_USER`
- `GRAFANA_ADMIN_PASSWORD`
- `AIRFLOW_UID`
- `AIRFLOW_EXECUTION_API_SERVER_URL`
- `AIRFLOW_SIMPLE_AUTH_MANAGER_USERS`
- `AIRFLOW_API_AUTH_JWT_SECRET`
- `MODEL_SERVING_MODE`
- `MLFLOW_SERVING_URL`

## Health Checks

Health checks are defined for:

- API `/ready`
- Frontend Nginx root page
- MLflow `/health`
- Airflow `/api/v2/monitor/health`
- Airflow scheduler job check
- Postgres `pg_isready`
- Prometheus `/-/ready`
- Grafana `/api/health`

These checks support demo reliability and make service readiness visible with:

```bash
docker compose ps
```

The smoke test additionally checks AlertManager `/-/ready` and node exporter `/metrics` from the host.

## Troubleshooting

- If Docker commands fail with a daemon socket error, start Docker Desktop or the local Docker daemon.
- If the API is unhealthy, check `docker compose logs api`.
- If Grafana has no data, submit predictions and then open `http://localhost:8000/metrics`.
- If Airflow is slow to become healthy, wait for `airflow-init` to finish and inspect `docker compose logs airflow-init`.
- If ports are already in use, stop the conflicting local service or change the published port in `docker-compose.yml`.

## Rollback

Rollback is implemented through `scripts/rollback_model.sh`. The script is intentionally non-destructive: when a Git/DVC revision is provided, it uses `dvc get` to copy the old model artifacts into `models/rollback/` instead of checking out the whole repository.

Generate a rollback env file for the current production artifact:

```bash
make rollback-current
```

Generate a rollback env file from a previous Git/DVC revision:

```bash
make rollback ROLLBACK_ARGS="--git-rev <commit-or-tag>"
```

Apply and verify the rollback against the running API:

```bash
make rollback-restart ROLLBACK_ARGS="--git-rev <commit-or-tag>"
```

The script writes `.env.rollback`, then applies it with:

```bash
docker compose --env-file .env.rollback up -d api
```

This gives a concrete rollback mechanism for failed deployments while keeping Git history, DVC artifacts, and MLflow run metadata traceable.
