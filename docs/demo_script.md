# Demo Script

## Pre-Demo Checklist

Run these before the live presentation:

```bash
source .venv/bin/activate
ruff check .
pytest
dvc status
dvc metrics show
cd apps/frontend && npm run build
cd ../..
make compose-config
```

For the full Docker demo:

```bash
make docker-up
make docker-smoke
```

Stop after rehearsal:

```bash
make docker-down
```

## Live Demo Flow

1. Open `docs/submission_checklist.md`.
   - Explain that this maps each rubric item to evidence.

2. Open `docs/architecture.md`.
   - Explain frontend, API, ML pipeline, DVC, Airflow, MLflow, Prometheus, Grafana, AlertManager, and Docker Compose.

3. Start or show the running stack.
   - Frontend: `http://localhost:5173`
   - API docs: `http://localhost:8000/docs`

4. Open the frontend Analyzer.
   - Mention it is for non-technical users.
   - Show the Guide panel.
   - Optionally show the Tour.

5. Submit a positive review.
   - Point out sentiment, confidence, probabilities, influential words, latency, model version, and MLflow run ID.

6. Submit a neutral review and a negative review.
   - Show that the UI handles all three classes.

7. Submit feedback.
   - Explain the ground-truth feedback loop.

8. Open the frontend MLOps tab.
   - Show KPI cards.
   - Show the status-aware pipeline lifecycle.
   - Show Recent Pipeline Events.
   - Show speed and throughput.
   - Show tool links.

9. Open Airflow at `http://localhost:8080`.
   - Login: `admin` / `admin`.
   - Show `sentiment_training_pipeline`.
   - Show `sentiment_batch_ingestion_pipeline`.
   - Explain retries, logs, DAG status, and batch quarantine behavior.

10. Open MLflow at `http://localhost:5001`.
    - Show candidate runs.
    - Show metrics and parameters.
    - Show selected run ID: `61f2ee995e7d4084a210a3513d83eec8`.
    - Show registered model name: `ProductReviewSentimentModel`.

11. Show DVC reproducibility.

    ```bash
    dvc dag
    dvc metrics show
    dvc status
    ```

    Explain that changing `params.yaml` reruns only affected stages.

12. Open Prometheus at `http://localhost:9091`.
    - Query `sentiment_model_test_macro_f1`.
    - Query `sentiment_pipeline_duration_seconds`.
    - Query `sentiment_data_rejected_ratio`.
    - Query `ALERTS`.

13. Open Grafana at `http://localhost:3001`.
    - Login: `admin` / `admin`.
    - Show API latency, request rate, error rate, prediction distribution, model acceptance, drift, pipeline duration, stage throughput, feedback, and host resource panels.

14. Open AlertManager at `http://localhost:19093`.
    - Explain alert grouping and silencing for maintenance.

15. Open `docs/test_report.md`.
    - Show current test evidence.

16. Explain rollback.
    - Previous model versions are retained through MLflow and DVC.
    - API can be pointed to a prior model artifact/version and restarted.

## Key Lines To Say

- The model is intentionally lightweight because this project is graded on MLOps completeness, not deep-learning novelty.
- Airflow gives visual orchestration; DVC gives exact reproducibility and artifact versioning.
- MLflow connects every model to parameters, metrics, artifacts, Git commit, DVC state, and run ID.
- The selected model is promoted through an explicit acceptance gate, not manually chosen.
- The frontend and backend are loosely coupled through REST APIs only.
- The MLOps dashboard is status-aware and surfaces both successful runs and warnings.
- Prometheus, AlertManager, node_exporter, and Grafana monitor application health, model health, data drift, data quality, feedback, pipeline speed, and host resources.
- Docker Compose keeps frontend, API, Airflow, MLflow, Prometheus, Grafana, AlertManager, node_exporter, and Postgres as separate local services.

## If Something Fails During Demo

| Failure | Quick response |
| --- | --- |
| Frontend cannot predict | Open `http://localhost:8000/ready` and check API logs |
| API not ready | Re-run `dvc repro` or restart API service |
| Grafana has no metrics | Submit a prediction, then call `/monitoring/refresh` |
| Airflow slow | Show DVC DAG and explain Airflow startup can take time locally |
| Docker service unhealthy | Run `docker compose ps` and `docker compose logs <service>` |
| Browser UI issue | Use API docs `/docs` to prove backend still works |
