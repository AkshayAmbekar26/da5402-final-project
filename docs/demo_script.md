# Demo Script

## Before The Demo

```bash
source .venv/bin/activate
dvc repro
pytest
cd apps/frontend && npm run build
cd ../..
docker compose config
```

## Live Demo Flow

1. Open `docs/architecture.md` and explain the service blocks.
2. Open the frontend at `http://localhost:5173`.
3. Submit one positive, one neutral, and one negative review.
4. Point out sentiment, confidence, influential words, latency, model version, and MLflow run ID.
5. Submit feedback and explain the ground-truth feedback loop.
6. Open the MLOps screen and show model metadata, drift status, and tool links.
7. Open API docs at `http://localhost:8000/docs`.
8. Show Airflow at `http://localhost:8080` and the `sentiment_training_pipeline` DAG.
9. Show MLflow at `http://localhost:5000` and the registered `ProductReviewSentimentModel`.
10. Show Prometheus at `http://localhost:9090` and Grafana at `http://localhost:3000`.
11. Run `dvc dag` or open `dvc.yaml` to explain reproducibility.
12. Show `docs/test_report.md` and explain acceptance criteria.

## Key Lines To Say

- The model is intentionally lightweight because the project is graded on MLOps completeness and local reproducibility.
- Airflow gives visual orchestration; DVC gives reproducibility and artifact versioning.
- MLflow connects every model to metrics, parameters, artifacts, and run ID.
- The frontend and backend are loosely coupled through REST APIs only.
- Prometheus and Grafana monitor latency, errors, prediction distribution, model loaded state, feedback, and drift.

