# User Manual

## Purpose

SentinelAI is a product review sentiment analyzer. It helps a non-technical user paste an e-commerce review and understand whether the customer sentiment is `positive`, `neutral`, or `negative`.

The application also includes an MLOps dashboard for showing model health, data quality, pipeline status, drift, and monitoring links during the project demo.

## Start The Application

Recommended Docker demo:

```bash
cp .env.example .env
make docker-up
make docker-smoke
```

Open:

```text
http://localhost:5173
```

For lightweight local frontend/API development:

```bash
source .venv/bin/activate
uvicorn apps.api.sentiment_api.main:app --host 0.0.0.0 --port 8000
```

In a second terminal:

```bash
cd apps/frontend
npm run dev -- --port 5173
```

## Main Screens

### Analyzer

Use this screen for normal product-review sentiment prediction.

What you can do:

1. Paste a product review.
2. Select a positive, neutral, or negative example if you do not have a review ready.
3. Press **Analyze Sentiment**.
4. Read the result card.
5. Submit optional feedback if the actual sentiment is known.

The result shows:

- Predicted sentiment
- Confidence percentage
- Class probability breakdown
- Influential words
- Request latency
- Model version
- MLflow run ID
- API target

### MLOps

Use this screen for demonstration and operational monitoring.

It shows:

- API health
- Model readiness
- Macro F1 score
- Drift status
- 9-stage pipeline lifecycle
- Status-aware stage labels: success, warning, failed, pending
- Recent pipeline events
- Raw, processed, and rejected rows
- Pipeline duration and stage throughput
- Model metadata and Git/DVC/MLflow traceability
- Data quality warnings
- Links to Airflow, MLflow, Prometheus, Grafana, and AlertManager

### Guide Panel

Click **Guide** in the top bar to open the in-app user manual. It explains the user flow in simple steps.

### Product Tour

Click **Tour** in the top bar to start the guided walkthrough. The tour highlights navigation, review input, example buttons, the Analyze button, the result area, and the guide.

## Analyze A Review

1. Open the **Analyzer** screen.
2. Paste a review such as:

   ```text
   Excellent quality and the delivery was faster than expected. I would buy this again.
   ```

3. Click **Analyze Sentiment**.
4. Confirm the result appears.
5. Read the confidence ring and probability bars.
6. Use influential words to explain why the model made the prediction.

## Submit Feedback

After a prediction:

1. Find the **Was this correct?** feedback section.
2. Select the actual sentiment label.
3. The feedback is stored for monitoring and future retraining analysis.

Feedback helps demonstrate the real-world loop where model predictions are compared against later ground-truth labels.

## View MLOps Status

1. Click the **MLOps** tab.
2. Press **Refresh**.
3. Check the KPI cards at the top.
4. Review the pipeline lifecycle timeline.
5. Review the **Recent Pipeline Events** panel.
6. Use the infrastructure links to open Airflow, MLflow, Prometheus, Grafana, and AlertManager.

During the demo, explain:

- Airflow is used for orchestration visibility and logs.
- DVC is used for reproducibility and artifact lineage.
- MLflow is used for experiment tracking and model registry metadata.
- Prometheus/Grafana are used for monitoring and alerting.

## Tool URLs

| Tool | URL |
| --- | --- |
| Frontend | `http://localhost:5173` |
| API | `http://localhost:8000` |
| API docs | `http://localhost:8000/docs` |
| MLflow | `http://localhost:5001` |
| Airflow | `http://localhost:8080` |
| Prometheus | `http://localhost:9091` |
| Grafana | `http://localhost:3001` |
| AlertManager | `http://localhost:19093` |
| Node exporter | `http://localhost:19100/metrics` |

Default local credentials:

| Tool | Username | Password |
| --- | --- | --- |
| Airflow | `admin` | `admin` |
| Grafana | `admin` | `admin` |

## Troubleshooting

| Problem | What to check |
| --- | --- |
| Prediction fails | Confirm API is running at `http://localhost:8000/ready` |
| Model not ready | Run `dvc repro` or start the Docker stack after model artifacts are available |
| MLOps screen says unavailable | Press Refresh and confirm `/metrics-summary` works |
| Grafana has no data | Submit a few predictions and call `/monitoring/refresh` |
| Airflow page unavailable | Check `docker compose ps` and `docker compose logs airflow-webserver` |
| MLflow page unavailable | Check `docker compose logs mlflow` |
| Docker stack unhealthy | Run `make docker-smoke` and inspect the failing service |

## Interpreting The Result

- **Positive:** review expresses satisfaction, praise, or willingness to buy again.
- **Neutral:** review is mixed, average, or factual without strong satisfaction/dissatisfaction.
- **Negative:** review expresses dissatisfaction, damage, failure, poor quality, or complaint.
- **Confidence:** model-estimated probability of the selected label.
- **Influential words:** terms that pushed the model toward or away from the predicted label.

The model is not a human judgment system. It is a local ML classifier used to demonstrate a complete MLOps lifecycle.

