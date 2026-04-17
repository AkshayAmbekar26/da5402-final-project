# User Manual

## Purpose

The application helps a non-technical user understand whether an e-commerce product review is positive, neutral, or negative.

## Start The Application

Recommended local demo:

```bash
cp .env.example .env
docker compose up --build
```

Open the web app:

```text
http://localhost:5173
```

## Analyze A Review

1. Open the **Analyzer** screen.
2. Paste or type a product review in the text area.
3. Select **Analyze**.
4. Read the result:
   - Sentiment label
   - Confidence percentage
   - Class probabilities
   - Influential words
   - Latency
   - Model version and MLflow run ID

## Submit Feedback

After a prediction, select the actual sentiment label. This stores feedback for monitoring real-world performance decay and future retraining.

## View MLOps Status

Open the **MLOps** screen to view:

- API health
- Model loading state
- Latest macro F1 score
- Drift status
- Pipeline stage timeline
- Model metadata
- Links to Airflow, MLflow, Prometheus, and Grafana

## Troubleshooting

- If prediction fails, confirm FastAPI is running at `http://localhost:8000`.
- If the model says fallback mode, run the training pipeline with `make demo` or `dvc repro`.
- If Grafana has no data, submit a few predictions so Prometheus has metrics to scrape.

