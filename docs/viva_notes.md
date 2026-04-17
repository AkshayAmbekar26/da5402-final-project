# Viva Notes

## One-Minute Explanation

This is an end-to-end MLOps project for product review sentiment analysis. The model classifies reviews into positive, neutral, or negative sentiment. The important part is not model novelty but the lifecycle: data ingestion, validation, preprocessing, training, experiment tracking, versioning, deployment, monitoring, feedback, and retraining readiness.

## Why This Problem

E-commerce platforms receive many reviews, and manually reading them is slow. A sentiment analyzer gives fast feedback summaries while still being simple enough to demonstrate strong MLOps practices.

## Dataset

The primary dataset is `SetFit/amazon_reviews_multi_en`, an English Amazon review dataset available through Hugging Face. The source label is mapped to a 1-5 rating and then grouped into negative, neutral, and positive sentiment. The default training dataset uses 1-star, 3-star, and 5-star reviews only so the three sentiment classes are unambiguous; 2-star and 4-star reviews are excluded as borderline or mixed sentiment. The local seed dataset is kept only as an offline fallback for demo reliability.

## Why TF-IDF + Logistic Regression

It is fast, explainable, reproducible, and suitable for local hardware. It supports latency under 200 ms and lets us show influential words. A transformer can be added later but is not necessary for the rubric.

## MLOps Tool Choices

- **DVC:** data/model versioning and reproducible DAG.
- **Airflow:** visual pipeline orchestration and run history.
- **MLflow:** experiment tracking and model registry.
- **FastAPI:** clean REST serving layer.
- **React:** polished and loosely coupled frontend.
- **Prometheus/Grafana:** near-real-time monitoring and alerting.
- **Docker Compose:** local environment parity without cloud services.

## Defensible Tradeoffs

- Public Amazon review data is used for training, and local seed data is included only so the demo works without internet.
- The API has fallback mode so the UI stays usable before training, while `/ready` clearly reports model state.
- Airflow and DVC both exist because they serve different rubric needs: orchestration visibility and reproducibility.

## Problems Faced And Mitigation

- **No cloud allowed:** all services run through Docker Compose.
- **Local hardware limits:** chose lightweight model.
- **Demo reliability:** included offline data and fallback prediction path.
- **Monitoring complexity:** exposed focused metrics instead of overengineering production observability.

## Incomplete Items To Admit Honestly

- TLS, authentication, and encrypted production storage are documented but not implemented for local demo.
- Category-level bias is not analyzed because the reduced SetFit dataset schema does not include product categories.
- Automated retraining trigger is represented by Airflow/DVC pipeline and feedback logging, but not scheduled continuously by default.
