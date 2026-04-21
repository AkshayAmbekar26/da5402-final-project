# Viva Notes

## One-Minute Explanation

This is an end-to-end local MLOps project for product review sentiment analysis. The app classifies e-commerce reviews as positive, neutral, or negative. The main achievement is not model novelty; it is the complete lifecycle: data ingestion, validation, EDA, preprocessing, feature baselines, experiment tracking, DVC reproducibility, Airflow orchestration, model acceptance, FastAPI deployment, React UI, Prometheus/Grafana monitoring, AlertManager alerting, feedback logging, Docker packaging, and documentation.

## Why This Problem

E-commerce platforms receive many customer reviews. A sentiment analyzer lets a business user quickly identify whether reviews are favorable, mixed, or negative. The problem is simple enough to demo clearly while still supporting a full MLOps lifecycle.

## Dataset

The primary dataset is `SetFit/amazon_reviews_multi_en`, an English Amazon review dataset available through Hugging Face. The project maps source labels to 1-5 star ratings and then maps ratings to sentiment. The default training data uses 1-star, 3-star, and 5-star reviews so the three sentiment classes are less ambiguous. A local seed fallback exists only for offline demo reliability.

## Why TF-IDF Plus Logistic Regression

This model is fast, explainable, reproducible, and practical on local hardware. It also supports word-level explanation through learned feature weights. The pipeline compares multiple candidate models in MLflow and promotes the best accepted candidate using a documented rule.

Current result:

- Selected model: `tfidf_logistic_tuned`
- Test macro F1: `0.7737`
- Latency: `0.0467 ms` per review
- MLflow run ID: `61f2ee995e7d4084a210a3513d83eec8`

## MLOps Tool Choices

| Tool | Why it is used |
| --- | --- |
| Git | Source, config, infrastructure, docs versioning |
| DVC | Data/model artifact versioning and reproducible pipeline DAG |
| Airflow | Visual orchestration, retries, logs, operational DAG console |
| MLflow | Experiment tracking, metrics, parameters, artifacts, model registry metadata |
| FastAPI | Typed REST inference and operational APIs |
| React/Vite | Polished and independent frontend |
| Prometheus | Metrics scraping and alert evaluation |
| Grafana | Monitoring visualization |
| AlertManager | Alert routing and silencing |
| node_exporter | CPU, memory, disk, filesystem, and load metrics |
| Docker Compose | Local environment parity and multi-service packaging |

## Key Defenses

### Why both Airflow and DVC?

Airflow and DVC solve different problems. Airflow is the operational orchestrator and visual pipeline console. DVC is the reproducibility and artifact-lineage system. Airflow answers "what ran and did it fail?" DVC answers "can I reproduce the exact data/model state?"

### How is reproducibility guaranteed?

Every run is tied to:

- Git commit hash
- DVC data/artifact state
- MLflow run ID
- fixed random seed
- `params.yaml`
- `dvc.yaml`
- Docker/MLproject environment definitions

### How do you know the model is accepted?

The acceptance gate checks:

- test macro F1 `>= 0.75`
- latency `< 200 ms`

The result is saved in `reports/acceptance_gate.json`. The DVC pipeline fails if the selected model does not pass.

### How is monitoring implemented?

FastAPI exposes `/metrics`. Prometheus scrapes it. Grafana visualizes API traffic, latency, errors, prediction distribution, model readiness, model acceptance, macro F1, drift, data quality, feedback accuracy, pipeline duration, stage throughput, and host resources. AlertManager receives alerts and supports silencing.

### How is the feedback loop implemented?

After prediction, the user can submit the actual sentiment. The API stores this feedback and updates Prometheus counters. This allows future calculation of real-world accuracy and retraining triggers.

### How would rollback work?

Use MLflow and DVC to identify and restore a previous accepted model. Then configure the API to load that model artifact or registry version, restart the API, and verify `/ready`, `/predict`, and monitoring.

## Problems Faced And Mitigations

| Problem | Mitigation |
| --- | --- |
| No cloud allowed | Used local Docker Compose services |
| Local hardware constraints | Used lightweight TF-IDF model |
| Internet may fail during demo | Added local seed fallback dataset |
| Need visible orchestration | Added Airflow DAGs |
| Need reproducible experiments | Added DVC stages and params |
| Need monitoring beyond API | Added node_exporter, AlertManager, Grafana dashboards |
| Need UI marks | Redesigned frontend with guide, tour, analyzer, and MLOps command center |
| Need failure visibility | Added status-aware pipeline stages and recent events panel |

## Incomplete Items To Admit Honestly

- TLS and authentication are documented but not enabled in the local course demo.
- Continuous scheduled retraining is not enabled by default, but DVC/Airflow can run retraining.
- Product-category bias analysis is limited because the selected dataset subset does not include category metadata.
- Transformer models are future work because the baseline already satisfies speed, explainability, and reproducibility needs.

## Short Answers For Likely Questions

**What dataset did you use?**
`SetFit/amazon_reviews_multi_en`, with a local seed fallback only for offline reliability.

**Why not use a transformer?**
The course is grading MLOps completeness. TF-IDF plus Logistic Regression is faster, explainable, reproducible, and sufficient for the target F1.

**Where is experiment tracking?**
MLflow logs candidate runs, hyperparameters, metrics, artifacts, Git commit, DVC data state, and selected model metadata.

**Where is the CI-style pipeline?**
DVC defines the reproducible DAG in `dvc.yaml`, and GitHub Actions runs tests/build checks.

**Where is the pipeline console?**
Airflow gives the DAG console, and the frontend MLOps dashboard gives a stakeholder-friendly summary.

**How do you monitor drift?**
Baseline statistics are computed during feature generation. A drift script compares current data against baseline and exports report-backed Prometheus gauges.

**How do you monitor infrastructure?**
node_exporter exports CPU, memory, disk, filesystem, and load metrics to Prometheus and Grafana.
