# Monitoring And Alerting

## Purpose

Monitoring is designed to show both application health and ML lifecycle health during the demo. The FastAPI service exports Prometheus metrics at `/metrics`, refreshes lifecycle gauges from the latest pipeline reports, and exposes a frontend-friendly summary at `/metrics-summary`.

## Metric Sources

| Source | Files or Endpoint | What It Represents |
| --- | --- | --- |
| FastAPI middleware | `/metrics` | Request count, status codes, request latency, and API errors |
| Prediction service | `/predict` | Inference latency and prediction distribution |
| Feedback endpoint | `/feedback` and `data/feedback/feedback.jsonl` | Feedback volume and feedback accuracy ratio |
| Training reports | `reports/evaluation.json`, `reports/model_comparison.json` | Macro F1, acceptance status, candidate counts |
| Data reports | `reports/ingestion_report.json`, `reports/preprocessing_report.json` | Raw rows, processed rows, rejected rows, rejected ratio |
| Drift report | `reports/drift_report.json` | Drift score and drift-detected flag |
| Maintenance report | `reports/maintenance_report.json` | Retraining decision, drift trigger state, and feedback degradation state |
| Pipeline performance | `reports/pipeline_performance.json` | Full pipeline duration, stage duration, and stage throughput |
| Node exporter | `node-exporter:9100/metrics` | CPU, memory, disk, filesystem, and host load |
| AlertManager | `alertmanager:9093` | Routed alert notifications and silence state |

## Prometheus Metrics

Key exported metrics include:

- `sentiment_api_requests_total`
- `sentiment_api_request_latency_seconds`
- `sentiment_api_errors_total`
- `sentiment_predictions_total`
- `sentiment_model_inference_latency_seconds`
- `sentiment_review_text_length_chars`
- `sentiment_model_loaded`
- `sentiment_model_fallback_mode`
- `sentiment_model_accepted`
- `sentiment_model_test_macro_f1`
- `sentiment_model_candidate_count`
- `sentiment_model_accepted_candidate_count`
- `sentiment_model_info_info`
- `sentiment_data_drift_score`
- `sentiment_data_drift_detected`
- `sentiment_pipeline_duration_seconds`
- `sentiment_pipeline_stage_duration_seconds`
- `sentiment_pipeline_stage_throughput_rows_per_second`
- `sentiment_data_raw_rows`
- `sentiment_data_processed_rows`
- `sentiment_data_rejected_rows`
- `sentiment_data_rejected_ratio`
- `sentiment_feedback_total`
- `sentiment_feedback_matches_total`
- `sentiment_feedback_accuracy_ratio`
- `sentiment_alert_notifications_total`

Infrastructure metrics from node exporter include:

- `node_cpu_seconds_total`
- `node_memory_MemAvailable_bytes`
- `node_memory_MemTotal_bytes`
- `node_filesystem_avail_bytes`
- `node_filesystem_size_bytes`
- `node_load1`

`POST /monitoring/refresh` can be called to force the API to reload report-backed metrics before opening Prometheus or Grafana.

## Maintenance Automation

Airflow includes `sentiment_monitoring_maintenance`, a scheduled maintenance DAG. It runs drift detection, evaluates the retraining policy, and triggers `sentiment_training_pipeline` when maintenance thresholds are crossed.

Configurable thresholds:

- `SENTIMENT_RETRAIN_DRIFT_THRESHOLD`
- `SENTIMENT_RETRAIN_MIN_FEEDBACK_COUNT`
- `SENTIMENT_RETRAIN_MIN_FEEDBACK_ACCURACY`
- `SENTIMENT_RETRAIN_COOLDOWN_HOURS`

Configurable schedules:

- `SENTIMENT_TRAINING_SCHEDULE`
- `SENTIMENT_MAINTENANCE_SCHEDULE`

The cooldown prevents the same unresolved drift condition from launching a new full training run every hour. The maintenance report records the last trigger time, cooldown window, decision, and reason.

## Alert Routing

Prometheus sends firing and resolved alerts to AlertManager at `alertmanager:9093`. AlertManager groups alerts by `alertname` and `severity`, then sends a demo webhook notification to the API endpoint `POST /ops/alerts`. The API counts received notifications with `sentiment_alert_notifications_total` so the dashboard can show alert-routing activity.

AlertManager also provides the silence workflow used during maintenance. During a demo, open `http://localhost:19093`, choose a firing alert, and create a silence to show how noisy alerts can be suppressed without deleting Prometheus rules.

## Grafana Dashboard

The provisioned dashboard is `Product Review Sentiment MLOps`. It contains panels for:

- Model loaded state
- Serving mode and fallback detection
- Model acceptance gate
- Test macro F1
- Pipeline duration
- Drift score
- API request throughput
- P95 inference latency
- Error rate
- Prediction distribution
- Raw, processed, and rejected row counts
- Rejected row ratio
- Pipeline stage duration and throughput
- Candidate model counts
- Feedback labels and feedback accuracy
- Host CPU, memory, disk, load, and filesystem availability
- Node exporter, AlertManager, and Prometheus scrape health
- Firing alerts and AlertManager notification activity
- Review text length Summary metric

## Alerts

Prometheus alert rules are defined in `infra/prometheus/alerts.yml`.

| Alert | Trigger | Demo Meaning |
| --- | --- | --- |
| `SentimentApiDown` | API scrape target is unavailable | Backend or Docker service is down |
| `SentimentApiHighErrorRate` | 5xx error rate is above 5% | API is failing requests |
| `SentimentModelNotLoaded` | `sentiment_model_loaded == 0` | Trained model artifact is missing or failed to load |
| `SentimentModelFallbackMode` | `sentiment_model_fallback_mode == 1` | API is serving fallback predictions |
| `SentimentModelAcceptanceFailed` | `sentiment_model_accepted == 0` | Latest model failed promotion criteria |
| `SentimentDriftDetected` | Drift flag or drift score exceeds threshold | Current data distribution differs from baseline |
| `SentimentPipelineDurationHigh` | Pipeline duration exceeds 5 minutes | Pipeline is too slow for expected local demo size |
| `SentimentRejectedRowsHigh` | More than 5% rejected rows | Data quality has degraded |
| `SentimentApiLatencyHigh` | P95 inference latency exceeds 200 ms | Business latency target is violated |
| `NodeExporterDown` | Node exporter target is unavailable | Infrastructure monitoring is unavailable |
| `HostCpuUsageHigh` | CPU usage exceeds 85% | Resource pressure may explain latency spikes |
| `HostMemoryUsageHigh` | Memory usage exceeds 85% | Services may become unstable |
| `HostDiskUsageHigh` | Disk usage exceeds 85% | DVC, MLflow, logs, or Docker volumes may fail |
| `AlertManagerDown` | AlertManager target is unavailable | Notifications and silences are unavailable |

## Demo Steps

1. Start the stack with `make docker-up` or `docker compose up`.
2. Submit several predictions in the frontend so request and prediction metrics change.
3. Submit at least one feedback label.
4. Call `POST http://localhost:8000/monitoring/refresh` or open `/metrics-summary`.
5. Open Prometheus at `http://localhost:9091` and query `sentiment_model_test_macro_f1`, `node_load1`, and `ALERTS`.
6. Open Grafana at `http://localhost:3001` and show the `Product Review Sentiment MLOps` dashboard.
7. Open AlertManager at `http://localhost:19093` and show alert groups plus the silence workflow.
