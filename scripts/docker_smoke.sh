#!/usr/bin/env bash
set -euo pipefail

API_BASE_URL="${API_BASE_URL:-http://localhost:8000}"
FRONTEND_URL="${FRONTEND_URL:-http://localhost:5173}"
MLFLOW_URL="${MLFLOW_URL:-http://localhost:${MLFLOW_HOST_PORT:-5001}}"
MLFLOW_MODEL_SERVER_URL="${MLFLOW_MODEL_SERVER_URL:-http://localhost:${MLFLOW_MODEL_SERVER_HOST_PORT:-5002}}"
AIRFLOW_URL="${AIRFLOW_URL:-http://localhost:8080}"
PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:${PROMETHEUS_HOST_PORT:-9091}}"
GRAFANA_URL="${GRAFANA_URL:-http://localhost:${GRAFANA_HOST_PORT:-3001}}"
ALERTMANAGER_URL="${ALERTMANAGER_URL:-http://localhost:${ALERTMANAGER_HOST_PORT:-19093}}"
NODE_EXPORTER_URL="${NODE_EXPORTER_URL:-http://localhost:${NODE_EXPORTER_HOST_PORT:-19100}}"
RETRIES="${RETRIES:-30}"
SLEEP_SECONDS="${SLEEP_SECONDS:-2}"

wait_for_url() {
  local name="$1"
  local url="$2"
  local method="${3:-GET}"
  local payload="${4:-}"

  for attempt in $(seq 1 "$RETRIES"); do
    if [[ "$method" == "POST" ]]; then
      if curl --fail --silent --show-error --max-time 10 \
        -H "Content-Type: application/json" \
        -X POST \
        -d "$payload" \
        "$url" >/dev/null; then
        printf "ok - %s\n" "$name"
        return 0
      fi
    else
      if curl --fail --silent --max-time 10 "$url" >/dev/null; then
        printf "ok - %s\n" "$name"
        return 0
      fi
    fi

    if [[ "$attempt" == "$RETRIES" ]]; then
      printf "failed - %s (%s)\n" "$name" "$url" >&2
      return 1
    fi
    sleep "$SLEEP_SECONDS"
  done
}

show_airflow_debug() {
  if command -v docker >/dev/null 2>&1; then
    docker compose --profile mlflow-serving ps airflow-webserver airflow-scheduler airflow-dag-processor || true
    docker compose --profile mlflow-serving logs --tail=250 --no-color airflow-webserver airflow-scheduler airflow-dag-processor || true
  fi
}

wait_for_airflow() {
  if [[ "${CHECK_AIRFLOW_INTERNAL:-false}" != "true" ]]; then
    wait_for_url "airflow" "$AIRFLOW_URL/api/v2/monitor/health"
    return $?
  fi

  for attempt in $(seq 1 "$RETRIES"); do
    if docker compose --profile mlflow-serving exec -T airflow-webserver \
      python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/v2/monitor/health', timeout=10)" >/dev/null 2>&1; then
      printf "ok - airflow\n"
      return 0
    fi

    if [[ "$attempt" == "$RETRIES" ]]; then
      printf "failed - airflow internal health check\n" >&2
      return 1
    fi
    sleep "$SLEEP_SECONDS"
  done
}

predict_payload='{"review_text":"Excellent product quality and fast delivery."}'

wait_for_url "frontend" "$FRONTEND_URL/"
wait_for_url "api health" "$API_BASE_URL/health"
wait_for_url "api readiness" "$API_BASE_URL/ready"
wait_for_url "api prediction" "$API_BASE_URL/predict" "POST" "$predict_payload"
wait_for_url "api prometheus metrics" "$API_BASE_URL/metrics"
wait_for_url "api monitoring refresh" "$API_BASE_URL/monitoring/refresh" "POST" "{}"
wait_for_url "mlflow" "$MLFLOW_URL/health"
if [[ "${RUN_MLFLOW_SERVING_SMOKE:-false}" == "true" ]]; then
  wait_for_url "mlflow model server health" "$MLFLOW_MODEL_SERVER_URL/ping"
  wait_for_url "mlflow model server prediction" "$MLFLOW_MODEL_SERVER_URL/invocations" "POST" '{"dataframe_records":[{"review_text":"Excellent product quality and fast delivery."}]}'
fi
if ! wait_for_airflow; then
  show_airflow_debug
  exit 1
fi
wait_for_url "prometheus" "$PROMETHEUS_URL/-/ready"
wait_for_url "grafana" "$GRAFANA_URL/api/health"
wait_for_url "alertmanager" "$ALERTMANAGER_URL/-/ready"
wait_for_url "node exporter" "$NODE_EXPORTER_URL/metrics"

printf "\nDocker smoke test passed.\n"
