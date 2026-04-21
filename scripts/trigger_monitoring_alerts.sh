#!/usr/bin/env bash
set -euo pipefail

API_BASE_URL="${API_BASE_URL:-http://localhost:8000}"
PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:${PROMETHEUS_HOST_PORT:-9091}}"
SCENARIO="${1:-invalid-reviews}"
REQUESTS="${REQUESTS:-80}"
SLEEP_SECONDS="${SLEEP_SECONDS:-0.05}"
HOLD_SECONDS="${HOLD_SECONDS:-90}"

post_json() {
  local url="$1"
  local payload="$2"
  curl --silent --show-error --max-time 10 \
    -H "Content-Type: application/json" \
    -X POST \
    -d "$payload" \
    "$url" >/dev/null || true
}

invalid_reviews() {
  printf "Sending %s invalid prediction requests to exercise input-quality metrics and alerts...\n" "$REQUESTS"
  for index in $(seq 1 "$REQUESTS"); do
    case $((index % 3)) in
      0) post_json "$API_BASE_URL/predict" '{}' ;;
      1) post_json "$API_BASE_URL/predict" '{"review_text":""}' ;;
      2) post_json "$API_BASE_URL/predict" "{\"review_text\":\"$(printf 'x%.0s' $(seq 1 5100))\"}" ;;
    esac
    sleep "$SLEEP_SECONDS"
  done
}

prediction_load() {
  printf "Sending %s normal prediction requests to populate throughput, latency, and distribution panels...\n" "$REQUESTS"
  for index in $(seq 1 "$REQUESTS"); do
    case $((index % 3)) in
      0) payload='{"review_text":"Excellent product quality, premium packaging, and very fast delivery."}' ;;
      1) payload='{"review_text":"The product is acceptable and works as described, but nothing special."}' ;;
      2) payload='{"review_text":"Terrible quality, damaged box, delayed delivery, and disappointing support."}' ;;
    esac
    post_json "$API_BASE_URL/predict" "$payload"
    sleep "$SLEEP_SECONDS"
  done
}

server_errors() {
  printf "Triggering demo 5xx errors. ENABLE_DEMO_OPS_ENDPOINTS must be true for the API service.\n"
  for _ in $(seq 1 "$REQUESTS"); do
    post_json "$API_BASE_URL/ops/demo/error" '{}'
    sleep "$SLEEP_SECONDS"
  done
}

api_down() {
  printf "Stopping the api container for %s seconds to demonstrate SentimentApiDown, then starting it again...\n" "$HOLD_SECONDS"
  docker compose stop api
  sleep "$HOLD_SECONDS"
  docker compose start api
}

case "$SCENARIO" in
  invalid-reviews)
    invalid_reviews
    ;;
  prediction-load)
    prediction_load
    ;;
  server-errors)
    server_errors
    ;;
  api-down)
    api_down
    ;;
  *)
    printf "Unknown scenario: %s\n" "$SCENARIO" >&2
    printf "Use one of: invalid-reviews, prediction-load, server-errors, api-down\n" >&2
    exit 1
    ;;
esac

printf "\nScenario complete. Check Prometheus alerts and Grafana panels:\n"
printf "  %s/alerts\n" "$PROMETHEUS_URL"
printf "  Grafana dashboard: Sentiment Operations - SLO and Alerts\n"
