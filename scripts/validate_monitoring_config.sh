#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

docker run --rm \
  --entrypoint promtool \
  -v "$ROOT_DIR/infra/prometheus:/etc/prometheus:ro" \
  prom/prometheus:v2.54.1 \
  check config /etc/prometheus/prometheus.yml

docker run --rm \
  --entrypoint amtool \
  -v "$ROOT_DIR/infra/alertmanager:/etc/alertmanager:ro" \
  prom/alertmanager:v0.27.0 \
  check-config /etc/alertmanager/alertmanager.yml

printf "\nMonitoring config validation passed.\n"
