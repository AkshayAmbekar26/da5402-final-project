#!/usr/bin/env bash
set -euo pipefail

API_BASE_URL="${API_BASE_URL:-http://localhost:8000}"
PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:${PROMETHEUS_HOST_PORT:-9091}}"
GRAFANA_URL="${GRAFANA_URL:-http://localhost:${GRAFANA_HOST_PORT:-3001}}"
ALERTMANAGER_URL="${ALERTMANAGER_URL:-http://localhost:${ALERTMANAGER_HOST_PORT:-19093}}"
NODE_EXPORTER_URL="${NODE_EXPORTER_URL:-http://localhost:${NODE_EXPORTER_HOST_PORT:-19100}}"

check_url() {
  local name="$1"
  local url="$2"
  if curl --fail --silent --show-error --max-time 10 "$url" >/dev/null; then
    printf "ok     %s\n" "$name"
  else
    printf "failed %s (%s)\n" "$name" "$url" >&2
    return 1
  fi
}

printf "Service health\n"
check_url "api /health" "$API_BASE_URL/health"
check_url "api /ready" "$API_BASE_URL/ready"
check_url "api /metrics" "$API_BASE_URL/metrics"
check_url "prometheus" "$PROMETHEUS_URL/-/ready"
check_url "grafana" "$GRAFANA_URL/api/health"
check_url "alertmanager" "$ALERTMANAGER_URL/-/ready"
check_url "node exporter" "$NODE_EXPORTER_URL/metrics"

printf "\nPrometheus active targets\n"
targets_json="$(mktemp)"
trap 'rm -f "$targets_json"' EXIT
curl --fail --silent --show-error --max-time 10 \
  "$PROMETHEUS_URL/api/v1/targets?state=active" > "$targets_json"

python3 - "$targets_json" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
targets = payload.get("data", {}).get("activeTargets", [])
if not targets:
    print("no active targets returned")
    raise SystemExit(1)

unhealthy = 0
for target in sorted(targets, key=lambda row: (row.get("labels", {}).get("job", ""), row.get("scrapeUrl", ""))):
    labels = target.get("labels", {})
    job = labels.get("job", "unknown")
    health = target.get("health", "unknown")
    scrape_url = target.get("scrapeUrl", "unknown")
    last_error = target.get("lastError") or ""
    marker = "ok    " if health == "up" else "failed"
    print(f"{marker} {job:18} {health:7} {scrape_url}")
    if last_error:
        print(f"       error: {last_error}")
    unhealthy += int(health != "up")

raise SystemExit(1 if unhealthy else 0)
PY

printf "\nMonitoring status check passed.\n"
