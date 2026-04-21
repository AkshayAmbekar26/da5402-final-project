#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export DVC_SITE_CACHE_DIR="${DVC_SITE_CACHE_DIR:-.dvc/tmp/site-cache}"
export PATH="$ROOT_DIR/.venv/bin:$PATH"

printf "== DVC status ==\n"
dvc status

printf "\n== DVC dag ==\n"
dvc dag

printf "\n== DVC repro ==\n"
dvc repro

printf "\n== DVC metrics show ==\n"
dvc metrics show

printf "\n== DVC plots show ==\n"
dvc plots show
