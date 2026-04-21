#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/rollback_model.sh --model-path models/sentiment_model.joblib [options]
  scripts/rollback_model.sh --git-rev <commit-or-tag> [options]
  scripts/rollback_model.sh --serving-mode mlflow --mlflow-serving-url http://localhost:5002/invocations [options]

Options:
  --git-rev <rev>                  Fetch model artifacts from a Git/DVC revision without changing the worktree.
  --model-path <path>              Local sklearn model artifact to serve.
  --metadata-path <path>           Model metadata JSON. Defaults to the model directory, then models/model_metadata.json.
  --feature-importance-path <path> Feature importance JSON. Defaults to the model directory, then models/feature_importance.json.
  --serving-mode <local|mlflow>    Serving backend to configure. Default: local.
  --mlflow-serving-url <url>       MLflow model server /invocations URL for mlflow serving mode.
  --rollback-dir <path>            Directory for artifacts fetched with --git-rev. Default: models/rollback.
  --env-file <path>                Rollback env file to write. Default: .env.rollback.
  --restart-api                    Recreate the API container using the rollback env file.
  --verify                         Check /ready and /predict after restart.
  -h, --help                       Show this help.

Examples:
  make rollback-current
  make rollback ROLLBACK_ARGS="--git-rev abc123"
  make rollback-restart ROLLBACK_ARGS="--model-path models/rollback/abc123/sentiment_model.joblib"
EOF
}

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$repo_root"

env_file=".env.rollback"
rollback_dir="models/rollback"
serving_mode="local"
mlflow_serving_url="${MLFLOW_SERVING_URL:-http://mlflow-model-server:5002/invocations}"
git_rev=""
model_path=""
metadata_path=""
feature_importance_path=""
restart_api=false
verify=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --git-rev)
      git_rev="${2:?Missing value for --git-rev}"
      shift 2
      ;;
    --model-path)
      model_path="${2:?Missing value for --model-path}"
      shift 2
      ;;
    --metadata-path)
      metadata_path="${2:?Missing value for --metadata-path}"
      shift 2
      ;;
    --feature-importance-path)
      feature_importance_path="${2:?Missing value for --feature-importance-path}"
      shift 2
      ;;
    --serving-mode)
      serving_mode="${2:?Missing value for --serving-mode}"
      shift 2
      ;;
    --mlflow-serving-url)
      mlflow_serving_url="${2:?Missing value for --mlflow-serving-url}"
      shift 2
      ;;
    --rollback-dir)
      rollback_dir="${2:?Missing value for --rollback-dir}"
      shift 2
      ;;
    --env-file)
      env_file="${2:?Missing value for --env-file}"
      shift 2
      ;;
    --restart-api)
      restart_api=true
      shift
      ;;
    --verify)
      verify=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf "Unknown option: %s\n\n" "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$serving_mode" != "local" && "$serving_mode" != "mlflow" ]]; then
  printf "serving mode must be local or mlflow, got: %s\n" "$serving_mode" >&2
  exit 2
fi

to_repo_relative_path() {
  local input_path="$1"
  if [[ "$input_path" = /* ]]; then
    case "$input_path" in
      "$repo_root"/*)
        input_path="${input_path#"$repo_root"/}"
        ;;
      *)
        printf "Path must live inside the repository for Docker Compose volume mounting: %s\n" "$input_path" >&2
        exit 2
        ;;
    esac
  fi
  input_path="${input_path#./}"
  printf "%s" "$input_path"
}

fetch_from_dvc_revision() {
  local rev="$1"
  local safe_rev="$2"
  local target_dir="$rollback_dir/$safe_rev"
  mkdir -p "$target_dir"

  # Keep rollback non-destructive: dvc get reads the revision without checking it out.
  if [[ ! -f "$target_dir/sentiment_model.joblib" ]]; then
    dvc get . models/sentiment_model.joblib --rev "$rev" -o "$target_dir/sentiment_model.joblib"
  fi
  if [[ ! -f "$target_dir/model_metadata.json" ]]; then
    dvc get . models/model_metadata.json --rev "$rev" -o "$target_dir/model_metadata.json"
  fi
  if [[ ! -f "$target_dir/feature_importance.json" ]]; then
    dvc get . models/feature_importance.json --rev "$rev" -o "$target_dir/feature_importance.json"
  fi

  model_path="$target_dir/sentiment_model.joblib"
  metadata_path="$target_dir/model_metadata.json"
  feature_importance_path="$target_dir/feature_importance.json"
}

if [[ -n "$git_rev" ]]; then
  if [[ "$serving_mode" == "mlflow" ]]; then
    printf "--git-rev rollback is only supported with local serving mode.\n" >&2
    exit 2
  fi
  safe_rev="$(printf "%s" "$git_rev" | tr -c 'A-Za-z0-9._-' '_')"
  fetch_from_dvc_revision "$git_rev" "$safe_rev"
fi

if [[ "$serving_mode" == "local" ]]; then
  if [[ -z "$model_path" ]]; then
    printf "Provide --model-path or --git-rev for local rollback.\n\n" >&2
    usage >&2
    exit 2
  fi

  model_path="$(to_repo_relative_path "$model_path")"
  model_dir="$(dirname "$model_path")"

  if [[ -z "$metadata_path" && -f "$model_dir/model_metadata.json" ]]; then
    metadata_path="$model_dir/model_metadata.json"
  fi
  if [[ -z "$metadata_path" ]]; then
    metadata_path="models/model_metadata.json"
  fi
  if [[ -z "$feature_importance_path" && -f "$model_dir/feature_importance.json" ]]; then
    feature_importance_path="$model_dir/feature_importance.json"
  fi
  if [[ -z "$feature_importance_path" ]]; then
    feature_importance_path="models/feature_importance.json"
  fi

  metadata_path="$(to_repo_relative_path "$metadata_path")"
  feature_importance_path="$(to_repo_relative_path "$feature_importance_path")"

  for required_path in "$model_path" "$metadata_path" "$feature_importance_path"; do
    if [[ ! -f "$required_path" ]]; then
      printf "Required rollback artifact not found: %s\n" "$required_path" >&2
      exit 1
    fi
  done
fi

{
  printf "# Generated by scripts/rollback_model.sh\n"
  printf "# Use with: docker compose --env-file %s up -d api\n" "$env_file"
  printf "MODEL_SERVING_MODE=%s\n" "$serving_mode"
  printf "ALLOW_FALLBACK_READY=false\n"
  if [[ "$serving_mode" == "local" ]]; then
    printf "MODEL_PATH=%s\n" "$model_path"
    printf "MODEL_METADATA_PATH=%s\n" "$metadata_path"
    printf "FEATURE_IMPORTANCE_PATH=%s\n" "$feature_importance_path"
  else
    printf "MLFLOW_SERVING_URL=%s\n" "$mlflow_serving_url"
  fi
} > "$env_file"

printf "Rollback environment written to %s\n" "$env_file"
if [[ "$serving_mode" == "local" ]]; then
  printf "Configured local model: %s\n" "$model_path"
else
  printf "Configured MLflow serving URL: %s\n" "$mlflow_serving_url"
fi

if [[ "$restart_api" == true ]]; then
  docker compose --env-file "$env_file" up -d api
fi

if [[ "$verify" == true ]]; then
  curl --fail --silent --show-error http://localhost:8000/ready >/dev/null
  curl --fail --silent --show-error \
    -H "Content-Type: application/json" \
    -X POST \
    -d '{"review_text":"Excellent product quality and fast delivery."}' \
    http://localhost:8000/predict >/dev/null
  printf "Rollback verification passed.\n"
fi

printf "Apply manually with: docker compose --env-file %s up -d api\n" "$env_file"
