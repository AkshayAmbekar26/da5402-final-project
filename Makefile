.PHONY: setup ingest validate eda preprocess baseline train evaluate acceptance drift dvc-repro dvc-check dvc-push test api frontend up down demo lint docker-build docker-up docker-down docker-logs docker-smoke docker-reset compose-config monitoring-config monitoring-status trigger-alerts rollback rollback-current rollback-restart mlflow-project mlflow-serve

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -e ".[dev]"

ingest:
	python -m ml.data_ingestion.ingest

validate:
	python -m ml.validation.validate_data

eda:
	python -m ml.eda.analyze

preprocess:
	python -m ml.preprocessing.preprocess

baseline:
	python -m ml.features.compute_baseline

train:
	python -m ml.training.train

mlflow-project:
	mlflow run . --env-manager local

mlflow-serve:
	mlflow models serve -m models/mlflow_model --host 0.0.0.0 --port 5002 --env-manager local

evaluate:
	python -m ml.evaluation.evaluate

acceptance:
	python -m ml.evaluation.check_acceptance

drift:
	python -m ml.monitoring.drift

dvc-repro:
	dvc repro

dvc-check:
	./scripts/dvc_repro_check.sh

dvc-push:
	dvc push

test:
	pytest

lint:
	ruff check .

api:
	uvicorn apps.api.sentiment_api.main:app --reload

frontend:
	cd apps/frontend && npm run dev

up:
	$(MAKE) docker-up

down:
	$(MAKE) docker-down

compose-config:
	docker compose --profile mlflow-serving config

monitoring-config:
	./scripts/validate_monitoring_config.sh

docker-build:
	docker compose --profile mlflow-serving build api frontend mlflow mlflow-model-server airflow-init airflow-webserver airflow-scheduler airflow-dag-processor

docker-up:
	docker compose --profile mlflow-serving up -d --build

docker-down:
	docker compose --profile mlflow-serving down

docker-logs:
	docker compose --profile mlflow-serving logs -f --tail=100

docker-smoke:
	./scripts/docker_smoke.sh

monitoring-status:
	./scripts/monitoring_status.sh

trigger-alerts:
	./scripts/trigger_monitoring_alerts.sh

rollback:
	./scripts/rollback_model.sh $(ROLLBACK_ARGS)

rollback-current:
	./scripts/rollback_model.sh --model-path models/sentiment_model.joblib --metadata-path models/model_metadata.json --feature-importance-path models/feature_importance.json

rollback-restart:
	./scripts/rollback_model.sh $(ROLLBACK_ARGS) --restart-api --verify

docker-reset:
	docker compose --profile mlflow-serving down --remove-orphans
	docker compose --profile mlflow-serving up -d --build

demo: ingest validate eda preprocess baseline train evaluate acceptance drift
	@echo "Demo artifacts are ready. Run 'make api' and 'make frontend' in separate terminals."
