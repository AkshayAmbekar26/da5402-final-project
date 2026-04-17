.PHONY: setup ingest validate preprocess baseline train evaluate drift dvc-repro test api frontend up down demo lint

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -e ".[dev]"

ingest:
	python -m ml.data_ingestion.ingest

validate:
	python -m ml.validation.validate_data

preprocess:
	python -m ml.preprocessing.preprocess

baseline:
	python -m ml.features.compute_baseline

train:
	python -m ml.training.train

evaluate:
	python -m ml.evaluation.evaluate

drift:
	python -m ml.monitoring.drift

dvc-repro:
	dvc repro

test:
	pytest

lint:
	ruff check .

api:
	uvicorn apps.api.sentiment_api.main:app --reload

frontend:
	cd apps/frontend && npm run dev

up:
	docker compose up --build

down:
	docker compose down

demo: ingest validate preprocess baseline train evaluate drift
	@echo "Demo artifacts are ready. Run 'make api' and 'make frontend' in separate terminals."

