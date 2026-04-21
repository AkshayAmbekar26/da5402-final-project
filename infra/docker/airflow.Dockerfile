FROM apache/airflow:3.2.0-python3.11

USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends git \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

USER airflow
WORKDIR /tmp/project
ENV PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10
COPY pyproject.toml README.md /tmp/project/
COPY ml /tmp/project/ml
COPY apps /tmp/project/apps
RUN pip install --default-timeout=300 --retries=10 --no-cache-dir \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.2.0/constraints-3.11.txt" \
  "datasets>=3.0.0" \
  "joblib>=1.4.0" \
  "matplotlib>=3.8.0" \
  "mlflow==3.11.1" \
  "pydantic-settings>=2.4.0" \
  "python-json-logger>=2.0.7" \
  "scikit-learn>=1.5.0"
RUN pip install --default-timeout=300 --retries=10 --no-cache-dir --no-deps /tmp/project
