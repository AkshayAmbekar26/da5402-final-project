FROM apache/airflow:2.10.2-python3.11

USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends git \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

USER airflow
WORKDIR /tmp/project
COPY pyproject.toml README.md /tmp/project/
COPY ml /tmp/project/ml
COPY apps /tmp/project/apps
RUN pip install --no-cache-dir /tmp/project

