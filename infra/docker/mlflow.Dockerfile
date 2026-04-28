FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10

COPY pyproject.toml constraints.txt README.md params.yaml /app/

RUN pip install --default-timeout=300 --retries=10 --no-cache-dir --no-binary pyasn1 pyasn1==0.6.3
RUN pip install --default-timeout=300 --retries=10 --no-cache-dir -c constraints.txt \
    alembic \
    cloudpickle \
    gunicorn \
    joblib \
    mlflow==3.11.1 \
    numpy \
    pandas \
    psutil \
    PyYAML \
    scikit-learn \
    scipy \
    sqlalchemy
COPY ml /app/ml
RUN pip install --default-timeout=300 --retries=10 --no-cache-dir --no-deps -e .
RUN mkdir -p /app/models /mlflow/artifacts

EXPOSE 5000
