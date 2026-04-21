FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10

COPY pyproject.toml constraints.txt README.md /app/
COPY ml /app/ml
COPY apps /app/apps
RUN pip install --default-timeout=300 --retries=10 --no-cache-dir --no-binary pyasn1 pyasn1==0.6.3
RUN pip install --default-timeout=300 --retries=10 --no-cache-dir -c constraints.txt -e .
RUN mkdir -p /app/models /app/reports /app/feedback

EXPOSE 8000
CMD ["uvicorn", "apps.api.sentiment_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
