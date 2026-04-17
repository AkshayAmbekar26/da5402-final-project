FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

COPY pyproject.toml README.md /app/
COPY ml /app/ml
COPY apps /app/apps
RUN pip install --no-cache-dir -e .

COPY models /app/models
COPY reports /app/reports
COPY feedback /app/feedback

EXPOSE 8000
CMD ["uvicorn", "apps.api.sentiment_api.main:app", "--host", "0.0.0.0", "--port", "8000"]

