from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from ml.common import path_for


class Settings(BaseSettings):
    api_name: str = "Product Review Sentiment API"
    model_path: Path = path_for("sentiment_model")
    model_metadata_path: Path = path_for("model_metadata")
    feature_importance_path: Path = path_for("feature_importance")
    model_serving_mode: str = "local"
    mlflow_serving_url: str = "http://mlflow-model-server:5002/invocations"
    mlflow_serving_timeout_seconds: float = 10.0
    feedback_path: Path = path_for("feedback_log")
    max_review_length: int = 5000
    cors_origins: str = "http://localhost:5173,http://localhost:3000,http://localhost:4173"
    airflow_url: str = "http://localhost:8080"
    mlflow_url: str = "http://localhost:5001"
    prometheus_url: str = "http://localhost:9091"
    grafana_url: str = "http://localhost:3001"
    alertmanager_url: str = "http://localhost:19093"
    enable_demo_ops_endpoints: bool = False
    allow_fallback_ready: bool = False

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


settings = Settings()
