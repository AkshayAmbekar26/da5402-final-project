from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    api_name: str = "Product Review Sentiment API"
    model_path: Path = Path("models/sentiment_model.joblib")
    model_metadata_path: Path = Path("models/model_metadata.json")
    feature_importance_path: Path = Path("models/feature_importance.json")
    feedback_path: Path = Path("feedback/feedback.jsonl")
    max_review_length: int = 5000
    cors_origins: str = "http://localhost:5173,http://localhost:3000,http://localhost:4173"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


settings = Settings()

