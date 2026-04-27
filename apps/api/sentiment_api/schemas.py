from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Sentiment = Literal["negative", "neutral", "positive"]


class PredictRequest(BaseModel):
    review_text: str = Field(..., min_length=1, max_length=5000)


class TokenWeight(BaseModel):
    token: str
    weight: float


class PredictResponse(BaseModel):
    sentiment: Sentiment
    confidence: float
    class_probabilities: dict[str, float]
    explanation: list[TokenWeight]
    model_version: str
    mlflow_run_id: str
    latency_ms: float


class FeedbackRequest(BaseModel):
    review_text: str = Field(..., min_length=1, max_length=5000)
    predicted_sentiment: Sentiment
    actual_sentiment: Sentiment
    source: str = "demo"


class FeedbackResponse(BaseModel):
    status: str
    stored: bool


class HealthResponse(BaseModel):
    status: str
    service: str


class ReadyResponse(BaseModel):
    ready: bool
    model_loaded: bool
    fallback_mode: bool
    model_path: str
