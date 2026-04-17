from __future__ import annotations

import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np

from apps.api.sentiment_api.config import settings

logger = logging.getLogger(__name__)


class ModelService:
    def __init__(self) -> None:
        self.model: Any | None = None
        self.metadata: dict[str, Any] = {
            "model_name": "keyword-fallback",
            "model_version": "fallback",
            "mlflow_run_id": "untrained",
            "labels": ["negative", "neutral", "positive"],
        }
        self.feature_importance: dict[str, Any] = {"feature_importance": []}
        self.load()

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @property
    def fallback_mode(self) -> bool:
        return self.model is None

    def load(self) -> None:
        if settings.model_path.exists():
            self.model = joblib.load(settings.model_path)
            logger.info("Loaded sentiment model from %s", settings.model_path)
        else:
            logger.warning("Model artifact not found at %s; using keyword fallback.", settings.model_path)

        if settings.model_metadata_path.exists():
            self.metadata = json.loads(settings.model_metadata_path.read_text(encoding="utf-8"))
        if settings.feature_importance_path.exists():
            self.feature_importance = json.loads(settings.feature_importance_path.read_text(encoding="utf-8"))

    def predict(self, review_text: str) -> dict[str, Any]:
        start = perf_counter()
        if self.model is not None:
            sentiment = str(self.model.predict([review_text])[0])
            probabilities = self._predict_proba(review_text)
            confidence = float(probabilities.get(sentiment, 0.0))
            explanation = self._explain_model_prediction(review_text, sentiment)
        else:
            sentiment, probabilities, explanation = self._fallback_predict(review_text)
            confidence = float(probabilities[sentiment])

        latency_ms = (perf_counter() - start) * 1000
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "class_probabilities": probabilities,
            "explanation": explanation,
            "model_version": str(self.metadata.get("model_version", "unknown")),
            "mlflow_run_id": str(self.metadata.get("mlflow_run_id", "unknown")),
            "latency_ms": float(latency_ms),
        }

    def _predict_proba(self, review_text: str) -> dict[str, float]:
        labels = list(getattr(self.model.named_steps["classifier"], "classes_", self.metadata["labels"]))
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba([review_text])[0]
            return {str(label): float(value) for label, value in zip(labels, proba, strict=False)}
        sentiment = str(self.model.predict([review_text])[0])
        return {label: float(label == sentiment) for label in labels}

    def _explain_model_prediction(self, review_text: str, sentiment: str) -> list[dict[str, float | str]]:
        if self.model is None:
            return []
        vectorizer = self.model.named_steps["tfidf"]
        classifier = self.model.named_steps["classifier"]
        features = vectorizer.get_feature_names_out()
        vector = vectorizer.transform([review_text])
        class_index = list(classifier.classes_).index(sentiment)
        contributions = vector.multiply(classifier.coef_[class_index]).toarray()[0]
        top_indices = np.argsort(np.abs(contributions))[-5:][::-1]
        explanation = [
            {"token": str(features[idx]), "weight": float(contributions[idx])}
            for idx in top_indices
            if abs(contributions[idx]) > 0
        ]
        return explanation

    def _fallback_predict(self, review_text: str) -> tuple[str, dict[str, float], list[dict[str, float | str]]]:
        text = review_text.lower()
        positive = ["excellent", "great", "love", "perfect", "fast", "premium", "durable", "reliable"]
        negative = ["bad", "poor", "terrible", "late", "damaged", "broken", "cheap", "disappointing"]
        positive_score = sum(token in text for token in positive)
        negative_score = sum(token in text for token in negative)
        if positive_score > negative_score:
            sentiment = "positive"
            probabilities = {"negative": 0.08, "neutral": 0.17, "positive": 0.75}
        elif negative_score > positive_score:
            sentiment = "negative"
            probabilities = {"negative": 0.75, "neutral": 0.17, "positive": 0.08}
        else:
            sentiment = "neutral"
            probabilities = {"negative": 0.20, "neutral": 0.60, "positive": 0.20}
        explanation = [
            {"token": token, "weight": 0.25}
            for token in positive + negative
            if token in text
        ][:5]
        return sentiment, probabilities, explanation

    def info(self) -> dict[str, Any]:
        return {
            "model_loaded": self.loaded,
            "fallback_mode": self.fallback_mode,
            "metadata": self.metadata,
            "model_path": str(Path(settings.model_path)),
        }

