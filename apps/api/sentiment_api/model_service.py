from __future__ import annotations

import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx
import joblib
import numpy as np

from apps.api.sentiment_api.config import settings
from apps.api.sentiment_api.metrics import observe_stage

logger = logging.getLogger(__name__)


class ModelService:
    """Small serving facade that hides local, MLflow, and fallback prediction modes."""

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
        return self.model is not None or self.serving_mode == "mlflow"

    @property
    def fallback_mode(self) -> bool:
        return self.model is None and self.serving_mode != "mlflow"

    @property
    def serving_mode(self) -> str:
        return settings.model_serving_mode.strip().lower()

    def load(self) -> None:
        """Load the production artifact when available; keep fallback mode for demo resilience."""
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
        """Return the stable API prediction contract regardless of serving backend."""
        if self.serving_mode == "mlflow":
            return self._predict_via_mlflow_serving(review_text)

        start = perf_counter()
        if self.model is not None:
            with observe_stage("model_predict_label"):
                sentiment = str(self.model.predict([review_text])[0])
            with observe_stage("model_predict_probabilities"):
                probabilities = self._predict_proba(review_text)
            confidence = float(probabilities.get(sentiment, 0.0))
            with observe_stage("model_explanation"):
                explanation = self._explain_model_prediction(review_text, sentiment)
        else:
            with observe_stage("fallback_rules"):
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

    def _predict_via_mlflow_serving(self, review_text: str) -> dict[str, Any]:
        """Adapt MLflow's generic serving response into the app's prediction schema."""
        start = perf_counter()
        payload = {"dataframe_records": [{"review_text": review_text}]}
        try:
            response = httpx.post(
                settings.mlflow_serving_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=settings.mlflow_serving_timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.exception("MLflow serving request failed")
            raise RuntimeError(f"MLflow model server request failed: {exc}") from exc

        prediction = self._extract_mlflow_prediction(response.json())
        probabilities = self._json_field(prediction, "class_probabilities_json", default={})
        explanation = self._json_field(prediction, "explanation_json", default=[])
        latency_ms = float(prediction.get("latency_ms") or 0.0)
        if latency_ms <= 0:
            latency_ms = (perf_counter() - start) * 1000
        return {
            "sentiment": str(prediction["sentiment"]),
            "confidence": float(prediction["confidence"]),
            "class_probabilities": {str(key): float(value) for key, value in probabilities.items()},
            "explanation": explanation,
            "model_version": str(prediction.get("model_version", self.metadata.get("model_version", "unknown"))),
            "mlflow_run_id": str(prediction.get("mlflow_run_id", self.metadata.get("mlflow_run_id", "unknown"))),
            "latency_ms": latency_ms,
        }

    def _extract_mlflow_prediction(self, body: Any) -> dict[str, Any]:
        if isinstance(body, dict) and "predictions" in body:
            predictions = body["predictions"]
            if isinstance(predictions, list) and predictions:
                first = predictions[0]
                if isinstance(first, dict):
                    return first
                if isinstance(first, list):
                    columns = [
                        "sentiment",
                        "confidence",
                        "class_probabilities_json",
                        "explanation_json",
                        "model_version",
                        "mlflow_run_id",
                        "latency_ms",
                    ]
                    return dict(zip(columns, first, strict=False))
        if isinstance(body, list) and body and isinstance(body[0], dict):
            return body[0]
        raise ValueError(f"Unexpected MLflow serving response: {body}")

    def _json_field(self, prediction: dict[str, Any], field: str, default: Any) -> Any:
        raw_value = prediction.get(field, default)
        if isinstance(raw_value, str):
            return json.loads(raw_value)
        return raw_value

    def _predict_proba(self, review_text: str) -> dict[str, float]:
        labels = list(getattr(self.model.named_steps["classifier"], "classes_", self.metadata["labels"]))
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba([review_text])[0]
            return {str(label): float(value) for label, value in zip(labels, proba, strict=False)}
        sentiment = str(self.model.predict([review_text])[0])
        return {label: float(label == sentiment) for label in labels}

    def _explain_model_prediction(self, review_text: str, sentiment: str) -> list[dict[str, float | str]]:
        """Use sparse feature contributions as a lightweight explanation for the predicted class."""
        if self.model is None:
            return []
        vectorizer = self.model.named_steps.get("features") or self.model.named_steps.get("tfidf")
        if vectorizer is None:
            return []
        classifier = self.model.named_steps["classifier"]
        features = vectorizer.get_feature_names_out()
        vector = vectorizer.transform([review_text])
        class_index = list(classifier.classes_).index(sentiment)
        if hasattr(classifier, "coef_"):
            weights = classifier.coef_[class_index]
        elif hasattr(classifier, "feature_log_prob_"):
            weights = classifier.feature_log_prob_[class_index]
        else:
            return []
        contributions = vector.multiply(weights).toarray()[0]
        top_indices = np.argsort(np.abs(contributions))[-5:][::-1]
        explanation = [
            {"token": str(features[idx]), "weight": float(contributions[idx])}
            for idx in top_indices
            if abs(contributions[idx]) > 0
        ]
        return explanation

    def _fallback_predict(self, review_text: str) -> tuple[str, dict[str, float], list[dict[str, float | str]]]:
        """Keyword fallback keeps the UI usable before a trained artifact exists."""
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
            "serving_mode": self.serving_mode,
            "metadata": self.metadata,
            "model_path": str(Path(settings.model_path)),
            "mlflow_serving_url": settings.mlflow_serving_url,
        }
