from __future__ import annotations

import json
from time import perf_counter
from typing import Any

import joblib
import mlflow.pyfunc
import numpy as np
import pandas as pd


class SentimentPyfuncModel(mlflow.pyfunc.PythonModel):
    """MLflow PyFunc wrapper with the same prediction contract as the FastAPI app."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        self.model = joblib.load(context.artifacts["model"])
        with open(context.artifacts["metadata"], encoding="utf-8") as handle:
            self.metadata = json.load(handle)
        with open(context.artifacts["feature_importance"], encoding="utf-8") as handle:
            self.feature_importance = json.load(handle)

    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame) -> pd.DataFrame:
        if "review_text" not in model_input.columns:
            raise ValueError("MLflow serving input must contain a review_text column.")

        rows: list[dict[str, Any]] = []
        for review_text in model_input["review_text"].astype(str).tolist():
            rows.append(self._predict_one(review_text))
        return pd.DataFrame(rows)

    def _predict_one(self, review_text: str) -> dict[str, Any]:
        start = perf_counter()
        sentiment = str(self.model.predict([review_text])[0])
        probabilities = self._predict_proba(review_text)
        confidence = float(probabilities.get(sentiment, 0.0))
        explanation = self._explain_model_prediction(review_text, sentiment)
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "class_probabilities_json": json.dumps(probabilities, sort_keys=True),
            "explanation_json": json.dumps(explanation, sort_keys=True),
            "model_version": str(self.metadata.get("model_version", "unknown")),
            "mlflow_run_id": str(self.metadata.get("mlflow_run_id", "unknown")),
            "latency_ms": float((perf_counter() - start) * 1000),
        }

    def _predict_proba(self, review_text: str) -> dict[str, float]:
        classifier = self.model.named_steps["classifier"]
        labels = list(getattr(classifier, "classes_", self.metadata.get("labels", [])))
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba([review_text])[0]
            return {str(label): float(value) for label, value in zip(labels, proba, strict=False)}
        sentiment = str(self.model.predict([review_text])[0])
        return {str(label): float(label == sentiment) for label in labels}

    def _explain_model_prediction(self, review_text: str, sentiment: str) -> list[dict[str, float | str]]:
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
        return [
            {"token": str(features[idx]), "weight": float(contributions[idx])}
            for idx in top_indices
            if abs(contributions[idx]) > 0
        ]
