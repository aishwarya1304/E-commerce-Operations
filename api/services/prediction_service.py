"""
api/services/prediction_service.py
────────────────────────────────────
Wraps the trained Random Forest model.
Loads the model once at startup (singleton pattern) for performance.

Assumptions:
- Model is trained with ecommerce_ai_project.py and saved via joblib.
- If no model file exists, falls back to a rule-based heuristic (so the
  API still works without running the training script first).
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib
from flask import current_app

logger = logging.getLogger(__name__)

# Feature order MUST match the training script exactly
FEATURE_COLUMNS = [
    "distance_km",
    "warehouse_load",
    "order_items",
    "past_delays",
    "order_value",
    "promised_delivery_days",
    "warehouse_WH_East",
    "warehouse_WH_North",
    "warehouse_WH_South",
    "warehouse_WH_West",
]

RISK_THRESHOLDS = {"Low": 0.30, "Medium": 0.60, "High": 1.01}


class PredictionService:
    """Singleton-like service for delay probability prediction."""

    _model = None
    _model_version = "unknown"

    @classmethod
    def load_model(cls):
        """Load (or reload) the trained sklearn model from disk."""
        model_path = current_app.config.get("MODEL_PATH", "models/random_forest_model.pkl")
        if os.path.exists(model_path):
            cls._model = joblib.load(model_path)
            cls._model_version = current_app.config.get("MODEL_VERSION", "1.0.0")
            logger.info(f"Model loaded from {model_path} (v{cls._model_version})")
        else:
            logger.warning(
                f"Model file not found at '{model_path}'. "
                "Using rule-based fallback. Run ecommerce_ai_project.py to train."
            )
            cls._model = None

    @classmethod
    def _rule_based_score(cls, features: dict) -> float:
        """
        Heuristic fallback when no model is available.
        Mimics the top 3 Random Forest features:
          distance (28%) + warehouse_load (24%) + past_delays (19%)
        """
        dist_score  = min(features.get("distance_km", 0) / 500, 1.0) * 0.35
        load_score  = min(features.get("warehouse_load", 0) / 250, 1.0) * 0.35
        delay_score = min(features.get("past_delays", 0) / 5, 1.0) * 0.30
        return round(dist_score + load_score + delay_score, 4)

    @classmethod
    def predict(cls, order_data: dict) -> dict:
        """
        Predict delay probability for a single order.

        Parameters
        ----------
        order_data : dict
            Keys: order_id, distance_km, warehouse_load, order_items,
                  past_delays, order_value, promised_delivery_days, warehouse_id

        Returns
        -------
        dict  with keys: delay_probability, risk_category, model_version
        """
        # Build feature vector
        features = {col: 0 for col in FEATURE_COLUMNS}
        features["distance_km"]            = order_data.get("distance_km", 0)
        features["warehouse_load"]         = order_data.get("warehouse_load", 100)
        features["order_items"]            = order_data.get("order_items", 1)
        features["past_delays"]            = order_data.get("past_delays", 0)
        features["order_value"]            = order_data.get("order_value", 1000)
        features["promised_delivery_days"] = order_data.get("promised_delivery_days", 3)

        # One-hot encode warehouse
        wh = order_data.get("warehouse_id", "")
        wh_col = f"warehouse_{wh}"
        if wh_col in features:
            features[wh_col] = 1

        # Predict
        if cls._model is not None:
            X = pd.DataFrame([features])[FEATURE_COLUMNS]
            probability = float(cls._model.predict_proba(X)[0][1])
            version = cls._model_version
        else:
            probability = cls._rule_based_score(features)
            version = "rule-based-fallback"

        # Categorise risk
        threshold = current_app.config.get("HIGH_RISK_THRESHOLD", 0.60)
        if probability >= threshold:
            risk = "High"
        elif probability >= 0.30:
            risk = "Medium"
        else:
            risk = "Low"

        return {
            "delay_probability": round(probability, 4),
            "risk_category":     risk,
            "model_version":     version,
            "features_snapshot": features,
        }

    @classmethod
    def batch_predict(cls, orders: list[dict]) -> list[dict]:
        """Predict for a list of orders efficiently (vectorised if model loaded)."""
        if cls._model is None or not orders:
            return [cls.predict(o) for o in orders]

        rows = []
        for o in orders:
            features = {col: 0 for col in FEATURE_COLUMNS}
            features["distance_km"]            = o.get("distance_km", 0)
            features["warehouse_load"]         = o.get("warehouse_load", 100)
            features["order_items"]            = o.get("order_items", 1)
            features["past_delays"]            = o.get("past_delays", 0)
            features["order_value"]            = o.get("order_value", 1000)
            features["promised_delivery_days"] = o.get("promised_delivery_days", 3)
            wh_col = f"warehouse_{o.get('warehouse_id', '')}"
            if wh_col in features:
                features[wh_col] = 1
            rows.append(features)

        X = pd.DataFrame(rows)[FEATURE_COLUMNS]
        probs = cls._model.predict_proba(X)[:, 1]
        threshold = 0.60  # can't access app context in static easily; use default

        results = []
        for prob in probs:
            risk = "High" if prob >= threshold else ("Medium" if prob >= 0.30 else "Low")
            results.append({
                "delay_probability": round(float(prob), 4),
                "risk_category":     risk,
                "model_version":     cls._model_version,
            })
        return results
