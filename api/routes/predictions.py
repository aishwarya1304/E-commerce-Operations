"""
api/routes/predictions.py
──────────────────────────
Endpoints for on-demand ML predictions.

Endpoints
---------
POST /api/predictions/single   Predict for one order payload
POST /api/predictions/batch    Predict for multiple orders
GET  /api/predictions/model-info  Model metadata
"""

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required

from api.services.prediction_service import PredictionService

predictions_bp = Blueprint("predictions", __name__)


@predictions_bp.post("/single")
@jwt_required()
def predict_single():
    """
    Predict delay probability for one order (does NOT save to DB).
    ---
    tags: [Predictions]
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required: [distance_km, warehouse_load, order_items,
                     past_delays, order_value, promised_delivery_days, warehouse_id]
    responses:
      200:
        description: Prediction result
    """
    body = request.get_json(force=True)
    required = ["distance_km", "warehouse_load", "order_items",
                "past_delays", "order_value", "promised_delivery_days", "warehouse_id"]
    missing = [f for f in required if f not in body]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    result = PredictionService.predict(body)
    return jsonify(result), 200


@predictions_bp.post("/batch")
@jwt_required()
def predict_batch():
    """
    Batch predict for multiple orders (max 500).
    ---
    tags: [Predictions]
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            orders:
              type: array
              items:
                type: object
    responses:
      200:
        description: List of predictions in same order as input
      400:
        description: Too many orders or validation error
    """
    body   = request.get_json(force=True)
    orders = body.get("orders", [])

    if not isinstance(orders, list) or len(orders) == 0:
        return jsonify({"error": "Provide a non-empty 'orders' list"}), 400
    if len(orders) > 500:
        return jsonify({"error": "Batch limit is 500 orders"}), 400

    results = PredictionService.batch_predict(orders)
    return jsonify({
        "count":   len(results),
        "results": results,
    }), 200


@predictions_bp.get("/model-info")
def model_info():
    """
    Return model metadata (no auth required — useful for health checks).
    ---
    tags: [Predictions]
    responses:
      200:
        description: Model version and status
    """
    model_loaded = PredictionService._model is not None
    return jsonify({
        "model_loaded":  model_loaded,
        "model_version": PredictionService._model_version,
        "mode":          "ml-model" if model_loaded else "rule-based-fallback",
    }), 200
