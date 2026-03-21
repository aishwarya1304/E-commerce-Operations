"""
api/routes/orders.py
─────────────────────
REST endpoints for order management.

Endpoints
---------
GET  /api/orders/                   List all orders (paginated)
GET  /api/orders/<order_id>         Get single order
POST /api/orders/                   Create new order (triggers prediction)
PUT  /api/orders/<order_id>/status  Update order delivery status
GET  /api/orders/high-risk          List high-risk orders
"""

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required

from api.extensions import db
from api.models.models import Order, Prediction, Alert
from api.services.prediction_service import PredictionService

orders_bp = Blueprint("orders", __name__)


@orders_bp.get("/")
@jwt_required()
def list_orders():
    """
    List all orders (paginated).
    ---
    tags: [Orders]
    parameters:
      - name: page
        in: query
        type: integer
        default: 1
      - name: per_page
        in: query
        type: integer
        default: 20
      - name: city
        in: query
        type: string
      - name: status
        in: query
        type: string
    responses:
      200:
        description: Paginated order list
    """
    page     = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    city     = request.args.get("city")
    status   = request.args.get("status")

    query = Order.query
    if city:
        query = query.filter(Order.city == city)
    if status:
        query = query.filter(Order.status == status)

    paginated = query.order_by(Order.order_date.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )

    return jsonify({
        "orders":   [o.to_dict() for o in paginated.items],
        "total":    paginated.total,
        "page":     paginated.page,
        "pages":    paginated.pages,
    }), 200


@orders_bp.get("/<order_id>")
@jwt_required()
def get_order(order_id):
    """
    Get a single order by ID.
    ---
    tags: [Orders]
    parameters:
      - name: order_id
        in: path
        required: true
        type: string
    responses:
      200:
        description: Order details
      404:
        description: Order not found
    """
    order = Order.query.get_or_404(order_id)
    data  = order.to_dict()

    # Attach latest prediction if exists
    latest_pred = order.predictions.order_by(Prediction.predicted_at.desc()).first()
    if latest_pred:
        data["prediction"] = latest_pred.to_dict()

    return jsonify(data), 200


@orders_bp.post("/")
@jwt_required()
def create_order():
    """
    Create a new order — automatically runs ML prediction.
    ---
    tags: [Orders]
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required: [order_id, customer_id, warehouse_id, city, distance_km,
                     order_value, order_items, promised_delivery_days]
          properties:
            order_id:               { type: string }
            customer_id:            { type: string }
            warehouse_id:           { type: string }
            city:                   { type: string }
            distance_km:            { type: integer }
            order_value:            { type: number }
            order_items:            { type: integer }
            promised_delivery_days: { type: integer }
            past_delays:            { type: integer }
            warehouse_load:         { type: integer }
    responses:
      201:
        description: Order created with prediction
      400:
        description: Validation error
    """
    body = request.get_json(force=True)
    required = ["order_id", "customer_id", "warehouse_id", "city",
                "distance_km", "order_value", "order_items", "promised_delivery_days"]

    missing = [f for f in required if f not in body]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    # Check duplicate
    if Order.query.get(body["order_id"]):
        return jsonify({"error": "Order already exists"}), 409

    # Create order record
    order = Order(
        order_id               = body["order_id"],
        customer_id            = body["customer_id"],
        warehouse_id           = body["warehouse_id"],
        city                   = body["city"],
        distance_km            = body["distance_km"],
        order_value            = body["order_value"],
        order_items            = body["order_items"],
        promised_delivery_days = body["promised_delivery_days"],
        warehouse_load         = body.get("warehouse_load", 100),
        past_delays            = body.get("past_delays", 0),
        status                 = "pending",
    )
    db.session.add(order)

    # Run prediction
    pred_result = PredictionService.predict(body)
    prediction = Prediction(
        order_id          = body["order_id"],
        model_version     = pred_result["model_version"],
        delay_probability = pred_result["delay_probability"],
        risk_category     = pred_result["risk_category"],
        features_snapshot = pred_result["features_snapshot"],
    )
    db.session.add(prediction)

    # Auto-create alert for high-risk orders
    if pred_result["risk_category"] == "High":
        alert = Alert(
            order_id   = body["order_id"],
            alert_type = "high_risk",
            message    = (f"Order {body['order_id']} flagged as HIGH RISK "
                          f"({pred_result['delay_probability']*100:.1f}% delay probability). "
                          f"Assign priority delivery partner."),
            severity   = "high",
        )
        db.session.add(alert)

    db.session.commit()

    return jsonify({
        "order":      order.to_dict(),
        "prediction": pred_result,
        "alert_created": pred_result["risk_category"] == "High",
    }), 201


@orders_bp.put("/<order_id>/status")
@jwt_required()
def update_order_status(order_id):
    """
    Update delivery status and actual delivery days.
    ---
    tags: [Orders]
    parameters:
      - name: order_id
        in: path
        required: true
        type: string
      - in: body
        name: body
        schema:
          type: object
          properties:
            status:              { type: string }
            actual_delivery_days:{ type: integer }
            is_returned:         { type: boolean }
    responses:
      200:
        description: Order updated
    """
    order = Order.query.get_or_404(order_id)
    body  = request.get_json(force=True)

    if "status" in body:
        order.status = body["status"]

    if "actual_delivery_days" in body:
        order.actual_delivery_days = body["actual_delivery_days"]
        order.is_delayed  = order.actual_delivery_days > order.promised_delivery_days
        order.delay_days  = max(0, order.actual_delivery_days - order.promised_delivery_days)
        order.delivery_cost = 50 + (order.distance_km * 2) + (order.delay_days * 100)

    if "is_returned" in body:
        order.is_returned = body["is_returned"]
        order.return_cost = 500 if order.is_returned else 0

    # Update prediction accuracy
    latest_pred = order.predictions.order_by(Prediction.predicted_at.desc()).first()
    if latest_pred and order.is_delayed is not None:
        predicted_delay = latest_pred.risk_category in ("Medium", "High")
        latest_pred.was_correct = (predicted_delay == order.is_delayed)

    db.session.commit()
    return jsonify(order.to_dict()), 200


@orders_bp.get("/high-risk")
@jwt_required()
def high_risk_orders():
    """
    List current high-risk undelivered orders.
    ---
    tags: [Orders]
    responses:
      200:
        description: High-risk order list
    """
    from api.services.analytics_service import get_high_risk_orders
    from flask import current_app

    threshold = current_app.config.get("HIGH_RISK_THRESHOLD", 0.60)
    orders = get_high_risk_orders(threshold=threshold)
    return jsonify({"count": len(orders), "orders": orders}), 200
