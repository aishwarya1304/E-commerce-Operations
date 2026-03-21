"""
api/routes/analytics.py
────────────────────────
Business intelligence endpoints — mirrors the SQL queries in
ecommerce_ai_project.py but runs against live database.

Endpoints
---------
GET /api/analytics/kpis              Real-time KPI dashboard
GET /api/analytics/city-analysis     Delay rate by city
GET /api/analytics/warehouse-perf    Warehouse performance table
GET /api/analytics/monthly-sla       Monthly SLA breach trend
GET /api/analytics/financial-impact  ROI / savings estimation
"""

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required

from api.services import analytics_service

analytics_bp = Blueprint("analytics", __name__)


@analytics_bp.get("/kpis")
@jwt_required()
def kpis():
    """
    Real-time KPI summary across all orders.
    ---
    tags: [Analytics]
    responses:
      200:
        description: |
          on_time_pct, sla_breach_pct, avg_delivery_days,
          return_rate_pct, avg_cost_per_order, total_operational_cost
    """
    data = analytics_service.get_kpi_summary()
    return jsonify(data), 200


@analytics_bp.get("/city-analysis")
@jwt_required()
def city_analysis():
    """
    Delay rate and average delay days grouped by delivery city.
    ---
    tags: [Analytics]
    responses:
      200:
        description: City-level delay breakdown
    """
    data = analytics_service.get_city_analysis()
    return jsonify(data), 200


@analytics_bp.get("/warehouse-perf")
@jwt_required()
def warehouse_perf():
    """
    Warehouse-level delay rate and average delivery cost.
    ---
    tags: [Analytics]
    responses:
      200:
        description: Warehouse performance metrics
    """
    data = analytics_service.get_warehouse_performance()
    return jsonify(data), 200


@analytics_bp.get("/monthly-sla")
@jwt_required()
def monthly_sla():
    """
    Monthly SLA breach trend.
    ---
    tags: [Analytics]
    parameters:
      - name: months
        in: query
        type: integer
        default: 12
    responses:
      200:
        description: Month-over-month SLA data
    """
    months = request.args.get("months", 12, type=int)
    data   = analytics_service.get_monthly_sla(months=months)
    return jsonify(data), 200


@analytics_bp.get("/financial-impact")
@jwt_required()
def financial_impact():
    """
    Estimated financial savings from AI-driven delay reduction.
    ---
    tags: [Analytics]
    responses:
      200:
        description: |
          prevented_delays, delay_savings_inr,
          return_savings_inr, total_monthly_savings_inr, annual_savings_inr
    """
    data = analytics_service.get_financial_impact()
    return jsonify(data), 200
