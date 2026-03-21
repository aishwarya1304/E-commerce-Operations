"""
api/services/analytics_service.py
───────────────────────────────────
Business analytics queries — pandas/SQL equivalents of the
SQL-style queries in ecommerce_ai_project.py, now running
against the live database through SQLAlchemy.
"""

from sqlalchemy import text
from api.extensions import db


def get_kpi_summary() -> dict:
    """Compute real-time KPIs from the orders table."""
    sql = text("""
        SELECT
            COUNT(*)                                         AS total_orders,
            SUM(CASE WHEN is_delayed THEN 1 ELSE 0 END)     AS delayed_orders,
            ROUND(100.0 * AVG(CASE WHEN NOT is_delayed THEN 1.0 ELSE 0.0 END), 2)
                                                             AS on_time_pct,
            ROUND(100.0 * AVG(CASE WHEN is_delayed THEN 1.0 ELSE 0.0 END), 2)
                                                             AS sla_breach_pct,
            ROUND(AVG(actual_delivery_days), 2)              AS avg_delivery_days,
            ROUND(100.0 * AVG(CASE WHEN is_returned THEN 1.0 ELSE 0.0 END), 2)
                                                             AS return_rate_pct,
            ROUND(AVG(delivery_cost), 2)                     AS avg_cost_per_order,
            ROUND(SUM(delivery_cost + return_cost), 2)       AS total_operational_cost
        FROM orders
    """)
    row = db.session.execute(sql).mappings().first()
    return dict(row) if row else {}


def get_city_analysis() -> list:
    """Delay rate and average delay days by city (Query 2 from project)."""
    sql = text("""
        SELECT
            city,
            COUNT(*)                                        AS total_orders,
            ROUND(100.0 * AVG(CASE WHEN is_delayed THEN 1.0 ELSE 0.0 END), 2)
                                                            AS delay_rate_pct,
            ROUND(AVG(delay_days), 2)                      AS avg_delay_days
        FROM orders
        GROUP BY city
        ORDER BY delay_rate_pct DESC
    """)
    rows = db.session.execute(sql).mappings().all()
    return [dict(r) for r in rows]


def get_warehouse_performance() -> list:
    """Warehouse-level delay rate and avg cost (Query 3 from project)."""
    sql = text("""
        SELECT
            w.warehouse_id,
            w.name,
            COUNT(o.order_id)                                AS total_orders,
            ROUND(100.0 * AVG(CASE WHEN o.is_delayed THEN 1.0 ELSE 0.0 END), 2)
                                                             AS delay_rate_pct,
            ROUND(AVG(o.delivery_cost), 2)                   AS avg_cost
        FROM warehouses w
        LEFT JOIN orders o ON o.warehouse_id = w.warehouse_id
        GROUP BY w.warehouse_id, w.name
        ORDER BY delay_rate_pct DESC
    """)
    rows = db.session.execute(sql).mappings().all()
    return [dict(r) for r in rows]


def get_monthly_sla(months: int = 12) -> list:
    """Monthly SLA breach trend — last N months (Query 1 from project)."""
    sql = text("""
        SELECT
            TO_CHAR(DATE_TRUNC('month', order_date), 'YYYY-MM') AS month,
            COUNT(*)                                              AS total_orders,
            SUM(CASE WHEN is_delayed THEN 1 ELSE 0 END)          AS sla_breaches,
            ROUND(100.0 * AVG(CASE WHEN is_delayed THEN 1.0 ELSE 0.0 END), 2)
                                                                  AS breach_rate_pct
        FROM orders
        WHERE order_date >= NOW() - INTERVAL ':months months'
        GROUP BY DATE_TRUNC('month', order_date)
        ORDER BY month DESC
    """).bindparams(months=months)
    rows = db.session.execute(sql).mappings().all()
    return [dict(r) for r in rows]


def get_high_risk_orders(threshold: float = 0.60, limit: int = 100) -> list:
    """Fetch orders flagged as high-risk by the prediction model."""
    sql = text("""
        SELECT
            o.order_id,
            o.city,
            o.warehouse_id,
            o.distance_km,
            o.order_date,
            o.status,
            p.delay_probability,
            p.risk_category,
            p.predicted_at
        FROM orders o
        JOIN predictions p ON p.order_id = o.order_id
        WHERE p.delay_probability >= :threshold
          AND o.status NOT IN ('delivered', 'returned')
        ORDER BY p.delay_probability DESC
        LIMIT :limit
    """).bindparams(threshold=threshold, limit=limit)
    rows = db.session.execute(sql).mappings().all()
    return [dict(r) for r in rows]


def get_financial_impact() -> dict:
    """Calculate potential savings from a 15% delay reduction."""
    sql = text("""
        SELECT
            COUNT(*)                                        AS total_orders,
            SUM(CASE WHEN is_delayed THEN 1 ELSE 0 END)    AS delayed_orders,
            SUM(CASE WHEN is_returned THEN 1 ELSE 0 END)   AS returned_orders,
            SUM(delivery_cost + return_cost)                AS total_cost
        FROM orders
    """)
    row = db.session.execute(sql).mappings().first()
    if not row:
        return {}

    delayed        = int(row["delayed_orders"] or 0)
    returned       = int(row["returned_orders"] or 0)
    prevented      = int(delayed * 0.15)
    delay_savings  = prevented * 100    # ₹100 per prevented delay
    return_savings = int(returned * 0.10) * 500  # ₹500 per prevented return

    return {
        "total_orders":       int(row["total_orders"]),
        "delayed_orders":     delayed,
        "returned_orders":    returned,
        "total_cost":         float(row["total_cost"] or 0),
        "prevented_delays":   prevented,
        "delay_savings_inr":  delay_savings,
        "return_savings_inr": return_savings,
        "total_monthly_savings_inr": delay_savings + return_savings,
        "annual_savings_inr": (delay_savings + return_savings) * 12,
    }
