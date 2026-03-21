"""
api/models/models.py
─────────────────────
SQLAlchemy ORM models — mirror the database/schema.sql tables.
"""

from datetime import datetime
from api.extensions import db


class Customer(db.Model):
    __tablename__ = "customers"

    customer_id = db.Column(db.String(10), primary_key=True)
    name        = db.Column(db.String(100))
    email       = db.Column(db.String(150), unique=True)
    phone       = db.Column(db.String(15))
    city        = db.Column(db.String(50))
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at  = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    orders = db.relationship("Order", back_populates="customer", lazy="dynamic")

    def to_dict(self):
        return {
            "customer_id": self.customer_id,
            "name": self.name,
            "email": self.email,
            "city": self.city,
        }


class Warehouse(db.Model):
    __tablename__ = "warehouses"

    warehouse_id  = db.Column(db.String(20), primary_key=True)
    name          = db.Column(db.String(100), nullable=False)
    city          = db.Column(db.String(50))
    state         = db.Column(db.String(50))
    capacity      = db.Column(db.Integer, default=500)
    current_load  = db.Column(db.Integer, default=0)
    is_active     = db.Column(db.Boolean, default=True)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)

    orders = db.relationship("Order", back_populates="warehouse", lazy="dynamic")

    def to_dict(self):
        return {
            "warehouse_id": self.warehouse_id,
            "name": self.name,
            "city": self.city,
            "capacity": self.capacity,
            "current_load": self.current_load,
            "utilisation_pct": round(self.current_load / max(self.capacity, 1) * 100, 1),
        }


class Order(db.Model):
    __tablename__ = "orders"

    order_id                = db.Column(db.String(10), primary_key=True)
    customer_id             = db.Column(db.String(10), db.ForeignKey("customers.customer_id"))
    warehouse_id            = db.Column(db.String(20), db.ForeignKey("warehouses.warehouse_id"))
    order_date              = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    city                    = db.Column(db.String(50), nullable=False)
    distance_km             = db.Column(db.Integer, nullable=False)
    order_value             = db.Column(db.Numeric(10, 2), nullable=False)
    order_items             = db.Column(db.Integer, nullable=False)
    warehouse_load          = db.Column(db.Integer)
    promised_delivery_days  = db.Column(db.Integer, nullable=False)
    actual_delivery_days    = db.Column(db.Integer)
    is_delayed              = db.Column(db.Boolean, default=False)
    delay_days              = db.Column(db.Integer, default=0)
    is_returned             = db.Column(db.Boolean, default=False)
    past_delays             = db.Column(db.Integer, default=0)
    delivery_cost           = db.Column(db.Numeric(10, 2))
    return_cost             = db.Column(db.Numeric(10, 2), default=0)
    status                  = db.Column(db.String(20), default="pending")
    created_at              = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at              = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    customer    = db.relationship("Customer", back_populates="orders")
    warehouse   = db.relationship("Warehouse", back_populates="orders")
    predictions = db.relationship("Prediction", back_populates="order", lazy="dynamic")
    alerts      = db.relationship("Alert", back_populates="order", lazy="dynamic")

    def to_dict(self):
        return {
            "order_id":               self.order_id,
            "customer_id":            self.customer_id,
            "warehouse_id":           self.warehouse_id,
            "order_date":             self.order_date.isoformat() if self.order_date else None,
            "city":                   self.city,
            "distance_km":            self.distance_km,
            "order_value":            float(self.order_value),
            "order_items":            self.order_items,
            "promised_delivery_days": self.promised_delivery_days,
            "actual_delivery_days":   self.actual_delivery_days,
            "is_delayed":             self.is_delayed,
            "delay_days":             self.delay_days,
            "is_returned":            self.is_returned,
            "status":                 self.status,
            "delivery_cost":          float(self.delivery_cost) if self.delivery_cost else None,
        }


class Prediction(db.Model):
    __tablename__ = "predictions"

    prediction_id      = db.Column(db.Integer, primary_key=True, autoincrement=True)
    order_id           = db.Column(db.String(10), db.ForeignKey("orders.order_id"))
    model_version      = db.Column(db.String(20), nullable=False)
    delay_probability  = db.Column(db.Numeric(5, 4), nullable=False)
    risk_category      = db.Column(db.String(10), nullable=False)  # Low|Medium|High
    features_snapshot  = db.Column(db.JSON)
    predicted_at       = db.Column(db.DateTime, default=datetime.utcnow)
    was_correct        = db.Column(db.Boolean)

    order = db.relationship("Order", back_populates="predictions")

    def to_dict(self):
        return {
            "prediction_id":     self.prediction_id,
            "order_id":          self.order_id,
            "model_version":     self.model_version,
            "delay_probability": float(self.delay_probability),
            "risk_category":     self.risk_category,
            "predicted_at":      self.predicted_at.isoformat(),
        }


class Alert(db.Model):
    __tablename__ = "alerts"

    alert_id    = db.Column(db.Integer, primary_key=True, autoincrement=True)
    order_id    = db.Column(db.String(10), db.ForeignKey("orders.order_id"))
    alert_type  = db.Column(db.String(30), nullable=False)
    message     = db.Column(db.Text)
    severity    = db.Column(db.String(10), default="medium")
    is_resolved = db.Column(db.Boolean, default=False)
    resolved_at = db.Column(db.DateTime)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)

    order = db.relationship("Order", back_populates="alerts")

    def to_dict(self):
        return {
            "alert_id":    self.alert_id,
            "order_id":    self.order_id,
            "alert_type":  self.alert_type,
            "message":     self.message,
            "severity":    self.severity,
            "is_resolved": self.is_resolved,
            "created_at":  self.created_at.isoformat(),
        }


class KpiSnapshot(db.Model):
    __tablename__ = "kpi_snapshots"

    snapshot_id       = db.Column(db.Integer, primary_key=True, autoincrement=True)
    snapshot_date     = db.Column(db.Date, nullable=False)
    period_type       = db.Column(db.String(10), default="daily")
    total_orders      = db.Column(db.Integer)
    delayed_orders    = db.Column(db.Integer)
    on_time_rate      = db.Column(db.Numeric(5, 2))
    sla_breach_rate   = db.Column(db.Numeric(5, 2))
    avg_delivery_days = db.Column(db.Numeric(4, 2))
    return_rate       = db.Column(db.Numeric(5, 2))
    cost_per_order    = db.Column(db.Numeric(10, 2))
    high_risk_flagged = db.Column(db.Integer)
    created_at        = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
