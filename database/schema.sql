-- ============================================================
-- E-Commerce Operations Optimization
-- Database Schema (PostgreSQL)
-- ============================================================
-- Run:  psql -U postgres -d ecommerce_ops -f schema.sql
-- ============================================================

-- ── 0. Create the database (run as superuser if needed) ──────
-- CREATE DATABASE ecommerce_ops;
-- \c ecommerce_ops;

-- ── 1. CUSTOMERS ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS customers (
    customer_id     VARCHAR(10)  PRIMARY KEY,          -- e.g. CUST0001
    name            VARCHAR(100),
    email           VARCHAR(150) UNIQUE,
    phone           VARCHAR(15),
    city            VARCHAR(50),
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);

-- ── 2. WAREHOUSES ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS warehouses (
    warehouse_id    VARCHAR(20)  PRIMARY KEY,          -- e.g. WH_North
    name            VARCHAR(100) NOT NULL,
    city            VARCHAR(50),
    state           VARCHAR(50),
    pincode         VARCHAR(10),
    latitude        DECIMAL(9,6),
    longitude       DECIMAL(9,6),
    capacity        INT          DEFAULT 500,          -- max orders/day
    current_load    INT          DEFAULT 0,
    is_active       BOOLEAN      DEFAULT TRUE,
    created_at      TIMESTAMP    DEFAULT NOW()
);

-- ── 3. ORDERS ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS orders (
    order_id                VARCHAR(10)   PRIMARY KEY,  -- e.g. ORD00001
    customer_id             VARCHAR(10)   REFERENCES customers(customer_id),
    warehouse_id            VARCHAR(20)   REFERENCES warehouses(warehouse_id),
    order_date              TIMESTAMP     NOT NULL,
    city                    VARCHAR(50)   NOT NULL,
    distance_km             INT           NOT NULL,
    order_value             DECIMAL(10,2) NOT NULL,
    order_items             INT           NOT NULL,
    promised_delivery_days  INT           NOT NULL,
    actual_delivery_days    INT,
    is_delayed              BOOLEAN       DEFAULT FALSE,
    delay_days              INT           DEFAULT 0,
    is_returned             BOOLEAN       DEFAULT FALSE,
    delivery_cost           DECIMAL(10,2),
    return_cost             DECIMAL(10,2) DEFAULT 0,
    status                  VARCHAR(20)   DEFAULT 'pending', -- pending|shipped|delivered|returned
    created_at              TIMESTAMP     DEFAULT NOW(),
    updated_at              TIMESTAMP     DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_orders_customer  ON orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_orders_warehouse ON orders(warehouse_id);
CREATE INDEX IF NOT EXISTS idx_orders_date      ON orders(order_date);
CREATE INDEX IF NOT EXISTS idx_orders_city      ON orders(city);
CREATE INDEX IF NOT EXISTS idx_orders_delayed   ON orders(is_delayed);

-- ── 4. MODEL PREDICTIONS ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id       SERIAL        PRIMARY KEY,
    order_id            VARCHAR(10)   REFERENCES orders(order_id),
    model_version       VARCHAR(20)   NOT NULL,
    delay_probability   DECIMAL(5,4)  NOT NULL,         -- 0.0000 – 1.0000
    risk_category       VARCHAR(10)   NOT NULL,          -- Low|Medium|High
    features_snapshot   JSONB,                           -- input features at prediction time
    predicted_at        TIMESTAMP     DEFAULT NOW(),
    was_correct         BOOLEAN                          -- filled after delivery
);

CREATE INDEX IF NOT EXISTS idx_pred_order   ON predictions(order_id);
CREATE INDEX IF NOT EXISTS idx_pred_risk    ON predictions(risk_category);

-- ── 5. KPI SNAPSHOTS (daily/weekly rollups) ───────────────────
CREATE TABLE IF NOT EXISTS kpi_snapshots (
    snapshot_id         SERIAL      PRIMARY KEY,
    snapshot_date       DATE        NOT NULL,
    period_type         VARCHAR(10) DEFAULT 'daily',     -- daily|weekly|monthly
    total_orders        INT,
    delayed_orders      INT,
    on_time_rate        DECIMAL(5,2),
    sla_breach_rate     DECIMAL(5,2),
    avg_delivery_days   DECIMAL(4,2),
    return_rate         DECIMAL(5,2),
    cost_per_order      DECIMAL(10,2),
    total_cost          DECIMAL(12,2),
    high_risk_flagged   INT,
    created_at          TIMESTAMP DEFAULT NOW()
);

-- ── 6. ALERTS (high-risk order actions) ──────────────────────
CREATE TABLE IF NOT EXISTS alerts (
    alert_id        SERIAL      PRIMARY KEY,
    order_id        VARCHAR(10) REFERENCES orders(order_id),
    alert_type      VARCHAR(30) NOT NULL,   -- high_risk|sla_breach|return_spike
    message         TEXT,
    severity        VARCHAR(10) DEFAULT 'medium', -- low|medium|high|critical
    is_resolved     BOOLEAN     DEFAULT FALSE,
    resolved_at     TIMESTAMP,
    created_at      TIMESTAMP   DEFAULT NOW()
);

-- ── 7. AUDIT LOG ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS audit_log (
    log_id      SERIAL      PRIMARY KEY,
    table_name  VARCHAR(50),
    record_id   VARCHAR(20),
    action      VARCHAR(10),                -- INSERT|UPDATE|DELETE
    old_values  JSONB,
    new_values  JSONB,
    changed_by  VARCHAR(100),
    changed_at  TIMESTAMP   DEFAULT NOW()
);

-- ── 8. AUTO-UPDATE updated_at TRIGGER ────────────────────────
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_orders_updated
    BEFORE UPDATE ON orders
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER trg_customers_updated
    BEFORE UPDATE ON customers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ── 9. USEFUL BUSINESS VIEWS ─────────────────────────────────

-- View: Current KPIs
CREATE OR REPLACE VIEW v_current_kpis AS
SELECT
    COUNT(*)                                        AS total_orders,
    SUM(is_delayed::INT)                            AS delayed_orders,
    ROUND(100.0 * (1 - AVG(is_delayed::INT)), 2)    AS on_time_pct,
    ROUND(100.0 * AVG(is_delayed::INT), 2)          AS sla_breach_pct,
    ROUND(AVG(actual_delivery_days), 2)             AS avg_delivery_days,
    ROUND(100.0 * AVG(is_returned::INT), 2)         AS return_rate_pct,
    ROUND(AVG(delivery_cost), 2)                    AS avg_cost_per_order
FROM orders
WHERE order_date >= NOW() - INTERVAL '30 days';

-- View: Warehouse performance
CREATE OR REPLACE VIEW v_warehouse_performance AS
SELECT
    w.warehouse_id,
    w.name,
    COUNT(o.order_id)                               AS total_orders,
    ROUND(100.0 * AVG(o.is_delayed::INT), 2)        AS delay_rate_pct,
    ROUND(AVG(o.delivery_cost), 2)                  AS avg_cost
FROM warehouses w
LEFT JOIN orders o ON o.warehouse_id = w.warehouse_id
GROUP BY w.warehouse_id, w.name
ORDER BY delay_rate_pct DESC;

-- View: City-level delay summary
CREATE OR REPLACE VIEW v_city_delay_summary AS
SELECT
    city,
    COUNT(*)                                        AS total_orders,
    SUM(is_delayed::INT)                            AS delayed_orders,
    ROUND(100.0 * AVG(is_delayed::INT), 2)          AS delay_rate_pct,
    ROUND(AVG(delay_days), 2)                       AS avg_delay_days
FROM orders
GROUP BY city
ORDER BY delay_rate_pct DESC;

-- ── 10. SEED: Sample warehouses ───────────────────────────────
INSERT INTO warehouses (warehouse_id, name, city, state, capacity)
VALUES
  ('WH_North', 'North Warehouse', 'Delhi',     'Delhi',      600),
  ('WH_South', 'South Warehouse', 'Chennai',   'Tamil Nadu', 550),
  ('WH_East',  'East Warehouse',  'Kolkata',   'West Bengal',500),
  ('WH_West',  'West Warehouse',  'Mumbai',    'Maharashtra',700)
ON CONFLICT DO NOTHING;
