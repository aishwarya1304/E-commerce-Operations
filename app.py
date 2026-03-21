"""
E-Commerce Operations Optimization API
=======================================
Flask application factory & entry point.

Run (dev):
    python app.py

Run (production):
    gunicorn "app:create_app()" --bind 0.0.0.0:5000 --workers 4
"""

import argparse
from flask import Flask, jsonify, render_template
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flasgger import Swagger

from api.extensions import db
from api.routes.orders import orders_bp
from api.routes.predictions import predictions_bp
from api.routes.analytics import analytics_bp
from api.routes.auth import auth_bp
from config.settings import get_config


def create_app(config_name: str = "development") -> Flask:
    """Application factory — creates and wires the Flask app."""
    app = Flask(__name__, template_folder="frontend/templates")

    # ── Config ──────────────────────────────────────────────
    cfg = get_config(config_name)
    app.config.from_object(cfg)

    # ── Extensions ──────────────────────────────────────────
    CORS(app, resources={r"/api/*": {"origins": "*"}})   # Restrict origins in prod
    db.init_app(app)
    JWTManager(app)

    # ── Swagger / OpenAPI docs at /apidocs ──────────────────
    Swagger(app, template={
        "info": {
            "title": "E-Commerce Operations API",
            "description": "Delivery delay prediction & operational analytics",
            "version": "1.0.0"
        },
        "securityDefinitions": {
            "Bearer": {
                "type": "apiKey",
                "name": "Authorization",
                "in": "header"
            }
        }
    })

    # ── Blueprints ───────────────────────────────────────────
    app.register_blueprint(auth_bp,        url_prefix="/api/auth")
    app.register_blueprint(orders_bp,      url_prefix="/api/orders")
    app.register_blueprint(predictions_bp, url_prefix="/api/predictions")
    app.register_blueprint(analytics_bp,   url_prefix="/api/analytics")

    # ── Dashboard ────────────────────────────────────────────
    @app.route('/')
    def index():
        return render_template('dashboard.html')

    # ── Health check ─────────────────────────────────────────
    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "version": "1.0.0"}), 200

    # ── DB auto-create tables (dev only) ─────────────────────
    with app.app_context():
        db.create_all()

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Flask app")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the app on")
    args = parser.parse_args()
    app = create_app()
    app.run(host="0.0.0.0", port=args.port, debug=True)
