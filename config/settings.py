"""
config/settings.py
──────────────────
Centralised configuration using environment variables.
All secrets come from .env (python-dotenv loads it at startup).
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env in project root


class BaseConfig:
    # Flask
    SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
    DEBUG = False
    TESTING = False

    # SQLAlchemy
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL", "sqlite:///ecommerce_ops.db"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = False

    # JWT
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-jwt-secret")
    JWT_ACCESS_TOKEN_EXPIRES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRES", 3600))

    # ML Model
    MODEL_PATH = os.getenv("MODEL_PATH", "models/random_forest_model.pkl")
    MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
    HIGH_RISK_THRESHOLD = float(os.getenv("HIGH_RISK_THRESHOLD", 0.60))

    # Business constants
    DELAY_COST_PER_ORDER = float(os.getenv("DELAY_COST_PER_ORDER", 100))
    RETURN_COST_PER_ORDER = float(os.getenv("RETURN_COST_PER_ORDER", 500))

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


class DevelopmentConfig(BaseConfig):
    DEBUG = True
    SQLALCHEMY_ECHO = True   # Print SQL queries to console


class ProductionConfig(BaseConfig):
    DEBUG = False
    SQLALCHEMY_ECHO = False


class TestingConfig(BaseConfig):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"  # in-memory DB for tests
    JWT_ACCESS_TOKEN_EXPIRES = 60


_config_map = {
    "development": DevelopmentConfig,
    "production":  ProductionConfig,
    "testing":     TestingConfig,
}


def get_config(name: str = "development"):
    """Return the correct config class by name."""
    return _config_map.get(name, DevelopmentConfig)
