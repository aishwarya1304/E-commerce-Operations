"""
api/extensions.py
─────────────────
Shared Flask extensions — import here to avoid circular imports.
"""

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
