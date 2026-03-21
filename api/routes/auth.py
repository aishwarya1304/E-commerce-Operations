"""
api/routes/auth.py
───────────────────
Simple JWT authentication endpoints.

Endpoints
---------
POST /api/auth/login    Issue access token
POST /api/auth/refresh  Refresh token (not implemented here — extend as needed)
"""

from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token

auth_bp = Blueprint("auth", __name__)

# ── Demo users (replace with DB-backed user table in production) ──
DEMO_USERS = {
    "admin": "admin123",
    "analyst": "analyst123",
}


@auth_bp.post("/login")
def login():
    """
    Login and receive a JWT access token.
    ---
    tags: [Auth]
    parameters:
      - in: body
        name: credentials
        required: true
        schema:
          type: object
          required: [username, password]
          properties:
            username: { type: string, example: admin }
            password: { type: string, example: admin123 }
    responses:
      200:
        description: JWT access token
      401:
        description: Invalid credentials
    """
    body     = request.get_json(force=True)
    username = body.get("username", "")
    password = body.get("password", "")

    if DEMO_USERS.get(username) == password:
        token = create_access_token(identity=username)
        return jsonify({"access_token": token, "username": username}), 200

    return jsonify({"error": "Invalid username or password"}), 401
