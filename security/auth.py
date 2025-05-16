#!/usr/bin/env python3
# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Authentication Module

Provides JWT authentication functionality for the Super AI API.
"""

import os
from functools import wraps
from flask import request, jsonify, current_app
from flask_jwt_extended import (
    create_access_token, create_refresh_token,
    jwt_required, get_jwt_identity, verify_jwt_in_request
)
import datetime
from werkzeug.security import generate_password_hash, check_password_hash

# Mock user database - in production, use a real database
_USERS = {
    "admin": {
        "password": generate_password_hash("admin_password"),  # Change in production!
        "roles": ["admin"]
    },
    "api_user": {
        "password": generate_password_hash("api_password"),  # Change in production!
        "roles": ["user"]
    }
}

def authenticate_user(username, password):
    """Authenticate a user"""
    if username not in _USERS:
        return None

    user = _USERS[username]
    if not check_password_hash(user["password"], password):
        return None

    return {"username": username, "roles": user["roles"]}

def generate_tokens(user_data):
    """Generate access and refresh tokens"""
    # Create access token (expires in 1 hour)
    access_token = create_access_token(
        identity=user_data["username"],
        additional_claims={"roles": user_data["roles"]},
        expires_delta=datetime.timedelta(hours=1)
    )

    # Create refresh token (expires in 30 days)
    refresh_token = create_refresh_token(
        identity=user_data["username"],
        additional_claims={"roles": user_data["roles"]},
        expires_delta=datetime.timedelta(days=30)
    )

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "user": {
            "username": user_data["username"],
            "roles": user_data["roles"]
        }
    }

def role_required(role):
    """Decorator to check if user has required role"""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            verify_jwt_in_request()
            claims = get_jwt_identity()
            roles = claims.get("roles", [])

            if role not in roles:
                return jsonify({"msg": "Access denied: insufficient privileges"}), 403

            return fn(*args, **kwargs)
        return wrapper
    return decorator

# Route handlers to be registered with the Flask app
def register_auth_routes(app):
    """Register authentication routes with the Flask app"""

    @app.route("/api/auth/login", methods=["POST"])
    def login():
        """Login and get JWT tokens"""
        username = request.json.get("username", None)
        password = request.json.get("password", None)

        if not username or not password:
            return jsonify({"error": "Missing username or password"}), 400

        user = authenticate_user(username, password)
        if not user:
            return jsonify({"error": "Invalid username or password"}), 401

        tokens = generate_tokens(user)
        return jsonify(tokens), 200

    @app.route("/api/auth/refresh", methods=["POST"])
    @jwt_required(refresh=True)
    def refresh():
        """Refresh access token"""
        identity = get_jwt_identity()
        roles = get_jwt_identity().get("roles", [])

        access_token = create_access_token(
            identity=identity,
            additional_claims={"roles": roles},
            expires_delta=datetime.timedelta(hours=1)
        )

        return jsonify(access_token=access_token), 200