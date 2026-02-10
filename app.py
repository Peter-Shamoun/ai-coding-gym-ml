"""
AI Coding Gym — Backend Server
================================
LeetCode-style ML challenge platform with AI coding agent.

Provides REST APIs for:
  - Browsing challenges (descriptions, starter code, datasets)
  - Running code for quick testing  (POST /api/challenges/<id>/run)
  - Submitting code for full grading (POST /api/challenges/<id>/submit)
  - AI agent for code generation     (POST /api/agent/chat)
"""

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

from config import Config


def create_app(config_class=Config):
    """Application factory."""
    app = Flask(__name__, static_folder="static")
    app.config.from_object(config_class)
    CORS(app)

    # ── Register API blueprints ──────────────────────────────
    from api.challenges import challenges_bp
    from api.execution import execution_bp
    from api.agent import agent_bp

    app.register_blueprint(challenges_bp, url_prefix="/api")
    app.register_blueprint(execution_bp, url_prefix="/api")
    app.register_blueprint(agent_bp, url_prefix="/api")

    # ── Utility routes ───────────────────────────────────────

    @app.route("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.route("/")
    def index():
        return send_from_directory("static", "index.html")

    return app


# ── Module-level app instance (used by gunicorn: app:app) ────
app = create_app()


# ── Entry point ──────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 58)
    print("  AI Coding Gym — Backend Server  (v2.0)")
    print("=" * 58)
    print(f"  Debug : {Config.DEBUG}")
    print(f"  Port  : {Config.PORT}")
    print(f"  URL   : http://localhost:{Config.PORT}")
    print()
    print("  API Endpoints:")
    print("    GET  /api/health                    Health check")
    print("    GET  /api/challenges                List challenges")
    print("    GET  /api/challenges/<id>           Challenge details")
    print("    POST /api/challenges/<id>/run       Quick test")
    print("    POST /api/challenges/<id>/submit    Full grading")
    print("    POST /api/agent/chat                AI coding agent")
    print("=" * 58)

    app.run(debug=Config.DEBUG, host="0.0.0.0", port=Config.PORT)
