"""
Application configuration.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file


class Config:
    """Central configuration — reads from env vars with sensible defaults."""

    # ── Flask ────────────────────────────────────────────────
    DEBUG = os.environ.get("FLASK_DEBUG", "false").lower() in ("true", "1", "yes")
    PORT = int(os.environ.get("PORT", 5000))
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-key-change-in-production")

    # ── Paths ────────────────────────────────────────────────
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CHALLENGES_DIR = os.path.join(BASE_DIR, "challenges")
    SUBMISSIONS_DIR = os.path.join(BASE_DIR, "submissions")

    # ── Execution sandbox ────────────────────────────────────
    EXECUTION_TIMEOUT = int(os.environ.get("EXECUTION_TIMEOUT", 300))   # submit mode
    RUN_TIMEOUT = int(os.environ.get("RUN_TIMEOUT", 120))               # run mode
    MAX_OUTPUT_SIZE = int(os.environ.get("MAX_OUTPUT_SIZE", 1_048_576))  # 1 MB
    MAX_CODE_SIZE = int(os.environ.get("MAX_CODE_SIZE", 102_400))       # 100 KB

    # ── Agent / LLM ──────────────────────────────────────────
    LLM_API_KEY = os.environ.get("OPENAI_API_KEY", os.environ.get("LLM_API_KEY", ""))
    LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
