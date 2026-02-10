"""
Submission stats persistence for social metrics.
Stores attempts and shortest passing prompt per challenge; exposes get_stats().
Uses SQLite in SUBMISSIONS_DIR (submissions/submissions.db).
"""

import os
import sqlite3
from typing import Optional

from config import Config


def _db_path() -> str:
    os.makedirs(Config.SUBMISSIONS_DIR, exist_ok=True)
    return os.path.join(Config.SUBMISSIONS_DIR, "submissions.db")


def _init_db() -> None:
    path = _db_path()
    with sqlite3.connect(path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                challenge_id TEXT NOT NULL,
                passed INTEGER NOT NULL,
                prompt_text TEXT NOT NULL,
                prompt_length INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_submissions_challenge ON submissions(challenge_id)"
        )


def record_submission(
    challenge_id: str,
    passed: bool,
    prompt_text: str = "",
) -> None:
    """Record one submission for stats (acceptance rate, prompt golf)."""
    _init_db()
    prompt_text = prompt_text or ""
    length = len(prompt_text)
    with sqlite3.connect(_db_path()) as conn:
        conn.execute(
            "INSERT INTO submissions (challenge_id, passed, prompt_text, prompt_length) VALUES (?, ?, ?, ?)",
            (challenge_id, 1 if passed else 0, prompt_text, length),
        )


def get_stats(challenge_id: str) -> dict:
    """
    Return social metrics for a challenge:
    - acceptance_rate (0..1)
    - total_attempts, passed_attempts
    - shortest_passing_prompt_length (or None)
    - shortest_passing_prompt_preview (first 50 chars, or None)
    """
    _init_db()
    with sqlite3.connect(_db_path()) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT passed, prompt_text, prompt_length FROM submissions WHERE challenge_id = ?",
            (challenge_id,),
        )
        rows = cur.fetchall()

    total = len(rows)
    passed_count = sum(1 for r in rows if r["passed"])
    acceptance_rate = (passed_count / total) if total else 0.0

    passing = [r for r in rows if r["passed"] and r["prompt_length"] is not None]
    shortest = min(passing, key=lambda r: r["prompt_length"]) if passing else None

    return {
        "acceptance_rate": round(acceptance_rate, 4),
        "total_attempts": total,
        "passed_attempts": passed_count,
        "shortest_passing_prompt_length": shortest["prompt_length"] if shortest else None,
        "shortest_passing_prompt_preview": (
            (shortest["prompt_text"] or "")[:50] if shortest else None
        ),
    }
