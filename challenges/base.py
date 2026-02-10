"""
Base challenge class.
=====================
All challenge types inherit from this and implement the
execution / grading contract.
"""

import os
from typing import Any


class BaseChallenge:
    """Abstract base for all coding challenges."""

    def __init__(self, config: dict, challenge_dir: str):
        self.config = config
        self.challenge_dir = challenge_dir

        # Project root (two levels up from challenges/<name>/)
        self.base_dir = os.path.dirname(
            os.path.dirname(os.path.abspath(challenge_dir))
        )

        # Core metadata
        self.id: str = config["id"]
        self.title: str = config["title"]
        self.difficulty: str = config.get("difficulty", "medium")
        self.tags: list = config.get("tags", [])
        self.description: str = config.get("description", "")
        self.objective: str = config.get("objective", "")
        self.timeout: int = config.get("timeout_seconds", 300)
        self.run_timeout: int = config.get("run_timeout_seconds", 120)

    # ── Serialisation ────────────────────────────────────────

    def get_summary(self) -> dict:
        """Short summary for challenge listings."""
        return {
            "id": self.id,
            "title": self.title,
            "difficulty": self.difficulty,
            "tags": self.tags,
            "objective": self.objective,
        }

    def get_details(self) -> dict:
        """Full details for the problem panel."""
        return {
            **self.get_summary(),
            "description": self.description,
            "starter_code": self.get_starter_code(),
            "dataset": self.config.get("dataset", {}),
            "scoring": self.config.get("scoring", {}),
            "examples": self.config.get("examples", []),
            "allowed_libraries": self.config.get("allowed_libraries", []),
        }

    # ── Starter code ─────────────────────────────────────────

    def get_starter_code(self) -> str:
        """Load the starter code template shown to users."""
        filename = self.config.get("starter_code_file", "starter_code.py")
        path = os.path.join(self.challenge_dir, filename)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        return "# No starter code available\n"

    # ── Execution (subclasses must override) ─────────────────

    def get_harness_code(self, mode: str) -> str:
        """Return the harness script code for sandbox execution."""
        raise NotImplementedError

    def get_data_files(self, mode: str) -> dict:
        """Return ``{sandbox_filename: source_path}`` for sandbox files."""
        raise NotImplementedError

    def get_harness_args(self, mode: str) -> list:
        """Return CLI args for the harness (before output_path)."""
        raise NotImplementedError

    # ── Grading (subclasses must override) ───────────────────

    def grade(self, user_code: str, execution_output: dict) -> dict:
        """
        Grade a submission.

        Parameters
        ----------
        user_code : str
            The user's source code (for static analysis).
        execution_output : dict
            Structured output from the harness (predictions, etc.).

        Returns
        -------
        dict with total_score, max_score, passed, categories, etc.
        """
        raise NotImplementedError
