"""
Challenge registry.
====================
Discovers and loads all challenge packages from the ``challenges/`` directory.
Each challenge is a sub-package that exports a ``Challenge`` class and ships
a ``challenge.yaml`` configuration file.
"""

import importlib
import os

import yaml


class ChallengeRegistry:
    """Singleton registry of available challenges."""

    _instance = None
    _challenges: dict = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._discover()
        return cls._instance

    # ── Discovery ────────────────────────────────────────────

    def _discover(self):
        """Scan the challenges/ directory for valid challenge packages."""
        challenges_dir = os.path.dirname(os.path.abspath(__file__))

        for entry in os.scandir(challenges_dir):
            if not entry.is_dir() or entry.name.startswith(("_", ".")):
                continue

            config_path = os.path.join(entry.path, "challenge.yaml")
            if not os.path.isfile(config_path):
                continue

            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)

                module = importlib.import_module(f"challenges.{entry.name}")
                challenge_cls = getattr(module, "Challenge")
                challenge = challenge_cls(config, entry.path)
                self._challenges[challenge.id] = challenge
            except Exception as exc:
                print(f"[registry] Failed to load challenge '{entry.name}': {exc}")

    # ── Public API ───────────────────────────────────────────

    def get(self, challenge_id: str):
        """Get a challenge by ID, or ``None``."""
        return self._challenges.get(challenge_id)

    def list_all(self) -> list:
        """Return summary list of every challenge."""
        return [c.get_summary() for c in self._challenges.values()]

    def list_ids(self) -> list:
        return list(self._challenges.keys())
