"""
Challenge API — browse challenges, get details, download data, stats.
"""

from flask import Blueprint, jsonify, send_file

from api.stats import get_stats
from challenges.registry import ChallengeRegistry

challenges_bp = Blueprint("challenges", __name__)


@challenges_bp.route("/challenges", methods=["GET"])
def list_challenges():
    """List all available challenges."""
    registry = ChallengeRegistry()
    return jsonify({"challenges": registry.list_all()})


@challenges_bp.route("/challenges/<challenge_id>", methods=["GET"])
def get_challenge(challenge_id: str):
    """Get full details for a specific challenge (problem panel data), including stats."""
    registry = ChallengeRegistry()
    challenge = registry.get(challenge_id)
    if challenge is None:
        return jsonify({"error": f"Challenge not found: {challenge_id}"}), 404
    details = challenge.get_details()
    details["stats"] = get_stats(challenge_id)
    return jsonify(details)


@challenges_bp.route("/challenges/<challenge_id>/stats", methods=["GET"])
def challenge_stats(challenge_id: str):
    """Get social metrics for a challenge (acceptance rate, prompt golf)."""
    registry = ChallengeRegistry()
    if registry.get(challenge_id) is None:
        return jsonify({"error": f"Challenge not found: {challenge_id}"}), 404
    return jsonify(get_stats(challenge_id))


@challenges_bp.route(
    "/challenges/<challenge_id>/dataset/<filename>", methods=["GET"]
)
def download_dataset(challenge_id: str, filename: str):
    """Download a user-accessible dataset file."""
    registry = ChallengeRegistry()
    challenge = registry.get(challenge_id)
    if challenge is None:
        return jsonify({"error": f"Challenge not found: {challenge_id}"}), 404

    # Only expose files that are copied into the sandbox (training data).
    data_files = challenge.get_data_files("run")
    if filename not in data_files:
        return jsonify({"error": f"File not available: {filename}"}), 404

    return send_file(
        data_files[filename], as_attachment=True, download_name=filename
    )
