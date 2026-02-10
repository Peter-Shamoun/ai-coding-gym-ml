"""
Challenge API — browse challenges, get details, download data.
"""

from flask import Blueprint, jsonify, send_file

from challenges.registry import ChallengeRegistry

challenges_bp = Blueprint("challenges", __name__)


@challenges_bp.route("/challenges", methods=["GET"])
def list_challenges():
    """List all available challenges."""
    registry = ChallengeRegistry()
    return jsonify({"challenges": registry.list_all()})


@challenges_bp.route("/challenges/<challenge_id>", methods=["GET"])
def get_challenge(challenge_id: str):
    """Get full details for a specific challenge (problem panel data)."""
    registry = ChallengeRegistry()
    challenge = registry.get(challenge_id)
    if challenge is None:
        return jsonify({"error": f"Challenge not found: {challenge_id}"}), 404
    return jsonify(challenge.get_details())


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
