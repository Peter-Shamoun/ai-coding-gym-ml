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


# ── Mock social metrics (to be replaced with real DB later) ───

_MOCK_METRICS = {
    "email-spam-detection": {
        "acceptance_rate": 68,
        "shortest_prompt": 80,
        "total_submissions": 142,
    },
    "mnist_digit_recognition": {
        "acceptance_rate": 72,
        "shortest_prompt": 95,
        "total_submissions": 98,
    },
    "imdb_sentiment_analysis": {
        "acceptance_rate": 55,
        "shortest_prompt": 120,
        "total_submissions": 73,
    },
    "customer_churn_prediction": {
        "acceptance_rate": 48,
        "shortest_prompt": 145,
        "total_submissions": 61,
    },
    "housing_price_prediction": {
        "acceptance_rate": 35,
        "shortest_prompt": 170,
        "total_submissions": 44,
    },
}


@challenges_bp.route("/challenges/<challenge_id>/metrics", methods=["GET"])
def get_challenge_metrics(challenge_id: str):
    """Return social metrics for a challenge (mock data for now)."""
    metrics = _MOCK_METRICS.get(
        challenge_id,
        {"acceptance_rate": 0, "shortest_prompt": 0, "total_submissions": 0},
    )
    return jsonify(metrics)
