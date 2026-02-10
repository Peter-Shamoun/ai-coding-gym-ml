"""
Execution API — run and submit code for challenges.
"""

from flask import Blueprint, jsonify, request

from api.stats import record_submission
from challenges.registry import ChallengeRegistry
from config import Config
from engine.executor import execute_challenge

execution_bp = Blueprint("execution", __name__)


@execution_bp.route("/challenges/<challenge_id>/run", methods=["POST"])
def run_code(challenge_id: str):
    """
    Run user code for quick testing (like LeetCode's "Run" button).

    Executes on a sample of training data.  Returns stdout, stderr,
    sample accuracy, and timing info — but does NOT run the full grading
    rubric.

    Request JSON::

        { "code": "<python source code>" }
    """
    registry = ChallengeRegistry()
    if registry.get(challenge_id) is None:
        return jsonify({"error": f"Challenge not found: {challenge_id}"}), 404

    data = request.get_json(silent=True) or {}
    code = data.get("code", "")

    if not code.strip():
        return jsonify({"error": "No code provided."}), 400
    if len(code) > Config.MAX_CODE_SIZE:
        return (
            jsonify(
                {"error": f"Code exceeds size limit ({Config.MAX_CODE_SIZE} bytes)."}
            ),
            400,
        )

    result = execute_challenge(challenge_id, code, mode="run")
    return jsonify(result.to_dict())


@execution_bp.route("/challenges/<challenge_id>/submit", methods=["POST"])
def submit_code(challenge_id: str):
    """
    Submit user code for full grading (like LeetCode's "Submit" button).

    Trains on the full training set, predicts on the hidden test set,
    and runs the complete grading rubric.

    Request JSON::

        { "code": "<python source code>", "prompt_text": "<optional user prompt>" }
    """
    registry = ChallengeRegistry()
    if registry.get(challenge_id) is None:
        return jsonify({"error": f"Challenge not found: {challenge_id}"}), 404

    data = request.get_json(silent=True) or {}
    code = data.get("code", "")
    prompt_text = data.get("prompt_text", "") or ""

    if not code.strip():
        return jsonify({"error": "No code provided."}), 400
    if len(code) > Config.MAX_CODE_SIZE:
        return (
            jsonify(
                {"error": f"Code exceeds size limit ({Config.MAX_CODE_SIZE} bytes)."}
            ),
            400,
        )

    result = execute_challenge(challenge_id, code, mode="submit")
    # Record for social metrics (acceptance rate, prompt golf)
    if result.grading is not None:
        passed = result.grading.get("passed", False)
        record_submission(challenge_id, passed=passed, prompt_text=prompt_text)
    return jsonify(result.to_dict())
