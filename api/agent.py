"""
Agent API — AI-powered coding assistant.
=========================================
Uses OpenAI's API to generate / refine code for challenges.
The agent receives the challenge context (description, starter code,
allowed libraries, current code) and returns updated Python code.

Supports multiple challenges generically — loads challenge details
from the registry and builds the system prompt accordingly.
"""

import json
import socket

from flask import Blueprint, jsonify, request

from challenges.registry import ChallengeRegistry
from config import Config

agent_bp = Blueprint("agent", __name__)


# ── Challenge-specific hints ────────────────────────────────────

_CHALLENGE_HINTS = {
    "email-spam-detection": """
CHALLENGE-SPECIFIC TIPS:
- This is a text classification problem (spam vs ham).
- TF-IDF with bigrams/trigrams is very effective for this task.
- Consider text preprocessing: lowercase, remove URLs, remove numbers, stemming/lemmatization.
- Handle class imbalance with class_weight='balanced'.
- LogisticRegression, SVM, or ensemble methods work well.
- The function receives a pandas DataFrame and must return a callable that accepts a list of strings.
""",
    "mnist_digit_recognition": """
CHALLENGE-SPECIFIC TIPS:
- This is an image classification problem (28x28 grayscale images, 10 digit classes).
- Images are uint8 with values 0-255. Consider normalizing to 0-1 range (divide by 255).
- Think about the spatial structure of images — pixels near each other are related.
- Common approaches: CNN, flatten + MLP, sklearn classifiers on flattened images.
- The function receives numpy arrays (X_train: (N, 28, 28), y_train: (N,)).
- Must return a callable that accepts X_test (N, 28, 28) and returns predictions (N,) with values 0-9.
- For best accuracy, consider using deep learning frameworks (TensorFlow/Keras or PyTorch).
- Data augmentation (rotation, shifts) can improve generalization.
""",
    "imdb_sentiment_analysis": """
CHALLENGE-SPECIFIC TIPS:
- This is a binary sentiment classification problem (positive/negative movie reviews).
- Reviews can be long — TF-IDF with n-grams is a strong baseline.
- Handle negation carefully: "not good" has opposite meaning to "good".
- Consider: text preprocessing (HTML tags, punctuation), n-gram features, sublinear TF.
- The function receives a pandas DataFrame with 'text' and 'sentiment' columns.
- Must return a callable that takes a DataFrame with 'text' column and returns 0/1 predictions.
- Advanced approaches: word embeddings, LSTM/GRU, or pre-trained transformers (BERT).
- LogisticRegression with good TF-IDF features can reach ~89%.
""",
    "customer_churn_prediction": """
CHALLENGE-SPECIFIC TIPS:
- This is an IMBALANCED binary classification problem (~27% churn, ~73% no churn).
- WARNING: Simply predicting "no churn" for everyone gives 73% accuracy but 0% F1!
- You MUST handle class imbalance: use class_weight='balanced', SMOTE, or oversampling.
- The dataset has MIXED types: categorical (Contract, InternetService) + numerical (tenure, MonthlyCharges).
- Watch for data quality: TotalCharges has blank values that need handling.
- Encode categoricals: OneHotEncoder, get_dummies, or LabelEncoder.
- The function receives a DataFrame with features + 'Churn' column ("Yes"/"No").
- Must return a callable that takes a DataFrame (without 'Churn') and returns 0/1 predictions.
- Key features: tenure, Contract type, MonthlyCharges, InternetService, PaymentMethod.
- Consider predict_proba with threshold tuning instead of default 0.5 threshold.
""",
    "housing_price_prediction": """
CHALLENGE-SPECIFIC TIPS:
- This is a REGRESSION problem — predict continuous house values, NOT classes!
- Use R² and RMSE as metrics, NOT accuracy.
- Simple linear regression only gets R² ≈ 0.60 — you need better approaches.
- Feature scaling is important (StandardScaler or similar).
- Consider polynomial and interaction features (especially Latitude × Longitude).
- Watch for outliers in AveRooms, AveOccup, Population — consider clipping or log transforms.
- The function receives a DataFrame with 8 features + 'MedHouseVal' target.
- Must return a callable that takes a DataFrame (without 'MedHouseVal') and returns float predictions.
- Tree-based models (Random Forest, Gradient Boosting) work well out of the box.
- Target values are in $100,000s (e.g., 2.5 means $250,000).
""",
}


def _build_system_prompt(challenge) -> str:
    """Build a system prompt with full challenge context."""
    details = challenge.get_details()
    allowed = ", ".join(sorted(details.get("allowed_libraries", [])))

    # Get function contract if available, otherwise use generic description
    function_contract = challenge.config.get("function_contract", "")
    function_name = challenge.config.get("function_name", "solve")
    dataset_info = details.get("dataset", {})

    # Build dataset description
    dataset_desc = ""
    if isinstance(dataset_info, dict):
        if "columns" in dataset_info:
            dataset_desc = f"DATASET COLUMNS:\n{json.dumps(dataset_info.get('columns', []), indent=2)}"
        elif "description" in dataset_info:
            dataset_desc = f"DATASET:\n{dataset_info['description']}"

    # Build scoring description
    scoring_info = details.get("scoring", {})
    scoring_desc = ""
    if scoring_info.get("categories"):
        scoring_desc = json.dumps(scoring_info["categories"], indent=2)

    # Get challenge-specific hints
    hints = _CHALLENGE_HINTS.get(challenge.id, "")

    prompt = f"""You are an expert ML engineer solving coding challenges.
You write clean, working Python code that solves the given problem.

CHALLENGE: {details['title']}
OBJECTIVE: {details.get('objective', '')}

DESCRIPTION:
{details['description']}

{dataset_desc}

FUNCTION CONTRACT:
- You must implement: {function_name}()
{function_contract if function_contract else '- See the description above for the expected function signature.'}

SCORING:
{scoring_desc}

ALLOWED LIBRARIES (ONLY these may be imported):
{allowed}
{hints}
RULES:
1. Return ONLY valid Python code. No markdown fences, no explanations outside comments.
2. The code must define the required function with the exact signature.
3. Only import from the allowed libraries list above.
4. CRITICAL: Include ALL necessary import statements at the top of the file.
   Every module you use (re, string, numpy, pandas, sklearn, etc.) MUST be explicitly imported.
5. Include clear comments explaining your approach.
6. Aim for the accuracy target specified in the objective.
7. The function must return a callable prediction function.
8. The predict function must work on NEW unseen data."""

    return prompt


def _call_openai(messages: list, api_key: str, model: str) -> str:
    """Call OpenAI chat completions API. Returns the assistant message content."""
    import urllib.error
    import urllib.request

    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 4096,
    }
    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API error ({e.code}): {error_body}")
    except (urllib.error.URLError, TimeoutError, socket.timeout) as e:
        raise RuntimeError(f"OpenAI API connection error: {e}")
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Invalid response from OpenAI API: {e}")


def _sanitize_history(history_raw) -> list:
    """
    Convert arbitrary history payload into a safe chat history list.

    Prevents malformed client payloads from causing uncaught server errors.
    """
    if not isinstance(history_raw, list):
        return []

    allowed_roles = {"system", "user", "assistant"}
    clean_history = []
    for turn in history_raw[-10:]:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role", "user")).strip().lower()
        if role not in allowed_roles:
            role = "user"
        content = turn.get("content", "")
        if content is None:
            continue
        content = str(content).strip()
        if not content:
            continue
        clean_history.append({"role": role, "content": content})
    return clean_history


def _extract_code(text: str) -> str:
    """Extract Python code from a response that may contain markdown fences."""
    # If wrapped in ```python ... ``` or ``` ... ```, strip fences
    import re

    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


@agent_bp.route("/agent/chat", methods=["POST"])
def agent_chat():
    """
    Send a message to the AI coding agent.

    Request JSON::

        {
            "challenge_id": "mnist_digit_recognition",
            "message": "Write a CNN classifier",
            "current_code": "...",         // optional: current editor contents
            "history": [...]               // optional: prior conversation turns
        }

    Response JSON::

        {
            "code": "<generated python code>",
            "message": "<explanation from the agent>"
        }
    """
    api_key = Config.LLM_API_KEY
    if not api_key:
        return jsonify({"error": "LLM_API_KEY not configured on the server."}), 503

    data = request.get_json(silent=True) or {}
    challenge_id = data.get("challenge_id", "")
    user_message = data.get("message", "")
    current_code = data.get("current_code", "")
    history = data.get("history", [])

    # Normalize potentially non-string client payload fields.
    challenge_id = str(challenge_id).strip() if challenge_id is not None else ""
    user_message = str(user_message) if user_message is not None else ""
    current_code = str(current_code) if current_code is not None else ""

    if not challenge_id:
        return jsonify({"error": "challenge_id is required."}), 400
    if not user_message.strip():
        return jsonify({"error": "message is required."}), 400

    registry = ChallengeRegistry()
    challenge = registry.get(challenge_id)
    if challenge is None:
        return jsonify({"error": f"Challenge not found: {challenge_id}"}), 404

    # Build messages
    system_prompt = _build_system_prompt(challenge)
    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history
    for turn in _sanitize_history(history):
        messages.append(turn)

    # Add current context
    user_content = user_message
    if current_code.strip():
        user_content += f"\n\nCurrent code in the editor:\n```python\n{current_code}\n```"

    messages.append({"role": "user", "content": user_content})

    try:
        model = Config.LLM_MODEL
        raw_response = _call_openai(messages, api_key, model)
        code = _extract_code(raw_response)

        # If the response is mostly code (no prose), use it directly
        # Otherwise separate code from explanation
        if code != raw_response.strip():
            explanation = raw_response.replace(f"```python\n{code}\n```", "").strip()
            explanation = explanation.replace("```", "").strip()
        else:
            explanation = ""

        return jsonify({"code": code, "message": explanation})

    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        return jsonify({"error": f"Agent error: {e}"}), 500
