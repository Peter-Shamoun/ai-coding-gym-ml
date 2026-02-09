"""
Automated Grading Script for Email Spam Detection Submissions
==============================================================
Usage:
    python grade_submission.py <submission_dir> [--test-data <path>]

Example:
    python grade_submission.py ./my_submission --test-data spam_test.csv

Expected submission structure:
    submission_dir/
        classifier.py        # Training/prediction code
        model.pkl (or model/) # Saved model artifact
        predictions.csv       # Predictions on test set (columns: prediction or label_num)
"""

import argparse
import ast
import json
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)


# ──────────────────────────────────────────────────────────────
#  Feedback helper
# ──────────────────────────────────────────────────────────────
class Feedback:
    """Collects per-section messages and scores for a readable report."""

    def __init__(self):
        self.sections = []

    def add(self, section: str, points: int, max_pts: int, message: str):
        self.sections.append(
            {"section": section, "points": points, "max": max_pts, "message": message}
        )

    def report(self) -> str:
        lines = [
            "",
            "=" * 62,
            "  EMAIL SPAM DETECTION -- GRADING REPORT",
            "=" * 62,
        ]
        total, total_max = 0, 0
        for s in self.sections:
            total += s["points"]
            total_max += s["max"]
            status = "PASS" if s["points"] == s["max"] else (
                "PARTIAL" if s["points"] > 0 else "FAIL"
            )
            lines.append(f"\n[{status}] {s['section']}  --  {s['points']}/{s['max']} pts")
            for msg_line in s["message"].strip().split("\n"):
                lines.append(f"       {msg_line}")
        lines.append("")
        lines.append("-" * 62)
        passed = total >= 80
        lines.append(f"  TOTAL SCORE : {total}/{total_max}")
        lines.append(f"  STATUS      : {'PASSED' if passed else 'NOT PASSED'}  (threshold: 80)")
        lines.append("-" * 62)
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
#  1. VALIDATION  (20 points)
# ──────────────────────────────────────────────────────────────
def grade_validation(submission_dir: str, feedback: Feedback):
    """Check that required files exist and model loads."""
    score = 0
    messages = []

    # --- classifier.py (5 pts) ---
    classifier_path = os.path.join(submission_dir, "classifier.py")
    has_classifier = os.path.isfile(classifier_path)
    if has_classifier:
        score += 5
        messages.append("classifier.py found.")
    else:
        messages.append("MISSING classifier.py -- expected in submission root.")

    # --- model.pkl or model/ directory (10 pts) ---
    model_pkl = os.path.join(submission_dir, "model.pkl")
    model_dir = os.path.join(submission_dir, "model")
    model_loaded = False
    model_path_used = None

    if os.path.isfile(model_pkl):
        model_path_used = model_pkl
        try:
            with open(model_pkl, "rb") as f:
                _ = pickle.load(f)
            model_loaded = True
            score += 10
            messages.append("model.pkl loaded successfully.")
        except Exception as exc:
            messages.append(f"model.pkl found but FAILED to load: {exc}")
    elif os.path.isdir(model_dir):
        model_path_used = model_dir
        # Accept directory-based models (e.g. joblib, tf saved model)
        contents = os.listdir(model_dir)
        if contents:
            model_loaded = True
            score += 10
            messages.append(f"model/ directory found with {len(contents)} file(s).")
        else:
            messages.append("model/ directory is empty.")
    else:
        messages.append("MISSING model artifact -- expected model.pkl or model/ directory.")

    # --- predictions.csv (5 pts) ---
    predictions_path = os.path.join(submission_dir, "predictions.csv")
    has_predictions = os.path.isfile(predictions_path)
    if has_predictions:
        score += 5
        messages.append("predictions.csv found.")
    else:
        messages.append("MISSING predictions.csv -- needed for accuracy grading.")

    feedback.add("1. VALIDATION", score, 20, "\n".join(messages))
    return score, has_classifier, has_predictions, classifier_path, predictions_path


# ──────────────────────────────────────────────────────────────
#  2. ACCURACY  (50 points)
# ──────────────────────────────────────────────────────────────
def grade_accuracy(predictions_path: str, test_data_path: str, feedback: Feedback):
    """Compare predictions.csv against test ground truth."""
    score = 0
    messages = []
    accuracy = None
    y_pred = None
    y_true = None

    try:
        test_df = pd.read_csv(test_data_path)
    except Exception as exc:
        messages.append(f"ERROR loading test data ({test_data_path}): {exc}")
        feedback.add("2. ACCURACY", 0, 50, "\n".join(messages))
        return 0, None, None, None

    try:
        pred_df = pd.read_csv(predictions_path)
    except Exception as exc:
        messages.append(f"ERROR loading predictions.csv: {exc}")
        feedback.add("2. ACCURACY", 0, 50, "\n".join(messages))
        return 0, None, None, None

    # Resolve prediction column
    pred_col = None
    for candidate in ["prediction", "label_num", "predicted", "pred", "spam", "label"]:
        if candidate in pred_df.columns:
            pred_col = candidate
            break
    if pred_col is None:
        # Fall back to first numeric-looking column
        for col in pred_df.columns:
            if pred_df[col].dtype in (np.int64, np.float64, int, float):
                pred_col = col
                break
    if pred_col is None:
        messages.append(
            "Could not identify prediction column in predictions.csv. "
            "Expected one of: prediction, label_num, predicted, pred."
        )
        feedback.add("2. ACCURACY", 0, 50, "\n".join(messages))
        return 0, None, None, None

    y_pred_raw = pred_df[pred_col].values

    # Handle string labels ('spam'/'ham') -> numeric
    if y_pred_raw.dtype == object:
        mapping = {"spam": 1, "ham": 0}
        try:
            y_pred = np.array([mapping[str(v).strip().lower()] for v in y_pred_raw])
        except KeyError:
            messages.append(
                "predictions.csv contains unrecognised string labels. "
                "Expected 'spam'/'ham' or 0/1."
            )
            feedback.add("2. ACCURACY", 0, 50, "\n".join(messages))
            return 0, None, None, None
    else:
        y_pred = y_pred_raw.astype(int)

    y_true = test_df["label_num"].values

    # Length check
    if len(y_pred) != len(y_true):
        messages.append(
            f"Row count mismatch: predictions.csv has {len(y_pred)} rows, "
            f"test set has {len(y_true)} rows."
        )
        feedback.add("2. ACCURACY", 0, 50, "\n".join(messages))
        return 0, None, y_pred, y_true

    accuracy = accuracy_score(y_true, y_pred)
    pct = accuracy * 100

    # Tiered scoring
    if accuracy >= 0.95:
        score = 50
    elif accuracy >= 0.93:
        score = 40
    elif accuracy >= 0.91:
        score = 30
    elif accuracy >= 0.89:
        score = 20
    elif accuracy >= 0.85:
        score = 10
    else:
        score = 0

    messages.append(f"Accuracy: {pct:.2f}%  ->  {score}/50 pts")
    messages.append(f"  Thresholds:  >=95%->50  >=93%->40  >=91%->30  >=89%->20  >=85%->10")
    messages.append(f"  Correct: {int(accuracy * len(y_true))}/{len(y_true)}")

    feedback.add("2. ACCURACY", score, 50, "\n".join(messages))
    return score, accuracy, y_pred, y_true


# ──────────────────────────────────────────────────────────────
#  3. PRECISION / RECALL  (20 points)
# ──────────────────────────────────────────────────────────────
def grade_precision_recall(y_true, y_pred, feedback: Feedback):
    """Evaluate spam-class precision and recall."""
    score = 0
    messages = []

    if y_true is None or y_pred is None:
        messages.append("Skipped -- predictions unavailable.")
        feedback.add("3. PRECISION / RECALL", 0, 20, "\n".join(messages))
        return 0, None, None

    # spam = positive class (label 1)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)

    if precision > 0.90:
        score += 10
        messages.append(f"Spam precision: {precision:.4f}  (>0.90 OK)  ->  10/10 pts")
    else:
        messages.append(f"Spam precision: {precision:.4f}  (<=0.90)    ->   0/10 pts")

    if recall > 0.90:
        score += 10
        messages.append(f"Spam recall:    {recall:.4f}  (>0.90 OK)  ->  10/10 pts")
    else:
        messages.append(f"Spam recall:    {recall:.4f}  (<=0.90)    ->   0/10 pts")

    # Extra context -- confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    messages.append(f"\nConfusion matrix (ham=0, spam=1):\n{cm}")

    # False-positive note (ham mistakenly flagged as spam)
    if cm.shape == (2, 2):
        fp = cm[0, 1]
        total_ham = cm[0].sum()
        fpr = fp / total_ham if total_ham else 0
        messages.append(f"False positive rate (ham->spam): {fpr:.4f}  ({fp}/{total_ham})")

    feedback.add("3. PRECISION / RECALL", score, 20, "\n".join(messages))
    return score, precision, recall


# ──────────────────────────────────────────────────────────────
#  4. CODE QUALITY  (10 points)
# ──────────────────────────────────────────────────────────────

# Patterns that indicate good practices
_PREPROCESSING_PATTERNS = [
    r"\bclean[_\s]?text\b",
    r"\blower\s*\(",
    r"\bre\.sub\b",
    r"\bstemm",
    r"\blemmat",
    r"\bstopwords\b",
    r"\btranslate\b",
    r"\bstrip\b.*html",
    r"\bBeautifulSoup\b",
    r"\bpreprocess",
    r"\btokeniz",
]

_IMBALANCE_PATTERNS = [
    r"\bclass_weight\b",
    r"\bbalanced\b",
    r"\bSMOTE\b",
    r"\bover_?sampl",
    r"\bunder_?sampl",
    r"\bresample\b",
    r"\bclass.?imbalance\b",
    r"\bsample_weight\b",
    r"\bRandomOverSampler\b",
    r"\bRandomUnderSampler\b",
]

_FEATURE_ENG_PATTERNS = [
    r"\bngram_range\b",
    r"\bGridSearchCV\b",
    r"\bRandomizedSearchCV\b",
    r"\bcross_val",
    r"\bfeature[_\s]?engineer",
    r"\bhstack\b",
    r"\bsublinear_tf\b",
    r"\bVotingClassifier\b",
    r"\bStackingClassifier\b",
    r"\bPipeline\b",
    r"\bparam_grid\b",
    r"\bbigram|trigram\b",
    r"\bensemble\b",
    r"\bmax_features\b",
    r"\bword2vec|Word2Vec\b",
    r"\bCountVectorizer\b.*\bTfidfTransformer\b",
    r"\bhyperparameter|hyper.?param\b",
    r"\bmin_df\b",
    r"\bmax_df\b",
]


def _scan_code(source: str, patterns: list[str]) -> list[str]:
    """Return list of matched pattern descriptions."""
    hits = []
    for pat in patterns:
        if re.search(pat, source, re.IGNORECASE):
            hits.append(pat)
    return hits


def grade_code_quality(classifier_path: str, feedback: Feedback):
    """Static analysis of classifier.py for best practices."""
    score = 0
    messages = []

    try:
        with open(classifier_path, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
    except Exception as exc:
        messages.append(f"Could not read classifier.py: {exc}")
        feedback.add("4. CODE QUALITY", 0, 10, "\n".join(messages))
        return 0

    # Check AST parses (basic sanity)
    try:
        ast.parse(source)
    except SyntaxError as exc:
        messages.append(f"WARNING: classifier.py has a SyntaxError at line {exc.lineno}.")

    # (a) Text preprocessing -- 3 pts
    prep_hits = _scan_code(source, _PREPROCESSING_PATTERNS)
    if prep_hits:
        score += 3
        messages.append(f"Text preprocessing detected ({len(prep_hits)} indicator(s)).  ->  3/3 pts")
    else:
        messages.append("No text preprocessing detected.  ->  0/3 pts")
        messages.append("  Tip: Add lowercasing, regex cleaning, stop-word removal, etc.")

    # (b) Class imbalance handling -- 3 pts
    imb_hits = _scan_code(source, _IMBALANCE_PATTERNS)
    if imb_hits:
        score += 3
        messages.append(f"Class imbalance handling detected ({len(imb_hits)} indicator(s)).  ->  3/3 pts")
    else:
        messages.append("No class imbalance handling detected.  ->  0/3 pts")
        messages.append("  Tip: Use class_weight='balanced' or SMOTE/resampling.")

    # (c) Feature engineering / tuning -- 4 pts
    feat_hits = _scan_code(source, _FEATURE_ENG_PATTERNS)
    if len(feat_hits) >= 2:
        score += 4
        messages.append(f"Feature engineering/tuning detected ({len(feat_hits)} indicator(s)).  ->  4/4 pts")
    elif len(feat_hits) == 1:
        score += 2
        messages.append(f"Some feature engineering detected (1 indicator).  ->  2/4 pts")
        messages.append("  Tip: Add n-gram ranges, hyperparameter tuning, or ensembling.")
    else:
        messages.append("No feature engineering or tuning detected.  ->  0/4 pts")
        messages.append("  Tip: Try ngram_range, GridSearchCV, ensembles, sublinear_tf, etc.")

    # Code length note (informational)
    loc = len([l for l in source.splitlines() if l.strip() and not l.strip().startswith("#")])
    messages.append(f"\nCode: {loc} non-comment lines of Python.")

    feedback.add("4. CODE QUALITY", score, 10, "\n".join(messages))
    return score


# ──────────────────────────────────────────────────────────────
#  Main grading entry point
# ──────────────────────────────────────────────────────────────
def grade_submission(submission_dir: str, test_data_path: str) -> dict:
    """
    Grade an email spam detection submission.

    Parameters
    ----------
    submission_dir : str
        Directory containing classifier.py, model.pkl / model/, predictions.csv.
    test_data_path : str
        Path to the test CSV with columns: text, label, label_num.

    Returns
    -------
    dict with total_score, accuracy, precision_spam, recall_spam, passed, feedback.
    """
    feedback = Feedback()
    total_score = 0

    # ── 1. Validation ────────────────────────────────────────
    val_score, has_classifier, has_predictions, classifier_path, predictions_path = (
        grade_validation(submission_dir, feedback)
    )
    total_score += val_score

    # ── 2. Accuracy ──────────────────────────────────────────
    accuracy, y_pred, y_true = None, None, None
    if has_predictions:
        acc_score, accuracy, y_pred, y_true = grade_accuracy(
            predictions_path, test_data_path, feedback
        )
        total_score += acc_score
    else:
        feedback.add(
            "2. ACCURACY", 0, 50,
            "Skipped -- predictions.csv not found."
        )

    # ── 3. Precision / Recall ────────────────────────────────
    pr_score, precision, recall = grade_precision_recall(y_true, y_pred, feedback)
    total_score += pr_score

    # ── 4. Code Quality ──────────────────────────────────────
    if has_classifier:
        cq_score = grade_code_quality(classifier_path, feedback)
        total_score += cq_score
    else:
        feedback.add(
            "4. CODE QUALITY", 0, 10,
            "Skipped -- classifier.py not found."
        )

    # ── Report ───────────────────────────────────────────────
    result = {
        "total_score": total_score,
        "max_score": 100,
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "precision_spam": round(precision, 4) if precision is not None else None,
        "recall_spam": round(recall, 4) if recall is not None else None,
        "passed": total_score >= 80,
    }

    report = feedback.report()
    print(report)

    # Also dump JSON for programmatic consumers
    result["feedback"] = report
    return result


# ──────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Grade an Email Spam Detection submission."
    )
    parser.add_argument(
        "submission_dir",
        help="Path to submission directory containing classifier.py, model.pkl, predictions.csv",
    )
    parser.add_argument(
        "--test-data",
        default="spam_test.csv",
        help="Path to test CSV (default: spam_test.csv)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also write results to grade_result.json",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.submission_dir):
        print(f"ERROR: '{args.submission_dir}' is not a directory.")
        sys.exit(1)
    if not os.path.isfile(args.test_data):
        print(f"ERROR: Test data '{args.test_data}' not found.")
        sys.exit(1)

    result = grade_submission(args.submission_dir, args.test_data)

    if args.json:
        out_path = os.path.join(args.submission_dir, "grade_result.json")
        with open(out_path, "w") as f:
            json.dump({k: v for k, v in result.items() if k != "feedback"}, f, indent=2)
        print(f"\nJSON results saved to {out_path}")

    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
