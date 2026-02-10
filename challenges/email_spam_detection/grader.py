"""
Email Spam Detection — execution harness & grading.
====================================================
Contains:
  - HARNESS_CODE: Python source that runs inside the sandbox subprocess.
  - EmailSpamChallenge: challenge class with grading rubric (100 pts).
"""

import ast
import json as _json
import os
import re
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

from challenges.base import BaseChallenge


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Harness code  (runs in sandbox subprocess)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HARNESS_CODE = r'''
"""
Execution harness for Email Spam Detection challenge.
Runs in an isolated subprocess.

Usage:  python _harness.py <mode> <train_path> <test_path> <output_path>
  mode: 'run' or 'submit'
"""
import sys
import json
import traceback
import time
import ast
import os


# ── Injected by get_harness_code() at runtime ────────────────
ALLOWED_LIBRARIES = __ALLOWED_PLACEHOLDER__

# Dangerous built-in calls that are never allowed in user code
BLOCKED_BUILTINS = {"exec", "eval", "compile", "__import__", "breakpoint"}


# ── Import / security validator ──────────────────────────────

def validate_user_code(filepath, allowed):
    """
    Static analysis of user code before execution.
    Checks:
      1. All import statements use only allowed top-level packages.
      2. No calls to dangerous built-in functions.
      3. No direct attribute access to __import__ / __builtins__.
    Returns a list of violation strings (empty = clean).
    """
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"SyntaxError at line {e.lineno}: {e.msg}"]

    allowed_set = set(allowed)
    violations = []

    for node in ast.walk(tree):
        # ── import X / import X.Y ────────────────────────────
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in allowed_set:
                    violations.append(
                        f"Line {node.lineno}: 'import {alias.name}' -- "
                        f"'{top}' is not an allowed library."
                    )

        # ── from X import Y ──────────────────────────────────
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top not in allowed_set:
                    violations.append(
                        f"Line {node.lineno}: 'from {node.module} import ...' -- "
                        f"'{top}' is not an allowed library."
                    )

        # ── Blocked built-in calls: exec(), eval(), etc. ─────
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in BLOCKED_BUILTINS:
                violations.append(
                    f"Line {node.lineno}: '{node.func.id}()' is not allowed "
                    f"for security reasons."
                )

        # ── Attribute access to __import__ / __builtins__ ────
        if isinstance(node, ast.Attribute):
            if node.attr in ("__import__", "__builtins__", "__subclasses__"):
                violations.append(
                    f"Line {node.lineno}: Access to '{node.attr}' is not "
                    f"allowed for security reasons."
                )

        # ── String-based access to dangerous names ───────────
        # Catches: globals()["__builtins__"], getattr(x, "__import__")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in ("__builtins__", "__import__", "__subclasses__",
                              "__globals__", "__code__", "__reduce__"):
                violations.append(
                    f"Line {node.lineno}: String reference to '{node.value}' "
                    f"is not allowed for security reasons."
                )

    return violations


def _write_and_exit(result, output_path):
    """Write result JSON and terminate."""
    with open(output_path, "w") as f:
        json.dump(result, f)
    sys.exit(0)


# ── Main harness ─────────────────────────────────────────────

def main():
    mode = sys.argv[1]
    train_path = sys.argv[2]
    test_path = sys.argv[3]
    output_path = sys.argv[4]

    result = {
        "success": False,
        "predictions": [],
        "num_predictions": 0,
        "error": None,
        "traceback_str": None,
    }

    # ── Step 1: Validate user code BEFORE running it ─────────
    solution_path = os.path.join(os.path.dirname(__file__), "solution.py")
    violations = validate_user_code(solution_path, ALLOWED_LIBRARIES)
    if violations:
        result["error"] = "Import validation failed"
        result["traceback_str"] = (
            "Your code uses libraries or functions that are not allowed.\n\n"
            + "\n".join(f"  - {v}" for v in violations)
            + "\n\nAllowed libraries:\n  "
            + ", ".join(sorted(ALLOWED_LIBRARIES))
        )
        _write_and_exit(result, output_path)
        return

    # ── Step 2: Execute user code ────────────────────────────
    try:
        import pandas as pd
        from sklearn.metrics import accuracy_score

        # Load training data
        train_df = pd.read_csv(train_path)

        # Import user solution (now safe — imports validated above)
        from solution import train_spam_detector

        # Train
        start = time.time()
        predict_fn = train_spam_detector(train_df)
        train_time = time.time() - start

        # Validate return value
        if predict_fn is None:
            raise ValueError(
                "train_spam_detector() returned None. "
                "It must return a callable prediction function."
            )
        if not callable(predict_fn):
            raise TypeError(
                f"train_spam_detector() returned {type(predict_fn).__name__}, "
                "expected a callable prediction function."
            )

        # ── Mode: run (quick test on training sample) ────────
        if mode == "run":
            sample = train_df.sample(min(200, len(train_df)), random_state=42)
            preds = predict_fn(sample["text"].tolist())
            preds = [int(p) for p in preds]

            acc = accuracy_score(sample["label_num"].values, preds)
            result["sample_accuracy"] = round(acc, 4)
            result["num_predictions"] = len(preds)
            result["train_time"] = round(train_time, 2)

        # ── Mode: submit (full evaluation on hidden test set)─
        elif mode == "submit":
            test_df = pd.read_csv(test_path)
            preds = predict_fn(test_df["text"].tolist())
            preds = [int(p) for p in preds]
            result["predictions"] = preds
            result["num_predictions"] = len(preds)
            result["train_time"] = round(train_time, 2)

        result["success"] = True

    except Exception as e:
        result["error"] = str(e)
        result["traceback_str"] = traceback.format_exc()

    with open(output_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
'''


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Code-quality patterns (static analysis)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
    r"\bhyperparameter|hyper.?param\b",
    r"\bmin_df\b",
    r"\bmax_df\b",
]


def _count_hits(source: str, patterns: list) -> int:
    return sum(1 for p in patterns if re.search(p, source, re.IGNORECASE))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Challenge class
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EmailSpamChallenge(BaseChallenge):
    """Email Spam Detection — binary classification challenge."""

    # ── Execution plumbing ───────────────────────────────────

    def get_harness_code(self, mode: str) -> str:
        """Inject the allowed-libraries list into the harness template."""
        allowed = self.config.get("allowed_libraries", [])
        return HARNESS_CODE.replace(
            "__ALLOWED_PLACEHOLDER__", repr(allowed)
        )

    def get_data_files(self, mode: str) -> dict:
        """Training data is copied into the sandbox as ``enron_spam.csv``."""
        train_src = os.path.join(
            self.base_dir, self.config["dataset"]["train_file"]
        )
        return {"enron_spam.csv": train_src}

    def get_harness_args(self, mode: str) -> list:
        """
        CLI args for the harness: ``mode  train_path  test_path``.
        ``output_path`` is appended by the sandbox automatically.
        """
        test_src = os.path.join(
            self.base_dir, self.config["dataset"]["test_file"]
        )
        # train_path is relative (inside sandbox); test_path is absolute.
        return [mode, "enron_spam.csv", test_src]

    # ── Grading (100 points) ────────────────────────────────

    def grade(self, user_code: str, execution_output: dict) -> dict:
        categories = []
        total = 0

        # 1 ─ Execution (20 pts)
        s, fb = self._grade_execution(execution_output)
        categories.append(self._cat("Execution", s, 20, fb))
        total += s

        predictions = execution_output.get("predictions", [])

        # 2 ─ Accuracy (50 pts)
        s, accuracy, fb, y_true, y_pred = self._grade_accuracy(predictions)
        cat = self._cat("Accuracy", s, 50, fb)
        cat["accuracy"] = accuracy
        categories.append(cat)
        total += s

        # 3 ─ Precision & Recall (20 pts)
        s, prec, rec, fb = self._grade_precision_recall(y_true, y_pred)
        cat = self._cat("Precision & Recall", s, 20, fb)
        cat["precision"] = prec
        cat["recall"] = rec
        categories.append(cat)
        total += s

        # 4 ─ Code Quality (10 pts)
        s, fb = self._grade_code_quality(user_code)
        categories.append(self._cat("Code Quality", s, 10, fb))
        total += s

        passed = total >= self.config["scoring"]["pass_threshold"]

        return {
            "total_score": total,
            "max_score": self.config["scoring"]["max_score"],
            "passed": passed,
            "categories": categories,
            "accuracy": accuracy,
            "precision_spam": prec,
            "recall_spam": rec,
        }

    # ── helpers ──────────────────────────────────────────────

    @staticmethod
    def _cat(name: str, score: int, max_score: int, feedback: list) -> dict:
        return {"name": name, "score": score, "max_score": max_score, "feedback": feedback}

    def _load_test_df(self) -> pd.DataFrame:
        path = os.path.join(self.base_dir, self.config["dataset"]["test_file"])
        return pd.read_csv(path)

    # ·· 1. Execution ·········································

    def _grade_execution(self, output: dict) -> Tuple[int, list]:
        score = 0
        fb: list = []

        if not output.get("success"):
            fb.append(f"Execution failed: {output.get('error', 'unknown')}")
            return score, fb

        score += 10
        fb.append("Code executed successfully.")

        n = output.get("num_predictions", 0)
        expected = len(self._load_test_df())
        if n == expected:
            score += 10
            fb.append(f"Prediction count matches test set ({n}).")
        elif n > 0:
            score += 5
            fb.append(f"Prediction count mismatch: got {n}, expected {expected}.")
        else:
            fb.append("No predictions produced.")

        return score, fb

    # ·· 2. Accuracy ··········································

    def _grade_accuracy(self, predictions: list) -> Tuple[int, Optional[float], list, Any, Any]:
        fb: list = []
        if not predictions:
            fb.append("No predictions to evaluate.")
            return 0, None, fb, None, None

        test_df = self._load_test_df()
        y_true = test_df["label_num"].values

        if len(predictions) != len(y_true):
            fb.append(
                f"Length mismatch: {len(predictions)} predictions "
                f"vs {len(y_true)} test samples."
            )
            return 0, None, fb, None, None

        y_pred = np.array(predictions, dtype=int)
        accuracy = float(accuracy_score(y_true, y_pred))
        pct = accuracy * 100

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

        fb.append(f"Accuracy: {pct:.2f}% -> {score}/50 pts")
        fb.append(f"Correct: {int(accuracy * len(y_true))}/{len(y_true)}")
        return score, round(accuracy, 4), fb, y_true, y_pred

    # ·· 3. Precision & Recall ································

    def _grade_precision_recall(
        self, y_true, y_pred
    ) -> Tuple[int, Optional[float], Optional[float], list]:
        fb: list = []
        if y_true is None or y_pred is None:
            fb.append("Skipped -- no valid predictions.")
            return 0, None, None, fb

        prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
        rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
        score = 0

        if prec > 0.90:
            score += 10
            fb.append(f"Spam precision: {prec:.4f} (>0.90) -> 10/10 pts")
        else:
            fb.append(f"Spam precision: {prec:.4f} (<=0.90) ->  0/10 pts")

        if rec > 0.90:
            score += 10
            fb.append(f"Spam recall:    {rec:.4f} (>0.90) -> 10/10 pts")
        else:
            fb.append(f"Spam recall:    {rec:.4f} (<=0.90) ->  0/10 pts")

        return score, round(prec, 4), round(rec, 4), fb

    # ·· 4. Code Quality ······································

    def _grade_code_quality(self, source: str) -> Tuple[int, list]:
        score = 0
        fb: list = []

        try:
            ast.parse(source)
        except SyntaxError as e:
            fb.append(f"SyntaxError at line {e.lineno}: {e.msg}")

        # (a) Preprocessing — 3 pts
        n = _count_hits(source, _PREPROCESSING_PATTERNS)
        if n:
            score += 3
            fb.append(f"Text preprocessing detected ({n} indicator(s)) -> 3/3 pts")
        else:
            fb.append("No text preprocessing detected -> 0/3 pts")

        # (b) Imbalance handling — 3 pts
        n = _count_hits(source, _IMBALANCE_PATTERNS)
        if n:
            score += 3
            fb.append(f"Class imbalance handling detected ({n} indicator(s)) -> 3/3 pts")
        else:
            fb.append("No class imbalance handling detected -> 0/3 pts")

        # (c) Feature engineering — 4 pts
        n = _count_hits(source, _FEATURE_ENG_PATTERNS)
        if n >= 2:
            score += 4
            fb.append(f"Feature engineering detected ({n} indicator(s)) -> 4/4 pts")
        elif n == 1:
            score += 2
            fb.append(f"Some feature engineering detected (1 indicator) -> 2/4 pts")
        else:
            fb.append("No feature engineering detected -> 0/4 pts")

        return score, fb
