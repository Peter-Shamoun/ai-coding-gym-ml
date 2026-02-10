"""
IMDB Sentiment Analysis — execution harness & grading.
======================================================
Contains:
  - HARNESS_CODE: Python source that runs inside the sandbox subprocess.
  - ImdbSentimentChallenge: challenge class with grading rubric (100 pts).
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
Execution harness for IMDB Sentiment Analysis challenge.
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

BLOCKED_BUILTINS = {"exec", "eval", "compile", "__import__", "breakpoint"}


def validate_user_code(filepath, allowed):
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"SyntaxError at line {e.lineno}: {e.msg}"]

    allowed_set = set(allowed)
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in allowed_set:
                    violations.append(
                        f"Line {node.lineno}: 'import {alias.name}' -- "
                        f"'{top}' is not an allowed library."
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top not in allowed_set:
                    violations.append(
                        f"Line {node.lineno}: 'from {node.module} import ...' -- "
                        f"'{top}' is not an allowed library."
                    )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in BLOCKED_BUILTINS:
                violations.append(
                    f"Line {node.lineno}: '{node.func.id}()' is not allowed."
                )
        if isinstance(node, ast.Attribute):
            if node.attr in ("__import__", "__builtins__", "__subclasses__"):
                violations.append(
                    f"Line {node.lineno}: Access to '{node.attr}' is not allowed."
                )
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in ("__builtins__", "__import__", "__subclasses__",
                              "__globals__", "__code__", "__reduce__"):
                violations.append(
                    f"Line {node.lineno}: String reference to '{node.value}' is not allowed."
                )
    return violations


def _write_and_exit(result, output_path):
    with open(output_path, "w") as f:
        json.dump(result, f)
    sys.exit(0)


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

    # Validate user code
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

    try:
        import pandas as pd
        from sklearn.metrics import accuracy_score

        train_df = pd.read_csv(train_path)

        from solution import train_sentiment_classifier

        start = time.time()
        predict_fn = train_sentiment_classifier(train_df)
        train_time = time.time() - start

        if predict_fn is None:
            raise ValueError(
                "train_sentiment_classifier() returned None. "
                "It must return a callable prediction function."
            )
        if not callable(predict_fn):
            raise TypeError(
                f"train_sentiment_classifier() returned {type(predict_fn).__name__}, "
                "expected a callable prediction function."
            )

        if mode == "run":
            sample = train_df.sample(min(200, len(train_df)), random_state=42)
            df_input = sample.drop('sentiment', axis=1)
            preds = predict_fn(df_input)
            preds = [int(p) for p in preds]

            acc = accuracy_score(sample['sentiment'].values, preds)
            result["sample_accuracy"] = round(acc, 4)
            result["num_predictions"] = len(preds)
            result["train_time"] = round(train_time, 2)

        elif mode == "submit":
            test_df = pd.read_csv(test_path)
            df_input = test_df.drop('sentiment', axis=1)
            preds = predict_fn(df_input)
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
#  Code-quality patterns
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_TEXT_CLEANING_PATTERNS = [
    r"\blower\s*\(",
    r"\bre\.sub\b",
    r"\bstemm",
    r"\blemmat",
    r"\bstopwords\b",
    r"\bpreprocess",
    r"\btokeniz",
    r"\bclean[_\s]?text\b",
    r"\bBeautifulSoup\b",
    r"\bstrip\b.*html",
    r"\bhtml",
    r"<br\s*/?>",
]

_NEGATION_PATTERNS = [
    r"\bnegat",
    r"\bnot_",
    r"\bn't",
    r"\bneg_",
    r"\bhandle.*negat",
    r"\bnegation_scope",
]

_ADVANCED_PATTERNS = [
    r"\bWord2Vec\b",
    r"\bword2vec\b",
    r"\bGloVe\b",
    r"\bglove\b",
    r"\bembedding\b",
    r"\bEmbedding\b",
    r"\bLSTM\b",
    r"\blstm\b",
    r"\bGRU\b",
    r"\bTransformer\b",
    r"\bBERT\b",
    r"\bbert\b",
    r"\bAutoModel\b",
    r"\bAutoTokenizer\b",
    r"\bngram_range\b",
    r"\bsublinear_tf\b",
]


def _count_hits(source: str, patterns: list) -> int:
    return sum(1 for p in patterns if re.search(p, source, re.IGNORECASE))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Challenge class
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ImdbSentimentChallenge(BaseChallenge):
    """IMDB Sentiment Analysis — binary text classification challenge."""

    def get_harness_code(self, mode: str) -> str:
        allowed = self.config.get("allowed_libraries", [])
        return HARNESS_CODE.replace("__ALLOWED_PLACEHOLDER__", repr(allowed))

    def get_data_files(self, mode: str) -> dict:
        train_src = os.path.join(
            self.challenge_dir, self.config["dataset"]["train_file"]
        )
        return {"imdb_train.csv": train_src}

    def get_harness_args(self, mode: str) -> list:
        test_src = os.path.join(
            self.challenge_dir, self.config["dataset"]["test_file"]
        )
        return [mode, "imdb_train.csv", test_src]

    # ── Grading (100 points) ────────────────────────────────

    def grade(self, user_code: str, execution_output: dict) -> dict:
        categories = []
        total = 0

        # 1 — Execution (20 pts)
        s, fb = self._grade_execution(execution_output)
        categories.append(self._cat("Execution", s, 20, fb))
        total += s

        predictions = execution_output.get("predictions", [])

        # 2 — Accuracy (50 pts)
        s, accuracy, fb, y_true, y_pred = self._grade_accuracy(predictions)
        cat = self._cat("Accuracy", s, 50, fb)
        cat["accuracy"] = accuracy
        categories.append(cat)
        total += s

        # 3 — Precision & Recall (20 pts)
        s, prec, rec, fb = self._grade_precision_recall(y_true, y_pred)
        cat = self._cat("Precision & Recall", s, 20, fb)
        cat["precision"] = prec
        cat["recall"] = rec
        categories.append(cat)
        total += s

        # 4 — Code Quality (10 pts)
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
        }

    # ── helpers ──────────────────────────────────────────────

    @staticmethod
    def _cat(name, score, max_score, feedback):
        return {"name": name, "score": score, "max_score": max_score, "feedback": feedback}

    def _load_test_df(self):
        path = os.path.join(self.challenge_dir, self.config["dataset"]["test_file"])
        return pd.read_csv(path)

    def _grade_execution(self, output):
        score = 0
        fb = []
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

    def _grade_accuracy(self, predictions):
        fb = []
        if not predictions:
            fb.append("No predictions to evaluate.")
            return 0, None, fb, None, None
        test_df = self._load_test_df()
        y_true = test_df["sentiment"].values
        if len(predictions) != len(y_true):
            fb.append(f"Length mismatch: {len(predictions)} vs {len(y_true)}.")
            return 0, None, fb, None, None
        y_pred = np.array(predictions, dtype=int)
        accuracy = float(accuracy_score(y_true, y_pred))
        pct = accuracy * 100
        if accuracy >= 0.91:
            score = 50
        elif accuracy >= 0.89:
            score = 40
        elif accuracy >= 0.87:
            score = 30
        elif accuracy >= 0.85:
            score = 20
        elif accuracy >= 0.82:
            score = 10
        else:
            score = 0
        fb.append(f"Accuracy: {pct:.2f}% -> {score}/50 pts")
        fb.append(f"Correct: {int(accuracy * len(y_true))}/{len(y_true)}")
        return score, round(accuracy, 4), fb, y_true, y_pred

    def _grade_precision_recall(self, y_true, y_pred):
        fb = []
        if y_true is None or y_pred is None:
            fb.append("Skipped — no valid predictions.")
            return 0, None, None, fb
        prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
        rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
        score = 0
        if prec > 0.85 and rec > 0.85:
            score = 20
            fb.append(f"Both precision ({prec:.4f}) and recall ({rec:.4f}) > 0.85 -> 20/20 pts")
        elif prec > 0.80 and rec > 0.80:
            score = 15
            fb.append(f"Both precision ({prec:.4f}) and recall ({rec:.4f}) > 0.80 -> 15/20 pts")
        else:
            score = 5
            fb.append(f"Precision: {prec:.4f}, Recall: {rec:.4f} -> 5/20 pts")
        return score, round(prec, 4), round(rec, 4), fb

    def _grade_code_quality(self, source):
        score = 0
        fb = []
        try:
            ast.parse(source)
        except SyntaxError as e:
            fb.append(f"SyntaxError at line {e.lineno}: {e.msg}")
        # (a) Text cleaning — 4 pts
        n = _count_hits(source, _TEXT_CLEANING_PATTERNS)
        if n:
            score += 4
            fb.append(f"Text cleaning detected ({n} indicator(s)) -> 4/4 pts")
        else:
            fb.append("No text cleaning detected -> 0/4 pts")
        # (b) Negation handling — 3 pts
        n = _count_hits(source, _NEGATION_PATTERNS)
        if n:
            score += 3
            fb.append(f"Negation handling detected ({n} indicator(s)) -> 3/3 pts")
        else:
            fb.append("No negation handling detected -> 0/3 pts")
        # (c) Advanced techniques — 3 pts
        n = _count_hits(source, _ADVANCED_PATTERNS)
        if n:
            score += 3
            fb.append(f"Advanced techniques detected ({n} indicator(s)) -> 3/3 pts")
        else:
            fb.append("No advanced techniques detected -> 0/3 pts")
        return score, fb
