"""
Customer Churn Prediction — execution harness & grading.
========================================================
Contains:
  - HARNESS_CODE: Python source that runs inside the sandbox subprocess.
  - CustomerChurnChallenge: challenge class with grading rubric (100 pts).

Key difference: This challenge emphasises F1 on the minority (churn) class,
not just overall accuracy.
"""

import ast
import json as _json
import os
import re
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from challenges.base import BaseChallenge


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Harness code  (runs in sandbox subprocess)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HARNESS_CODE = r'''
"""
Execution harness for Customer Churn Prediction challenge.
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

        from solution import train_churn_predictor

        start = time.time()
        predict_fn = train_churn_predictor(train_df)
        train_time = time.time() - start

        if predict_fn is None:
            raise ValueError(
                "train_churn_predictor() returned None. "
                "It must return a callable prediction function."
            )
        if not callable(predict_fn):
            raise TypeError(
                f"train_churn_predictor() returned {type(predict_fn).__name__}, "
                "expected a callable prediction function."
            )

        if mode == "run":
            sample = train_df.sample(min(200, len(train_df)), random_state=42)
            y_sample = (sample['Churn'] == 'Yes').astype(int).values
            df_input = sample.drop('Churn', axis=1)
            preds = predict_fn(df_input)
            preds = [int(p) for p in preds]

            acc = accuracy_score(y_sample, preds)
            result["sample_accuracy"] = round(acc, 4)
            result["num_predictions"] = len(preds)
            result["train_time"] = round(train_time, 2)

        elif mode == "submit":
            test_df = pd.read_csv(test_path)
            df_input = test_df.drop('Churn', axis=1)
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

_CLASS_BALANCE_PATTERNS = [
    r"\bclass_weight\b",
    r"\bbalanced\b",
    r"\bSMOTE\b",
    r"\bover_?sampl",
    r"\bunder_?sampl",
    r"\bresample\b",
    r"\bsample_weight\b",
    r"\bRandomOverSampler\b",
    r"\bRandomUnderSampler\b",
    r"\bclass.?imbalance\b",
]

_FEATURE_ENG_PATTERNS = [
    r"\bOneHotEncoder\b",
    r"\bget_dummies\b",
    r"\bLabelEncoder\b",
    r"\bColumnTransformer\b",
    r"\bOrdinalEncoder\b",
    r"\bPipeline\b",
    r"\bpd\.cut\b",
    r"\bpd\.qcut\b",
    r"\binteraction\b",
    r"\bStandardScaler\b",
    r"\bMinMaxScaler\b",
    r"\bRobustScaler\b",
    r"\bto_numeric\b",
    r"\bfillna\b",
    r"\bdropna\b",
]

_THRESHOLD_PATTERNS = [
    r"\bpredict_proba\b",
    r"\bthreshold\b",
    r"\broc_curve\b",
    r"\bprecision_recall_curve\b",
    r"\bGridSearchCV\b",
    r"\bRandomizedSearchCV\b",
    r"\bcross_val\b",
]


def _count_hits(source: str, patterns: list) -> int:
    return sum(1 for p in patterns if re.search(p, source, re.IGNORECASE))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Challenge class
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CustomerChurnChallenge(BaseChallenge):
    """Customer Churn Prediction — imbalanced binary classification."""

    def get_harness_code(self, mode: str) -> str:
        allowed = self.config.get("allowed_libraries", [])
        return HARNESS_CODE.replace("__ALLOWED_PLACEHOLDER__", repr(allowed))

    def get_data_files(self, mode: str) -> dict:
        train_src = os.path.join(
            self.challenge_dir, self.config["dataset"]["train_file"]
        )
        return {"churn_train.csv": train_src}

    def get_harness_args(self, mode: str) -> list:
        test_src = os.path.join(
            self.challenge_dir, self.config["dataset"]["test_file"]
        )
        return [mode, "churn_train.csv", test_src]

    # ── Grading (100 points) ────────────────────────────────

    def grade(self, user_code: str, execution_output: dict) -> dict:
        categories = []
        total = 0

        # 1 — Execution (20 pts)
        s, fb = self._grade_execution(execution_output)
        categories.append(self._cat("Execution", s, 20, fb))
        total += s

        predictions = execution_output.get("predictions", [])

        # 2 — Accuracy (40 pts)
        s, accuracy, fb, y_true, y_pred = self._grade_accuracy(predictions)
        cat = self._cat("Accuracy", s, 40, fb)
        cat["accuracy"] = accuracy
        categories.append(cat)
        total += s

        # 3 — Churn Class F1 (30 pts)
        s, churn_f1, fb = self._grade_churn_f1(y_true, y_pred)
        cat = self._cat("Churn Class F1", s, 30, fb)
        cat["churn_f1"] = churn_f1
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
        test_df = self._load_test_df()
        expected = len(test_df)
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
        y_true = (test_df["Churn"] == "Yes").astype(int).values
        if len(predictions) != len(y_true):
            fb.append(f"Length mismatch: {len(predictions)} vs {len(y_true)}.")
            return 0, None, fb, None, None
        y_pred = np.array(predictions, dtype=int)
        accuracy = float(accuracy_score(y_true, y_pred))
        pct = accuracy * 100
        if accuracy >= 0.87:
            score = 40
        elif accuracy >= 0.85:
            score = 30
        elif accuracy >= 0.83:
            score = 20
        elif accuracy >= 0.80:
            score = 10
        else:
            score = 0
        fb.append(f"Accuracy: {pct:.2f}% -> {score}/40 pts")
        fb.append(f"Correct: {int(accuracy * len(y_true))}/{len(y_true)}")
        return score, round(accuracy, 4), fb, y_true, y_pred

    def _grade_churn_f1(self, y_true, y_pred):
        fb = []
        if y_true is None or y_pred is None:
            fb.append("Skipped — no valid predictions.")
            return 0, None, fb
        # F1 for the churn class (positive = 1)
        churn_f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
        churn_prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
        churn_rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))

        if churn_f1 >= 0.75:
            score = 30
        elif churn_f1 >= 0.70:
            score = 25
        elif churn_f1 >= 0.65:
            score = 15
        else:
            score = 5
        fb.append(f"Churn F1: {churn_f1:.4f} -> {score}/30 pts")
        fb.append(f"Churn Precision: {churn_prec:.4f}")
        fb.append(f"Churn Recall: {churn_rec:.4f}")

        # Additional context
        n_churn = int(y_true.sum())
        n_total = len(y_true)
        fb.append(f"Churn rate in test: {n_churn}/{n_total} ({n_churn/n_total*100:.1f}%)")

        return score, round(churn_f1, 4), fb

    def _grade_code_quality(self, source):
        score = 0
        fb = []
        try:
            ast.parse(source)
        except SyntaxError as e:
            fb.append(f"SyntaxError at line {e.lineno}: {e.msg}")
        # (a) Class balancing — 4 pts
        n = _count_hits(source, _CLASS_BALANCE_PATTERNS)
        if n:
            score += 4
            fb.append(f"Class balancing detected ({n} indicator(s)) -> 4/4 pts")
        else:
            fb.append("No class balancing detected -> 0/4 pts")
        # (b) Feature engineering — 3 pts
        n = _count_hits(source, _FEATURE_ENG_PATTERNS)
        if n >= 2:
            score += 3
            fb.append(f"Feature engineering detected ({n} indicator(s)) -> 3/3 pts")
        elif n == 1:
            score += 1
            fb.append(f"Some feature engineering detected (1 indicator) -> 1/3 pts")
        else:
            fb.append("No feature engineering detected -> 0/3 pts")
        # (c) Threshold / tuning — 3 pts
        n = _count_hits(source, _THRESHOLD_PATTERNS)
        if n:
            score += 3
            fb.append(f"Threshold/tuning detected ({n} indicator(s)) -> 3/3 pts")
        else:
            fb.append("No threshold/tuning detected -> 0/3 pts")
        return score, fb
