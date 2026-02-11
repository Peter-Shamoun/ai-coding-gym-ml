"""
California Housing Price Prediction — execution harness & grading.
==================================================================
Contains:
  - HARNESS_CODE: Python source that runs inside the sandbox subprocess.
  - HousingPriceChallenge: challenge class with grading rubric (100 pts).

Key difference: This is REGRESSION. Uses R² and RMSE instead of accuracy.
"""

import ast
import json as _json
import os
import re
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from challenges.base import BaseChallenge


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Harness code  (runs in sandbox subprocess)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HARNESS_CODE = r'''
"""
Execution harness for California Housing Price Prediction challenge.
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
import math


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
        import numpy as np
        from sklearn.metrics import r2_score, mean_squared_error

        train_df = pd.read_csv(train_path)

        from solution import train_price_predictor

        start = time.time()
        predict_fn = train_price_predictor(train_df)
        train_time = time.time() - start

        if predict_fn is None:
            raise ValueError(
                "train_price_predictor() returned None. "
                "It must return a callable prediction function."
            )
        if not callable(predict_fn):
            raise TypeError(
                f"train_price_predictor() returned {type(predict_fn).__name__}, "
                "expected a callable prediction function."
            )

        if mode == "run":
            sample = train_df.sample(min(200, len(train_df)), random_state=42)
            y_sample = sample['MedHouseVal'].values
            df_input = sample.drop('MedHouseVal', axis=1)
            preds = predict_fn(df_input)
            preds = np.asarray(preds, dtype=float).flatten()

            if len(preds) != len(y_sample):
                raise ValueError(
                    f"Predictions length {len(preds)} != sample size {len(y_sample)}"
                )

            r2 = r2_score(y_sample, preds)
            rmse = float(np.sqrt(mean_squared_error(y_sample, preds)))

            # Use r2 as "sample_accuracy" for compatibility with the output format
            result["sample_accuracy"] = round(r2, 4)
            result["sample_r2"] = round(r2, 4)
            result["sample_rmse"] = round(rmse, 4)
            result["num_predictions"] = len(preds)
            result["train_time"] = round(train_time, 2)

        elif mode == "submit":
            test_df = pd.read_csv(test_path)
            # Randomly sample to save computation
            if len(test_df) > 100:
                test_df = test_df.sample(n=100)
            df_input = test_df.drop('MedHouseVal', axis=1)
            preds = predict_fn(df_input)
            preds = np.asarray(preds, dtype=float).flatten()

            if len(preds) != len(test_df):
                raise ValueError(
                    f"Predictions must have length {len(test_df)}, got {len(preds)}"
                )

            # Check for NaN / inf
            if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                raise ValueError(
                    "Predictions contain NaN or Inf values. "
                    "Ensure your model handles all inputs."
                )

            result["predictions"] = preds.tolist()
            result["y_true"] = test_df["MedHouseVal"].tolist()
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

_SCALING_PATTERNS = [
    r"\bStandardScaler\b",
    r"\bMinMaxScaler\b",
    r"\bRobustScaler\b",
    r"\bnormalize\b",
    r"\bscale\b",
    r"\bfit_transform\b",
]

_POLYNOMIAL_PATTERNS = [
    r"\bPolynomialFeatures\b",
    r"\bpoly\b",
    r"\binteraction\b",
    r"\bfeature_inter",
    r"\bcross_feature",
    r"\blatitude.*longitude|longitude.*latitude\b",
]

_OUTLIER_PATTERNS = [
    r"\bclip\b",
    r"\bquantile\b",
    r"\bIQR\b",
    r"\boutlier\b",
    r"\bwinsor",
    r"\bpercentile\b",
    r"\bnp\.log\b",
    r"\bnp\.log1p\b",
    r"\blog_transform\b",
]


def _count_hits(source: str, patterns: list) -> int:
    return sum(1 for p in patterns if re.search(p, source, re.IGNORECASE))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Challenge class
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class HousingPriceChallenge(BaseChallenge):
    """California Housing Price Prediction — regression challenge."""

    def get_harness_code(self, mode: str) -> str:
        allowed = self.config.get("allowed_libraries", [])
        return HARNESS_CODE.replace("__ALLOWED_PLACEHOLDER__", repr(allowed))

    def get_data_files(self, mode: str) -> dict:
        train_src = os.path.join(
            self.challenge_dir, self.config["dataset"]["train_file"]
        )
        return {"housing_train.csv": train_src}

    def get_harness_args(self, mode: str) -> list:
        test_src = os.path.join(
            self.challenge_dir, self.config["dataset"]["test_file"]
        )
        return [mode, "housing_train.csv", test_src]

    # ── Grading (100 points) ────────────────────────────────

    def grade(self, user_code: str, execution_output: dict) -> dict:
        categories = []
        total = 0

        # 1 — Execution (20 pts)
        s, fb = self._grade_execution(execution_output)
        categories.append(self._cat("Execution", s, 20, fb))
        total += s

        predictions = execution_output.get("predictions", [])
        y_true_from_output = execution_output.get("y_true", [])

        # 2 — R² Score (40 pts)
        s, r2, rmse, fb, y_true, y_pred = self._grade_r2(predictions, y_true_from_output)
        cat = self._cat("R² Score", s, 40, fb)
        cat["r2"] = r2
        categories.append(cat)
        total += s

        # 3 — RMSE (30 pts)
        s, fb = self._grade_rmse(rmse)
        cat = self._cat("RMSE", s, 30, fb)
        cat["rmse"] = rmse
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
            "accuracy": r2,  # Use R² as the "accuracy" metric for display
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
        y_true_out = output.get("y_true", [])
        expected = len(y_true_out) if y_true_out else len(self._load_test_df())
        if n == expected:
            score += 10
            fb.append(f"Prediction count matches test set ({n}).")
        elif n > 0:
            score += 5
            fb.append(f"Prediction count mismatch: got {n}, expected {expected}.")
        else:
            fb.append("No predictions produced.")
        return score, fb

    def _grade_r2(self, predictions, y_true_list=None):
        fb = []
        if not predictions:
            fb.append("No predictions to evaluate.")
            return 0, None, None, fb, None, None
        if y_true_list:
            y_true = np.array(y_true_list)
        else:
            test_df = self._load_test_df()
            y_true = test_df["MedHouseVal"].values
        if len(predictions) != len(y_true):
            fb.append(f"Length mismatch: {len(predictions)} vs {len(y_true)}.")
            return 0, None, None, fb, None, None

        y_pred = np.array(predictions, dtype=float)
        r2 = float(r2_score(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

        if r2 >= 0.85:
            score = 40
        elif r2 >= 0.82:
            score = 30
        elif r2 >= 0.78:
            score = 20
        elif r2 >= 0.75:
            score = 10
        else:
            score = 0
        fb.append(f"R² Score: {r2:.4f} -> {score}/40 pts")
        return score, round(r2, 4), round(rmse, 4), fb, y_true, y_pred

    def _grade_rmse(self, rmse):
        fb = []
        if rmse is None:
            fb.append("Skipped — no valid predictions.")
            return 0, fb
        if rmse <= 0.50:
            score = 30
        elif rmse <= 0.55:
            score = 25
        elif rmse <= 0.60:
            score = 15
        else:
            score = 5
        fb.append(f"RMSE: {rmse:.4f} -> {score}/30 pts")
        return score, fb

    def _grade_code_quality(self, source):
        score = 0
        fb = []
        try:
            ast.parse(source)
        except SyntaxError as e:
            fb.append(f"SyntaxError at line {e.lineno}: {e.msg}")
        # (a) Feature scaling — 4 pts
        n = _count_hits(source, _SCALING_PATTERNS)
        if n:
            score += 4
            fb.append(f"Feature scaling detected ({n} indicator(s)) -> 4/4 pts")
        else:
            fb.append("No feature scaling detected -> 0/4 pts")
        # (b) Polynomial / interaction features — 3 pts
        n = _count_hits(source, _POLYNOMIAL_PATTERNS)
        if n:
            score += 3
            fb.append(f"Polynomial/interaction features detected ({n} indicator(s)) -> 3/3 pts")
        else:
            fb.append("No polynomial/interaction features detected -> 0/3 pts")
        # (c) Outlier handling — 3 pts
        n = _count_hits(source, _OUTLIER_PATTERNS)
        if n:
            score += 3
            fb.append(f"Outlier handling detected ({n} indicator(s)) -> 3/3 pts")
        else:
            fb.append("No outlier handling detected -> 0/3 pts")
        return score, fb
