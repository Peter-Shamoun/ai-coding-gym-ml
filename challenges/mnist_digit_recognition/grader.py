"""
MNIST Digit Recognition — execution harness & grading.
=======================================================
Contains:
  - HARNESS_CODE: Python source that runs inside the sandbox subprocess.
  - MnistDigitChallenge: challenge class with grading rubric (100 pts).
"""

import ast
import json as _json
import os
import re
from typing import Any, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from challenges.base import BaseChallenge


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Harness code  (runs in sandbox subprocess)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HARNESS_CODE = r'''
"""
Execution harness for MNIST Digit Recognition challenge.
Runs in an isolated subprocess.

Usage:  python _harness.py <mode> <train_images> <train_labels> <test_images> <test_labels> <output_path>
  mode: 'run' or 'submit'
"""
import sys
import json
import traceback
import time
import ast
import os
import numpy as np


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
    train_images_path = sys.argv[2]
    train_labels_path = sys.argv[3]
    test_images_path = sys.argv[4]
    test_labels_path = sys.argv[5]
    output_path = sys.argv[6]

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
        # Load training data
        X_train = np.load(train_images_path)
        y_train = np.load(train_labels_path)

        # Import user solution (now safe — imports validated above)
        from solution import train_digit_classifier

        # Train
        start = time.time()
        predict_fn = train_digit_classifier(X_train, y_train)
        train_time = time.time() - start

        # Validate return value
        if predict_fn is None:
            raise ValueError(
                "train_digit_classifier() returned None. "
                "It must return a callable prediction function."
            )
        if not callable(predict_fn):
            raise TypeError(
                f"train_digit_classifier() returned {type(predict_fn).__name__}, "
                "expected a callable prediction function."
            )

        # ── Mode: run (quick test on training sample) ────────
        if mode == "run":
            # Use a small sample for quick testing
            n_sample = min(500, len(X_train))
            indices = np.random.RandomState(42).choice(len(X_train), n_sample, replace=False)
            X_sample = X_train[indices]
            y_sample = y_train[indices]

            preds = predict_fn(X_sample)
            preds = np.asarray(preds, dtype=int).flatten()

            if len(preds) != n_sample:
                raise ValueError(
                    f"Predictions must have length {n_sample}, got {len(preds)}. "
                    f"Make sure predict returns one label per image."
                )

            from sklearn.metrics import accuracy_score
            acc = accuracy_score(y_sample, preds)
            result["sample_accuracy"] = round(float(acc), 4)
            result["num_predictions"] = len(preds)
            result["train_time"] = round(train_time, 2)

        # ── Mode: submit (evaluation on sampled test set) ─────
        elif mode == "submit":
            X_test = np.load(test_images_path)
            y_test_labels = np.load(test_labels_path)

            # Randomly sample to save computation
            if len(X_test) > 100:
                indices = np.random.choice(len(X_test), 100, replace=False)
                X_test = X_test[indices]
                y_test_labels = y_test_labels[indices]

            preds = predict_fn(X_test)
            preds = np.asarray(preds, dtype=int).flatten()

            if len(preds) != len(X_test):
                raise ValueError(
                    f"Predictions must be numpy array of shape ({len(X_test)},), "
                    f"got {len(preds)} predictions."
                )

            if preds.min() < 0 or preds.max() > 9:
                raise ValueError(
                    f"Predictions must be integers 0-9, "
                    f"got values in range [{preds.min()}, {preds.max()}]."
                )

            result["predictions"] = preds.tolist()
            result["y_true"] = y_test_labels.tolist()
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

_NORMALIZATION_PATTERNS = [
    r"/\s*255",
    r"/\s*255\.0",
    r"/\s*255\.",
    r"\.astype\s*\(\s*['\"]?float",
    r"normalize",
    r"Normalize",
    r"StandardScaler",
    r"MinMaxScaler",
]

_CNN_PATTERNS = [
    r"\bConv2D\b",
    r"\bconv2d\b",
    r"\bConv2d\b",
    r"\bnn\.Conv",
    r"\bConvolution",
    r"\bMaxPool",
    r"\bmax_pool",
    r"\bMaxPooling",
    r"\bAvgPool",
    r"\bBatchNorm",
    r"\bbatch_norm",
    r"\bBatchNormalization\b",
]

_REGULARIZATION_PATTERNS = [
    r"\bDropout\b",
    r"\bdropout\b",
    r"\bweight_decay\b",
    r"\bl2_regularizer\b",
    r"\bregularizers\.l",
    r"\bkernel_regularizer\b",
    r"\bEarlyStopping\b",
    r"\bearly_stopping\b",
]

_AUGMENTATION_PATTERNS = [
    r"\bImageDataGenerator\b",
    r"\btransforms\.",
    r"\bRandomRotation\b",
    r"\bRandomAffine\b",
    r"\bRandomCrop\b",
    r"\bRandomHorizontalFlip\b",
    r"\baugment",
    r"\bAugment",
    r"\bdata_augmentation\b",
]


def _has_match(source: str, patterns: list) -> bool:
    """Return True if any pattern matches in the source code."""
    return any(re.search(p, source) for p in patterns)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Challenge class
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MnistDigitChallenge(BaseChallenge):
    """MNIST Digit Recognition — 10-class image classification challenge."""

    # ── Execution plumbing ───────────────────────────────────

    def get_harness_code(self, mode: str) -> str:
        """Inject the allowed-libraries list into the harness template."""
        allowed = self.config.get("allowed_libraries", [])
        return HARNESS_CODE.replace(
            "__ALLOWED_PLACEHOLDER__", repr(allowed)
        )

    def get_data_files(self, mode: str) -> dict:
        """
        Training data is copied into the sandbox.
        Test data is accessed via absolute paths passed as harness args.
        """
        ds = self.config.get("dataset", {})
        return {
            ds.get("train_images", "mnist_train_images.npy"): os.path.join(
                self.challenge_dir, ds.get("train_images", "mnist_train_images.npy")
            ),
            ds.get("train_labels", "mnist_train_labels.npy"): os.path.join(
                self.challenge_dir, ds.get("train_labels", "mnist_train_labels.npy")
            ),
        }

    def get_harness_args(self, mode: str) -> list:
        """
        CLI args for the harness:
          mode  train_images  train_labels  test_images  test_labels
        output_path is appended by the sandbox automatically.
        """
        ds = self.config.get("dataset", {})
        train_images = ds.get("train_images", "mnist_train_images.npy")
        train_labels = ds.get("train_labels", "mnist_train_labels.npy")
        # Test data: absolute paths (not copied into sandbox)
        test_images_abs = os.path.join(
            self.challenge_dir, ds.get("test_images", "mnist_test_images.npy")
        )
        test_labels_abs = os.path.join(
            self.challenge_dir, ds.get("test_labels", "mnist_test_labels.npy")
        )
        return [mode, train_images, train_labels, test_images_abs, test_labels_abs]

    # ── Grading (100 points) ────────────────────────────────

    def grade(self, user_code: str, execution_output: dict) -> dict:
        categories = []
        total = 0

        # 1 ─ Execution (20 pts)
        s, fb = self._grade_execution(execution_output)
        categories.append(self._cat("Execution", s, 20, fb))
        total += s

        predictions = execution_output.get("predictions", [])
        y_true_from_output = execution_output.get("y_true", [])

        # 2 ─ Accuracy (50 pts)
        s, accuracy, fb, y_true, y_pred = self._grade_accuracy(predictions, y_true_from_output)
        cat = self._cat("Accuracy", s, 50, fb)
        cat["accuracy"] = accuracy
        categories.append(cat)
        total += s

        # 3 ─ Per-Class Performance (20 pts)
        s, per_class, fb = self._grade_per_class(y_true, y_pred)
        cat = self._cat("Per-Class Performance", s, 20, fb)
        cat["per_class_f1"] = per_class
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
        }

    # ── helpers ──────────────────────────────────────────────

    @staticmethod
    def _cat(name: str, score: int, max_score: int, feedback: list) -> dict:
        return {"name": name, "score": score, "max_score": max_score, "feedback": feedback}

    def _load_test_labels(self) -> np.ndarray:
        ds = self.config.get("dataset", {})
        path = os.path.join(
            self.challenge_dir, ds.get("test_labels", "mnist_test_labels.npy")
        )
        return np.load(path)

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
        y_true_out = output.get("y_true", [])
        expected = len(y_true_out) if y_true_out else len(self._load_test_labels())
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

    def _grade_accuracy(self, predictions: list, y_true_list: list = None) -> Tuple[int, Optional[float], list, Any, Any]:
        fb: list = []
        if not predictions:
            fb.append("No predictions to evaluate.")
            return 0, None, fb, None, None

        if y_true_list:
            y_true = np.array(y_true_list)
        else:
            y_true = self._load_test_labels()

        if len(predictions) != len(y_true):
            fb.append(
                f"Length mismatch: {len(predictions)} predictions "
                f"vs {len(y_true)} test samples."
            )
            return 0, None, fb, None, None

        y_pred = np.array(predictions, dtype=int)
        accuracy = float(accuracy_score(y_true, y_pred))
        pct = accuracy * 100

        # Tiered scoring
        if accuracy >= 0.98:
            score = 50
        elif accuracy >= 0.97:
            score = 40
        elif accuracy >= 0.95:
            score = 30
        elif accuracy >= 0.93:
            score = 20
        elif accuracy >= 0.90:
            score = 10
        else:
            score = 0

        fb.append(f"Accuracy: {pct:.2f}% -> {score}/50 pts")
        fb.append(f"Correct: {int(accuracy * len(y_true))}/{len(y_true)}")
        return score, round(accuracy, 4), fb, y_true, y_pred

    # ·· 3. Per-Class Performance ·····························

    def _grade_per_class(
        self, y_true, y_pred
    ) -> Tuple[int, Optional[dict], list]:
        fb: list = []
        if y_true is None or y_pred is None:
            fb.append("Skipped — no valid predictions.")
            return 0, None, fb

        # Per-class F1 scores for digits 0-9
        f1_per_class = f1_score(y_true, y_pred, labels=list(range(10)), average=None)
        per_class_dict = {str(d): round(float(f1_per_class[d]), 4) for d in range(10)}

        min_f1 = float(f1_per_class.min())
        all_above_095 = all(f >= 0.95 for f in f1_per_class)
        all_above_090 = all(f >= 0.90 for f in f1_per_class)

        if all_above_095:
            score = 20
            fb.append(f"All digits F1 > 0.95 -> 20/20 pts")
        elif all_above_090:
            score = 15
            fb.append(f"All digits F1 > 0.90 -> 15/20 pts")
        else:
            score = 5
            weak_digits = [d for d in range(10) if f1_per_class[d] < 0.90]
            fb.append(f"Some digits F1 < 0.90: {weak_digits} -> 5/20 pts")

        # Add per-digit breakdown
        for d in range(10):
            fb.append(f"  Digit {d}: F1 = {f1_per_class[d]:.4f}")

        fb.append(f"  Min F1: {min_f1:.4f} (digit {int(np.argmin(f1_per_class))})")

        return score, per_class_dict, fb

    # ·· 4. Code Quality ······································

    def _grade_code_quality(self, source: str) -> Tuple[int, list]:
        score = 0
        fb: list = []

        try:
            ast.parse(source)
        except SyntaxError as e:
            fb.append(f"SyntaxError at line {e.lineno}: {e.msg}")

        # (a) Normalization — 3 pts
        if _has_match(source, _NORMALIZATION_PATTERNS):
            score += 3
            fb.append("Normalization detected -> 3/3 pts")
        else:
            fb.append("No normalization detected -> 0/3 pts")

        # (b) CNN architecture — 3 pts
        if _has_match(source, _CNN_PATTERNS):
            score += 3
            fb.append("CNN architecture detected -> 3/3 pts")
        else:
            fb.append("No CNN architecture detected -> 0/3 pts")

        # (c) Dropout/regularization — 2 pts
        if _has_match(source, _REGULARIZATION_PATTERNS):
            score += 2
            fb.append("Regularization detected -> 2/2 pts")
        else:
            fb.append("No regularization detected -> 0/2 pts")

        # (d) Data augmentation — 2 pts
        if _has_match(source, _AUGMENTATION_PATTERNS):
            score += 2
            fb.append("Data augmentation detected -> 2/2 pts")
        else:
            fb.append("No data augmentation detected -> 0/2 pts")

        return score, fb
