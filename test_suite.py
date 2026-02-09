"""
Test Suite for Email Spam Detection Grading System
====================================================
Generates synthetic submissions at various quality levels and grades them,
verifying the grader produces scores in the expected ranges.

Usage:
    python test_suite.py                   # Run all tests
    python test_suite.py --list            # List available test cases
    python test_suite.py --case excellent  # Run a single test case
    python test_suite.py --keep            # Keep generated submission dirs for inspection

Quality tiers tested:
    excellent         -> ~97-100 pts  (PASS)   All criteria maxed
    good              -> ~78-90 pts   (PASS)   Solid accuracy + code quality
    satisfactory      -> ~80-86 pts   (PASS)   Borderline pass
    unsatisfactory    -> ~40-65 pts   (FAIL)   Some effort but below threshold
    poor              -> ~15-25 pts   (FAIL)   Low accuracy, no code quality
    broken_model      -> ~40-65 pts   (FAIL)   Corrupt model.pkl
    missing_files     -> ~0 pts       (FAIL)   Nothing submitted
    wrong_format      -> ~15-25 pts   (FAIL)   Unrecognisable predictions.csv
    all_spam          -> ~25-40 pts   (FAIL)   Predicts everything as spam
    all_ham           -> ~15-25 pts   (FAIL)   Predicts everything as ham
    string_labels     -> ~90-100 pts  (PASS)   Uses 'spam'/'ham' strings
    model_dir         -> ~85-100 pts  (PASS)   Uses model/ directory
"""

import argparse
import os
import pickle
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure we can import the grader from the project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from grade_submission import grade_submission

TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "spam_test.csv")


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_ground_truth():
    """Load the test set ground truth labels."""
    df = pd.read_csv(TEST_DATA_PATH)
    return df["label_num"].values  # shape (1035,)


def craft_predictions(y_true, target_accuracy, bias="random", seed=42):
    """
    Create a predictions array that achieves approximately `target_accuracy`.

    Parameters
    ----------
    y_true : array-like of {0, 1}
    target_accuracy : float in (0, 1)
    bias : str
        'random'     – flip random samples from either class
        'spam_only'  – only flip spam->ham labels (hurts recall, keeps precision)
        'ham_only'   – only flip ham->spam labels (hurts precision, keeps recall)
    seed : int

    Returns
    -------
    np.ndarray of predicted labels (0/1)
    """
    rng = np.random.RandomState(seed)
    y = np.array(y_true, copy=True)
    n = len(y)
    n_wrong = int(round(n * (1 - target_accuracy)))

    if n_wrong == 0:
        return y  # perfect predictions

    if bias == "random":
        flip_idx = rng.choice(n, size=n_wrong, replace=False)
    elif bias == "spam_only":
        # Only flip spam (1) -> ham (0): hurts recall, keeps precision perfect
        spam_idx = np.where(y == 1)[0]
        n_wrong = min(n_wrong, len(spam_idx))
        flip_idx = rng.choice(spam_idx, size=n_wrong, replace=False)
    elif bias == "ham_only":
        # Only flip ham (0) -> spam (1): hurts precision, keeps recall perfect
        ham_idx = np.where(y == 0)[0]
        n_wrong = min(n_wrong, len(ham_idx))
        flip_idx = rng.choice(ham_idx, size=n_wrong, replace=False)
    else:
        raise ValueError(f"Unknown bias: {bias}")

    y[flip_idx] = 1 - y[flip_idx]  # flip 0->1 or 1->0
    return y


def save_predictions(dirpath, y_pred, col_name="prediction"):
    """Write predictions to a CSV."""
    pd.DataFrame({col_name: y_pred}).to_csv(
        os.path.join(dirpath, "predictions.csv"), index=False
    )


def save_model(dirpath, valid=True):
    """Save a model.pkl (real sklearn model or deliberately corrupt)."""
    pkl_path = os.path.join(dirpath, "model.pkl")
    if valid:
        # Save a simple, valid sklearn model
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.feature_extraction.text import TfidfVectorizer
        model = {"vectorizer": TfidfVectorizer(), "classifier": MultinomialNB()}
        with open(pkl_path, "wb") as f:
            pickle.dump(model, f)
    else:
        # Write garbage bytes so pickle.load() will fail
        with open(pkl_path, "wb") as f:
            f.write(b"NOT_A_PICKLE_FILE__corrupt_data_here")


def save_classifier(dirpath, quality="excellent"):
    """
    Write a classifier.py whose static-analysis code-quality score matches
    the desired tier.

    Tiers and expected code quality scores
    ----------------------------------------
    excellent    -> preprocessing + imbalance + feature eng  =  10/10
    good         -> preprocessing + imbalance + 1 feat eng   =   8/10
    satisfactory -> preprocessing only                       =   3/10
    poor         -> nothing detected                         =   0/10
    syntax_error -> has a Python syntax error                =   0/10
    """
    code_map = {
        # ────────────── EXCELLENT (10/10) ──────────────
        "excellent": '''\
"""Spam classifier – production quality."""
import re
import string
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from scipy.sparse import hstack


def clean_text(text):
    """Preprocess text: lowercasing, URL/email removal, stemming-ready."""
    text = text.lower()
    text = re.sub(r"http\\S+|www\\.\\S+", " url ", text)
    text = re.sub(r"\\S+@\\S+", " email ", text)
    text = re.sub(r"\\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\\s+", " ", text).strip()
    return text


# ── Feature engineering ──
subject_vec = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
    max_df=0.95,
)
body_vec = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
)

# ── Ensemble with class_weight=balanced ──
ensemble = VotingClassifier(
    estimators=[
        ("lr", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)),
        ("svc", CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", max_iter=2000), cv=3)),
        ("sgd", SGDClassifier(loss="log_loss", class_weight="balanced", max_iter=1000)),
    ],
    voting="soft",
)

if __name__ == "__main__":
    train = pd.read_csv("spam_train.csv")
    test = pd.read_csv("spam_test.csv")
    for df in [train, test]:
        df["clean"] = df["text"].apply(clean_text)
    X_tr = body_vec.fit_transform(train["clean"])
    X_te = body_vec.transform(test["clean"])
    ensemble.fit(X_tr, train["label_num"])
    preds = ensemble.predict(X_te)
    pd.DataFrame({"prediction": preds}).to_csv("predictions.csv", index=False)
    pickle.dump({"vec": body_vec, "model": ensemble}, open("model.pkl", "wb"))
''',

        # ────────────── GOOD (8/10 or 10/10) ──────────────
        "good": '''\
"""Spam classifier – good quality."""
import re
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def preprocess(text):
    """Basic cleaning: lowercase, remove numbers."""
    text = text.lower()
    text = re.sub(r"\\d+", "", text)
    return text.strip()


# class_weight='balanced' handles imbalance
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
model = LogisticRegression(class_weight="balanced", max_iter=1000)

if __name__ == "__main__":
    train = pd.read_csv("spam_train.csv")
    test = pd.read_csv("spam_test.csv")
    train["clean"] = train["text"].apply(preprocess)
    test["clean"] = test["text"].apply(preprocess)
    X_tr = vectorizer.fit_transform(train["clean"])
    X_te = vectorizer.transform(test["clean"])
    model.fit(X_tr, train["label_num"])
    preds = model.predict(X_te)
    pd.DataFrame({"prediction": preds}).to_csv("predictions.csv", index=False)
    pickle.dump({"vec": vectorizer, "model": model}, open("model.pkl", "wb"))
''',

        # ────────────── SATISFACTORY (3/10) ──────────────
        "satisfactory": '''\
"""Spam classifier – minimal effort."""
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def clean_text(text):
    return text.lower().strip()


vectorizer = TfidfVectorizer()
model = MultinomialNB()

if __name__ == "__main__":
    train = pd.read_csv("spam_train.csv")
    test = pd.read_csv("spam_test.csv")
    train["text"] = train["text"].apply(clean_text)
    test["text"] = test["text"].apply(clean_text)
    X_tr = vectorizer.fit_transform(train["text"])
    X_te = vectorizer.transform(test["text"])
    model.fit(X_tr, train["label_num"])
    preds = model.predict(X_te)
    pd.DataFrame({"prediction": preds}).to_csv("predictions.csv", index=False)
    pickle.dump({"vec": vectorizer, "model": model}, open("model.pkl", "wb"))
''',

        # ────────────── POOR (0/10 code quality) ──────────────
        "poor": '''\
"""Quick spam classifier."""
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

vectorizer = TfidfVectorizer()
model = MultinomialNB()

if __name__ == "__main__":
    train = pd.read_csv("spam_train.csv")
    test = pd.read_csv("spam_test.csv")
    X_tr = vectorizer.fit_transform(train["text"])
    X_te = vectorizer.transform(test["text"])
    model.fit(X_tr, train["label_num"])
    preds = model.predict(X_te)
    pd.DataFrame({"prediction": preds}).to_csv("predictions.csv", index=False)
    pickle.dump({"vec": vectorizer, "model": model}, open("model.pkl", "wb"))
''',

        # ────────────── SYNTAX ERROR ──────────────
        "syntax_error": '''\
"""Classifier with syntax error."""
import pandas as pd

def train_model(
    # Oops, missing closing paren and colon
    x, y

if __name__ == "__main__":
    print("broken")
''',
    }

    source = code_map.get(quality, code_map["poor"])
    with open(os.path.join(dirpath, "classifier.py"), "w", encoding="utf-8") as f:
        f.write(source)


# ═══════════════════════════════════════════════════════════════════════════
#  Test-case builders
# ═══════════════════════════════════════════════════════════════════════════
#
#  Scoring recap (100 total, pass >= 80):
#    1. Validation   20 pts  (classifier.py=5, model loads=10, predictions.csv=5)
#    2. Accuracy     50 pts  (>=95%→50, >=93%→40, >=91%→30, >=89%→20, >=85%→10)
#    3. P/R          20 pts  (spam precision>0.90→10, spam recall>0.90→10)
#    4. Code qual    10 pts  (preprocess=3, imbalance=3, feat eng(>=2)=4)
#
#  NOTE on precision/recall math (1035 samples: 735 ham, 300 spam):
#    Both P>0.90 and R>0.90 requires total errors <= ~58 → accuracy >= ~94.4%.
#    With random bias, precision often fails first since more ham can flip.
#    Use spam_only bias (hurts recall, keeps precision) or ham_only (hurts
#    precision, keeps recall) to control which one passes.
# ═══════════════════════════════════════════════════════════════════════════

def build_excellent(base_dir, y_true):
    """
    EXCELLENT – should score ~97-100.
      Validation:  20  (all files present, model loads)
      Accuracy:    50  (97% → highest tier)
      P/R:         20  (random bias at 97% keeps both P and R above 0.90)
      Code:        10  (full patterns)
      Total:       100
    """
    d = os.path.join(base_dir, "excellent")
    os.makedirs(d)
    y_pred = craft_predictions(y_true, 0.97, bias="random")
    save_predictions(d, y_pred)
    save_model(d, valid=True)
    save_classifier(d, quality="excellent")
    return d, {"min": 95, "max": 100, "passed": True,
               "label": "EXCELLENT – all criteria maxed, ~97% accuracy"}


def build_good(base_dir, y_true):
    """
    GOOD – should score ~78-90.
      Validation:  20
      Accuracy:    40  (94% → second tier)
      P/R:       0-20  (random bias at 94% → recall likely passes, precision iffy)
      Code:     8-10   (preprocessing + imbalance + at least 1 feature eng)
      Total:    ~78-90
    """
    d = os.path.join(base_dir, "good")
    os.makedirs(d)
    y_pred = craft_predictions(y_true, 0.94, bias="random")
    save_predictions(d, y_pred)
    save_model(d, valid=True)
    save_classifier(d, quality="good")
    return d, {"min": 75, "max": 95, "passed": True,
               "label": "GOOD – 94% accuracy, decent code quality"}


def build_satisfactory(base_dir, y_true):
    """
    SATISFACTORY / borderline pass – should score ~80-86.
      Validation:  20
      Accuracy:    50  (96% → top tier, using ham_only bias)
      P/R:         10  (ham_only: recall=1.0 passes, precision~0.88 fails)
      Code:         3  (preprocessing only)
      Total:       83
    """
    d = os.path.join(base_dir, "satisfactory")
    os.makedirs(d)
    # ham_only bias: all errors are ham→spam, so recall stays perfect but
    # precision drops. Target 96% to safely clear the 95% threshold.
    y_pred = craft_predictions(y_true, 0.96, bias="ham_only")
    save_predictions(d, y_pred)
    save_model(d, valid=True)
    save_classifier(d, quality="satisfactory")
    return d, {"min": 78, "max": 88, "passed": True,
               "label": "SATISFACTORY – borderline pass (~83 pts)"}


def build_unsatisfactory(base_dir, y_true):
    """
    UNSATISFACTORY – should score ~40-65 (FAIL).
      Validation:  20
      Accuracy:    20  (90% → fourth tier)
      P/R:       0-10  (random at 90% → maybe recall passes)
      Code:         0  (no patterns)
      Total:    ~40-60
    """
    d = os.path.join(base_dir, "unsatisfactory")
    os.makedirs(d)
    y_pred = craft_predictions(y_true, 0.90, bias="random")
    save_predictions(d, y_pred)
    save_model(d, valid=True)
    save_classifier(d, quality="poor")
    return d, {"min": 35, "max": 70, "passed": False,
               "label": "UNSATISFACTORY – 90% accuracy, no code quality"}


def build_poor(base_dir, y_true):
    """
    POOR – should score ~15-25 (FAIL).
      Validation:  20
      Accuracy:     0  (80% → below all tiers)
      P/R:        0-10 (likely both fail at 80%)
      Code:         0
      Total:      ~20
    """
    d = os.path.join(base_dir, "poor")
    os.makedirs(d)
    y_pred = craft_predictions(y_true, 0.80, bias="random")
    save_predictions(d, y_pred)
    save_model(d, valid=True)
    save_classifier(d, quality="poor")
    return d, {"min": 15, "max": 35, "passed": False,
               "label": "POOR – 80% accuracy, no code quality"}


def build_broken_model(base_dir, y_true):
    """
    BROKEN MODEL – should score ~40-60 (FAIL).
      Validation:  10  (classifier=5, model FAILS=0, predictions=5)
      Accuracy:    30  (91% → third tier)
      P/R:       0-10  (random at 91%)
      Code:         0  (no patterns)
      Total:     ~40-60
    """
    d = os.path.join(base_dir, "broken_model")
    os.makedirs(d)
    y_pred = craft_predictions(y_true, 0.91, bias="random")
    save_predictions(d, y_pred)
    save_model(d, valid=False)  # corrupt model
    save_classifier(d, quality="poor")
    return d, {"min": 35, "max": 65, "passed": False,
               "label": "BROKEN MODEL – model.pkl fails to load"}


def build_missing_files(base_dir, y_true):
    """
    MISSING FILES – should score 0 (FAIL).
      No classifier.py, no model, no predictions.csv.
      All sections skipped → 0/100.
    """
    d = os.path.join(base_dir, "missing_files")
    os.makedirs(d)
    # Write a dummy file so the directory isn't empty
    with open(os.path.join(d, "readme.txt"), "w") as f:
        f.write("I forgot to include my files.\n")
    return d, {"min": 0, "max": 5, "passed": False,
               "label": "MISSING FILES – empty submission"}


def build_wrong_format(base_dir, y_true):
    """
    WRONG FORMAT – predictions.csv has unrecognisable values (FAIL).
      Correct row count (1035) but column contains strings the grader
      cannot map ('positive'/'negative' instead of 'spam'/'ham' or 0/1).
      Validation:  20  (all files present)
      Accuracy:     0  (unrecognised labels → 0 pts)
      P/R:          0  (skipped)
      Code:         0  (no patterns)
      Total:       ~20
    """
    d = os.path.join(base_dir, "wrong_format")
    os.makedirs(d)
    save_model(d, valid=True)
    save_classifier(d, quality="poor")
    # Use unrecognisable string labels in a valid column name
    n = len(y_true)
    bad_labels = np.where(y_true == 1, "positive", "negative")
    pd.DataFrame({"prediction": bad_labels}).to_csv(
        os.path.join(d, "predictions.csv"), index=False
    )
    return d, {"min": 15, "max": 25, "passed": False,
               "label": "WRONG FORMAT – unrecognisable prediction values"}


def build_all_spam(base_dir, y_true):
    """
    ALL SPAM – predicts every email as spam (FAIL).
      Validation:  20
      Accuracy:     0  (300/1035 ≈ 29% → below all tiers)
      P/R:         10  (recall=1.0 passes, precision=300/1035≈0.29 fails)
      Code:         0
      Total:       ~30
    """
    d = os.path.join(base_dir, "all_spam")
    os.makedirs(d)
    y_pred = np.ones(len(y_true), dtype=int)  # everything is spam
    save_predictions(d, y_pred)
    save_model(d, valid=True)
    save_classifier(d, quality="poor")
    return d, {"min": 25, "max": 40, "passed": False,
               "label": "ALL SPAM – predicts everything as spam"}


def build_all_ham(base_dir, y_true):
    """
    ALL HAM – predicts every email as ham (FAIL).
      Validation:  20
      Accuracy:     0  (735/1035 ≈ 71% → below all tiers)
      P/R:          0  (precision=0, recall=0 both fail)
      Code:         0
      Total:       ~20
    """
    d = os.path.join(base_dir, "all_ham")
    os.makedirs(d)
    y_pred = np.zeros(len(y_true), dtype=int)  # everything is ham
    save_predictions(d, y_pred)
    save_model(d, valid=True)
    save_classifier(d, quality="poor")
    return d, {"min": 15, "max": 25, "passed": False,
               "label": "ALL HAM – majority-class baseline (0 recall)"}


def build_string_labels(base_dir, y_true):
    """
    STRING LABELS – uses 'spam'/'ham' strings instead of 0/1 (PASS).
      Verifies the grader handles string-to-numeric conversion.
      Validation:  20
      Accuracy:    50  (96% → top tier)
      P/R:       ~20  (at 96%, both P and R should pass)
      Code:        10
      Total:     ~95-100
    """
    d = os.path.join(base_dir, "string_labels")
    os.makedirs(d)
    y_pred_num = craft_predictions(y_true, 0.96, bias="random")
    y_pred_str = np.where(y_pred_num == 1, "spam", "ham")
    pd.DataFrame({"prediction": y_pred_str}).to_csv(
        os.path.join(d, "predictions.csv"), index=False
    )
    save_model(d, valid=True)
    save_classifier(d, quality="excellent")
    return d, {"min": 85, "max": 100, "passed": True,
               "label": "STRING LABELS – uses 'spam'/'ham' strings (should pass)"}


def build_model_dir(base_dir, y_true):
    """
    MODEL DIRECTORY – uses model/ directory instead of model.pkl (PASS).
      Verifies the grader accepts directory-based models.
      Validation:  20
      Accuracy:    50  (95% → top tier)
      P/R:       0-20
      Code:     8-10
      Total:    ~80-100
    """
    d = os.path.join(base_dir, "model_dir")
    os.makedirs(d)
    y_pred = craft_predictions(y_true, 0.95, bias="random")
    save_predictions(d, y_pred)
    # Use model/ directory instead of model.pkl
    model_subdir = os.path.join(d, "model")
    os.makedirs(model_subdir)
    with open(os.path.join(model_subdir, "weights.bin"), "wb") as f:
        f.write(b"fake_model_weights")
    with open(os.path.join(model_subdir, "config.json"), "w") as f:
        f.write('{"type": "test_model"}')
    save_classifier(d, quality="good")
    return d, {"min": 78, "max": 100, "passed": True,
               "label": "MODEL DIR – uses model/ directory (should pass)"}


# ═══════════════════════════════════════════════════════════════════════════
#  Registry of all test cases
# ═══════════════════════════════════════════════════════════════════════════

TEST_CASES = {
    # ── Correct / passing submissions ──
    "excellent":      build_excellent,
    "good":           build_good,
    "satisfactory":   build_satisfactory,
    # ── Incorrect / failing submissions ──
    "unsatisfactory": build_unsatisfactory,
    "poor":           build_poor,
    "broken_model":   build_broken_model,
    "missing_files":  build_missing_files,
    "wrong_format":   build_wrong_format,
    # ── Degenerate predictions ──
    "all_spam":       build_all_spam,
    "all_ham":        build_all_ham,
    # ── Edge-case formats that should still pass ──
    "string_labels":  build_string_labels,
    "model_dir":      build_model_dir,
}


# ═══════════════════════════════════════════════════════════════════════════
#  Runner
# ═══════════════════════════════════════════════════════════════════════════

def run_test_case(name, builder, y_true, base_dir):
    """Build a submission, grade it, return structured result."""
    sub_dir, expect = builder(base_dir, y_true)

    # Suppress the grader's print output during batch runs
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = grade_submission(sub_dir, TEST_DATA_PATH)

    score = result["total_score"]
    passed = result["passed"]
    accuracy = result.get("accuracy")

    score_ok = expect["min"] <= score <= expect["max"]
    pass_ok = passed == expect["passed"]
    ok = score_ok and pass_ok

    return {
        "name": name,
        "label": expect["label"],
        "score": score,
        "expected_range": f"{expect['min']}-{expect['max']}",
        "passed": passed,
        "expected_passed": expect["passed"],
        "accuracy": f"{accuracy*100:.1f}%" if accuracy else "N/A",
        "score_in_range": score_ok,
        "pass_matches": pass_ok,
        "test_ok": ok,
        "feedback": result.get("feedback", ""),
    }


def print_results(results):
    """Print a formatted results table."""
    # Header
    print("\n")
    print("=" * 100)
    print("  TEST SUITE RESULTS")
    print("=" * 100)
    print()

    # Column widths
    name_w = max(len(r["name"]) for r in results) + 2
    header = (
        f"  {'Case':<{name_w}}  {'Score':>5}  {'Expected':>10}  "
        f"{'Pass?':>5}  {'Expect':>6}  {'Acc':>7}  {'Result':>8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    n_pass = 0
    for r in results:
        status = "OK" if r["test_ok"] else "FAIL"
        marker = "  " if r["test_ok"] else ">>"
        if r["test_ok"]:
            n_pass += 1

        print(
            f"{marker}{'':1}{r['name']:<{name_w}}  "
            f"{r['score']:>5}  "
            f"{r['expected_range']:>10}  "
            f"{'Yes' if r['passed'] else 'No':>5}  "
            f"{'Yes' if r['expected_passed'] else 'No':>6}  "
            f"{r['accuracy']:>7}  "
            f"{status:>8}"
        )

    print()
    print("-" * 100)
    print(f"  Tests passed: {n_pass}/{len(results)}")

    if n_pass == len(results):
        print("  ALL TESTS PASSED - Grading system is working correctly.")
    else:
        failed = [r for r in results if not r["test_ok"]]
        print(f"  FAILURES ({len(failed)}):")
        for r in failed:
            reason = []
            if not r["score_in_range"]:
                reason.append(f"score {r['score']} not in [{r['expected_range']}]")
            if not r["pass_matches"]:
                reason.append(
                    f"passed={r['passed']} but expected={r['expected_passed']}"
                )
            print(f"    - {r['name']}: {'; '.join(reason)}")

    print("-" * 100)
    print()

    # Detailed breakdown
    print("DETAILED BREAKDOWN:")
    print()
    for r in results:
        icon = "[OK]  " if r["test_ok"] else "[FAIL]"
        print(f"  {icon} {r['label']}")
        print(f"         Score: {r['score']}  |  Range: {r['expected_range']}  "
              f"|  Passed: {r['passed']}  |  Accuracy: {r['accuracy']}")
        print()

    return n_pass == len(results)


def print_single_result(r):
    """Print detailed output for a single test case (shows full grader feedback)."""
    print("\n")
    print("=" * 70)
    print(f"  TEST CASE: {r['name']}")
    print("=" * 70)
    print(f"  Expected: {r['label']}")
    print(f"  Score:    {r['score']}")
    print(f"  Range:    {r['expected_range']}")
    print(f"  Passed:   {r['passed']}  (expected: {r['expected_passed']})")
    print(f"  Accuracy: {r['accuracy']}")
    status = "OK" if r["test_ok"] else "FAIL"
    print(f"  Test:     {status}")
    print()
    if r.get("feedback"):
        print(r["feedback"])
    print()


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Test suite for the email spam detection grading system"
    )
    parser.add_argument("--list", action="store_true", help="List available test cases")
    parser.add_argument("--case", type=str, help="Run a single test case by name")
    parser.add_argument("--keep", action="store_true",
                        help="Keep generated submission dirs (in ./test_submissions/)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show full grader feedback for each case")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable test cases:")
        print("-" * 60)
        print(f"  {'Name':<18} {'Expected':>12}  Description")
        print("-" * 60)
        # Build a quick reference by inspecting docstrings
        descs = {
            "excellent":      ("95-100 PASS", "All criteria maxed, ~97% accuracy"),
            "good":           ("75-95  PASS", "94% accuracy, solid code quality"),
            "satisfactory":   ("78-88  PASS", "Borderline pass, 95% acc + minimal code"),
            "unsatisfactory": ("35-70  FAIL", "90% accuracy, no code quality"),
            "poor":           ("15-35  FAIL", "80% accuracy, everything minimal"),
            "broken_model":   ("35-65  FAIL", "Corrupt model.pkl"),
            "missing_files":  (" 0-5   FAIL", "Empty submission"),
            "wrong_format":   ("15-25  FAIL", "Unrecognisable prediction values"),
            "all_spam":       ("25-40  FAIL", "Predicts everything as spam"),
            "all_ham":        ("15-25  FAIL", "Predicts everything as ham"),
            "string_labels":  ("85-100 PASS", "Uses 'spam'/'ham' string labels"),
            "model_dir":      ("78-100 PASS", "Uses model/ directory"),
        }
        for name in TEST_CASES:
            rng, desc = descs.get(name, ("?", ""))
            print(f"  {name:<18} {rng:>12}  {desc}")
        print(f"\nTotal: {len(TEST_CASES)} cases")
        return 0

    # Verify test data exists
    if not os.path.isfile(TEST_DATA_PATH):
        print(f"ERROR: Test data not found at {TEST_DATA_PATH}")
        print("Make sure spam_test.csv is in the project root.")
        return 1

    y_true = load_ground_truth()
    print(f"\nLoaded ground truth: {len(y_true)} samples "
          f"({(y_true == 0).sum()} ham, {(y_true == 1).sum()} spam)")

    # Choose output directory
    if args.keep:
        base_dir = os.path.join(PROJECT_ROOT, "test_submissions")
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)
        os.makedirs(base_dir)
    else:
        base_dir = tempfile.mkdtemp(prefix="spam_test_suite_")

    # Select cases to run
    if args.case:
        if args.case not in TEST_CASES:
            print(f"ERROR: Unknown test case '{args.case}'")
            print(f"Available: {', '.join(TEST_CASES.keys())}")
            return 1
        cases = {args.case: TEST_CASES[args.case]}
    else:
        cases = TEST_CASES

    # Run
    print(f"Running {len(cases)} test case(s)...\n")
    results = []
    for name, builder in cases.items():
        print(f"  Grading: {name} ...", end=" ", flush=True)
        t0 = time.time()
        r = run_test_case(name, builder, y_true, base_dir)
        elapsed = time.time() - t0
        status = "OK" if r["test_ok"] else "FAIL"
        print(f"[{status}]  score={r['score']:>3}  ({elapsed:.1f}s)")
        results.append(r)

    # Output
    if args.case and len(results) == 1:
        print_single_result(results[0])
    elif args.verbose:
        for r in results:
            print_single_result(r)

    all_ok = print_results(results)

    # Cleanup
    if not args.keep:
        shutil.rmtree(base_dir, ignore_errors=True)
        print("(Temp dirs cleaned up. Use --keep to preserve them for inspection.)\n")
    else:
        print(f"Submission dirs preserved at: {base_dir}\n")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
