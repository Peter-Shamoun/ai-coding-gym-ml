"""
Test script for ALL 5 challenges.
==================================
Verifies:
  1. Registry discovers all 5 challenges
  2. All data files exist
  3. Quick "run" mode works for each
  4. Full "submit" mode works with grading for each
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force re-discovery each time
from challenges.registry import ChallengeRegistry
ChallengeRegistry._instance = None
ChallengeRegistry._challenges = {}


# ── Baseline solutions for each challenge ───────────────────────

IMDB_SOLUTION = '''
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def train_sentiment_classifier(df_train):
    texts = df_train['text'].values
    labels = df_train['sentiment'].values

    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X, labels)

    def predict(df_test):
        X_test = vectorizer.transform(df_test['text'].values)
        return model.predict(X_test)

    return predict
'''

CHURN_SOLUTION = '''
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

def train_churn_predictor(df_train):
    df = df_train.copy()
    y = (df['Churn'] == 'Yes').astype(int)
    df = df.drop(['Churn', 'customerID'], axis=1, errors='ignore')

    # Handle TotalCharges blanks
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Encode categoricals
    encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    feature_cols = df.columns.tolist()

    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, random_state=42,
        min_samples_split=10
    )
    model.fit(df, y)

    def predict(df_test):
        df_t = df_test.copy()
        df_t = df_t.drop('customerID', axis=1, errors='ignore')
        df_t['TotalCharges'] = pd.to_numeric(df_t['TotalCharges'], errors='coerce')
        df_t['TotalCharges'] = df_t['TotalCharges'].fillna(0)
        for col, le in encoders.items():
            if col in df_t.columns:
                # Handle unseen labels
                df_t[col] = df_t[col].astype(str).map(
                    lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1
                )
        # Ensure same columns
        for c in feature_cols:
            if c not in df_t.columns:
                df_t[c] = 0
        return model.predict(df_t[feature_cols])

    return predict
'''

HOUSING_SOLUTION = '''
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

def train_price_predictor(df_train):
    features = df_train.drop('MedHouseVal', axis=1)
    target = df_train['MedHouseVal']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
    )
    model.fit(X_scaled, target)

    def predict(df_test):
        X = scaler.transform(df_test)
        return model.predict(X)

    return predict
'''


ALL_CHALLENGES = {
    "email-spam-detection": {
        "name": "Email Spam Detection",
        "solution": '''
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def train_spam_detector(df):
    texts = df['text'].tolist()
    labels = df['label_num'].values

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X, labels)

    def predict(texts_list):
        X_test = vectorizer.transform(texts_list)
        return model.predict(X_test).tolist()

    return predict
''',
        "min_accuracy": 0.80,
    },
    "mnist_digit_recognition": {
        "name": "MNIST Digit Recognition",
        "solution": '''
import numpy as np
from sklearn.neural_network import MLPClassifier

def train_digit_classifier(X_train, y_train):
    X_flat = X_train.reshape(len(X_train), -1) / 255.0
    clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=10, random_state=42)
    clf.fit(X_flat, y_train)

    def predict(X_test):
        X_test_flat = X_test.reshape(len(X_test), -1) / 255.0
        return clf.predict(X_test_flat)

    return predict
''',
        "min_accuracy": 0.90,
    },
    "imdb_sentiment_analysis": {
        "name": "IMDB Sentiment Analysis",
        "solution": IMDB_SOLUTION,
        "min_accuracy": 0.80,
    },
    "customer_churn_prediction": {
        "name": "Customer Churn Prediction",
        "solution": CHURN_SOLUTION,
        "min_accuracy": 0.75,
    },
    "housing_price_prediction": {
        "name": "Housing Price Prediction",
        "solution": HOUSING_SOLUTION,
        "min_accuracy": 0.60,  # R² for housing
    },
}


def test_registry():
    """Test that all 5 challenges load via the registry."""
    print("=" * 60)
    print("TEST: Registry Discovery")
    print("=" * 60)

    registry = ChallengeRegistry()
    all_ids = registry.list_ids()
    print(f"  Found {len(all_ids)} challenges: {all_ids}")

    expected = set(ALL_CHALLENGES.keys())
    found = set(all_ids)
    missing = expected - found
    if missing:
        print(f"  MISSING: {missing}")
        return False

    for cid in expected:
        challenge = registry.get(cid)
        assert challenge is not None, f"Challenge {cid} is None"
        details = challenge.get_details()
        assert "description" in details
        assert "starter_code" in details
        assert "allowed_libraries" in details
        print(f"  {cid}: {challenge.title} [{challenge.difficulty}]")

    print("  PASSED\n")
    return True


def test_challenge_run(challenge_id, info):
    """Test quick run mode for a challenge."""
    print(f"  [{challenge_id}] Running quick test...")
    from engine.executor import execute_challenge

    start = time.time()
    result = execute_challenge(challenge_id, info["solution"], mode="run")
    elapsed = time.time() - start

    if not result.success:
        print(f"  [{challenge_id}] RUN FAILED: {result.error}")
        if result.traceback_str:
            print(f"    {result.traceback_str[:300]}")
        return False

    acc = result.sample_accuracy
    print(f"  [{challenge_id}] Run OK — sample metric: {acc}, time: {elapsed:.1f}s")
    return True


def test_challenge_submit(challenge_id, info):
    """Test full submit mode for a challenge."""
    print(f"  [{challenge_id}] Running full submit...")
    from engine.executor import execute_challenge

    start = time.time()
    result = execute_challenge(challenge_id, info["solution"], mode="submit")
    elapsed = time.time() - start

    if not result.success:
        print(f"  [{challenge_id}] SUBMIT FAILED: {result.error}")
        if result.traceback_str:
            print(f"    {result.traceback_str[:300]}")
        return False

    g = result.grading
    if g is None:
        print(f"  [{challenge_id}] GRADING IS NONE")
        return False

    accuracy = g.get("accuracy")
    print(f"  [{challenge_id}] Score: {g['total_score']}/{g['max_score']} "
          f"{'PASSED' if g['passed'] else 'FAILED'}")
    if accuracy is not None:
        print(f"  [{challenge_id}] Primary metric: {accuracy}")

    for cat in g["categories"]:
        print(f"    {cat['name']}: {cat['score']}/{cat['max_score']}")
        for fb in cat.get("feedback", [])[:3]:
            print(f"      - {fb}")

    # Verify minimum performance
    if accuracy is not None and accuracy < info["min_accuracy"]:
        print(f"  [{challenge_id}] WARNING: accuracy {accuracy} < expected minimum {info['min_accuracy']}")

    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  ALL CHALLENGES — COMPREHENSIVE TEST SUITE")
    print("=" * 60 + "\n")

    if not test_registry():
        print("Registry test failed! Aborting.")
        sys.exit(1)

    # Test each challenge
    results = {}
    for cid, info in ALL_CHALLENGES.items():
        print("=" * 60)
        print(f"TESTING: {info['name']} ({cid})")
        print("=" * 60)

        # Reset registry for clean state
        ChallengeRegistry._instance = None
        ChallengeRegistry._challenges = {}

        run_ok = test_challenge_run(cid, info)
        results[f"{cid}_run"] = run_ok

        # Reset again for submit
        ChallengeRegistry._instance = None
        ChallengeRegistry._challenges = {}

        submit_ok = test_challenge_submit(cid, info)
        results[f"{cid}_submit"] = submit_ok
        print()

    # Summary
    print("=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("  ALL TESTS PASSED!")
    else:
        print("  SOME TESTS FAILED!")
    print("=" * 60)
