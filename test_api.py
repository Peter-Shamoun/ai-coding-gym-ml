"""Quick API test script for the new backend."""

import json
import time
import urllib.request

BASE = "http://localhost:5050/api"

# A minimal working solution for testing
TEST_CODE = '''
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_spam_detector(df):
    X_train, X_val, y_train, y_val = train_test_split(
        df["text"], df["label_num"], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    model = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    model.fit(X_train_vec, y_train)

    val_acc = accuracy_score(y_val, model.predict(X_val_vec))
    print(f"Validation accuracy: {val_acc:.4f}")

    def predict(texts):
        X = vectorizer.transform(texts)
        return model.predict(X).tolist()

    return predict
'''


def api_call(method, path, data=None):
    url = f"{BASE}{path}"
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, method=method)
    if body:
        req.add_header("Content-Type", "application/json")
    try:
        resp = urllib.request.urlopen(req, timeout=600)
        return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_list():
    print("\n=== GET /challenges ===")
    status, data = api_call("GET", "/challenges")
    print(f"  Status: {status}")
    print(f"  Challenges: {len(data['challenges'])}")
    for c in data["challenges"]:
        print(f"    - {c['id']} ({c['difficulty']}): {c['title']}")


def test_detail():
    print("\n=== GET /challenges/email-spam-detection ===")
    status, data = api_call("GET", "/challenges/email-spam-detection")
    print(f"  Status: {status}")
    print(f"  Title: {data['title']}")
    print(f"  Starter code length: {len(data['starter_code'])} chars")
    print(f"  Scoring categories: {len(data['scoring']['categories'])}")


def test_run():
    print("\n=== POST /challenges/email-spam-detection/run ===")
    print("  (running user code on training sample...)")
    start = time.time()
    status, data = api_call("POST", "/challenges/email-spam-detection/run", {"code": TEST_CODE})
    elapsed = time.time() - start
    print(f"  Status: {status}")
    print(f"  Success: {data.get('success')}")
    print(f"  Sample accuracy: {data.get('sample_accuracy')}")
    print(f"  Train time: {data.get('train_time')}s")
    print(f"  Execution time: {data.get('execution_time')}s")
    print(f"  Wall time: {elapsed:.1f}s")
    if data.get("stdout"):
        print(f"  Stdout: {data['stdout'].strip()}")
    if data.get("error"):
        print(f"  Error: {data['error']}")


def test_submit():
    print("\n=== POST /challenges/email-spam-detection/submit ===")
    print("  (full grading on hidden test set...)")
    start = time.time()
    status, data = api_call("POST", "/challenges/email-spam-detection/submit", {"code": TEST_CODE})
    elapsed = time.time() - start
    print(f"  Status: {status}")
    print(f"  Success: {data.get('success')}")
    print(f"  Execution time: {data.get('execution_time')}s")
    print(f"  Wall time: {elapsed:.1f}s")
    if data.get("stdout"):
        print(f"  Stdout: {data['stdout'].strip()}")
    if data.get("error"):
        print(f"  Error: {data['error']}")
    if data.get("grading"):
        g = data["grading"]
        print(f"\n  --- Grading Results ---")
        print(f"  Total: {g['total_score']}/{g['max_score']}  {'PASSED' if g['passed'] else 'FAILED'}")
        print(f"  Accuracy: {g.get('accuracy')}")
        print(f"  Precision: {g.get('precision_spam')}")
        print(f"  Recall: {g.get('recall_spam')}")
        for cat in g.get("categories", []):
            print(f"    [{cat['score']}/{cat['max_score']}] {cat['name']}")
            for fb in cat.get("feedback", []):
                print(f"           {fb}")


if __name__ == "__main__":
    test_list()
    test_detail()
    test_run()
    test_submit()
    print("\nAll tests completed successfully.")
