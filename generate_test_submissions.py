"""
Generate test submission folders for manual upload testing.
Each folder contains the files you'd upload via the web UI:
    - classifier.py
    - model.pkl
    - predictions.csv

Run once:  python generate_test_submissions.py
Then browse to test_submissions/<tier>/ and upload those files.
"""

import os
import pickle
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "test_submissions")
TEST_CSV = os.path.join(PROJECT_ROOT, "spam_test.csv")


def load_truth():
    return pd.read_csv(TEST_CSV)["label_num"].values


def flip(y, n_errors, bias, rng):
    y = y.copy()
    if bias == "random":
        idx = rng.choice(len(y), n_errors, replace=False)
    elif bias == "ham_only":
        idx = rng.choice(np.where(y == 0)[0], n_errors, replace=False)
    elif bias == "spam_only":
        idx = rng.choice(np.where(y == 1)[0], n_errors, replace=False)
    else:
        idx = []
    y[idx] = 1 - y[idx]
    return y


def write(folder, classifier_code, model_valid, predictions, pred_col="prediction"):
    os.makedirs(folder, exist_ok=True)

    # classifier.py
    with open(os.path.join(folder, "classifier.py"), "w", encoding="utf-8") as f:
        f.write(classifier_code)

    # model.pkl
    pkl = os.path.join(folder, "model.pkl")
    if model_valid:
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.feature_extraction.text import TfidfVectorizer
        with open(pkl, "wb") as f:
            pickle.dump({"vec": TfidfVectorizer(), "clf": MultinomialNB()}, f)
    else:
        with open(pkl, "wb") as f:
            f.write(b"CORRUPT_NOT_A_PICKLE")

    # predictions.csv
    if predictions is not None:
        pd.DataFrame({pred_col: predictions}).to_csv(
            os.path.join(folder, "predictions.csv"), index=False
        )


# ── classifier.py templates ─────────────────────────────────────────────

EXCELLENT_CODE = '''\
"""Spam classifier - production quality."""
import re, string, pickle
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
    """Preprocess: lowercase, URL/email removal, strip punctuation."""
    text = text.lower()
    text = re.sub(r"http\\S+|www\\.\\S+", " url ", text)
    text = re.sub(r"\\S+@\\S+", " email ", text)
    text = re.sub(r"\\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\\s+", " ", text).strip()
    return text


subject_vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                               sublinear_tf=True, min_df=2, max_df=0.95)
body_vec = TfidfVectorizer(max_features=15000, ngram_range=(1, 2),
                            sublinear_tf=True, min_df=2)

ensemble = VotingClassifier(estimators=[
    ("lr", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)),
    ("svc", CalibratedClassifierCV(
        LinearSVC(class_weight="balanced", max_iter=2000), cv=3)),
    ("sgd", SGDClassifier(loss="log_loss", class_weight="balanced", max_iter=1000)),
], voting="soft")

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
'''

GOOD_CODE = '''\
"""Spam classifier - good quality."""
import re, pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def preprocess(text):
    """Basic cleaning: lowercase, remove numbers."""
    text = text.lower()
    text = re.sub(r"\\d+", "", text)
    return text.strip()


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
'''

MINIMAL_CODE = '''\
"""Spam classifier - minimal."""
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
'''

BARE_CODE = '''\
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
'''


# ── Generate all tiers ──────────────────────────────────────────────────

def main():
    if os.path.exists(OUTPUT_DIR):
        import shutil
        shutil.rmtree(OUTPUT_DIR)

    y = load_truth()
    rng = np.random.RandomState(42)
    n = len(y)  # 1035

    print(f"Ground truth: {n} samples ({(y==0).sum()} ham, {(y==1).sum()} spam)\n")

    # ─────────────────────────────────────────────────────────
    # 1) PASS - Excellent  (~100 pts)
    #    97% accuracy, full code quality, both P/R pass
    # ─────────────────────────────────────────────────────────
    preds = flip(y, int(round(n * 0.03)), "random", rng)
    write(os.path.join(OUTPUT_DIR, "1_excellent_PASS"), EXCELLENT_CODE, True, preds)
    print("  1_excellent_PASS         ->  ~100 pts  (should PASS)")

    # ─────────────────────────────────────────────────────────
    # 2) PASS - Good  (~80 pts)
    #    94% accuracy, good code quality
    # ─────────────────────────────────────────────────────────
    preds = flip(y, int(round(n * 0.06)), "random", rng)
    write(os.path.join(OUTPUT_DIR, "2_good_PASS"), GOOD_CODE, True, preds)
    print("  2_good_PASS              ->  ~80 pts   (should PASS)")

    # ─────────────────────────────────────────────────────────
    # 3) PASS - Satisfactory  (~83 pts)
    #    96% accuracy (compensates for low code quality)
    # ─────────────────────────────────────────────────────────
    preds = flip(y, int(round(n * 0.04)), "ham_only", rng)
    write(os.path.join(OUTPUT_DIR, "3_satisfactory_PASS"), MINIMAL_CODE, True, preds)
    print("  3_satisfactory_PASS      ->  ~83 pts   (should PASS, borderline)")

    # ─────────────────────────────────────────────────────────
    # 4) FAIL - Unsatisfactory  (~50 pts)
    #    90% accuracy, no code quality patterns
    # ─────────────────────────────────────────────────────────
    preds = flip(y, int(round(n * 0.10)), "random", rng)
    write(os.path.join(OUTPUT_DIR, "4_unsatisfactory_FAIL"), BARE_CODE, True, preds)
    print("  4_unsatisfactory_FAIL    ->  ~50 pts   (should FAIL)")

    # ─────────────────────────────────────────────────────────
    # 5) FAIL - Poor  (~20 pts)
    #    80% accuracy, below all accuracy tiers
    # ─────────────────────────────────────────────────────────
    preds = flip(y, int(round(n * 0.20)), "random", rng)
    write(os.path.join(OUTPUT_DIR, "5_poor_FAIL"), BARE_CODE, True, preds)
    print("  5_poor_FAIL              ->  ~20 pts   (should FAIL)")

    # ─────────────────────────────────────────────────────────
    # 6) FAIL - Broken Model  (~50 pts)
    #    model.pkl is corrupt, but predictions.csv is fine (91%)
    # ─────────────────────────────────────────────────────────
    preds = flip(y, int(round(n * 0.09)), "random", rng)
    write(os.path.join(OUTPUT_DIR, "6_broken_model_FAIL"), BARE_CODE, False, preds)
    print("  6_broken_model_FAIL      ->  ~50 pts   (should FAIL, corrupt model)")

    # ─────────────────────────────────────────────────────────
    # 7) FAIL - Wrong Format  (~20 pts)
    #    predictions.csv uses unrecognisable labels
    # ─────────────────────────────────────────────────────────
    bad = np.where(y == 1, "positive", "negative")
    write(os.path.join(OUTPUT_DIR, "7_wrong_format_FAIL"), BARE_CODE, True, bad)
    print("  7_wrong_format_FAIL      ->  ~20 pts   (should FAIL, bad labels)")

    # ─────────────────────────────────────────────────────────
    # 8) FAIL - Missing predictions  (~15 pts)
    #    classifier.py and model.pkl present, but no predictions.csv
    # ─────────────────────────────────────────────────────────
    write(os.path.join(OUTPUT_DIR, "8_no_predictions_FAIL"), BARE_CODE, True, None)
    print("  8_no_predictions_FAIL    ->  ~15 pts   (should FAIL, no predictions)")

    print(f"\nDone! Files are in: {OUTPUT_DIR}")
    print("\nTo test: upload the 3 files from any folder via the web UI.")


if __name__ == "__main__":
    main()
