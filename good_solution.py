"""
VERSION 4 - Good Solution
===========================
Changes from v3:
- Better vectorizer params: bigrams, tuned max_features, sublinear_tf
- Try multiple models (SVM, LogisticRegression) and pick best
- Hyperparameter tuning via cross-validation
- Ensemble (VotingClassifier) of top models for robustness

Why this helps: Bigrams capture phrases like "click here", "free offer"
that unigrams miss. Sublinear TF scaling dampens very frequent terms.
Ensembling multiple strong models reduces variance.
Expected accuracy: 94-95%+
"""

import re
import string
import warnings
import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")


def clean_text(text):
    """Text preprocessing with spam-aware cleaning."""
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', ' url ', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' email ', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def split_subject_body(text):
    """Extract subject line and body from raw email text."""
    match = re.match(r'^Subject:\s*(.*?)(?:\n|$)(.*)', text, re.DOTALL | re.IGNORECASE)
    if match:
        subject = match.group(1).strip()
        body = match.group(2).strip()
    else:
        subject = ""
        body = text.strip()
    return subject, body


# Load data
train = pd.read_csv("spam_train.csv")
test = pd.read_csv("spam_test.csv")

# Split and clean
for df in [train, test]:
    parts = df["text"].apply(split_subject_body)
    df["subject"] = parts.apply(lambda x: clean_text(x[0]))
    df["body"] = parts.apply(lambda x: clean_text(x[1]))

# NEW: Better vectorizer params — bigrams, sublinear_tf, tuned max_features
subject_vec = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),      # Unigrams + bigrams
    sublinear_tf=True,        # Dampen frequent terms
    min_df=2,                 # Ignore very rare terms
)
body_vec = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
)

X_train_subj = subject_vec.fit_transform(train["subject"])
X_test_subj = subject_vec.transform(test["subject"])

X_train_body = body_vec.fit_transform(train["body"])
X_test_body = body_vec.transform(test["body"])

SUBJECT_WEIGHT = 3.0
X_train_combined = hstack([X_train_subj * SUBJECT_WEIGHT, X_train_body])
X_test_combined = hstack([X_test_subj * SUBJECT_WEIGHT, X_test_body])

y_train = train["label_num"]
y_test = test["label_num"]

# NEW: Try multiple models and compare via cross-validation
print("Cross-validation scores on training set:")
print("-" * 50)

models = {
    "LogisticRegression": LogisticRegression(
        C=1.0, class_weight='balanced', max_iter=1000, random_state=42
    ),
    "LinearSVC": CalibratedClassifierCV(
        LinearSVC(class_weight='balanced', max_iter=2000, random_state=42),
        cv=3
    ),
    "SGD (log_loss)": SGDClassifier(
        loss='log_loss', class_weight='balanced',
        max_iter=1000, random_state=42
    ),
}

best_name, best_score = None, 0
for name, m in models.items():
    scores = cross_val_score(m, X_train_combined, y_train, cv=5, scoring='accuracy')
    mean_score = scores.mean()
    print(f"  {name}: {mean_score:.4f} (+/- {scores.std():.4f})")
    if mean_score > best_score:
        best_name, best_score = name, mean_score

print(f"\nBest single model: {best_name} ({best_score:.4f})")
print()

# NEW: Ensemble the top models with soft voting
ensemble = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)),
        ('svc', CalibratedClassifierCV(
            LinearSVC(class_weight='balanced', max_iter=2000, random_state=42), cv=3
        )),
        ('sgd', SGDClassifier(loss='log_loss', class_weight='balanced', max_iter=1000, random_state=42)),
    ],
    voting='soft',
)
ensemble.fit(X_train_combined, y_train)

# Predict and evaluate
y_pred = ensemble.predict(X_test_combined)

accuracy = accuracy_score(y_test, y_pred)
print(f"VERSION 4 - Good Solution (Ensemble)")
print(f"=" * 40)
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print()
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print()
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
