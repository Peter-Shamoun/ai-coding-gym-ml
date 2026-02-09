"""
VERSION 3 - Better Features
=============================
Changes from v2:
- Separate subject and body from the email text
- Create TF-IDF features for each, weight subject higher (subjects are
  highly indicative of spam — e.g. "Subject: free viagra")
- Use class_weight='balanced' via complement NB or switch to a model
  that supports it (SGDClassifier) to handle 71/29 ham/spam imbalance

Why this helps: Subject lines are short but packed with spam signals.
Weighting them separately prevents them from being drowned out by
longer body text. Handling class imbalance ensures the model doesn't
just predict "ham" for borderline cases.
Expected accuracy: ~93-94%
"""

import re
import string
import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def clean_text(text):
    """Basic text preprocessing."""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def split_subject_body(text):
    """Extract subject line and body from raw email text."""
    # Emails start with "Subject: ..." followed by body on the next line
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

# NEW: Split into subject and body, then clean each
for df in [train, test]:
    parts = df["text"].apply(split_subject_body)
    df["subject"] = parts.apply(lambda x: clean_text(x[0]))
    df["body"] = parts.apply(lambda x: clean_text(x[1]))

# NEW: Separate vectorizers for subject and body
subject_vec = TfidfVectorizer(max_features=5000)
body_vec = TfidfVectorizer(max_features=10000)

X_train_subj = subject_vec.fit_transform(train["subject"])
X_test_subj = subject_vec.transform(test["subject"])

X_train_body = body_vec.fit_transform(train["body"])
X_test_body = body_vec.transform(test["body"])

# NEW: Weight subject features 3x higher (subjects are very indicative)
SUBJECT_WEIGHT = 3.0
X_train_combined = hstack([X_train_subj * SUBJECT_WEIGHT, X_train_body])
X_test_combined = hstack([X_test_subj * SUBJECT_WEIGHT, X_test_body])

y_train = train["label_num"]
y_test = test["label_num"]

# NEW: SGDClassifier with class_weight='balanced' to handle imbalance
# (hinge loss = linear SVM, log_loss = logistic regression)
model = SGDClassifier(
    loss='modified_huber',    # Probability-calibrated SVM-like
    class_weight='balanced',  # Handle 71/29 imbalance
    random_state=42,
    max_iter=1000,
)
model.fit(X_train_combined, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_combined)

accuracy = accuracy_score(y_test, y_pred)
print(f"VERSION 3 - Better Features")
print(f"=" * 40)
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print()
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print()
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
