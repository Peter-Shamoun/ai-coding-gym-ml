"""
VERSION 2 - Basic Cleaning
===========================
Changes from v1:
- Added text preprocessing: lowercase, remove punctuation, remove numbers
- Added stop_words='english' to filter common words
- Limited vocabulary (max_features=1000) to focus on most informative terms
- Same model family (MultinomialNB)

Why this helps: Cleaning normalizes case so "FREE" and "free" are the same.
Limiting vocabulary + stop word removal forces the model to focus on the
most discriminative terms rather than being diluted by rare/common words.
Expected accuracy: ~91-92%
"""

import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def clean_text(text):
    """Basic text preprocessing."""
    text = text.lower()                              # Lowercase
    text = re.sub(r'\d+', '', text)                  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()         # Collapse whitespace
    return text


# Load data
train = pd.read_csv("spam_train.csv")
test = pd.read_csv("spam_test.csv")

# NEW: Apply text cleaning before vectorizing
train["clean_text"] = train["text"].apply(clean_text)
test["clean_text"] = test["text"].apply(clean_text)

X_train, y_train = train["clean_text"], train["label_num"]
X_test, y_test = test["clean_text"], test["label_num"]

# NEW: Better vectorizer params — stop words + limited vocabulary
vectorizer = TfidfVectorizer(
    stop_words='english',   # Remove common English words
    max_features=1000,       # Keep only top 1000 terms (reduces noise)
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Same model as v1
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f"VERSION 2 - Basic Cleaning")
print(f"=" * 40)
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print()
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print()
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
