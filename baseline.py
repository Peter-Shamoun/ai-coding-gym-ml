"""
VERSION 1 - Naive Baseline
===========================
Simple approach: TfidfVectorizer on raw text + MultinomialNB
- No text preprocessing
- No handling of class imbalance
- No feature engineering
This is what a "one-shot" attempt typically produces.
Expected accuracy: ~88-90%
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data
train = pd.read_csv("spam_train.csv")
test = pd.read_csv("spam_test.csv")

X_train, y_train = train["text"], train["label_num"]
X_test, y_test = test["text"], test["label_num"]

# Vectorize raw text — no preprocessing at all
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a simple Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f"VERSION 1 - Naive Baseline")
print(f"=" * 40)
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print()
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print()
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
