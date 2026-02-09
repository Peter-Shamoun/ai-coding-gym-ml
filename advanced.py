"""
VERSION 5 - Advanced (Deep Learning)
======================================
Changes from v4:
- Uses a bidirectional LSTM with learned word embeddings
- More sophisticated preprocessing (keeps some spam signal chars)
- Handles sequence padding for variable-length emails
- Uses class weights in the loss function

Why this helps: Neural networks can learn non-linear feature interactions
and contextual patterns that linear models miss. The LSTM captures
sequential dependencies in email text.
Expected accuracy: 96-98%

Requirements: pip install tensorflow
(Falls back to the v4 ensemble if tensorflow is unavailable)
"""

import re
import string
import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

warnings.filterwarnings("ignore")

# Load data
train = pd.read_csv("spam_train.csv")
test = pd.read_csv("spam_test.csv")


def clean_text_light(text):
    """Lighter preprocessing — keep some structure for the neural net."""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' urltoken ', text)
    text = re.sub(r'\S+@\S+', ' emailtoken ', text)
    text = re.sub(r'\d+', ' numtoken ', text)
    # Keep some punctuation (! and $) as they're spam signals
    text = re.sub(r'[^\w\s!$]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


train["clean"] = train["text"].apply(clean_text_light)
test["clean"] = test["text"].apply(clean_text_light)

y_train = train["label_num"].values
y_test = test["label_num"].values

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Embedding, Bidirectional, LSTM, Dense, Dropout, GlobalMaxPooling1D
    )
    from tensorflow.keras.callbacks import EarlyStopping

    USE_DL = True
    print("TensorFlow found — using LSTM model")
except ImportError:
    USE_DL = False
    print("TensorFlow not found — falling back to sklearn ensemble")

if USE_DL:
    # ---- Tokenize and pad sequences ----
    MAX_WORDS = 20000   # Vocabulary size
    MAX_LEN = 300       # Max sequence length (words)
    EMBED_DIM = 64      # Embedding dimensions

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(train["clean"])

    X_train_seq = tokenizer.texts_to_sequences(train["clean"])
    X_test_seq = tokenizer.texts_to_sequences(test["clean"])

    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post')

    # ---- Compute class weights for imbalanced data ----
    n_ham = (y_train == 0).sum()
    n_spam = (y_train == 1).sum()
    total = len(y_train)
    class_weights = {
        0: total / (2 * n_ham),
        1: total / (2 * n_spam),
    }
    print(f"Class weights: {class_weights}")

    # ---- Build LSTM model ----
    model = Sequential([
        Embedding(MAX_WORDS, EMBED_DIM, input_length=MAX_LEN),
        Bidirectional(LSTM(64, return_sequences=True)),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    model.summary()

    # ---- Train with early stopping ----
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train_pad, y_train,
        epochs=15,
        batch_size=32,
        validation_split=0.1,
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=1,
    )

    # ---- Predict ----
    y_pred_prob = model.predict(X_test_pad).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)

else:
    # Fallback: enhanced sklearn ensemble (same as v4 but tuned further)
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV

    def split_subject_body(text):
        match = re.match(r'^Subject:\s*(.*?)(?:\n|$)(.*)', text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return "", text.strip()

    for df in [train, test]:
        parts = df["text"].apply(split_subject_body)
        df["subject"] = parts.apply(lambda x: clean_text_light(x[0]))
        df["body"] = parts.apply(lambda x: clean_text_light(x[1]))

    subj_vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    body_vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 3), sublinear_tf=True, min_df=2)

    X_train_s = subj_vec.fit_transform(train["subject"])
    X_test_s = subj_vec.transform(test["subject"])
    X_train_b = body_vec.fit_transform(train["body"])
    X_test_b = body_vec.transform(test["body"])

    X_train_combined = hstack([X_train_s * 3.0, X_train_b])
    X_test_combined = hstack([X_test_s * 3.0, X_test_b])

    ensemble = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)),
            ('svc', CalibratedClassifierCV(
                LinearSVC(class_weight='balanced', C=0.5, max_iter=3000, random_state=42), cv=3
            )),
            ('sgd', SGDClassifier(loss='log_loss', class_weight='balanced',
                                  alpha=1e-4, max_iter=1000, random_state=42)),
        ],
        voting='soft',
    )
    ensemble.fit(X_train_combined, y_train)
    y_pred = ensemble.predict(X_test_combined)


# ---- Evaluate ----
accuracy = accuracy_score(y_test, y_pred)
print()
print(f"VERSION 5 - Advanced {'(LSTM)' if USE_DL else '(Ensemble Fallback)'}")
print(f"=" * 40)
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print()
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print()
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
