import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score


def train_spam_detector(df):
    """
    Train a spam detector on the Enron dataset.

    Args:
        df: pandas DataFrame with columns:
            - 'text': raw email content (str)
            - 'label': 'spam' or 'ham' (str)
            - 'label_num': 1 (spam) or 0 (ham) (int)

    Returns:
        predict_fn: a callable that takes a list of email text strings
                    and returns a list of predictions (0=ham, 1=spam)

    Target: >= 93% accuracy on the hidden test set
    """
    # TODO: Implement your solution here
    #
    # Suggested steps:
    # 1. Preprocess text (lowercase, remove special chars, etc.)
    # 2. Split data for local validation
    # 3. Vectorize text (e.g., TfidfVectorizer)
    # 4. Train a classifier (e.g., LogisticRegression, SVM, etc.)
    # 5. Evaluate locally
    # 6. Return a prediction function
    #
    # Example structure:
    # -----------------
    # X_train, X_val, y_train, y_val = train_test_split(
    #     df['text'], df['label_num'], test_size=0.2, random_state=42
    # )
    #
    # vectorizer = TfidfVectorizer(max_features=10000)
    # X_train_vec = vectorizer.fit_transform(X_train)
    # X_val_vec = vectorizer.transform(X_val)
    #
    # model = SomeClassifier()
    # model.fit(X_train_vec, y_train)
    #
    # val_acc = accuracy_score(y_val, model.predict(X_val_vec))
    # print(f"Validation accuracy: {val_acc:.4f}")
    #
    # def predict(texts):
    #     X = vectorizer.transform(texts)
    #     return model.predict(X).tolist()
    #
    # return predict

    pass
