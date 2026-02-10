import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score


def train_sentiment_classifier(df_train):
    """
    Train a sentiment classifier on IMDB movie reviews.

    Args:
        df_train: pandas DataFrame with columns:
            - 'text': movie review text (str)
            - 'sentiment': 0 (negative) or 1 (positive)

    Returns:
        predict_fn: a callable that takes a DataFrame with 'text' column
                    and returns an array of predictions (0 or 1)

    Target: >= 89% accuracy on the hidden test set
    """
    # TODO: Implement your solution here
    #
    # Suggested steps:
    # 1. Preprocess text (lowercase, remove HTML tags, handle punctuation)
    # 2. Split data for local validation
    # 3. Vectorize text (e.g., TfidfVectorizer with n-grams)
    # 4. Train a classifier
    # 5. Return a prediction function
    #
    # Example structure:
    # -----------------
    # from sklearn.linear_model import LogisticRegression
    #
    # texts = df_train['text'].values
    # labels = df_train['sentiment'].values
    #
    # vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
    # X_train = vectorizer.fit_transform(texts)
    #
    # model = LogisticRegression(max_iter=1000)
    # model.fit(X_train, labels)
    #
    # def predict(df_test):
    #     X = vectorizer.transform(df_test['text'].values)
    #     return model.predict(X)
    #
    # return predict

    pass
