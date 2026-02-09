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
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    text = re.sub(r"\S+@\S+", " email ", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
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
