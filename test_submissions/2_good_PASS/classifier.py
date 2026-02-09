"""Spam classifier - good quality."""
import re, pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def preprocess(text):
    """Basic cleaning: lowercase, remove numbers."""
    text = text.lower()
    text = re.sub(r"\d+", "", text)
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
