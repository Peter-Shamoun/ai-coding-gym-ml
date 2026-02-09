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
