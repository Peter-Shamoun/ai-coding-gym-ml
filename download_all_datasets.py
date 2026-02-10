"""
Download / generate datasets for all 3 new challenges:
  1. IMDB Sentiment Analysis  → imdb_train.csv, imdb_test.csv
  2. Customer Churn Prediction → churn_train.csv, churn_test.csv
  3. Housing Price Prediction  → housing_train.csv, housing_test.csv
"""

import os
import sys
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1. IMDB Sentiment Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def download_imdb():
    """Download IMDB reviews and save as CSV.
    Decodes keras word-index representation back to text."""
    out_dir = os.path.join(BASE, "challenges", "imdb_sentiment_analysis")
    train_path = os.path.join(out_dir, "imdb_train.csv")
    test_path = os.path.join(out_dir, "imdb_test.csv")

    if os.path.isfile(train_path) and os.path.isfile(test_path):
        print("[IMDB] Already exists, skipping.")
        return

    print("[IMDB] Downloading via keras...")
    from tensorflow.keras.datasets import imdb

    # Load all data (no vocab limit)
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=None)

    # Build reverse word index
    word_index = imdb.get_word_index()
    reverse_index = {v + 3: k for k, v in word_index.items()}
    reverse_index[0] = ""      # padding
    reverse_index[1] = ""      # start
    reverse_index[2] = ""      # unknown
    reverse_index[3] = ""      # unused

    def decode(seq):
        return " ".join(reverse_index.get(i, "") for i in seq).strip()

    print("[IMDB] Decoding reviews...")
    all_texts = [decode(s) for s in list(x_train) + list(x_test)]
    all_labels = list(y_train) + list(y_test)

    # Clean up: collapse multiple spaces
    import re
    all_texts = [re.sub(r"\s+", " ", t).strip() for t in all_texts]

    # Shuffle and split: 40k train, 10k test
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(all_texts))
    n_train = 40000

    train_idx = indices[:n_train]
    test_idx = indices[n_train:n_train + 10000]

    train_df = pd.DataFrame({
        "text": [all_texts[i] for i in train_idx],
        "sentiment": [all_labels[i] for i in train_idx],
    })
    test_df = pd.DataFrame({
        "text": [all_texts[i] for i in test_idx],
        "sentiment": [all_labels[i] for i in test_idx],
    })

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[IMDB] Train: {len(train_df)} ({train_df['sentiment'].mean():.2%} positive)")
    print(f"[IMDB] Test:  {len(test_df)} ({test_df['sentiment'].mean():.2%} positive)")
    print(f"[IMDB] Saved to {out_dir}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  2. Customer Churn Prediction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def download_churn():
    """Download Telco Churn dataset. Falls back to synthetic data."""
    out_dir = os.path.join(BASE, "challenges", "customer_churn_prediction")
    train_path = os.path.join(out_dir, "churn_train.csv")
    test_path = os.path.join(out_dir, "churn_test.csv")

    if os.path.isfile(train_path) and os.path.isfile(test_path):
        print("[CHURN] Already exists, skipping.")
        return

    df = None

    # Try downloading from known URLs
    urls = [
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
        "https://raw.githubusercontent.com/datasciencedojo/datasets/master/Telco%20Customer%20Churn.csv",
    ]

    for url in urls:
        try:
            print(f"[CHURN] Trying {url[:80]}...")
            df = pd.read_csv(url)
            if "Churn" in df.columns and len(df) > 5000:
                print(f"[CHURN] Downloaded {len(df)} rows")
                break
            df = None
        except Exception as e:
            print(f"[CHURN] Failed: {e}")
            df = None

    if df is None:
        print("[CHURN] Generating synthetic dataset...")
        df = _generate_synthetic_churn()

    # Clean up
    if "customerID" not in df.columns:
        df.insert(0, "customerID", [f"C{i:05d}" for i in range(len(df))])

    # Ensure Churn is Yes/No strings
    if df["Churn"].dtype != object:
        df["Churn"] = df["Churn"].map({1: "Yes", 0: "No", True: "Yes", False: "No"})

    # Shuffle and split 80/20
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_train = int(len(df) * 0.8)
    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_train:]

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    churn_rate = (train_df["Churn"] == "Yes").mean()
    print(f"[CHURN] Train: {len(train_df)} (churn rate: {churn_rate:.1%})")
    print(f"[CHURN] Test:  {len(test_df)}")
    print(f"[CHURN] Saved to {out_dir}")


def _generate_synthetic_churn():
    """Generate a synthetic telco churn dataset with realistic distributions."""
    rng = np.random.RandomState(42)
    n = 7043

    gender = rng.choice(["Male", "Female"], n)
    senior = rng.choice([0, 1], n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], n, p=[0.30, 0.70])
    tenure = rng.randint(0, 73, n)
    phone = rng.choice(["Yes", "No"], n, p=[0.90, 0.10])
    multi = np.where(
        phone == "No", "No phone service",
        rng.choice(["Yes", "No"], n, p=[0.42, 0.58])
    )
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    no_internet = internet == "No"

    def internet_service(p_yes=0.40):
        vals = rng.choice(["Yes", "No"], n, p=[p_yes, 1 - p_yes])
        vals[no_internet] = "No internet service"
        return vals

    online_sec = internet_service(0.29)
    online_bak = internet_service(0.34)
    dev_prot = internet_service(0.34)
    tech_sup = internet_service(0.29)
    streaming_tv = internet_service(0.38)
    streaming_mov = internet_service(0.39)

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24]
    )
    paperless = rng.choice(["Yes", "No"], n, p=[0.59, 0.41])
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)",
         "Credit card (automatic)"],
        n, p=[0.34, 0.23, 0.22, 0.21]
    )

    monthly = np.where(
        internet == "Fiber optic", rng.normal(80, 20, n).clip(18, 118),
        np.where(
            internet == "DSL", rng.normal(55, 15, n).clip(18, 90),
            rng.normal(22, 5, n).clip(18, 35)
        )
    ).round(2)

    total = (monthly * tenure + rng.normal(0, 50, n)).clip(0).round(2)
    # Introduce ~11 blank TotalCharges values (like real dataset)
    total_str = [str(t) for t in total]
    for i in rng.choice(n, 11, replace=False):
        total_str[i] = " "

    # Generate churn labels with realistic correlations
    churn_prob = np.full(n, 0.15)
    churn_prob[contract == "Month-to-month"] += 0.20
    churn_prob[internet == "Fiber optic"] += 0.10
    churn_prob[payment == "Electronic check"] += 0.08
    churn_prob[tenure < 12] += 0.15
    churn_prob[tenure > 48] -= 0.10
    churn_prob[senior == 1] += 0.05
    churn_prob[online_sec == "Yes"] -= 0.05
    churn_prob[tech_sup == "Yes"] -= 0.05
    churn_prob[contract == "Two year"] -= 0.10
    churn_prob = churn_prob.clip(0.02, 0.90)

    churn = np.array(["Yes" if rng.random() < p else "No" for p in churn_prob])

    df = pd.DataFrame({
        "customerID": [f"C{i:05d}" for i in range(n)],
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multi,
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_bak,
        "DeviceProtection": dev_prot,
        "TechSupport": tech_sup,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_mov,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total_str,
        "Churn": churn,
    })
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3. Housing Price Prediction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def download_housing():
    """Download California Housing dataset from sklearn."""
    out_dir = os.path.join(BASE, "challenges", "housing_price_prediction")
    train_path = os.path.join(out_dir, "housing_train.csv")
    test_path = os.path.join(out_dir, "housing_test.csv")

    if os.path.isfile(train_path) and os.path.isfile(test_path):
        print("[HOUSING] Already exists, skipping.")
        return

    print("[HOUSING] Loading California Housing from sklearn...")
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split

    data = fetch_california_housing(as_frame=True)
    df = data.frame  # DataFrame with feature columns + 'MedHouseVal'

    # 80/20 split → 16512 / 4128
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[HOUSING] Train: {len(train_df)} samples")
    print(f"[HOUSING] Test:  {len(test_df)} samples")
    print(f"[HOUSING] Target range: {train_df['MedHouseVal'].min():.2f} - {train_df['MedHouseVal'].max():.2f}")
    print(f"[HOUSING] Saved to {out_dir}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("=" * 60)
    print("  Downloading datasets for 3 new challenges")
    print("=" * 60)

    download_imdb()
    print()
    download_churn()
    print()
    download_housing()

    print()
    print("=" * 60)
    print("  All datasets ready!")
    print("=" * 60)
