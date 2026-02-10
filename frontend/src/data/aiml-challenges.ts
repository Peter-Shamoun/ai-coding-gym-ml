import type { Challenge } from "./challenges-data";

export const aimlChallenges: Challenge[] = [
  {
    id: "aiml-001",
    backendId: "email-spam-detection",
    title: "Email Spam Detection",
    type: "AI-ML",
    difficulty: "Easy",
    status: "new",
    tags: ["Classification", "NLP", "Binary"],
    acceptance: 68,
    description:
      "Build a binary classifier to detect spam emails using the Enron Email dataset. Target: ≥93% accuracy.",
    problemStatement: `## Email Spam Detection

Classify emails as **spam** or **ham** (not spam) using the Enron Email Spam Dataset.

### Objective
Build a model that achieves **≥93% accuracy** on the held-out test set.

### Dataset
The Enron Email Spam Dataset contains labeled email messages. Each row has the raw email text and a binary label (spam / ham).`,
    testCases: [
      { input: "Email: 'Congratulations! You won $1M...'", output: "spam" },
      { input: "Email: 'Meeting at 3pm to discuss Q4 budget'", output: "ham" },
      { input: "Email: 'URGENT: Verify your account NOW'", output: "spam" },
    ],
    codeStub: `import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("enron_spam.csv")

def train_spam_detector(df):
    """
    Train a spam detector on the Enron dataset.
    Returns: trained model and accuracy on test set.
    Target: >= 93% accuracy
    """
    # TODO: Preprocess, vectorize, train, evaluate
    pass`,
    dataset: "Enron Email Spam Dataset",
    taskType: "Binary Classification (spam vs ham)",
    targetMetrics: "Accuracy ≥ 93%",
    dataDownloadUrl: "https://www.cs.cmu.edu/~enron/",
    validationScript: `from sklearn.metrics import accuracy_score, classification_report

def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["ham", "spam"])
    print(f"Accuracy: {acc:.4f}")
    print(report)
    return acc >= 0.93`,
    datasetSamples: [
      { features: "Subject: Meeting Tomorrow\\nHi team, let's sync at 10am...", label: "ham" },
      { features: "Subject: YOU WON!!!\\nClick here to claim your prize...", label: "spam" },
      { features: "Subject: Q3 Report\\nAttached is the quarterly summary...", label: "ham" },
    ],
    gradingRubric: "**Primary metric:** Accuracy on held-out test set (≥93% to pass).\n\nSecondary considerations:\n- Precision and recall balance\n- No data leakage between train/test\n- Reproducible results (set random seed)",
    deliverables: "A Python script that loads the dataset, trains a classifier, and prints the accuracy on a test split. The script should be self-contained and runnable.",
  },
  {
    id: "aiml-002",
    backendId: "mnist_digit_recognition",
    title: "Handwritten Digit Recognition",
    type: "AI-ML",
    difficulty: "Easy",
    status: "new",
    tags: ["Classification", "Computer Vision", "Multi-class"],
    acceptance: 72,
    description:
      "Build a multi-class image classifier to recognize handwritten digits (0–9) using MNIST. Target: ≥97% accuracy.",
    problemStatement: `## Handwritten Digit Recognition

Classify 28×28 grayscale images of handwritten digits (0–9) using the MNIST dataset.

### Objective
Achieve **≥97% accuracy** on the MNIST test set.

### Dataset
MNIST contains 60,000 training and 10,000 test images of handwritten digits. Each image is 28×28 pixels, grayscale.`,
    testCases: [
      { input: "Image of handwritten '7'", output: "7" },
      { input: "Image of handwritten '3'", output: "3" },
      { input: "Image of handwritten '0'", output: "0" },
    ],
    codeStub: `import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

def train_digit_recognizer(X_train, y_train, X_test, y_test):
    """
    Train a digit recognizer on MNIST.
    Target: >= 97% accuracy on test set.
    """
    # TODO: Preprocess, build model, train, evaluate
    pass`,
    dataset: "MNIST",
    taskType: "Multi-class Image Classification (digits 0–9)",
    targetMetrics: "Accuracy ≥ 97%",
    dataDownloadUrl: "https://yann.lecun.com/exdb/mnist/",
    validationScript: `from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    return acc >= 0.97`,
    datasetSamples: [
      { features: "28×28 grayscale image (pixel values 0-255)", label: "5" },
      { features: "28×28 grayscale image (pixel values 0-255)", label: "0" },
      { features: "28×28 grayscale image (pixel values 0-255)", label: "9" },
    ],
    gradingRubric: "**Primary metric:** Accuracy on MNIST test set (≥97% to pass).\n\nSecondary considerations:\n- Per-class accuracy (no digit should fall below 90%)\n- Model should be trained from scratch (no pretrained weights)\n- Reproducible results",
    deliverables: "A Python script that loads MNIST, trains a classifier, and reports test accuracy. Include the model architecture summary.",
  },
  {
    id: "aiml-003",
    backendId: "imdb_sentiment_analysis",
    title: "Sentiment Analysis on Movie Reviews",
    type: "AI-ML",
    difficulty: "Medium",
    status: "new",
    tags: ["NLP", "Classification", "Sentiment"],
    acceptance: 55,
    description:
      "Build a binary sentiment classifier on IMDB movie reviews. Target: ≥89% accuracy.",
    problemStatement: `## Sentiment Analysis on Movie Reviews

Classify IMDB movie reviews as **positive** or **negative** sentiment.

### Objective
Achieve **≥89% accuracy** on the IMDB test set.

### Dataset
The IMDB dataset contains 50,000 movie reviews (25K train, 25K test), evenly split between positive and negative.`,
    testCases: [
      { input: "\"This movie was absolutely fantastic, a masterpiece!\"", output: "positive" },
      { input: "\"Terrible acting, boring plot. Complete waste of time.\"", output: "negative" },
      { input: "\"A beautifully crafted film with stunning visuals.\"", output: "positive" },
    ],
    codeStub: `import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load data
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

def train_sentiment_model(X_train, y_train, X_test, y_test, max_len=200):
    """
    Train a sentiment classifier on IMDB reviews.
    Target: >= 89% accuracy on test set.
    """
    # TODO: Pad sequences, build model, train, evaluate
    pass`,
    dataset: "IMDB Movie Reviews",
    taskType: "Binary Sentiment Classification",
    targetMetrics: "Accuracy ≥ 89%",
    dataDownloadUrl: "https://ai.stanford.edu/~amaas/data/sentiment/",
    validationScript: `from sklearn.metrics import accuracy_score, f1_score

def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    return acc >= 0.89`,
    datasetSamples: [
      { features: "\"One of the best movies I have ever seen. The performances were incredible...\"", label: "positive" },
      { features: "\"I couldn't sit through this. The dialogue was painfully bad...\"", label: "negative" },
      { features: "\"A decent watch but nothing groundbreaking. Some good moments...\"", label: "positive" },
    ],
    gradingRubric: "**Primary metric:** Accuracy on IMDB test set (≥89% to pass).\n\nSecondary considerations:\n- F1 score should be reasonable (>0.85)\n- No data leakage\n- Proper text preprocessing pipeline",
    deliverables: "A Python script that loads the IMDB dataset, trains a sentiment classifier, and reports test accuracy and F1 score.",
  },
  {
    id: "aiml-004",
    backendId: "customer_churn_prediction",
    title: "Customer Churn Prediction",
    type: "AI-ML",
    difficulty: "Medium",
    status: "new",
    tags: ["Classification", "Tabular", "Imbalanced"],
    acceptance: 48,
    description:
      "Predict customer churn from the Telco dataset. Target: ≥85% accuracy AND F1 ≥ 0.70 on the churn class.",
    problemStatement: `## Customer Churn Prediction

Predict which customers are likely to churn using the Telco Customer Churn dataset.

### Objective
Achieve **≥85% accuracy** AND **F1 ≥ 0.70 on the churn class**.

### Dataset
The Telco Customer Churn dataset contains ~7,000 customer records with demographic info, account details, and service usage. The target is a binary churn indicator.

The dataset is imbalanced (~26% churn). Accuracy alone is misleading — a model predicting "no churn" for everything gets ~74% accuracy. You must balance precision and recall on the churn class.`,
    testCases: [
      { input: "Customer: 2-month tenure, high charges, no contract", output: "churn (high risk)" },
      { input: "Customer: 48-month tenure, low charges, 2-year contract", output: "no churn (low risk)" },
      { input: "Customer: 6-month tenure, fiber optic, month-to-month", output: "churn (medium risk)" },
    ],
    codeStub: `import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Load data
df = pd.read_csv("telco_churn.csv")

def train_churn_predictor(df):
    """
    Train a churn predictor on the Telco dataset.
    Target: >= 85% accuracy AND F1 >= 0.70 on churn class.
    """
    # TODO: Feature engineering, handle imbalance, train, evaluate
    pass`,
    dataset: "Telco Customer Churn",
    taskType: "Binary Classification (churn vs no churn)",
    targetMetrics: "Accuracy ≥ 85% AND F1 ≥ 0.70 on churn class",
    dataDownloadUrl: "https://www.kaggle.com/datasets/blastchar/telco-customer-churn",
    validationScript: `from sklearn.metrics import accuracy_score, f1_score, classification_report

def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1_churn = f1_score(y_true, y_pred, pos_label=1)
    report = classification_report(y_true, y_pred, target_names=["No Churn", "Churn"])
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 (Churn): {f1_churn:.4f}")
    print(report)
    return acc >= 0.85 and f1_churn >= 0.70`,
    datasetSamples: [
      { features: "Gender: Female, Tenure: 1, Contract: Month-to-month, MonthlyCharges: 29.85", label: "No" },
      { features: "Gender: Male, Tenure: 34, Contract: One year, MonthlyCharges: 56.95", label: "No" },
      { features: "Gender: Male, Tenure: 2, Contract: Month-to-month, MonthlyCharges: 53.85", label: "Yes" },
    ],
    gradingRubric: "**Primary metrics:** Both must be met:\n1. Overall accuracy ≥ 85%\n2. F1 score on the churn class ≥ 0.70\n\nAccuracy alone is insufficient — a naive 'no churn' predictor would fail the F1 requirement.\n\nSecondary considerations:\n- Proper handling of class imbalance\n- No data leakage from test set\n- Feature preprocessing documented",
    deliverables: "A Python script that loads the Telco dataset, trains a churn model, and reports accuracy, F1 (churn class), and a full classification report.",
  },
  {
    id: "aiml-005",
    backendId: "housing_price_prediction",
    title: "House Price Prediction",
    type: "AI-ML",
    difficulty: "Hard",
    status: "new",
    tags: ["Regression", "Tabular", "Feature Engineering"],
    acceptance: 35,
    description:
      "Predict house prices using the California Housing dataset. Target: R² ≥ 0.85 and RMSE within 15% of mean price.",
    problemStatement: `## House Price Prediction

Predict median house values using the California Housing dataset.

### Objective
Achieve **R² ≥ 0.85** and **RMSE within 15% of the mean price**.

### Dataset
The California Housing dataset contains ~20,000 block groups with features like median income, house age, average rooms, population, and geographic coordinates. The target is the median house value.`,
    testCases: [
      { input: "MedInc: 8.3, HouseAge: 41, AveRooms: 6.9, Lat: 37.88", output: "$452,600 (predicted)" },
      { input: "MedInc: 3.5, HouseAge: 30, AveRooms: 5.2, Lat: 34.05", output: "$156,200 (predicted)" },
      { input: "MedInc: 5.0, HouseAge: 15, AveRooms: 6.0, Lat: 38.56", output: "$228,400 (predicted)" },
    ],
    codeStub: `import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Load data
data = fetch_california_housing()
X, y = data.data, data.target

def train_price_predictor(X, y):
    """
    Train a house price predictor on California Housing.
    Target: R² >= 0.85 and RMSE within 15% of mean price.
    """
    # TODO: Feature engineering, train model, evaluate
    pass`,
    dataset: "California Housing",
    taskType: "Regression",
    targetMetrics: "R² ≥ 0.85 AND RMSE within 15% of mean price",
    dataDownloadUrl: "https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset",
    validationScript: `import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

def evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mean_price = np.mean(y_true)
    rmse_pct = (rmse / mean_price) * 100
    print(f"R²: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Mean Price: {mean_price:.4f}")
    print(f"RMSE as % of mean: {rmse_pct:.1f}%")
    return r2 >= 0.85 and rmse_pct <= 15.0`,
    datasetSamples: [
      { features: "MedInc: 8.3252, HouseAge: 41, AveRooms: 6.98, AveBedrms: 1.02, Population: 322", label: "4.526 ($452,600)" },
      { features: "MedInc: 8.3014, HouseAge: 21, AveRooms: 6.24, AveBedrms: 0.97, Population: 2401", label: "3.585 ($358,500)" },
      { features: "MedInc: 3.8462, HouseAge: 52, AveRooms: 5.32, AveBedrms: 1.07, Population: 558", label: "1.725 ($172,500)" },
    ],
    gradingRubric: "**Primary metrics:** Both must be met:\n1. R² score ≥ 0.85\n2. RMSE ≤ 15% of mean target price\n\nSecondary considerations:\n- Residual analysis (no systematic bias)\n- Proper train/test split\n- Feature engineering quality\n- Handling of outliers",
    deliverables: "A Python script that loads California Housing, engineers features, trains a regression model, and reports R², RMSE, and RMSE as percentage of mean price.",
  },
];
