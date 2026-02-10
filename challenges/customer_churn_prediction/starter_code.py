import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def train_churn_predictor(df_train):
    """
    Train a customer churn predictor on the Telco dataset.

    Args:
        df_train: pandas DataFrame with customer features + 'Churn' column
            - Churn: "Yes" or "No" (target)
            - ~20 feature columns (demographics, account, services)
            - Mixed types: categorical + numerical
            - Watch for data quality issues (e.g., TotalCharges blanks)

    Returns:
        predict_fn: a callable that takes a DataFrame (without 'Churn' column)
                    and returns an array of predictions (0=no churn, 1=churn)

    Target: >= 85% accuracy AND F1 >= 0.70 on churn class
    """
    # TODO: Implement your solution here
    #
    # Key challenges:
    # 1. Class imbalance (~27% churn) - handle with class_weight or SMOTE
    # 2. Mixed feature types - encode categorical variables
    # 3. Data quality - TotalCharges may have blank values
    # 4. Feature engineering - interactions, binning, etc.
    #
    # Example structure:
    # -----------------
    # from sklearn.ensemble import GradientBoostingClassifier
    # from sklearn.preprocessing import LabelEncoder
    #
    # df = df_train.copy()
    # y = (df['Churn'] == 'Yes').astype(int)
    # df = df.drop(['Churn', 'customerID'], axis=1)
    #
    # # Encode categoricals
    # # ... your encoding logic ...
    #
    # model = GradientBoostingClassifier()
    # model.fit(X_processed, y)
    #
    # def predict(df_test):
    #     # Apply same preprocessing
    #     # ... your preprocessing ...
    #     return model.predict(X_test_processed)
    #
    # return predict

    pass
