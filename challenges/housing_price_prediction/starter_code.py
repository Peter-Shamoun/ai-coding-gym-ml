import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error


def train_price_predictor(df_train):
    """
    Train a housing price predictor on the California Housing dataset.

    Args:
        df_train: pandas DataFrame with 8 feature columns + 'MedHouseVal' target
            Features: MedInc, HouseAge, AveRooms, AveBedrms,
                      Population, AveOccup, Latitude, Longitude
            Target: MedHouseVal (median house value in $100,000s)

    Returns:
        predict_fn: a callable that takes a DataFrame (without 'MedHouseVal')
                    and returns predicted prices (array of floats)

    Target: R² >= 0.82 AND RMSE <= 0.55
    """
    # TODO: Implement your solution here
    #
    # This is REGRESSION, not classification!
    # Predict continuous house values, not categories.
    #
    # Key challenges:
    # 1. Feature scaling is important for many models
    # 2. Outliers in AveRooms, AveOccup, Population
    # 3. Geographic features (Lat/Long) encode location
    # 4. Simple linear regression only gets R² ~ 0.60
    #
    # Example structure:
    # -----------------
    # from sklearn.ensemble import GradientBoostingRegressor
    # from sklearn.preprocessing import StandardScaler
    #
    # features = df_train.drop('MedHouseVal', axis=1)
    # target = df_train['MedHouseVal']
    #
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(features)
    #
    # model = GradientBoostingRegressor(n_estimators=200)
    # model.fit(X_scaled, target)
    #
    # def predict(df_test):
    #     X = scaler.transform(df_test)
    #     return model.predict(X)
    #
    # return predict

    pass
