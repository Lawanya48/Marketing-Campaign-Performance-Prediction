import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import xgboost as xgb

# =========================
# LOAD DATASET
# =========================

combined_df = pd.read_csv(
    "data/processed/feature_engineered_data.csv"
)

print(combined_df.head())

print(combined_df.shape)
# Features and target
X = combined_df.drop([
    'campaign_id',
    'channel_used',
    'date',
    'revenue',
    'profit_flag'
], axis=1, errors='ignore')

# Keep only numeric columns
X = X.select_dtypes(include=['int64', 'float64'])

y = combined_df['revenue']


# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
# Models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'XGBoost': xgb.XGBRegressor()
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    results.append([
        name,
        mae,
        mse,
        rmse,
        r2
    ])

    print(name)
    print('MAE:', mae)
    print('RMSE:', rmse)
    print('R2:', r2)
    print()

# Save model comparison
results_df = pd.DataFrame(results, columns=[
    'Model',
    'MAE',
    'MSE',
    'RMSE',
    'R2'
])

results_df.to_csv("data/processed/model_metrics_summary.csv", index=False)

# Save best model
best_model = RandomForestRegressor()
best_model.fit(X_train, y_train)

X_columns = X.columns.tolist()

joblib.dump(
X_columns,
"models/regression_features.pkl"
)

joblib.dump(best_model, "models/regression_model.pkl")
# Prediction file
prediction_df = pd.DataFrame({
    'Actual_Revenue': y_test,
    'Predicted_Revenue': best_model.predict(X_test)
})

prediction_df.to_csv(
    "data/processed/regression_predictions.csv",
    index=False
)

print("Regression completed")