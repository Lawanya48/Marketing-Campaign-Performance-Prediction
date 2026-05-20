
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
import os

os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
# =========================
# LOAD DATASET
# =========================

combined_df = pd.read_csv(
    "data/processed/feature_engineered_data.csv"
)
print(combined_df.head())

print(combined_df.shape)

# =========================
# FEATURES & TARGET
# =========================

# Remove leakage + string columns
X = combined_df.drop([

'campaign_id',
'channel_used',
'date',

# classification target
'profit_flag',

# remove leakage
'roi'

], axis=1, errors='ignore')

# Keep only numeric columns

X = X.select_dtypes(
include=['int64', 'float64']
)

# Target column
y = combined_df['profit_flag']

print("\nFeature Columns:\n")
print(X.columns)

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# MODELS
# =========================

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier()
}

classification_results = []

# =========================
# TRAINING
# =========================

for name, model in models.items():

    print(f"\nTraining {name}...\n")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    classification_results.append([
        name,
        accuracy,
        precision,
        recall,
        f1
    ])

    print(classification_report(y_test, predictions))

# =========================
# SAVE RESULTS
# =========================

classification_df = pd.DataFrame(
    classification_results,
    columns=[
        'Model',
        'Accuracy',
        'Precision',
        'Recall',
        'F1_Score'
    ]
)

classification_df.to_csv(
    "data/processed/classification_predictions.csv",
    index=False
)

# =========================
# SAVE BEST MODEL
# =========================

best_classifier = RandomForestClassifier()

best_classifier.fit(X_train, y_train)

X_columns = X.columns.tolist()

joblib.dump(
X_columns,
"models/classification_features.pkl"
)

joblib.dump(
    best_classifier,
    "models/classification_model.pkl"
)

print("\nClassification completed successfully")

