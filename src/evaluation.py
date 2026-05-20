import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load regression results

regression_df = pd.read_csv(
"data/processed/model_metrics_summary.csv"
)

# Load classification results

classification_df = pd.read_csv(
"data/processed/classification_predictions.csv"
)

print(regression_df)

print(classification_df)

# =========================

# REGRESSION VISUALIZATION

# =========================

plt.figure(figsize=(10,5))

sns.barplot(
x='Model',
y='RMSE',
data=regression_df
)

plt.xticks(rotation=45)

plt.title("Regression Model Comparison")

plt.show()

# =========================

# CLASSIFICATION VISUALIZATION

# =========================

plt.figure(figsize=(10,5))

sns.barplot(
x='Model',
y='Accuracy',
data=classification_df
)

plt.xticks(rotation=45)

plt.title("Classification Model Accuracy")

plt.show()

# =========================

# BEST MODELS

# =========================

best_regression = regression_df.loc[
regression_df['R2'].idxmax()
]

best_classification = classification_df.loc[
classification_df['Accuracy'].idxmax()
]

print("\nBest Regression Model:")
print(best_regression)

print("\nBest Classification Model:")
print(best_classification)
