import pandas as pd
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

# =============================
# CREATE FOLDERS
# =============================

os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)

# =============================
# LOAD DATASETS
# =============================

nykaa = pd.read_csv("data/raw/nykaa_campaign_data_with_nulls.csv")
purplle = pd.read_csv("data/raw/purplle_campaign_data_with_nulls.csv")
tira = pd.read_csv("data/raw/tira_campaign_data_with_nulls.csv")

# =============================
# ADD BRAND COLUMN
# =============================

nykaa['Brand'] = 'Nykaa'
purplle['Brand'] = 'Purplle'
tira['Brand'] = 'Tira'

# =============================
# COMBINE DATASETS
# =============================

combined_df = pd.concat([
    nykaa,
    purplle,
    tira
], ignore_index=True)

print("Combined Shape:", combined_df.shape)

# Save combined dataset
combined_df.to_csv(
    "data/processed/combined_campaign_data.csv",
    index=False
)
# =============================
# REMOVE DUPLICATES
# =============================

combined_df.drop_duplicates(inplace=True)

# =============================
# STANDARDIZE COLUMN NAMES
# =============================

combined_df.columns = combined_df.columns.str.strip()
combined_df.columns = combined_df.columns.str.lower()
combined_df.columns = combined_df.columns.str.replace(' ', '_')

print("Columns:")
print(combined_df.columns)

# =============================
# DATE CONVERSION
# =============================

if 'date' in combined_df.columns:
    combined_df['date'] = pd.to_datetime(
        combined_df['date'],
        errors='coerce'
    )

# =============================
# CHECK NULL VALUES
# =============================

print("Missing Values Before Cleaning:")
print(combined_df.isnull().sum())
# =============================
# NUMERICAL & CATEGORICAL COLUMNS
# =============================

numerical_cols = [
    'duration',
    'impressions',
    'clicks',
    'leads',
    'conversions',
    'revenue',
    'acquisition_cost',
    'roi',
    'engagement_score'
]

categorical_cols = [
    'campaign_type',
    'target_audience',
    'channel_used',
    'language',
    'customer_segment',
    'brand'
]

# Keep only existing columns
numerical_cols = [
    col for col in numerical_cols
    if col in combined_df.columns
]

categorical_cols = [
    col for col in categorical_cols
    if col in combined_df.columns
]

# =============================
# HANDLE NULL VALUES
# =============================

# Numerical columns
num_imputer = SimpleImputer(strategy='median')
combined_df[numerical_cols] = num_imputer.fit_transform(
    combined_df[numerical_cols]
)

# Categorical columns
cat_imputer = SimpleImputer(strategy='most_frequent')
combined_df[categorical_cols] = cat_imputer.fit_transform(
    combined_df[categorical_cols]
)
# =============================
# CHECK NULL VALUES AFTER CLEANING
# =============================

print("Missing Values After Cleaning:")
print(combined_df.isnull().sum())

# =============================
# VALIDATE ROI
# =============================

if 'revenue' in combined_df.columns and 'acquisition_cost' in combined_df.columns:

    combined_df['calculated_roi'] = np.where(
        combined_df['acquisition_cost'] != 0,
        (
            (
                combined_df['revenue'] -
                combined_df['acquisition_cost']
            ) /
            combined_df['acquisition_cost']
        ) * 100,
        0
    )

# =============================
# OUTLIER HANDLING USING IQR
# =============================

for col in numerical_cols:

    Q1 = combined_df[col].quantile(0.25)
    Q3 = combined_df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    combined_df[col] = np.where(
        combined_df[col] < lower_bound,
        lower_bound,
        combined_df[col]
    )

    combined_df[col] = np.where(
        combined_df[col] > upper_bound,
        upper_bound,
        combined_df[col]
    )

# =============================
# FEATURE ENGINEERING
# =============================

# Profit Flag
combined_df['profit_flag'] = np.where(
    combined_df['roi'] > 0,
    1,
    0
)
# CTR
combined_df['ctr'] = np.where(
    combined_df['impressions'] != 0,
    (
        combined_df['clicks'] /
        combined_df['impressions']
    ) * 100,
    0
)

# Conversion Rate
combined_df['conversion_rate'] = np.where(
    combined_df['clicks'] != 0,
    (
        combined_df['conversions'] /
        combined_df['clicks']
    ) * 100,
    0
)

# Cost Per Lead
combined_df['cost_per_lead'] = np.where(
    combined_df['leads'] != 0,
    (
        combined_df['acquisition_cost'] /
        combined_df['leads']
    ),
    0
)

# Revenue Per Click
combined_df['revenue_per_click'] = np.where(
    combined_df['clicks'] != 0,
    (
        combined_df['revenue'] /
        combined_df['clicks']
    ),
    0
)
# =============================

if 'channel_used' in combined_df.columns:

    combined_df['channel_used'] = combined_df['channel_used'].astype(str)

    combined_df['channel_used'] = combined_df[
        'channel_used'
    ].apply(lambda x: x.split(','))

    mlb = MultiLabelBinarizer()

    encoded_channels = pd.DataFrame(
        mlb.fit_transform(combined_df['channel_used']),
        columns=mlb.classes_
    )

    combined_df = pd.concat([
        combined_df,
        encoded_channels
    ], axis=1)

# =============================
# LABEL ENCODING
# =============================

label_columns = [
    'campaign_type',
    'target_audience',
    'language',
    'customer_segment',
    'brand'
]

encoder = LabelEncoder()

for col in label_columns:

    if col in combined_df.columns:

        combined_df[col] = encoder.fit_transform(
            combined_df[col].astype(str)
        )
# =============================
# SAVE OUTPUT FILES
# =============================

combined_df.to_csv(
    "data/processed/cleaned_data.csv",
    index=False
)

combined_df.to_csv(
    "data/processed/feature_engineered_data.csv",
    index=False
)

combined_df.to_csv(
    "data/processed/encoded_data.csv",
    index=False
)

combined_df.to_csv(
    "data/processed/outlier_handled_data.csv",
    index=False
)

print("All preprocessing completed successfully")
print("Final Shape:", combined_df.shape)

print("Saved Files:")
print("1. combined_campaign_data.csv")
print("2. cleaned_data.csv")
print("3. feature_engineered_data.csv")
print("4. encoded_data.csv")
print("5. outlier_handled_data.csv")

# IQR Outlier Treatment
for col in numerical_cols:
    Q1 = combined_df[col].quantile(0.25)
    Q3 = combined_df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    combined_df[col] = np.where(combined_df[col] < lower, lower, combined_df[col])
    combined_df[col] = np.where(combined_df[col] > upper, upper, combined_df[col])

combined_df.to_csv("data/processed/outlier_handled_data.csv", index=False)

print("Outlier handling completed")
