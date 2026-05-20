# Profit/Loss Flag
combined_df['ctr'] = (
    combined_df['clicks'] /
    combined_df['impressions']
) * 100

# Conversion Rate
combined_df['conversion_rate'] = (
    combined_df['conversions'] /
    combined_df['clicks']
) * 100

# Cost Per Lead
combined_df['cost_per_lead'] = (
    combined_df['acquisition_cost'] /
    combined_df['leads']
)

# Revenue Per Click
combined_df['revenue_per_click'] = (
    combined_df['revenue'] /
    combined_df['clicks']
)

# Replace infinity
combined_df.replace([np.inf, -np.inf], 0, inplace=True)
combined_df.fillna(0, inplace=True)

# Multi-label encoding
combined_df['channel_used'] = combined_df['channel_used'].astype(str)
combined_df['channel_used'] = combined_df['channel_used'].apply(lambda x: x.split(','))

mlb = MultiLabelBinarizer()
channel_encoded = pd.DataFrame(
    mlb.fit_transform(combined_df['channel_used']),
    columns=mlb.classes_
)

combined_df = pd.concat([combined_df, channel_encoded], axis=1)

# Label encoding
label_cols = [
    'campaign_type',
    'target_audience',
    'language',
    'customer_segment',
    'brand'
]

encoder = LabelEncoder()

for col in label_cols:
    combined_df[col] = encoder.fit_transform(combined_df[col])

combined_df.to_csv("data/processed/feature_engineered_data.csv", index=False)

print("Feature engineering completed")