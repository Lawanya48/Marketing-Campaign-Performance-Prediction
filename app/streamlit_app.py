import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# =========================

# PAGE CONFIG

# =========================

st.set_page_config(
page_title="Marketing Campaign Analytics Dashboard",
layout="wide"
)

# =========================

# LOAD DATASETS

# =========================

combined_df = pd.read_csv(
"data/processed/feature_engineered_data.csv"
)

regression_df = pd.read_csv(
"data/processed/model_metrics_summary.csv"
)

classification_df = pd.read_csv(
"data/processed/classification_predictions.csv"
)

cluster_df = pd.read_csv(
"data/processed/clustered_campaigns.csv"
)

# =========================

# LOAD MODELS

# =========================

regression_model = joblib.load(
"models/regression_model.pkl"
)

classification_model = joblib.load(
"models/classification_model.pkl"
)

# =========================

# LOAD FEATURE ORDERS

# =========================

regression_features = joblib.load(
"models/regression_features.pkl"
)

classification_features = joblib.load(
"models/classification_features.pkl"
)

# =========================

# SIDEBAR

# =========================

st.sidebar.title("Navigation")

page = st.sidebar.selectbox(
"Select Page",
[
"Home",
"EDA",
"Brand Analysis",
"Regression",
"Classification",
"Clustering",
"Prediction",
"Insights"
]
)

# =========================

# HOME PAGE

# =========================

if page == "Home":


    st.title("Marketing Campaign Analytics Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Campaigns",
        len(combined_df)
    )

    col2.metric(
        "Total Revenue",
        f"{combined_df['revenue'].sum():,.0f}"
    )

    col3.metric(
        "Average ROI",
        f"{combined_df['roi'].mean():.2f}"
    )

    col4.metric(
        "Total Conversions",
        f"{combined_df['conversions'].sum():,.0f}"
    )

    st.dataframe(combined_df.head())


# =========================

# EDA PAGE

# =========================

elif page == "EDA":

    st.title("Exploratory Data Analysis Dashboard")

    st.subheader("Marketing Campaign Insights")

# =========================
# KPI METRICS
# =========================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Revenue",
        f"{combined_df['revenue'].sum():,.0f}"
    )

    col2.metric(
        "Average ROI",
        f"{combined_df['roi'].mean():.2f}"
    )

    col3.metric(
        "Total Clicks",
        f"{combined_df['clicks'].sum():,.0f}"
    )

    col4.metric(
        "Total Conversions",
        f"{combined_df['conversions'].sum():,.0f}"
    )

# =========================
# REVENUE DISTRIBUTION
# =========================

    st.subheader("Revenue Distribution")

    fig1 = px.histogram(

        combined_df,

        x='revenue',
        nbins=50,
        color_discrete_sequence=['#00CC96']

    )

    st.plotly_chart(
        fig1,
        width='stretch'
    )

# =========================
# ROI DISTRIBUTION
# =========================

    st.subheader("ROI Distribution")

    fig2 = px.box(

        combined_df,

        y='roi',
        color_discrete_sequence=['#636EFA']

    )

    st.plotly_chart(
        fig2,
        width='stretch'
    )

# =========================
# CLICKS VS REVENUE
# =========================

    st.subheader("Clicks vs Revenue")

    fig3 = px.scatter(

        combined_df.sample(3000),

        x='clicks',
        y='revenue',
        color='roi',
        size='engagement_score',

        title='Clicks Impact on Revenue'

    )

    st.plotly_chart(
        fig3,
        width='stretch'
    )

# =========================
# CONVERSIONS VS REVENUE
# =========================

    st.subheader("Conversions vs Revenue")

    fig4 = px.scatter(

        combined_df.sample(3000),

        x='conversions',
        y='revenue',
        color='engagement_score',

        title='Conversions Impact on Revenue'

    )

    st.plotly_chart(
        fig4,
        width='stretch'
    )

# =========================
# CHANNEL PERFORMANCE
# =========================

    st.subheader("Marketing Channel Usage")

    channel_columns = [

        'Email',
        'Facebook',
        'Google',
        'Instagram',
        'WhatsApp',
        'YouTube'

    ]

    channel_usage = combined_df[
        channel_columns
        ].sum().reset_index()

    channel_usage.columns = [
        'Channel',
        'Usage'
    ]

    fig5 = px.bar(

        channel_usage,

        x='Channel',
        y='Usage',
        color='Channel',

        title='Marketing Channel Effectiveness'

    )

    st.plotly_chart(
        fig5,
        width='stretch'
    )

# =========================
# ENGAGEMENT ANALYSIS
# =========================

    st.subheader("Engagement Score Analysis")

    fig6 = px.histogram(

        combined_df,

        x='engagement_score',
        nbins=40,
        color_discrete_sequence=['#EF553B']

    )

    st.plotly_chart(
        fig6,
        width='stretch'
    )

# =========================
# BRAND PERFORMANCE
# =========================

    st.subheader("Brand Revenue Comparison")

    brand_mapping = {

     0: 'Nykaa',
     1: 'Purplle',
     2: 'Tira'

    }

    brand_df = combined_df.copy()

    brand_df['brand_name'] = brand_df[
        'brand'
        ].map(brand_mapping)

    brand_revenue = brand_df.groupby(
        'brand_name'
        )['revenue'].mean().reset_index()

    fig7 = px.bar(

        brand_revenue,

        x='brand_name',
        y='revenue',
        color='brand_name',

        title='Average Revenue by Brand'

    )

    st.plotly_chart(
        fig7,
        width='stretch'
    )

# =========================
# CORRELATION HEATMAP
# =========================

    st.subheader("Feature Correlation Heatmap")

    correlation_df = combined_df[[

        'revenue',
        'roi',
        'clicks',
        'conversions',
        'engagement_score',
        'impressions',
        'acquisition_cost'

    ]].corr()

    fig8 = px.imshow(

        correlation_df,

        text_auto=True,
        aspect="auto",

        title='Feature Relationships'

    )

    st.plotly_chart(
        fig8,
        width='stretch'
    )

# =========================
# FINAL EDA INSIGHTS
# =========================

    st.subheader("EDA Insights")

    st.success(
       "Campaigns with higher engagement scores generated stronger revenue."
    )

    st.success(
        "Conversions and clicks showed strong positive correlation with revenue."
    )

    st.success(
        "Instagram and Google campaigns produced high engagement."
    )

    st.success(
        "High ROI campaigns generally achieved better conversion performance."
    )

    st.info(
        "EDA reveals that customer engagement plays a major role in campaign profitability."
    )



# =========================

# BRAND ANALYSIS

# =========================



elif page == "Brand Analysis":


    st.title("Brand Performance Analysis Dashboard")

# =========================
# BRAND SUMMARY
# =========================

    brand_summary = combined_df.groupby(
        'brand'
        ).agg({

            'revenue': 'mean',
            'roi': 'mean',
            'conversions': 'sum',
            'clicks': 'sum',
            'impressions': 'sum',
            'engagement_score': 'mean'

        }).reset_index()

# =========================
# BRAND NAME MAPPING
# =========================

    brand_mapping = {

        0: 'Nykaa',
        1: 'Purplle',
        2: 'Tira'

    }

    brand_summary['brand_name'] = brand_summary[
        'brand'
    ].map(brand_mapping)

# =========================
# BEST BRANDS
# =========================

    best_revenue_brand = brand_summary.loc[
        brand_summary['revenue'].idxmax()
    ]

    best_roi_brand = brand_summary.loc[
        brand_summary['roi'].idxmax()
    ]

    best_conversion_brand = brand_summary.loc[
        brand_summary['conversions'].idxmax()
    ]

# =========================
# KPI METRICS
# =========================

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Highest Revenue Brand",
         best_revenue_brand['brand_name']
    )

    col2.metric(
        "Highest ROI Brand",
         best_roi_brand['brand_name']
    )

    col3.metric(
        "Highest Conversion Brand",
        best_conversion_brand['brand_name']
    )

# =========================
# SUMMARY TABLE
# =========================

    st.subheader("Brand Summary Table")

    st.dataframe(

        brand_summary[[

            'brand_name',
            'revenue',
            'roi',
            'conversions',
            'clicks',
            'impressions',
            'engagement_score'

        ]]

    )

# =========================
# REVENUE CHART
# =========================

    st.subheader("Average Revenue by Brand")

    fig1 = px.bar(

        brand_summary,

        x='brand_name',
        y='revenue',
        color='brand_name',
        text_auto=True,

        title='Average Revenue Comparison'

    )

    st.plotly_chart(
        fig1,
        width='stretch'
    )

# =========================
# ROI CHART
# =========================

    st.subheader("Average ROI by Brand")

    fig2 = px.bar(

        brand_summary,

        x='brand_name',
        y='roi',
        color='brand_name',
        text_auto=True,

        title='ROI Comparison'

    )

    st.plotly_chart(
        fig2,
        width='stretch'
    )

# =========================
# CONVERSION PIE CHART
# =========================

    st.subheader("Conversion Share")

    fig3 = px.pie(

        brand_summary,

        names='brand_name',
        values='conversions',

        title='Conversion Distribution'

    )

    st.plotly_chart(
        fig3,
        width='stretch'
    )

# =========================
# ENGAGEMENT ANALYSIS
# =========================

    st.subheader("Engagement Score Trends")

    fig4 = px.line(

        brand_summary,

        x='brand_name',
        y='engagement_score',
        markers=True,

        title='Engagement Analysis'

    )

    st.plotly_chart(
        fig4,
        width='stretch'
    )

# =========================
# CLICKS VS IMPRESSIONS
# =========================

    st.subheader("Clicks vs Impressions")

    fig5 = px.scatter(

        brand_summary,

        x='impressions',
        y='clicks',
        color='brand_name',
        size='revenue',

        title='Reach vs Engagement'

    )

    st.plotly_chart(
        fig5,
        width='stretch'
    )

# =========================
# FINAL INSIGHTS
# =========================

    st.subheader("Brand Insights")

    st.success(
        f"Best Revenue Brand: {best_revenue_brand['brand_name']}"
    )

    st.success(
        f"Best ROI Brand: {best_roi_brand['brand_name']}"
    )

    st.success(
        f"Most Customer Conversions: {best_conversion_brand['brand_name']}"
    )

    st.info(
        "Higher engagement campaigns generally produced stronger ROI and conversions."
    )



# =========================

# REGRESSION PAGE

# =========================

elif page == "Regression":


    st.title("Regression Model Evaluation")

    st.dataframe(regression_df)

    fig = px.bar(
        regression_df,
        x='Model',
        y='R2',
        color='Model',
        title='Regression Model Comparison'
    )

    st.plotly_chart(fig)

    best_regression = regression_df.loc[
    regression_df['R2'].idxmax()
    ]

    st.success(
        f"Best Regression Model: {best_regression['Model']}"
    )


# =========================

# CLASSIFICATION PAGE

# =========================

elif page == "Classification":


    st.title("Classification Model Evaluation")

    st.dataframe(classification_df)

    fig = px.bar(
        classification_df,
        x='Model',
        y='Accuracy',
        color='Model',
        title='Classification Accuracy Comparison'
    )

    st.plotly_chart(fig)

    best_classification = classification_df.loc[
        classification_df['Accuracy'].idxmax()
    ]

    st.success(
        f"Best Classification Model: {best_classification['Model']}"
    )

# =========================

# PREDICTION PAGE

# =========================

elif page == "Prediction":

    st.title("Campaign Prediction System")

    campaign_type_option = st.selectbox(

        "Campaign Type",

        [
            "Awareness",
            "Conversion",
            "Retention",
            "Promotion",
            "Lead Generation"
        ]

    )

    campaign_type_mapping = {

        "Awareness": 0,
        "Conversion": 1,
        "Retention": 2,
        "Promotion": 3,
        "Lead Generation": 4

    }

    campaign_type = campaign_type_mapping[
        campaign_type_option
    ]

    target_audience_option = st.selectbox(


        "Target Audience",

        [
            "Students",
            "Working Professionals",
            "Parents",
            "Premium Customers",
            "General Audience"
        ]


    )

    target_audience_mapping = {


        "Students": 0,
        "Working Professionals": 1,
        "Parents": 2,
        "Premium Customers": 3,
        "General Audience": 4


    }

    target_audience = target_audience_mapping[
    target_audience_option
    ]


    duration = st.number_input(
        "Duration",
        min_value=1
    )

    impressions = st.number_input(
        "Impressions",
        min_value=0
    )

    clicks = st.number_input(
        "Clicks",
        min_value=0
    )

    leads = st.number_input(
        "Leads",
        min_value=0
    )

    conversions = st.number_input(
        "Conversions",
        min_value=0
    )

    acquisition_cost = st.number_input(
        "Acquisition Cost",
        min_value=0.0
    )

    engagement_score = st.number_input(
        "Engagement Score",
        min_value=0.0
    )

# =========================
# FEATURE ENGINEERING
# =========================

    ctr = 0

    if impressions != 0:
        ctr = (clicks / impressions) * 100

    conversion_rate = 0

    if clicks != 0:
        conversion_rate = (
            conversions / clicks
        ) * 100

    cost_per_lead = 0

    if leads != 0:
        cost_per_lead = (
            acquisition_cost / leads
        )

# =========================
# PREDICT BUTTON
# =========================

    if st.button("Predict"):

        input_data = pd.DataFrame({

            'campaign_type': [campaign_type],
            'target_audience': [target_audience],
            'duration': [duration],
            'impressions': [impressions],
            'clicks': [clicks],
            'leads': [leads],
            'conversions': [conversions],
            'roi': [0],
            'acquisition_cost': [acquisition_cost],
            'language': [0],
            'engagement_score': [engagement_score],
            'customer_segment': [0],
            'brand': [0],
            'calculated_roi': [0],
            'ctr': [ctr],
            'conversion_rate': [conversion_rate],
            'cost_per_lead': [cost_per_lead],
            'revenue_per_click': [0],

            ' Email': [0],
            ' Facebook': [0],
            ' Google': [0],
            ' Instagram': [0],
            ' WhatsApp': [0],
            ' YouTube': [0],

            'Email': [0],
            'Facebook': [0],
            'Google': [0],
            'Instagram': [0],
            'WhatsApp': [0],
            'YouTube': [0]

        })

    # =========================
    # MATCH TRAINING FEATURES
    # =========================

        regression_input = input_data.reindex(
            columns=regression_features,
            fill_value=0
        )

        classification_input = input_data.reindex(
            columns=classification_features,
            fill_value=0
        )

    # =========================
    # PREDICTIONS
    # =========================

        revenue_prediction = regression_model.predict(
            regression_input
        )

        profit_prediction = classification_model.predict(
            classification_input
        )

    # =========================
    # OUTPUTS
    # =========================

        st.subheader("Prediction Results")

        st.success(
            f"Predicted Revenue: {revenue_prediction[0]:,.2f}"
        )

        if profit_prediction[0] == 1:

            st.success(
                "Campaign Prediction: PROFIT"
            )

        else:

            st.error(
                "Campaign Prediction: LOSS"
            )


# =========================

# INSIGHTS PAGE

# =========================

elif page == "Insights":


    st.title("Business Insights")

    st.write(
        "• Random Forest achieved the best performance in regression and classification."
    )

    st.write(
        "• Higher engagement campaigns produced better revenue."
    )

    st.write(
        "• ROI strongly correlated with conversions."
    )

    st.write(
        "• High CTR campaigns generated higher profitability."
    )

    st.write(
        "• Brand performance varies significantly across campaign types."
    )
