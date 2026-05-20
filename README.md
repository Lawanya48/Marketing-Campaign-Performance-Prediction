# Marketing Campaign Analytics & Prediction System

## Project Overview

The Marketing Campaign Analytics & Prediction System is an end-to-end Machine Learning project developed to analyze, preprocess, visualize, and predict marketing campaign performance using datasets from multiple brands such as Nykaa, Purplle, and Tira.

This project demonstrates the complete Machine Learning lifecycle including:

* Data Collection
* Data Cleaning & Preprocessing
* Feature Engineering
* Exploratory Data Analysis (EDA)
* Regression Modeling
* Classification Modeling
* Clustering & PCA
* Model Evaluation
* Streamlit Deployment

The system helps marketing teams understand campaign performance, predict future revenue, classify campaigns as profitable or non-profitable, and identify campaign clusters for business decision-making.

---

# Problem Statement

Marketing companies generate large amounts of campaign data including:

* Impressions
* Clicks
* Leads
* Conversions
* Revenue
* ROI
* Acquisition Cost
* Engagement Scores

However, this data is often stored in raw CSV format with:

* Missing values
* Duplicate records
* Multi-label categorical features
* Inconsistent ROI values

The objective of this project is to clean and preprocess the raw campaign data, perform analysis, engineer meaningful features, and build Machine Learning models for prediction and business insights.

---

# Objectives

The major objectives of the project are:

* Clean and preprocess marketing datasets
* Handle missing values and duplicates
* Perform feature engineering
* Apply multi-label encoding
* Perform Exploratory Data Analysis (EDA)
* Predict campaign revenue using regression
* Predict campaign profitability using classification
* Segment campaigns using clustering
* Visualize campaign insights using Streamlit dashboard
* Generate business insights for decision-making

---

# Dataset Description

The datasets contain detailed marketing campaign information.

| Column           | Description                   |
| ---------------- | ----------------------------- |
| Campaign_ID      | Unique campaign identifier    |
| Campaign_Type    | Type of campaign              |
| Target_Audience  | Target customer category      |
| Duration         | Campaign duration             |
| Channel_Used     | Marketing channels used       |
| Impressions      | Total campaign views          |
| Clicks           | Total user clicks             |
| Leads            | Potential customers generated |
| Conversions      | Successful conversions        |
| Revenue          | Revenue generated             |
| Acquisition_Cost | Campaign cost                 |
| ROI              | Return on Investment          |
| Language         | Campaign language             |
| Engagement_Score | Customer engagement metric    |
| Customer_Segment | Customer type                 |
| Date             | Campaign date                 |

Datasets Used:

* Nykaa Campaign Dataset
* Purplle Campaign Dataset
* Tira Campaign Dataset

---

# Technologies Used

| Category         | Technologies                |
| ---------------- | --------------------------- |
| Programming      | Python                      |
| Data Analysis    | Pandas, NumPy               |
| Visualization    | Plotly, Matplotlib, Seaborn |
| Machine Learning | Scikit-learn                |
| Deployment       | Streamlit                   |
| Model Storage    | Joblib                      |
| Version Control  | Git & GitHub                |

---

# Machine Learning Workflow

## 1. Data Collection

* Imported multiple CSV datasets
* Converted CSV files into Pandas DataFrames
* Combined datasets into a single dataset

## 2. Data Preprocessing

Performed:

* Missing value handling
* Duplicate removal
* Datatype conversion
* ROI correction
* Data standardization
* Outlier handling

## 3. Feature Engineering

Created:

* Profit_Flag
* CTR (Click Through Rate)
* Conversion Rate
* Cost Per Lead
* Revenue Per Click

Applied:

* Label Encoding
* Multi-Label Encoding

## 4. Exploratory Data Analysis (EDA)

Performed:

* Revenue analysis
* ROI analysis
* Brand analysis
* Channel analysis
* Correlation analysis
* Engagement analysis

## 5. Regression Modeling

Goal:
Predict campaign revenue.

Models Used:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor

Evaluation Metrics:

* MAE
* MSE
* RMSE
* R² Score

## 6. Classification Modeling

Goal:
Predict campaign profit or loss.

Models Used:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* SVM
* KNN

Evaluation Metrics:

* Accuracy
* Precision
* Recall
* F1-Score

## 7. Clustering

Used:

* K-Means Clustering
* PCA (Principal Component Analysis)

Clusters:

* High Performance Campaigns
* Medium Performance Campaigns
* Low Performance Campaigns

---

# Streamlit Dashboard Features

The project includes an interactive Streamlit dashboard.

Dashboard Pages:

| Page           | Purpose                         |
| -------------- | ------------------------------- |
| Home           | Project overview                |
| EDA            | Exploratory Data Analysis       |
| Brand Analysis | Brand comparison                |
| Regression     | Regression model performance    |
| Classification | Classification model evaluation |
| Clustering     | Campaign segmentation           |
| Prediction     | Revenue & Profit prediction     |
| Insights       | Business recommendations        |

---

# Prediction System

Users can enter campaign information such as:

* Campaign Type
* Target Audience
* Impressions
* Clicks
* Leads
* Conversions
* Acquisition Cost
* Engagement Score

Outputs:

* Predicted Revenue
* Predicted Profit/Loss

---

# Business Insights

The project generates important business insights such as:

* Best performing brand
* Most profitable campaigns
* Best marketing channels
* Revenue-driving campaign types
* High engagement campaign analysis
* Cluster-based segmentation

---

# Project Structure

```text
marketing_campaign_project/
│
├── app/
│   └── streamlit_app.py
│
├── src/
│   ├── preprocessing.py
│   ├── regression_models.py
│   ├── classification_models.py
│   ├── clustering.py
│   └── evaluation.py
│
├── data/
│   ├── raw/
│   ├── sample_campaign_data.csv
│   └── processed/
│
├── models/
│   ├── regression_model.pkl
│   ├── classification_model.pkl
│   ├── regression_features.pkl
│   └── classification_features.pkl
│
├── outputs/
│   ├── regression_results.csv
│   ├── classification_results.csv
│   └── cluster_summary.csv
│
├── notebooks/
│   └── eda.ipynb
│
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

---

# How to Run the Project

## Step 1: Install Requirements

```bash
pip install -r requirements.txt
```

## Step 2: Run Preprocessing

```bash
python src/preprocessing.py
```

## Step 3: Train Regression Models

```bash
python src/regression_models.py
```

## Step 4: Train Classification Models

```bash
python src/classification_models.py
```


```

## Step 6: Launch Streamlit Application

```bash
python -m streamlit run app/streamlit_app.py
```

---

# GitHub Note

Large processed datasets are excluded from the repository due to GitHub file size limitations.

The preprocessing pipeline automatically recreates cleaned and feature-engineered datasets from the raw CSV files.

---

# Results

The project successfully achieved:

* Data cleaning and preprocessing
* Feature engineering
* Multi-label encoding
* Revenue prediction
* Profit/Loss classification
* Campaign clustering
* Business insight generation
* Interactive Streamlit deployment

---

# Future Improvements

Future enhancements can include:

* Deep Learning models
* Real-time API integration
* Cloud deployment
* AutoML integration
* Advanced NLP-based campaign analysis
* Recommendation systems

---

# Conclusion

The Marketing Campaign Analytics & Prediction System is a complete end-to-end Machine Learning project that demonstrates practical applications of:

* Data Science
* Machine Learning
* Business Analytics
* Dashboard Development
* Marketing Intelligence

The system helps organizations make data-driven marketing decisions using predictive analytics and visualization.

---

# Author

Lawanya Duraisamy
