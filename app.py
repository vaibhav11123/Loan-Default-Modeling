import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(level=logging.INFO, filename='logs/dashboard.log', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("Starting Streamlit dashboard on June 20, 2025")

# Streamlit app
st.title('Loan Default Prediction Dashboard')

# Load data and model
try:
    df = pd.read_csv('/Users/vaibhavsingh/loan_default_project/data/raw/loan_data.csv')
    logger.info(f"Loaded dataset: {df.shape}")
except FileNotFoundError:
    st.error("Dataset not found at /Users/vaibhavsingh/loan_default_project/data/raw/loan_data.csv")
    logger.error("Dataset not found")
    st.stop()

try:
    model = joblib.load('/Users/vaibhavsingh/loan_default_project/models/random_forest_model.pkl')
    logger.info("Loaded model: random_forest_model.pkl")
except FileNotFoundError:
    st.error("Model not found at models/random_forest_model.pkl")
    logger.error("Model not found")
    st.stop()

# Feature engineering (aligned with 2.0-modeling.ipynb)
def engineer_features(df):
    """Create features consistent with modeling notebook."""
    df['debt_to_income'] = df['dtir1'] / 100
    df['loan_to_income'] = df['loan_amount'] / df.get('income', df['loan_amount']).replace(0, 1)
    df['credit_score_bin'] = pd.cut(df['Credit_Score'], bins=[0, 600, 700, 800, 900], labels=['Low', 'Medium', 'High', 'Excellent'])
    df['loan_to_value'] = df['loan_amount'] / df.get('property_value', df['loan_amount']).replace(0, 1)
    logger.info("Engineered features: debt_to_income, loan_to_income, credit_score_bin, loan_to_value")
    return df

df = engineer_features(df.copy())

# Add clustering for segmentation
features_for_clustering = ['Credit_Score', 'dtir1', 'loan_amount', 'LTV']
scaler = StandardScaler()
try:
    X_scaled = scaler.fit_transform(df[features_for_clustering].dropna())
    logger.info(f"Clustering: {len(df) - len(df[features_for_clustering].dropna())} rows dropped due to missing values")
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['cluster'] = pd.Series(kmeans.fit_predict(X_scaled), index=df[features_for_clustering].dropna().index).reindex(df.index, fill_value=-1)
    logger.info("Added K-means clustering with 4 clusters")
except Exception as e:
    st.error(f"Error in clustering: {e}")
    logger.error(f"Error in clustering: {e}")
    df['cluster'] = -1

# Page navigation
page = st.sidebar.selectbox('Select Page', ['EDA', 'Model Results', 'Segmentation'])

if page == 'EDA':
    st.header('Exploratory Data Analysis')
    
    st.subheader('Credit Score Distribution by Region')
    try:
        with open('reports/figures/credit_score_dist.html', 'r') as f:
            st.components.v1.html(f.read(), height=500)
        logger.info("Displayed credit_score_dist.html")
    except FileNotFoundError:
        st.warning("Credit score distribution plot not found. Run 1.0-exploratory-data-analysis.ipynb first.")
        logger.warning("Credit score distribution plot not found")

    st.subheader('DTI by Default Status')
    try:
        with open('reports/figures/dti_by_default.html', 'r') as f:
            st.components.v1.html(f.read(), height=500)
        logger.info("Displayed dti_by_default.html")
    except FileNotFoundError:
        st.warning("DTI boxplot not found. Run 1.0-exploratory-data-analysis.ipynb first.")
        logger.warning("DTI boxplot not found")

    st.subheader('Loan Amount vs. Credit Score with Outliers')
    try:
        with open('reports/figures/outliers.html', 'r') as f:
            st.components.v1.html(f.read(), height=500)
        logger.info("Displayed outliers.html")
    except FileNotFoundError:
        st.warning("Outliers plot not found. Run 1.0-exploratory-data-analysis.ipynb first.")
        logger.warning("Outliers plot not found")

    st.subheader('Anomalies in Loan Amount vs. Credit Score')
    try:
        with open('reports/figures/anomalies.html', 'r') as f:
            st.components.v1.html(f.read(), height=500)
        logger.info("Displayed anomalies.html")
    except FileNotFoundError:
        st.warning("Anomalies plot not found. Run 1.0-exploratory-data-analysis.ipynb first.")
        logger.warning("Anomalies plot not found")

    st.subheader('Default Rate by Year')
    try:
        with open('reports/figures/yearly_defaults.html', 'r') as f:
            st.components.v1.html(f.read(), height=500)
        logger.info("Displayed yearly_defaults.html")
    except FileNotFoundError:
        st.warning("Yearly defaults plot not found. Run 1.0-exploratory-data-analysis.ipynb first.")
        logger.warning("Yearly defaults plot not found")

    st.subheader('Correlation Heatmap (Defaulted Loans)')
    try:
        with open('reports/figures/corr_defaulted.html', 'r') as f:
            st.components.v1.html(f.read(), height=500)
        logger.info("Displayed corr_defaulted.html")
    except FileNotFoundError:
        st.warning("Correlation heatmap not found. Run 1.0-exploratory-data-analysis.ipynb first.")
        logger.warning("Correlation heatmap not found")

    st.subheader('Pair Plot of Key Features')
    try:
        with open('reports/figures/pair_plot.html', 'r') as f:
            st.components.v1.html(f.read(), height=800)
        logger.info("Displayed pair_plot.html")
    except FileNotFoundError:
        st.warning("Pair plot not found. Run 1.0-exploratory-data-analysis.ipynb first.")
        logger.warning("Pair plot not found")

    st.subheader('Default Rates by Loan Purpose')
    try:
        with open('reports/figures/purpose_defaults.html', 'r') as f:
            st.components.v1.html(f.read(), height=500)
        logger.info("Displayed purpose_defaults.html")
    except FileNotFoundError:
        st.warning("Loan purpose defaults plot not found. Run 1.0-exploratory-data-analysis.ipynb first.")
        logger.warning("Loan purpose defaults plot not found")

elif page == 'Model Results':
    st.header('Model Performance')
    
    st.subheader('Feature Importances')
    try:
        with open('reports/figures/feature_importance.html', 'r') as f:
            st.components.v1.html(f.read(), height=500)
        logger.info("Displayed feature_importance.html")
    except FileNotFoundError:
        st.warning("Feature importance plot not found. Run 2.0-modeling.ipynb first.")
        logger.warning("Feature importance plot not found")

    st.subheader('Confusion Matrix')
    try:
        with open('reports/figures/confusion_matrix.html', 'r') as f:
            st.components.v1.html(f.read(), height=500)
        logger.info("Displayed confusion_matrix.html")
    except FileNotFoundError:
        st.warning("Confusion matrix plot not found. Run 2.0-modeling.ipynb first.")
        logger.warning("Confusion matrix plot not found")

    st.subheader('Precision-Recall Curve')
    try:
        with open('reports/figures/pr_curve.html', 'r') as f:
            st.components.v1.html(f.read(), height=500)
        logger.info("Displayed pr_curve.html")
    except FileNotFoundError:
        st.warning("Precision-recall curve not found. Run 2.0-modeling.ipynb first.")
        logger.warning("Precision-recall curve not found")

    st.subheader('Model Metrics')
    try:
        metrics_df = pd.read_csv('reports/model_summary.csv')
        st.write(metrics_df)
        logger.info("Displayed model metrics from model_summary.csv")
    except FileNotFoundError:
        st.warning("Model metrics not found. Run 2.0-modeling.ipynb first.")
        logger.warning("Model metrics not found")

elif page == 'Segmentation':
    st.header('Customer Segmentation')
    cluster = st.slider('Select Cluster', 0, 3, 0)
    cluster_data = df[df['cluster'] == cluster]
    if not cluster_data.empty:
        default_rate = cluster_data['Status'].mean()
        st.write(f'Default Rate for Cluster {cluster}: {default_rate:.2%}')
        st.write(f'Cluster Size: {len(cluster_data)}')
        fig = px.scatter(cluster_data, x='Credit_Score', y='dtir1', color='Status',
                         title=f'Cluster {cluster} Analysis',
                         hover_data=['loan_amount', 'LTV'])
        st.plotly_chart(fig, use_container_width=True)
        logger.info(f"Displayed segmentation for cluster {cluster}")
    else:
        st.warning(f"No data for cluster {cluster}")
        logger.warning(f"No data for cluster {cluster}")

# Save dashboard state
logger.info(f"Dashboard page displayed: {page}") 