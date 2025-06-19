# Loan Default Prediction

A data science project to predict loan defaults using a Kaggle dataset (~148k loans, 34 features).

## Setup
1. Clone repo: `git clone https://github.com/vaibhav11123/Loan-Default-Modeling.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Download dataset from [Kaggle](https://www.kaggle.com/datasets/yasserh/loan-default-dataset) and place in `data/raw/`
4. Run Streamlit app: `streamlit run app.py`

## Project Structure
```
loan_default_project/
├── app.py                   # Streamlit dashboard
├── data/
│   ├── processed/          # Cleaned and processed data
│   └── raw/                # Original dataset
├── models/                 # Trained models
├── notebooks/
│   ├── exploratory/       # EDA notebooks
│   └── reports/           # Final analysis notebooks
├── reports/
│   └── figures/           # Generated graphics
└── src/
    ├── data/              # Data processing scripts
    ├── features/          # Feature engineering
    ├── models/            # Model training code
    └── visualization/     # Plotting utilities
```

## Project Overview
This project aims to predict loan defaults using machine learning techniques. The dataset contains information about ~148,000 loans with 34 features including borrower information, loan characteristics, and payment history.

### Key Features
- Data preprocessing and feature engineering pipeline
- Multiple ML models comparison (Random Forest, XGBoost)
- Interactive Streamlit dashboard for predictions
- Comprehensive EDA and model evaluation notebooks

## Development Status
🚧 Project is currently under development 