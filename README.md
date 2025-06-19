# Loan Default Prediction Project

This project analyzes and models loan default risk using a real-world banking dataset (~148,670 loans, 34 features).

## Structure
- `data/raw/loan_data.csv`: Original dataset
- `data/processed/loans_cleaned.csv`: Cleaned data
- `data/processed/loans_encoded.csv`: Encoded data for modeling
- `notebooks/exploratory/1.0-eda.ipynb`: EDA notebook
- `src/data/prepare_data.py`: Data cleaning and encoding script
- `reports/figures/`: EDA plots

## Setup
Install requirements:
```bash
pip install -r requirements.txt
```

## Usage
- Run data cleaning: `python src/data/prepare_data.py`
- Explore data: Open the EDA notebook 