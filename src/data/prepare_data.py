import pandas as pd

def clean_data(input_path, output_path):
    df = pd.read_csv(input_path)
    if 'ID' in df.columns:
        df.drop('ID', axis=1, inplace=True)
    if 'rate_of_interest' in df.columns:
        df['rate_of_interest'].fillna(df['rate_of_interest'].median(), inplace=True)
    if 'Gender' in df.columns:
        df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    categorical_cols = ['Gender', 'loan_type', 'loan_purpose', 'Region']
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    df.to_csv(output_path, index=False)

if __name__ == '__main__':
    clean_data('data/raw/loan_data.csv', 'data/processed/loans_cleaned.csv') 