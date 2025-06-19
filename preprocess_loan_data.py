import pandas as pd

# Load the dataset
file_path = 'data/raw/loan_data.csv'
df = pd.read_csv(file_path)

# 1. Identify columns with missing data
missing = df.isnull().sum()
print('Columns with missing values:')
print(missing[missing > 0])

# 2. Impute missing values
# Numerical columns
if 'rate_of_interest' in df.columns:
    df['rate_of_interest'].fillna(df['rate_of_interest'].median(), inplace=True)
if 'dtir1' in df.columns:
    df['dtir1'].fillna(df['dtir1'].median(), inplace=True)

# Categorical columns
categorical_to_impute = ['Gender', 'loan_purpose']
for col in categorical_to_impute:
    if col in df.columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

# 3. Save cleaned data
cleaned_path = 'data/processed/loans_cleaned.csv'
df.to_csv(cleaned_path, index=False)
print(f'Cleaned data saved to {cleaned_path}')

# 4. Encode categorical variables
categorical_cols = ['Gender', 'loan_type', 'loan_purpose', 'Region']
categorical_cols = [col for col in categorical_cols if col in df.columns]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 5. Drop irrelevant columns
if 'ID' in df_encoded.columns:
    df_encoded.drop('ID', axis=1, inplace=True)

# 6. Save encoded data
encoded_path = 'data/processed/loans_encoded.csv'
df_encoded.to_csv(encoded_path, index=False)
print(f'Encoded data saved to {encoded_path}') 