import pandas as pd

# Load the dataset
file_path = 'data/raw/loan_data.csv'
df = pd.read_csv(file_path)

# Print the shape of the DataFrame
print('Shape:', df.shape)  # Should be ~148,670 rows, 34 columns

# Print the column names
print('Columns:', df.columns.tolist())

# Print the first 5 rows
print('Head:')
print(df.head()) 