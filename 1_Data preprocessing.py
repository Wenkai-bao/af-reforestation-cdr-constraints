# Data preprocessing
import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv('.../Data S1.csv')

# Specify the response and explanatory variable column positions
response_col_idx = ...
predictor_cols_idx = ...
columns_to_normalize = [df.columns[response_col_idx]] + df.columns[predictor_cols_idx].tolist()

# Apply min-max normalization while preserving 9999 as missing-value
for column in columns_to_normalize:
    valid_mask = df[column] != 9999
    valid_values = df.loc[valid_mask, column]
    min_value = valid_values.min()
    max_value = valid_values.max()
    df.loc[valid_mask, column] = (valid_values - min_value) / (max_value - min_value)

# Log-transform the response variable. The 1e-4 offset is added to retain zero-valued observations without producing log(0).
valid_response = df[df.columns[response_col_idx]] != 9999
df.loc[valid_response, df.columns[response_col_idx]] = np.log(
    df.loc[valid_response, df.columns[response_col_idx]] + 1e-4)

# Save the normalized csv dataset
df.to_csv('.../normalized data.csv', index=False)