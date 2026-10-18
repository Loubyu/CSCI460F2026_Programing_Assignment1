import csv
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('bank-full.csv', sep=',',quoting=csv.QUOTE_NONE)
df.columns = df.columns.str.strip('"')
df["age"] = df["age"].str.strip('"').astype(int)

for column in df.columns:
    if pd.api.types.is_string_dtype(df[column]):
        df[column] = df[column].str.strip('"')
        unique_values = df[column].unique()
        mapping = {}
        for i, value in enumerate(unique_values):
            mapping[value] = i
        df[column] = df[column].map(mapping)

df = df.dropna()

y = df["y"]
X = df.drop(columns=["y"])

print(X.shape)
print(y.shape)