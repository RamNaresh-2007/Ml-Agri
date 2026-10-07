import os
import pandas as pd

csv_path = os.path.join(os.path.dirname(__file__), "Iris.csv")
df = pd.read_csv(csv_path)

print(df.head())
print(df.info())
print(df.tail())
print(df.shape)
print(df.columns)
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())