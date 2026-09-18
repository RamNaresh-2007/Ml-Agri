import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "01_Datasets", "crop_yield.csv")

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("  AGRICULTURAL DATASET - CATEGORICAL FEATURES OVERVIEW")
print("=" * 60)

print(f"Total Records: {len(df)}")
print(f"Unique Crops ({df['Crop'].nunique()}):")
print(df['Crop'].value_counts().head(10))

print("\n" + "=" * 60)
print(f"Unique States ({df['State'].nunique()}):")
print(df['State'].value_counts().head(10))

print("\n" + "=" * 60)
print(f"Unique Seasons ({df['Season'].nunique()}):")
print(df['Season'].value_counts())
