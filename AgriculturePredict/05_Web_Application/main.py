import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "01_Datasets", "crop_yield.csv")

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("  AGRIYIELD AI - DATASET & RUNNER CHECK")
print("=" * 60)
print("Shape:", df.shape)
print("\nFirst 5 Records:")
print(df.head())
print("\nData Info:")
print(df.info())
