import os
import shutil
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, "..", "01_Datasets", "crop_yield.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "..", "01_Datasets", "processed_crop_yield.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "processed_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("  DATA CLEANING & LEAKAGE PREVENTION PIPELINE")
print("=" * 65)

print(f"Loading raw dataset from {RAW_DATA_PATH}...")
df = pd.read_csv(RAW_DATA_PATH)
initial_shape = df.shape
print(f"Initial raw shape: {initial_shape}")

# 1. Drop 'Production' to prevent data leakage (since Yield = Production / Area)
if "Production" in df.columns:
    print("Dropping 'Production' column to strictly prevent target leakage...")
    df = df.drop(columns=["Production"])

# 2. Strip whitespace from text columns
for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = df[col].apply(lambda s: s.strip() if isinstance(s, str) else s)

# 3. Numeric conversion and null handling
numeric_cols = ['Crop_Year', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()
cleaned_shape = df.shape
print(f"Cleaned dataset shape (dropped {initial_shape[0] - cleaned_shape[0]} invalid rows): {cleaned_shape}")

# 4. Outlier summary via IQR
print("\n--- Outlier Detection via IQR (1.5 x IQR) ---")
outlier_summary = []
for col in ['Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield']:
    q25 = df[col].quantile(0.25)
    q75 = df[col].quantile(0.75)
    iqr = q75 - q25
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_pct = round((len(outliers) / len(df)) * 100, 2)
    outlier_summary.append({
        "Feature": col,
        "Q25": round(q25, 2),
        "Q75": round(q75, 2),
        "IQR": round(iqr, 2),
        "Outlier_Count": len(outliers),
        "Outlier_Pct": outlier_pct
    })
    print(f"  {col}: {len(outliers)} outliers ({outlier_pct}%)")

pd.DataFrame(outlier_summary).to_csv(os.path.join(OUTPUT_DIR, "outlier_analysis.csv"), index=False)

# 5. Save cleaned dataset
df.to_csv(PROCESSED_DATA_PATH, index=False)
# Also copy to 06_Outputs_and_Utils/processed_data/cleaned_data.csv
df.to_csv(os.path.join(OUTPUT_DIR, "cleaned_data.csv"), index=False)
print(f"\nCleaned dataset saved to:\n  - {PROCESSED_DATA_PATH}\n  - {os.path.join(OUTPUT_DIR, 'cleaned_data.csv')}")
