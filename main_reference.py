import os
import pandas as pd

# Check relative path or project splits / datasets
CANDIDATE_PATHS = [
    os.path.join(os.path.dirname(__file__), "data", "train.csv"),
    os.path.join(os.path.dirname(__file__), "AgriculturePredict", "06_Outputs_and_Utils", "splits", "train.csv"),
    os.path.join(os.path.dirname(__file__), "AgriculturePredict", "01_Datasets", "crop_yield.csv"),
]

data_file = None
for p in CANDIDATE_PATHS:
    if os.path.exists(p):
        data_file = p
        break

if data_file:
    print(f"Loading reference dataset from: {data_file}")
    df = pd.read_csv(data_file)
    print("\nDataset Head (first 5 records):")
    print(df.head())
    print("\nDataset Info:")
    df.info()
else:
    print("Warning: No candidate training dataset found in workspace.")