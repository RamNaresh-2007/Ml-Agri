import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "01_Datasets", "crop_yield.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "processed_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH).dropna()

print("=" * 65)
print("  CATEGORICAL ENCODING PIPELINE (One-Hot & Frequency)")
print("=" * 65)

cat_features = ["Crop", "Season", "State"]
# Strip whitespace
for col in cat_features:
    df[col] = df[col].astype(str).str.strip()

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_matrix = encoder.fit_transform(df[cat_features])
encoded_cols = encoder.get_feature_names_out(cat_features)
encoded_df = pd.DataFrame(encoded_matrix, columns=encoded_cols)

print(f"Original Categorical Features: {cat_features}")
print(f"One-Hot Encoded Columns Generated: {len(encoded_cols)}")
print("\nSample Encoded Columns:")
print(encoded_cols[:15])

# Save encoded sample and column list
feature_names_path = os.path.join(OUTPUT_DIR, "categorical_feature_names.csv")
pd.DataFrame({"feature_name": encoded_cols}).to_csv(feature_names_path, index=False)
print(f"\nCategorical feature names saved to: {feature_names_path}")
