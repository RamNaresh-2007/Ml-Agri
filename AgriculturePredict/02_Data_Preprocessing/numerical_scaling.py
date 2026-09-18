import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "01_Datasets", "crop_yield.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "06_Outputs_and_Utils", "processed_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH).dropna()

print("=" * 65)
print("  NUMERICAL SCALING & NORMALIZATION PIPELINE")
print("=" * 65)

numeric_features = ["Crop_Year", "Area", "Annual_Rainfall", "Fertilizer", "Pesticide"]
for col in numeric_features + ["Yield"]:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna()

X_num = df[numeric_features]
y = df["Yield"]

# 1. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X_num, y, test_size=0.2, random_state=42)

# 2. Standard Scaling (Z-score normalization)
std_scaler = StandardScaler()
X_train_std = std_scaler.fit_transform(X_train)
X_test_std = std_scaler.transform(X_test)

# 3. MinMax Scaling (Range [0, 1])
minmax_scaler = MinMaxScaler()
X_train_minmax = minmax_scaler.fit_transform(X_train)
X_test_minmax = minmax_scaler.transform(X_test)

print(f"Numerical Features: {numeric_features}")
print(f"X_train Shape: {X_train.shape} | X_test Shape: {X_test.shape}")
print("\n--- StandardScaler (Zero Mean, Unit Variance) Summary ---")
print("Train Means:", [round(m, 3) for m in X_train_std.mean(axis=0)])
print("Train Stds:", [round(s, 3) for s in X_train_std.std(axis=0)])

print("\n--- MinMaxScaler ([0, 1] Bounded) Summary ---")
print("Train Mins:", [round(m, 3) for m in X_train_minmax.min(axis=0)])
print("Train Maxs:", [round(m, 3) for m in X_train_minmax.max(axis=0)])

# Save outputs
pd.DataFrame(X_train_std, columns=numeric_features).to_csv(os.path.join(OUTPUT_DIR, "X_train_num_standard.csv"), index=False)
pd.DataFrame(X_test_std, columns=numeric_features).to_csv(os.path.join(OUTPUT_DIR, "X_test_num_standard.csv"), index=False)
pd.DataFrame(X_train_minmax, columns=numeric_features).to_csv(os.path.join(OUTPUT_DIR, "X_train_num_minmax.csv"), index=False)
pd.DataFrame(X_test_minmax, columns=numeric_features).to_csv(os.path.join(OUTPUT_DIR, "X_test_num_minmax.csv"), index=False)
y_train.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)

print(f"\nScaled splits saved to: {OUTPUT_DIR}")
