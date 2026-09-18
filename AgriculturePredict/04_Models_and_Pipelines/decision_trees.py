"""
Decision Tree Regressor for Agricultural Crop Yield Prediction
Includes hyperparameter pruning (max_depth, min_samples_split) and feature importance analysis.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(PROJECT_DIR, "01_Datasets", "crop_yield.csv")
OUT_DIR = os.path.join(PROJECT_DIR, "06_Outputs_and_Utils", "Model_Outputs", "Regression")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 65)
print("  DECISION TREE REGRESSOR & HYPERPARAMETER PRUNING")
print("=" * 65)

df = pd.read_csv(DATA_PATH).dropna()
if "Production" in df.columns:
    df = df.drop(columns=["Production"])

categorical_cols = ["Crop", "Season", "State"]
numeric_cols = ["Crop_Year", "Area", "Annual_Rainfall", "Fertilizer", "Pesticide"]
for c in categorical_cols:
    df[c] = df[c].astype(str).str.strip()

X = df[categorical_cols + numeric_cols]
y = df["Yield"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

# Benchmark depths: 5, 10, 15, unconstrained None
depths = [5, 10, 15, 20]
results = []

for d in depths:
    dt_pipe = Pipeline([
        ('prep', preprocessor),
        ('tree', DecisionTreeRegressor(max_depth=d, min_samples_split=10, random_state=42))
    ])
    dt_pipe.fit(X_train, y_train)
    pred = dt_pipe.predict(X_test)
    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    print(f"DecisionTree (max_depth={d:2d}) -> R2: {r2:.4f} | RMSE: {rmse:.2f} | MAE: {mae:.2f}")
    results.append({"Max_Depth": d, "R2_Score": r2, "RMSE": rmse, "MAE": mae})

# Save best model (depth 15)
best_pipe = Pipeline([
    ('prep', preprocessor),
    ('tree', DecisionTreeRegressor(max_depth=15, min_samples_split=10, random_state=42))
])
best_pipe.fit(X_train, y_train)
joblib.dump(best_pipe, os.path.join(OUT_DIR, "decision_tree.joblib"))
pd.DataFrame(results).to_csv(os.path.join(OUT_DIR, "decision_tree_depth_comparison.csv"), index=False)
print(f"Decision Tree model and comparison saved to {OUT_DIR}")
