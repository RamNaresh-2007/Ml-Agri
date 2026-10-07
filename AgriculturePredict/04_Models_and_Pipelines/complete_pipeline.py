"""
Master Complete Machine Learning Pipeline for AgriculturePredict
Performs end-to-end dataset preprocessing, model evaluation, artifact serialization, and metric logging.
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(PROJECT_DIR, "01_Datasets", "crop_yield.csv")
PROCESSED_PATH = os.path.join(PROJECT_DIR, "01_Datasets", "processed_crop_yield.csv")
OUT_REG_DIR = os.path.join(PROJECT_DIR, "06_Outputs_and_Utils", "Model_Outputs", "Regression")
OUT_DATA_DIR = os.path.join(PROJECT_DIR, "06_Outputs_and_Utils", "processed_data")
OUT_UTILS_DIR = os.path.join(PROJECT_DIR, "06_Outputs_and_Utils", "outputs")

for d in [OUT_REG_DIR, OUT_DATA_DIR, OUT_UTILS_DIR]:
    os.makedirs(d, exist_ok=True)

print("=" * 70)
print("  MASTER END-TO-END MACHINE LEARNING PIPELINE")
print("=" * 70)

# Step 1: Load and clean
print("Step 1: Ingesting raw dataset...")
df = pd.read_csv(DATA_PATH)
if "Production" in df.columns:
    df = df.drop(columns=["Production"])

for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = df[col].astype(str).str.strip()

for col in ['Crop_Year', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()
df.to_csv(PROCESSED_PATH, index=False)
print(f"Cleaned dataset: {len(df)} records.")

# Step 2: Preprocessor definition
categorical_cols = ["Crop", "Season", "State"]
numeric_cols = ["Crop_Year", "Area", "Annual_Rainfall", "Fertilizer", "Pesticide"]
X = df[categorical_cols + numeric_cols]
y = df["Yield"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save train/test splits for reproducibility
X_train.to_csv(os.path.join(OUT_DATA_DIR, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(OUT_DATA_DIR, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(OUT_DATA_DIR, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(OUT_DATA_DIR, "y_test.csv"), index=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

# Step 3: Benchmarking multiple models
models = {
    "Linear Regression (OLS)": LinearRegression(),
    "Ridge Regression": RidgeCV(alphas=np.logspace(-2, 3, 20)),
    "Decision Tree": DecisionTreeRegressor(max_depth=15, min_samples_split=10, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=22, random_state=42, n_jobs=-1)
}

comparison_results = []
trained_pipelines = {}

print("\nStep 2: Training & Benchmarking Models:")
for name, model_obj in models.items():
    t0 = time.time()
    pipe = Pipeline([
        ('prep', preprocessor),
        ('model', model_obj)
    ])
    pipe.fit(X_train, y_train)
    duration = time.time() - t0
    
    pred = pipe.predict(X_test)
    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    
    print(f"  {name:25s} | R2: {r2:.4f} | RMSE: {rmse:8.2f} | MAE: {mae:6.2f} | Time: {duration:5.2f}s")
    comparison_results.append({
        "Model": name,
        "R2_Score": round(float(r2), 4),
        "RMSE": round(float(rmse), 4),
        "MAE": round(float(mae), 4),
        "Training_Time_Sec": round(duration, 2)
    })
    trained_pipelines[name] = pipe

# Save model comparison table
comp_df = pd.DataFrame(comparison_results)
comp_df.to_csv(os.path.join(OUT_REG_DIR, "regression_model_comparison.csv"), index=False)
comp_df.to_csv(os.path.join(OUT_UTILS_DIR, "regression_model_comparison.csv"), index=False)

# Step 4: Serialize Champion Model & Artifacts
champion_name = "Random Forest"
champion_pipe = trained_pipelines[champion_name]

joblib.dump(champion_pipe, os.path.join(OUT_REG_DIR, "random_forest.joblib"), compress=3)
joblib.dump(champion_pipe, os.path.join(OUT_REG_DIR, "model.joblib"), compress=3)
joblib.dump(champion_pipe.named_steps['prep'], os.path.join(OUT_DATA_DIR, "preprocessor.joblib"), compress=3)

print("\nStep 3: Exporting Summary Metrics & Categories...")
summary_metrics = {
    "champion_model": champion_name,
    "r2_score": comp_df.loc[comp_df["Model"] == champion_name, "R2_Score"].values[0],
    "rmse": comp_df.loc[comp_df["Model"] == champion_name, "RMSE"].values[0],
    "mae": comp_df.loc[comp_df["Model"] == champion_name, "MAE"].values[0],
    "dataset": {
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "crops_count": df["Crop"].nunique(),
        "states_count": df["State"].nunique(),
        "seasons_count": df["Season"].nunique()
    },
    "benchmarks": comparison_results
}

with open(os.path.join(OUT_UTILS_DIR, "complete_project_summary.json"), "w") as f:
    json.dump(summary_metrics, f, indent=2)

print("=" * 70)
print(f"Master Pipeline Finished Successfully! All artifacts written to {OUT_REG_DIR}")
print("=" * 70)
