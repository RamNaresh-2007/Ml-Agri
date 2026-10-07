"""
Random Forest & Gradient Boosting Regressor for Agricultural Crop Yield Prediction
Includes feature importance extraction and serialization.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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
print("  RANDOM FOREST ENSEMBLE REGRESSOR")
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

rf_pipe = Pipeline([
    ('prep', preprocessor),
    ('rf', RandomForestRegressor(n_estimators=100, max_depth=22, random_state=42, n_jobs=-1))
])

print("Training RandomForestRegressor (100 estimators, max_depth=22)...")
rf_pipe.fit(X_train, y_train)

pred = rf_pipe.predict(X_test)
r2 = r2_score(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)

print(f"Random Forest Performance -> R2: {r2:.4f} | RMSE: {rmse:.2f} | MAE: {mae:.2f}")

# Extract top features
ohe = rf_pipe.named_steps['prep'].named_transformers_['cat']
cat_feat_names = list(ohe.get_feature_names_out(categorical_cols))
all_features = numeric_cols + cat_feat_names
importances = rf_pipe.named_steps['rf'].feature_importances_

feat_df = pd.DataFrame({
    "Feature": all_features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Feature Importances:")
print(feat_df.head(10).to_string(index=False))

# Save
feat_df.to_csv(os.path.join(OUT_DIR, "random_forest_feature_importances.csv"), index=False)
joblib.dump(rf_pipe, os.path.join(OUT_DIR, "random_forest.joblib"), compress=3)
# Also save as primary model.joblib in regression outputs and root models
joblib.dump(rf_pipe, os.path.join(OUT_DIR, "model.joblib"), compress=3)
print(f"Models and feature importances saved to {OUT_DIR}")
