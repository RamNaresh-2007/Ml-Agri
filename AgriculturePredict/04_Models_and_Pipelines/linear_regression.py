import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(PROJECT_DIR, "01_Datasets", "crop_yield.csv")
OUT_DIR = os.path.join(PROJECT_DIR, "06_Outputs_and_Utils", "Model_Outputs", "Regression")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 65)
print("  LINEAR & REGULARIZED REGRESSION (OLS, Ridge, Lasso)")
print("=" * 65)

df = pd.read_csv(DATA_PATH).dropna()
if "Production" in df.columns:
    df = df.drop(columns=["Production"])

# Features & Target
categorical_cols = ["Crop", "Season", "State"]
numeric_cols = ["Crop_Year", "Area", "Annual_Rainfall", "Fertilizer", "Pesticide"]
for c in categorical_cols:
    df[c] = df[c].astype(str).str.strip()

X = df[categorical_cols + numeric_cols]
y = df["Yield"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

# 1. Linear Regression (OLS)
lr_pipe = Pipeline([
    ('prep', preprocessor),
    ('reg', LinearRegression())
])
lr_pipe.fit(X_train, y_train)
lr_pred = lr_pipe.predict(X_test)
lr_r2 = r2_score(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_mae = mean_absolute_error(y_test, lr_pred)

print(f"Linear Regression -> R2: {lr_r2:.4f} | RMSE: {lr_rmse:.4f} | MAE: {lr_mae:.4f}")

# 2. Ridge Regression
ridge_pipe = Pipeline([
    ('prep', preprocessor),
    ('reg', RidgeCV(alphas=np.logspace(-2, 3, 20)))
])
ridge_pipe.fit(X_train, y_train)
ridge_pred = ridge_pipe.predict(X_test)
ridge_r2 = r2_score(y_test, ridge_pred)
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
ridge_mae = mean_absolute_error(y_test, ridge_pred)

print(f"Ridge Regression  -> R2: {ridge_r2:.4f} | RMSE: {ridge_rmse:.4f} | MAE: {ridge_mae:.4f} (Best Alpha: {ridge_pipe.named_steps['reg'].alpha_:.2f})")

# Save comparison and model
metrics_df = pd.DataFrame([
    {"Model": "Linear Regression (OLS)", "R2_Score": lr_r2, "RMSE": lr_rmse, "MAE": lr_mae},
    {"Model": "Ridge Regression", "R2_Score": ridge_r2, "RMSE": ridge_rmse, "MAE": ridge_mae}
])
metrics_df.to_csv(os.path.join(OUT_DIR, "linear_models_comparison.csv"), index=False)
joblib.dump(ridge_pipe, os.path.join(OUT_DIR, "linear_regression.joblib"))
print(f"Artifacts saved to {OUT_DIR}")
