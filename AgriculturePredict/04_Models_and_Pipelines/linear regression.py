# Multilinear regression for Agricultural Crop Yield
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "01_Datasets", "crop_yield.csv")

if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(BASE_DIR, "..", "01_Datasets", "crop_yield_sample.csv")

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

# Features
features = ["Area", "Annual_Rainfall", "Fertilizer", "Pesticide"]
X = df[features]

# Target
y = df["Yield"]

# Remove missing values
data = pd.concat([X, y], axis=1).dropna()
X = data[features]
y = data["Yield"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Create model
model = LinearRegression()

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("=" * 50)
print("AGRICULTURE MULTILINEAR REGRESSION")
print("=" * 50)
print("Intercept:", model.intercept_)
print("\nCoefficients:")
for feature, coefficient in zip(features, model.coef_):
    print(f"  {feature:18s} : {coefficient:.6f}")

print("\nModel Performance")
print("-----------------------")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R2   : {r2:.4f}")

# Sample prediction for a new farm
new_farm = pd.DataFrame([[
    1000.0,   # Area (hectares)
    1250.0,   # Annual_Rainfall (mm)
    85000.0,  # Fertilizer (kg)
    120.0     # Pesticide (kg)
]], columns=features)

predicted_yield = model.predict(new_farm)
print(f"\nPredicted Crop Yield for New Farm: {predicted_yield[0]:.2f} tons/hectare")
print("=" * 50)
