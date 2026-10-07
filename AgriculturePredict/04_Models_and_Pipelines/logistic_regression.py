# Logistic Regression - sigmoid, cross-entropy, decision boundary, classification
# =================================================================================
# Adapted for AgriculturePredict to classify High-Yield (1) vs Standard-Yield (0) crops.

import sys, os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(BASE_DIR, "..", "05_Web_Application")
if APP_DIR not in sys.path:
    sys.path.append(APP_DIR)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, classification_report
from sklearn.model_selection import train_test_split
import config

os.makedirs(config.LOGISTIC_DIR, exist_ok=True)

# 1. Load Data
raw_path = getattr(config, "RAW_DATA_PATH", "")
if not os.path.exists(raw_path):
    raw_path = os.path.join(BASE_DIR, "..", "01_Datasets", "crop_yield_sample.csv")

df = pd.read_csv(raw_path)
df.columns = df.columns.str.strip()

# Target binary: High Yield threshold (Top 50th percentile)
median_yield = df["Yield"].median()
df["HighYield"] = (df["Yield"] >= median_yield).astype(int)

train, val = train_test_split(df, test_size=0.2, random_state=config.RANDOM_STATE)

feature_cols = [c for c in config.NUMERIC_COLUMNS if c in train.columns and c != "Yield"]
x_train = train[feature_cols].copy()
y_train = train["HighYield"]
x_val = val[feature_cols].copy()
y_val = val["HighYield"]

for col in feature_cols:
    med = x_train[col].median()
    x_train[col] = x_train[col].fillna(med)
    x_val[col] = x_val[col].fillna(med)

high_yield_pct = y_train.mean() * 100
print(f"High Yield Farms: {high_yield_pct:.1f}% | Standard: {100-high_yield_pct:.1f}%")

# 2. Rainfall vs High Yield: Empirical S-curve
if "Annual_Rainfall" in train.columns:
    bins = pd.cut(train["Annual_Rainfall"], bins=15)
    fraction_high = train.groupby(bins, observed=True)["HighYield"].mean()
    bin_centers = [interval.mid for interval in fraction_high.index]

    plt.figure(figsize=(7, 5))
    plt.plot(bin_centers, fraction_high.values, marker="o", color="teal", lw=2)
    plt.xlabel("Annual Rainfall (mm)")
    plt.ylabel("Fraction High Yield")
    plt.title("Rainfall vs High Yield - Empirical S-Curve", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(config.LOGISTIC_DIR, "rainfall_vs_high_yield_curve.png"), dpi=150)
    plt.close()

# 3. Sigmoid Function Illustration
z = np.linspace(-6, 6, 200)
sigma = 1 / (1 + np.exp(-z))
plt.figure(figsize=(7, 4.5))
plt.plot(z, sigma, color="darkblue", lw=2.5, label=r"$\sigma(z) = \frac{1}{1 + e^{-z}}$")
plt.axhline(0.5, color="red", linestyle="--", alpha=0.7, label="Decision Threshold (0.5)")
plt.axvline(0, color="gray", linestyle=":", alpha=0.6)
plt.title("Logistic Sigmoid Activation Function", fontsize=13, fontweight="bold")
plt.xlabel("Linear Predictor z = w^T x + b")
plt.ylabel("P(Y = 1 | x)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(config.LOGISTIC_DIR, "sigmoid_function.png"), dpi=150)
plt.close()

# 4. Standardize & Fit Logistic Regression
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)

clf = LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)
clf.fit(x_train_scaled, y_train)

y_pred = clf.predict(x_val_scaled)
y_prob = clf.predict_proba(x_val_scaled)[:, 1]

acc = accuracy_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_prob)
loss = log_loss(y_val, y_prob)

print(f"Validation Accuracy: {acc:.4f} | ROC AUC: {auc:.4f} | Log Loss: {loss:.4f}")

# 5. 2D Decision Boundary (Rainfall vs Fertilizer)
idx1, idx2 = 0, 1
if len(feature_cols) >= 2:
    f1, f2 = feature_cols[idx1], feature_cols[idx2]
    clf_2d = LogisticRegression(max_iter=500, random_state=config.RANDOM_STATE)
    scaler_2d = StandardScaler()
    X_train_2d = scaler_2d.fit_transform(train[[f1, f2]])
    clf_2d.fit(X_train_2d, y_train)

    X_val_2d = scaler_2d.transform(val[[f1, f2]])
    x_min, x_max = X_val_2d[:, 0].min() - 1, X_val_2d[:, 0].max() + 1
    y_min, y_max = X_val_2d[:, 1].min() - 1, X_val_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    plt.scatter(X_val_2d[:, 0], X_val_2d[:, 1], c=y_val, edgecolors="k", cmap="coolwarm", s=30, alpha=0.6)
    plt.title(f"Logistic Decision Boundary: {f1} vs {f2}", fontsize=13, fontweight="bold")
    plt.xlabel(f"{f1} (standardized)")
    plt.ylabel(f"{f2} (standardized)")
    plt.tight_layout()
    plt.savefig(os.path.join(config.LOGISTIC_DIR, "decision_boundary.png"), dpi=150)
    plt.close()

# 6. Report
rep_path = os.path.join(config.LOGISTIC_DIR, "logistic_regression_report.txt")
with open(rep_path, "w") as f:
    f.write("=" * 60 + "\n")
    f.write("LOGISTIC REGRESSION CLASSIFICATION REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Target: HighYield (Threshold Yield >= {median_yield:.2f} t/ha)\n")
    f.write(f"Features: {', '.join(feature_cols)}\n\n")
    f.write(f"Metrics on Validation Split:\n")
    f.write(f"  - Accuracy: {acc:.4f}\n")
    f.write(f"  - ROC AUC:  {auc:.4f}\n")
    f.write(f"  - Log Loss: {loss:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_val, y_pred, target_names=["Standard Yield", "High Yield"]))

print(f"Saved logistic regression report: {rep_path}")
