# Models.py - Systematic Evaluation of Feature Scaling Across ML Families
# =========================================================================
# Compares model performance and training time (Scaled vs Unscaled) for:
# 1. Distance-based (KNN, SVM RBF)
# 2. Gradient-descent (Logistic Regression, Neural Net MLP)
# 3. Clustering (K-Means)
# 4. Regularized Regression (Ridge, Lasso)
# Saves comprehensive results to Outputs/model_scaling_results.csv

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, silhouette_score, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.neural_network import MLPClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(PROJECT_DIR, "01_Datasets", "crop_yield.csv")

if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(PROJECT_DIR, "01_Datasets", "crop_yield_sample.csv")

def load_and_preprocess_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    # Numerical imputation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Regression target: Yield
    y_reg = df["Yield"]

    # Classification target: High Yield (>= median)
    y_class = (df["Yield"] >= df["Yield"].median()).astype(int)

    # Feature matrix (excluding leakage columns like Production and target Yield)
    drop_cols = ["Yield", "Production"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # One-hot encode categoricals
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    return X_encoded, y_class, y_reg

def evaluate_models():
    X, y_class, y_reg = load_and_preprocess_data()

    # Classification Split
    X_tr_cls, X_ts_cls, y_tr_cls, y_ts_cls = train_test_split(
        X, y_class, test_size=0.2, random_state=42
    )

    # Regression Split
    X_tr_reg, X_ts_reg, y_tr_reg, y_ts_reg = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    # Standardization
    scaler_cls = StandardScaler()
    X_tr_cls_sc = scaler_cls.fit_transform(X_tr_cls)
    X_ts_cls_sc = scaler_cls.transform(X_ts_cls)

    scaler_reg = StandardScaler()
    X_tr_reg_sc = scaler_reg.fit_transform(X_tr_reg)
    X_ts_reg_sc = scaler_reg.transform(X_ts_reg)

    # Subsample for compute efficiency on SVM / MLP
    sample_size = min(4000, len(X_tr_cls))
    X_tr_cls_sub = X_tr_cls.iloc[:sample_size]
    y_tr_cls_sub = y_tr_cls.iloc[:sample_size]
    X_ts_cls_sub = X_ts_cls.iloc[:1000]
    y_ts_cls_sub = y_ts_cls.iloc[:1000]

    X_tr_cls_sc_sub = X_tr_cls_sc[:sample_size]
    X_ts_cls_sc_sub = X_ts_cls_sc[:1000]

    results = []

    def train_and_eval(model, name, family, X_train, y_train, X_test, y_test, is_scaled, task="Classification", is_cluster=False):
        start = time.time()
        try:
            if is_cluster:
                model.fit(X_train)
                train_time = time.time() - start
                s_idx = np.random.choice(len(X_train), min(1500, len(X_train)), replace=False)
                x_sub = X_train.values[s_idx] if hasattr(X_train, "values") else X_train[s_idx]
                labels = model.labels_[s_idx]
                score = silhouette_score(x_sub, labels) if len(set(labels)) > 1 else 0.0
                metric_name = "Silhouette Score"
            else:
                model.fit(X_train, y_train)
                train_time = time.time() - start
                preds = model.predict(X_test)
                if task == "Classification":
                    score = accuracy_score(y_test, preds)
                    metric_name = "Accuracy"
                else:
                    score = r2_score(y_test, preds)
                    metric_name = "R2 Score"

            results.append({
                "Family": family,
                "Model Name": name,
                "Task": task,
                "Scaled": "Yes" if is_scaled else "No",
                "Metric": metric_name,
                "Score": round(float(score), 4),
                "Train Time (s)": round(float(train_time), 4)
            })
        except Exception as e:
            print(f"Error {name} (scaled={is_scaled}): {e}")

    # 1. Classification Models
    cls_configs = [
        ("Distance-based", "KNN", lambda: KNeighborsClassifier(n_neighbors=5)),
        ("Distance-based", "SVM (RBF)", lambda: SVC(kernel="rbf", max_iter=2000, random_state=42)),
        ("Gradient-descent", "Logistic Regression", lambda: LogisticRegression(max_iter=500, random_state=42)),
        ("Gradient-descent", "Neural Net", lambda: MLPClassifier(hidden_layer_sizes=(50,), max_iter=200, early_stopping=True, random_state=42))
    ]

    for family, name, model_fn in cls_configs:
        # Unscaled
        train_and_eval(model_fn(), name, family, X_tr_cls_sub, y_tr_cls_sub, X_ts_cls_sub, y_ts_cls_sub, False, "Classification")
        # Scaled
        train_and_eval(model_fn(), name, family, X_tr_cls_sc_sub, y_tr_cls_sub, X_ts_cls_sc_sub, y_ts_cls_sub, True, "Classification")

    # 2. Clustering
    family = "Distance-based"
    name = "K-Means (k=3)"
    train_and_eval(KMeans(n_clusters=3, random_state=42, n_init=10), name, family, X_tr_cls_sub, None, None, None, False, "Clustering", True)
    train_and_eval(KMeans(n_clusters=3, random_state=42, n_init=10), name, family, X_tr_cls_sc_sub, None, None, None, True, "Clustering", True)

    # 3. Regression Models
    reg_configs = [
        ("Regularized", "Ridge", lambda: Ridge(alpha=1.0, random_state=42)),
        ("Regularized", "Lasso", lambda: Lasso(alpha=0.1, max_iter=200, tol=1e-3, random_state=42))
    ]

    for family, name, model_fn in reg_configs:
        train_and_eval(model_fn(), name, family, X_tr_reg, y_tr_reg, X_ts_reg, y_ts_reg, False, "Regression")
        train_and_eval(model_fn(), name, family, X_tr_reg_sc, y_tr_reg, X_ts_reg_sc, y_ts_reg, True, "Regression")

    return results

if __name__ == "__main__":
    print("Evaluating models with and without scaling across machine learning families...")
    res = evaluate_models()
    df_res = pd.DataFrame(res)
    
    # Save to both standard locations for compatibility
    out_dir = os.path.join(PROJECT_DIR, "06_Outputs_and_Utils", "Outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model_scaling_results.csv")
    df_res.to_csv(out_path, index=False)

    alt_out_dir = os.path.join(PROJECT_DIR, "06_Outputs_and_Utils", "outputs")
    os.makedirs(alt_out_dir, exist_ok=True)
    df_res.to_csv(os.path.join(alt_out_dir, "model_scaling_results.csv"), index=False)

    print(f"\nResults successfully saved to: {out_path}")
    print("\n" + df_res.to_string(index=False))
