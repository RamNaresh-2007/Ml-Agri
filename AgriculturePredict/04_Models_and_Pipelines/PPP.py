# ============================================================
# AGRICULTURAL PREDICTION PROJECT (APP / PPP)
# EDA -> CLEANING -> ENCODING -> FEATURE SCALING -> MODEL READY
# ============================================================
# Generates model-ready splits and preprocessed artifacts:
# - splits/train.csv, splits/test.csv
# - processed/cleaned_data.csv
# - Scaled feature matrices

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(PROJECT_DIR, "01_Datasets", "crop_yield.csv")

if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(PROJECT_DIR, "01_Datasets", "crop_yield_sample.csv")

OUT_DIR = os.path.join(PROJECT_DIR, "06_Outputs_and_Utils")
SPLITS_DIR = os.path.join(OUT_DIR, "splits")
PROCESSED_DIR = os.path.join(OUT_DIR, "processed")
PROCESSED_DATA_DIR = os.path.join(OUT_DIR, "processed_data")

os.makedirs(SPLITS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

print("=" * 60)
print("1. LOADING RAW AGRICULTURAL DATASET")
print("=" * 60)
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
print(f"Loaded records: {df.shape[0]:,}, Features: {df.shape[1]}")

print("\n2. BASIC CLEANING")
# Clean missing values if any
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

clean_path = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
df.to_csv(clean_path, index=False)
print(f"Saved cleaned dataset to: {clean_path}")

print("\n3. TRAIN / TEST DATASET SPLITTING (BEFORE PREPROCESSING)")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_path = os.path.join(SPLITS_DIR, "train.csv")
test_path = os.path.join(SPLITS_DIR, "test.csv")
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)
print(f"Train set saved: {train_path} ({len(train_df):,} rows)")
print(f"Test set saved:  {test_path} ({len(test_df):,} rows)")

print("\n4. ENCODING & FEATURE SCALING")
# Define feature sets
cat_cols = ["Crop", "Season", "State"]
num_cols = ["Crop_Year", "Area", "Annual_Rainfall", "Fertilizer", "Pesticide"]
target_col = "Yield"

# Fit one-hot encoder or dummy variables on train
X_train_raw = train_df[cat_cols + num_cols]
X_test_raw = test_df[cat_cols + num_cols]
y_train = train_df[target_col]
y_test = test_df[target_col]

X_train_encoded = pd.get_dummies(X_train_raw, columns=cat_cols, drop_first=True)
X_test_encoded = pd.get_dummies(X_test_raw, columns=cat_cols, drop_first=True)

# Align columns
X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join="left", axis=1, fill_value=0)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_encoded),
    columns=X_train_encoded.columns,
    index=X_train_encoded.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_encoded),
    columns=X_test_encoded.columns,
    index=X_test_encoded.index
)

# Save processed matrices
X_train_scaled.to_csv(os.path.join(PROCESSED_DATA_DIR, "X_train_standard.csv"), index=False)
X_test_scaled.to_csv(os.path.join(PROCESSED_DATA_DIR, "X_test_standard.csv"), index=False)
y_train.to_csv(os.path.join(PROCESSED_DATA_DIR, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(PROCESSED_DATA_DIR, "y_test.csv"), index=False)

# MinMax scaling
minmax = MinMaxScaler()
X_train_minmax = pd.DataFrame(minmax.fit_transform(X_train_encoded), columns=X_train_encoded.columns)
X_test_minmax = pd.DataFrame(minmax.transform(X_test_encoded), columns=X_test_encoded.columns)
X_train_minmax.to_csv(os.path.join(PROCESSED_DATA_DIR, "X_train_minmax.csv"), index=False)
X_test_minmax.to_csv(os.path.join(PROCESSED_DATA_DIR, "X_test_minmax.csv"), index=False)

print("Saved standardized & minmax matrices to processed_data/")
print("\n" + "=" * 60)
print("AGRICULTURAL PIPELINE PREPARATION COMPLETE")
print("=" * 60)
