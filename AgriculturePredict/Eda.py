"""
AgriYield Quick Exploratory Data Analysis (EDA) Script
Standalone EDA analysis script adapted from the reference architecture.
Loads the agricultural crop yield dataset, computes statistical summaries,
and saves core exploratory visualizations to the 'plots/' directory.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "01_Datasets", "crop_yield.csv")

if not os.path.exists(DATA_PATH):
    # fallback
    DATA_PATH = os.path.join(BASE_DIR, "01_Datasets", "crop_yield_sample.csv")

print("=" * 60)
print(f"Loading Dataset: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

# Summary statistics
print("=" * 60)
print("First 5 Rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nShape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nMissing Values:\n", df.isnull().sum())
print("\nStatistical Summary:\n", df.describe())

# Create plots folder
PLOT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
sns.set_style("whitegrid")

# 1. Distribution of Yield
plt.figure(figsize=(8, 5))
sns.histplot(df["Yield"], kde=True, color="forestgreen", bins=40)
plt.title("Yield Distribution (log-normal profile)", fontsize=14, fontweight="bold")
plt.xlabel("Yield (t/ha)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "Yield_Distribution.png"), dpi=300)
plt.close()

# 2. BoxPlot of Yield
plt.figure(figsize=(7, 4))
sns.boxplot(x=df["Yield"], color="lightgreen")
plt.title("BoxPlot of Crop Yield", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "BoxPlot_Yield.png"), dpi=300)
plt.close()

# 3. Numeric Columns Distributions & Boxplots
numeric_cols = ["Area", "Annual_Rainfall", "Fertilizer", "Pesticide", "Yield"]
for col in numeric_cols:
    if col in df.columns:
        plt.figure(figsize=(7, 4))
        sns.boxplot(x=df[col], color="skyblue")
        plt.title(f"BoxPlot: {col}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"BoxPlot_{col}.png"), dpi=300)
        plt.close()

# 4. Correlation Heatmap
plt.figure(figsize=(8, 6))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True)
plt.title("Correlation Heatmap of Agricultural Features", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "Correlation_Heatmap.png"), dpi=300)
plt.close()

# 5. Scatter Plot: Rainfall vs Yield
if "Annual_Rainfall" in df.columns and "Yield" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=df["Annual_Rainfall"], y=df["Yield"], alpha=0.4, color="teal")
    plt.title("Annual Rainfall vs Crop Yield", fontsize=14, fontweight="bold")
    plt.xlabel("Annual Rainfall (mm)")
    plt.ylabel("Yield (t/ha)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "Scatter_Rainfall_Yield.png"), dpi=300)
    plt.close()

# 6. Season vs Yield BoxPlot
if "Season" in df.columns and "Yield" in df.columns:
    plt.figure(figsize=(9, 5))
    sns.boxplot(x="Season", y="Yield", data=df, palette="Set2")
    plt.title("Yield Distribution Across Seasons", fontsize=14, fontweight="bold")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "Season_vs_Yield_BoxPlot.png"), dpi=300)
    plt.close()

print(f"\n[OK] EDA completed successfully! Plots saved to {PLOT_DIR}")
