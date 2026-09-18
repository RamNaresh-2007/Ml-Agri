import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(PROJECT_DIR, "01_Datasets", "processed_crop_yield.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
EDA_OUT_DIR = os.path.join(BASE_DIR, "EDA_Outputs")
UTILS_OUT_DIR = os.path.join(PROJECT_DIR, "06_Outputs_and_Utils", "outputs")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(EDA_OUT_DIR, exist_ok=True)
os.makedirs(UTILS_OUT_DIR, exist_ok=True)

if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(PROJECT_DIR, "01_Datasets", "crop_yield.csv")

df = pd.read_csv(DATA_PATH).dropna()

print("=" * 65)
print("  EXPLORATORY DATA ANALYSIS (EDA) & VISUALIZATION PIPELINE")
print("=" * 65)

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
palette = ['#10b981', '#38bdf8', '#a855f7', '#f59e0b', '#ef4444', '#06b6d4']

# 1. Distribution Plots for Numerical Features
numeric_cols = ['Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield']
for col in numeric_cols:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[col], kde=True, color='#10b981', ax=ax, bins=30)
    ax.set_title(f'Distribution of {col}', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel(col, fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    plt.tight_layout()
    plot_file = f"01_distribution_{col}.png"
    fig.savefig(os.path.join(PLOTS_DIR, plot_file), dpi=150)
    fig.savefig(os.path.join(EDA_OUT_DIR, plot_file), dpi=150)
    plt.close(fig)

# 2. Boxplots for Outlier Analysis
for col in numeric_cols:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(x=df[col], color='#38bdf8', ax=ax)
    ax.set_title(f'Boxplot & Outlier Spread: {col}', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel(col, fontsize=11)
    plt.tight_layout()
    plot_file = f"02_boxplot_{col}.png"
    fig.savefig(os.path.join(PLOTS_DIR, plot_file), dpi=150)
    fig.savefig(os.path.join(EDA_OUT_DIR, plot_file), dpi=150)
    plt.close(fig)

# 3. Correlation Matrix Heatmap
fig, ax = plt.subplots(figsize=(8, 6))
corr_matrix = df[numeric_cols + (['Crop_Year'] if 'Crop_Year' in df.columns else [])].corr()
sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt='.2f', linewidths=0.5, ax=ax)
ax.set_title('Feature Correlation Matrix Heatmap', fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "14_correlation_heatmap.png"), dpi=150)
fig.savefig(os.path.join(EDA_OUT_DIR, "14_correlation_heatmap.png"), dpi=150)
plt.close(fig)

corr_matrix.to_csv(os.path.join(EDA_OUT_DIR, "13_correlation_matrix.csv"))

# 4. Seasonal Yield Spread
if "Season" in df.columns:
    fig, ax = plt.subplots(figsize=(10, 5))
    season_df = df.groupby('Season')['Yield'].mean().sort_values(ascending=False).reset_index()
    sns.barplot(data=season_df, x='Season', y='Yield', palette='viridis', ax=ax)
    ax.set_title('Average Agricultural Yield by Crop Season', fontsize=14, fontweight='bold', pad=12)
    ax.set_ylabel('Mean Yield', fontsize=11)
    plt.xticks(rotation=25)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "eda_seasonal_yield.png"), dpi=150)
    fig.savefig(os.path.join(EDA_OUT_DIR, "eda_seasonal_yield.png"), dpi=150)
    plt.close(fig)

# 5. Top 10 High-Yielding Crops
top_crops = df.groupby('Crop')['Yield'].mean().sort_values(ascending=False).head(10).reset_index()
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=top_crops, x='Yield', y='Crop', palette='crest', ax=ax)
ax.set_title('Top 10 High-Yielding Crops (Mean Yield)', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Mean Yield', fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "eda_crops_distribution.png"), dpi=150)
fig.savefig(os.path.join(EDA_OUT_DIR, "eda_crops_distribution.png"), dpi=150)
plt.close(fig)

# 6. EDA Summary JSON
summary = {
    "total_records": len(df),
    "features": list(df.columns),
    "numeric_features": numeric_cols,
    "top_crops": top_crops.to_dict(orient='records'),
    "generated_plots": [
        "14_correlation_heatmap.png",
        "eda_seasonal_yield.png",
        "eda_crops_distribution.png"
    ] + [f"01_distribution_{c}.png" for c in numeric_cols]
      + [f"02_boxplot_{c}.png" for c in numeric_cols]
}

with open(os.path.join(EDA_OUT_DIR, "eda_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

with open(os.path.join(UTILS_OUT_DIR, "eda_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"Generated {len(summary['generated_plots'])} high-resolution plots in:")
print(f"  - {PLOTS_DIR}")
print(f"  - {EDA_OUT_DIR}")
print("EDA Pipeline Complete.")
