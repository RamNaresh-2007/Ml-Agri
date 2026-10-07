# Principal Component Analysis (PCA) for AgriculturePredict
# ==========================================================
# Dimensionality Reduction, Latent Agro-Climatic Space Decomposition,
# Feature Loadings Biplot, and Reconstruction Variance Analysis.

import os
import sys

print("=" * 65, flush=True)
print("  PRINCIPAL COMPONENT ANALYSIS (PCA) - AGRI-YIELD", flush=True)
print("=" * 65, flush=True)
print("[*] Initializing environment and loading libraries...", flush=True)

import json
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(BASE_DIR, "..", "05_Web_Application")
if APP_DIR not in sys.path:
    sys.path.append(APP_DIR)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import config

# Ensure destination directory exists
pca_dir = getattr(config, "PCA_DIR", os.path.join(config.OUTPUTS_DIR, "outputs", "pca"))
os.makedirs(pca_dir, exist_ok=True)

# 1. Load Dataset
train_csv_path = os.path.join(getattr(config, "SPLITS_DIR", ""), "train.csv")
if os.path.exists(train_csv_path):
    df_full = pd.read_csv(train_csv_path)
elif hasattr(config, "RAW_DATA_PATH") and os.path.exists(config.RAW_DATA_PATH):
    df_full = pd.read_csv(config.RAW_DATA_PATH)
elif hasattr(config, "SAMPLE_DATA_PATH") and os.path.exists(config.SAMPLE_DATA_PATH):
    df_full = pd.read_csv(config.SAMPLE_DATA_PATH)
else:
    raise FileNotFoundError("Could not find agricultural dataset for PCA analysis.")

df_full.columns = df_full.columns.str.strip()
if "Production" in df_full.columns:
    # Drop production to prevent direct target leakage
    pass

# Select numeric agricultural features
core_numeric = ["Area", "Annual_Rainfall", "Fertilizer", "Pesticide", "Yield"]
feature_cols = [c for c in core_numeric if c in df_full.columns]
if len(feature_cols) < 3:
    feature_cols = [c for c in config.NUMERIC_COLUMNS if c in df_full.columns]

print(f"Loaded {len(df_full):,} agricultural records.")
print(f"Features for PCA decomposition: {feature_cols}")

# 2. Imputation and Scaling
X_raw = df_full[feature_cols].copy()
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X_raw), columns=feature_cols)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 3. Fit Full Dimensionality PCA
n_features = len(feature_cols)
pca_full = PCA(n_components=n_features, random_state=config.RANDOM_STATE)
X_pca_full = pca_full.fit_transform(X_scaled)

eigenvalues = pca_full.explained_variance_
explained_var_ratio = pca_full.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var_ratio)
component_names = [f"PC{i+1}" for i in range(n_features)]

# Identify components meeting Kaiser Criterion (eigenvalue > 1.0)
kaiser_components = int(np.sum(eigenvalues >= 1.0))
kaiser_components = max(1, kaiser_components)

# Variance thresholds
k_80 = int(np.argmax(cumulative_var >= 0.80)) + 1
k_90 = int(np.argmax(cumulative_var >= 0.90)) + 1
k_95 = int(np.argmax(cumulative_var >= 0.95)) + 1

print("\n--- Eigenvalue & Explained Variance Decomposition ---")
for i in range(n_features):
    kaiser_flag = " (Kaiser > 1.0)" if eigenvalues[i] >= 1.0 else ""
    print(f"  {component_names[i]}: Eigenvalue={eigenvalues[i]:.4f}, "
          f"Var Explained={explained_var_ratio[i]*100:.2f}%, "
          f"Cumulative={cumulative_var[i]*100:.2f}%{kaiser_flag}")

# 4. Feature Loadings (Eigenvector Weights)
# Loading = eigenvector * sqrt(eigenvalue) representing correlation between original feature and PC
loadings = pca_full.components_.T * np.sqrt(eigenvalues)
loadings_df = pd.DataFrame(loadings, index=feature_cols, columns=component_names)

print("\n--- Principal Component Loadings Matrix ---")
print(loadings_df.round(4).to_string())

# 5. Reconstruction Error (MSE across components 1 to n_features)
reconstruction_errors = []
for k in range(1, n_features + 1):
    pca_k = PCA(n_components=k, random_state=config.RANDOM_STATE)
    X_k = pca_k.fit_transform(X_scaled)
    X_reconstructed = pca_k.inverse_transform(X_k)
    mse = mean_squared_error(X_scaled, X_reconstructed)
    reconstruction_errors.append(mse)

# 6. Downstream Comparison (Ordinary Least Squares: Original Features vs Top 2 PCs)
if "Yield" in feature_cols:
    y_reg = df_full["Yield"].fillna(df_full["Yield"].median())
    x_features = [c for c in feature_cols if c != "Yield"]
    if len(x_features) >= 2:
        x_orig = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(df_full[x_features]))
        lr_orig = LinearRegression().fit(x_orig, y_reg)
        r2_orig = r2_score(y_reg, lr_orig.predict(x_orig))

        # PCA on features excluding target
        pca_x = PCA(n_components=2, random_state=config.RANDOM_STATE)
        x_pca_top2 = pca_x.fit_transform(x_orig)
        lr_pca = LinearRegression().fit(x_pca_top2, y_reg)
        r2_pca = r2_score(y_reg, lr_pca.predict(x_pca_top2))
    else:
        r2_orig, r2_pca = 0.0, 0.0
else:
    r2_orig, r2_pca = 0.0, 0.0

# -------------------------------------------------------------
# PLOT 1: Scree Plot & Cumulative Explained Variance
# -------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(9, 5))
x_indices = np.arange(1, n_features + 1)

color_bar = "#3b82f6"
color_line = "#10b981"

bars = ax1.bar(x_indices, explained_var_ratio * 100, color=color_bar, alpha=0.75, width=0.45, label="Individual Variance (%)")
ax1.set_xlabel("Principal Components", fontsize=11, fontweight="bold")
ax1.set_ylabel("Variance Explained (%)", fontsize=11, color="navy", fontweight="bold")
ax1.set_xticks(x_indices)
ax1.set_xticklabels(component_names)
ax1.set_ylim(0, max(explained_var_ratio * 100) * 1.3)

for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2, yval + 1.2, f"{yval:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax2 = ax1.twinx()
ax2.plot(x_indices, cumulative_var * 100, color=color_line, marker="o", lw=2.5, label="Cumulative Variance (%)")
ax2.axhline(90, color="#ef4444", linestyle="--", alpha=0.8, label="90% Benchmark")
ax2.axhline(80, color="#f59e0b", linestyle=":", alpha=0.8, label="80% Benchmark")
ax2.set_ylabel("Cumulative Variance (%)", fontsize=11, color="darkgreen", fontweight="bold")
ax2.set_ylim(0, 110)

plt.title("PCA Scree Plot & Cumulative Explained Variance", fontsize=13, fontweight="bold", pad=12)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", framealpha=0.9)
plt.tight_layout()

scree_path = os.path.join(pca_dir, "pca_scree_variance.png")
plt.savefig(scree_path, dpi=300)
plt.close(fig)
print(f"Saved Scree plot: {scree_path}")

# -------------------------------------------------------------
# PLOT 2: Feature Loadings Heatmap
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
cax = ax.imshow(loadings_df.values, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
ax.set_xticks(np.arange(n_features))
ax.set_yticks(np.arange(len(feature_cols)))
ax.set_xticklabels(component_names, fontsize=10, fontweight="bold")
ax.set_yticklabels(feature_cols, fontsize=10, fontweight="bold")

for i in range(len(feature_cols)):
    for j in range(n_features):
        val = loadings_df.iloc[i, j]
        text_color = "white" if abs(val) > 0.5 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontweight="bold", fontsize=9)

cbar = fig.colorbar(cax, ax=ax)
cbar.set_label("Correlation / Loading Weight", rotation=270, labelpad=15, fontweight="bold")
ax.set_title("PCA Feature Loadings Matrix (Correlation with Components)", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()

loadings_path = os.path.join(pca_dir, "pca_feature_loadings_heatmap.png")
plt.savefig(loadings_path, dpi=300)
plt.close(fig)
print(f"Saved loadings heatmap: {loadings_path}")

# -------------------------------------------------------------
# PLOT 3: 2D PCA Biplot with Feature Vectors
# -------------------------------------------------------------
# Subsample points for a visually clean biplot
subsample_n = min(2500, len(X_pca_full))
rng = np.random.RandomState(config.RANDOM_STATE)
sample_idx = rng.choice(len(X_pca_full), size=subsample_n, replace=False)

fig, ax = plt.subplots(figsize=(10, 7))

# Color points by Yield or Rainfall if present
if "Yield" in df_full.columns:
    color_vals = df_full["Yield"].iloc[sample_idx]
    c_label = "Crop Yield (t/ha)"
else:
    color_vals = df_full["Annual_Rainfall"].iloc[sample_idx]
    c_label = "Annual Rainfall (mm)"

scatter = ax.scatter(
    X_pca_full[sample_idx, 0],
    X_pca_full[sample_idx, 1],
    c=color_vals,
    cmap="viridis",
    alpha=0.45,
    s=24,
    edgecolors="none"
)
cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label(c_label, fontsize=10, fontweight="bold")

# Plot feature vectors (scaled for visibility)
scale_arrow = np.max(np.abs(X_pca_full[sample_idx, :2])) * 0.8
for i, feature in enumerate(feature_cols):
    vec_x = pca_full.components_[0, i] * scale_arrow
    vec_y = pca_full.components_[1, i] * scale_arrow
    ax.arrow(0, 0, vec_x, vec_y, color="#dc2626", width=0.03, head_width=0.25, length_includes_head=True, zorder=5)
    ax.text(vec_x * 1.15, vec_y * 1.15, feature, color="#991b1b", fontweight="bold", fontsize=10, ha="center", va="center", zorder=6)

ax.axhline(0, color="gray", linestyle="--", lw=0.8, alpha=0.6)
ax.axvline(0, color="gray", linestyle="--", lw=0.8, alpha=0.6)
ax.set_title(f"PCA 2D Biplot: Agro-Ecological Space & Feature Vectors", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel(f"PC1 ({explained_var_ratio[0]*100:.1f}% Variance Explained)", fontsize=11, fontweight="bold")
ax.set_ylabel(f"PC2 ({explained_var_ratio[1]*100:.1f}% Variance Explained)", fontsize=11, fontweight="bold")
ax.grid(True, linestyle=":", alpha=0.5)
plt.tight_layout()

biplot_path = os.path.join(pca_dir, "pca_2d_biplot.png")
plt.savefig(biplot_path, dpi=300)
plt.close(fig)
print(f"Saved PCA 2D biplot: {biplot_path}")

# -------------------------------------------------------------
# PLOT 4: Reconstruction Error vs Component Count
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(range(1, n_features + 1), reconstruction_errors, marker="s", color="#8b5cf6", lw=2.2, markersize=7)
ax.set_title("PCA Information Preservation: Reconstruction MSE vs Components", fontsize=12, fontweight="bold", pad=10)
ax.set_xlabel("Number of Principal Components Retained", fontsize=11, fontweight="bold")
ax.set_ylabel("Mean Squared Reconstruction Error", fontsize=11, fontweight="bold")
ax.set_xticks(range(1, n_features + 1))
ax.set_xticklabels(range(1, n_features + 1))
ax.grid(True, linestyle="--", alpha=0.6)

for k, err in enumerate(reconstruction_errors, start=1):
    ax.annotate(f"{err:.3f}", (k, err), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
recon_path = os.path.join(pca_dir, "pca_reconstruction_error.png")
plt.savefig(recon_path, dpi=300)
plt.close(fig)
print(f"Saved reconstruction error plot: {recon_path}")

# 7. Agronomic Interpretation of Components
agronomic_interpretations = []
for idx, pc in enumerate(component_names):
    strongest_pos = loadings_df[pc].idxmax()
    strongest_pos_val = loadings_df[pc].max()
    strongest_neg = loadings_df[pc].idxmin()
    strongest_neg_val = loadings_df[pc].min()
    
    interp = f"{pc} ({explained_var_ratio[idx]*100:.1f}% var): "
    if abs(strongest_pos_val) >= abs(strongest_neg_val):
        interp += f"Primarily driven positively by '{strongest_pos}' (+{strongest_pos_val:.2f})"
        if abs(strongest_neg_val) > 0.3:
            interp += f" with counter-balance from '{strongest_neg}' ({strongest_neg_val:.2f})"
    else:
        interp += f"Primarily driven inversely by '{strongest_neg}' ({strongest_neg_val:.2f})"
    agronomic_interpretations.append(interp)

# 8. Save Serialized Model and Scaler
pca_artifact = {
    "pca": pca_full,
    "scaler": scaler,
    "imputer": imputer,
    "feature_names": feature_cols,
    "explained_variance_ratio": explained_var_ratio.tolist(),
    "loadings": loadings_df.to_dict()
}
joblib_path = os.path.join(pca_dir, "pca_model.joblib")
joblib.dump(pca_artifact, joblib_path)
print(f"Saved serialized PCA model: {joblib_path}")

# 9. Save JSON Summary
summary_dict = {
    "n_samples": int(len(df_full)),
    "n_features": n_features,
    "feature_names": feature_cols,
    "eigenvalues": [round(float(v), 4) for v in eigenvalues],
    "explained_variance_ratio": [round(float(v), 4) for v in explained_var_ratio],
    "cumulative_variance_ratio": [round(float(v), 4) for v in cumulative_var],
    "kaiser_components_count": kaiser_components,
    "components_needed_80pct": k_80,
    "components_needed_90pct": k_90,
    "components_needed_95pct": k_95,
    "reconstruction_errors": [round(float(v), 4) for v in reconstruction_errors],
    "downstream_r2": {
        "all_original_features": round(float(r2_orig), 4),
        "top_2_pca_components": round(float(r2_pca), 4)
    },
    "interpretations": agronomic_interpretations
}
json_path = os.path.join(pca_dir, "pca_summary.json")
with open(json_path, "w") as f:
    json.dump(summary_dict, f, indent=2)
print(f"Saved PCA summary JSON: {json_path}")

# Also copy summary to outputs/
shared_json = os.path.join(config.OUTPUTS_DIR, "outputs", "co4_pca_summary.json")
with open(shared_json, "w") as f:
    json.dump(summary_dict, f, indent=2)

# 10. Save Detailed Technical Report
report_path = os.path.join(pca_dir, "pca_report.txt")
with open(report_path, "w") as f:
    f.write("=" * 65 + "\n")
    f.write("PRINCIPAL COMPONENT ANALYSIS (PCA) - TECHNICAL REPORT\n")
    f.write("=" * 65 + "\n\n")
    f.write(f"Dataset Size: {len(df_full):,} agricultural records\n")
    f.write(f"Original Feature Count: {n_features} features\n")
    f.write(f"Analyzed Features: {', '.join(feature_cols)}\n\n")
    f.write("1. EIGENVALUE & VARIANCE DECOMPOSITION:\n")
    for i in range(n_features):
        kaiser = " [Kaiser Criterion: Retain]" if eigenvalues[i] >= 1.0 else ""
        f.write(f"   {component_names[i]}: Eigenvalue={eigenvalues[i]:.4f} | "
                f"Variance={explained_var_ratio[i]*100:.2f}% | "
                f"Cumulative={cumulative_var[i]*100:.2f}%{kaiser}\n")
    f.write(f"\n2. DIMENSIONALITY REDUCTION THRESHOLDS:\n")
    f.write(f"   - Kaiser Criterion (Eigenvalue > 1.0): {kaiser_components} components\n")
    f.write(f"   - Components for 80% Variance: {k_80} components ({cumulative_var[k_80-1]*100:.1f}%)\n")
    f.write(f"   - Components for 90% Variance: {k_90} components ({cumulative_var[k_90-1]*100:.1f}%)\n")
    f.write(f"   - Components for 95% Variance: {k_95} components ({cumulative_var[k_95-1]*100:.1f}%)\n\n")
    f.write("3. FACTOR LOADINGS MATRIX (Feature Correlations with PCs):\n")
    f.write(loadings_df.round(4).to_string() + "\n\n")
    f.write("4. AGRONOMIC INTERPRETATION:\n")
    for interp in agronomic_interpretations:
        f.write(f"   * {interp}\n")
    f.write(f"\n5. RECONSTRUCTION ERROR (MSE):\n")
    for k, err in enumerate(reconstruction_errors, start=1):
        f.write(f"   - Retaining {k} components: MSE = {err:.4f}\n")
    f.write(f"\n6. DOWNSTREAM REGRESSION VALIDATION:\n")
    f.write(f"   - OLS Linear Model with all original features: R2 = {r2_orig:.4f}\n")
    f.write(f"   - OLS Linear Model with top 2 PCA components: R2 = {r2_pca:.4f}\n")
    f.write("=" * 65 + "\n")

print(f"Saved PCA report: {report_path}")
print("PCA analysis execution complete.\n")
