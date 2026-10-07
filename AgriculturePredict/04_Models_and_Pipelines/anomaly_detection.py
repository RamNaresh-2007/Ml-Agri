# Anomaly Detection & Agricultural Outlier Diagnostics for AgriculturePredict
# ==============================================================================
# Detects agro-ecological anomalies, extreme weather outliers, input dosage errors,
# and yield irregularities using an ensemble of unsupervised algorithms:
# 1. Isolation Forest (Tree-based partitioning depth)
# 2. Local Outlier Factor (LOF - Density-based local reachability)
# 3. Robust Elliptic Envelope (Mahalanobis covariance distance)
# 4. One-Class Support Vector Machine (OC-SVM with RBF kernel)

import os
import sys

print("=" * 65, flush=True)
print("  AGRICULTURAL ANOMALY DETECTION & OUTLIER PROFILING", flush=True)
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
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
import config

anomaly_dir = getattr(config, "ANOMALY_DIR", os.path.join(config.OUTPUTS_DIR, "outputs", "anomaly"))
os.makedirs(anomaly_dir, exist_ok=True)

# 1. Load Data
train_csv_path = os.path.join(getattr(config, "SPLITS_DIR", ""), "train.csv")
if os.path.exists(train_csv_path):
    df_full = pd.read_csv(train_csv_path)
elif hasattr(config, "RAW_DATA_PATH") and os.path.exists(config.RAW_DATA_PATH):
    df_full = pd.read_csv(config.RAW_DATA_PATH)
elif hasattr(config, "SAMPLE_DATA_PATH") and os.path.exists(config.SAMPLE_DATA_PATH):
    df_full = pd.read_csv(config.SAMPLE_DATA_PATH)
else:
    raise FileNotFoundError("Could not find agricultural dataset for anomaly detection.")

df_full.columns = df_full.columns.str.strip()

# Subsample if dataset is large for fast multi-algorithm training
MAX_SAMPLES = 5000
if len(df_full) > MAX_SAMPLES:
    rng = np.random.RandomState(config.RANDOM_STATE)
    sample_idx = rng.choice(len(df_full), size=MAX_SAMPLES, replace=False)
    df = df_full.iloc[sample_idx].copy().reset_index(drop=True)
    print(f"Subsampled {len(df):,} of {len(df_full):,} records for comprehensive anomaly diagnostics.")
else:
    df = df_full.copy().reset_index(drop=True)
    print(f"Loaded {len(df):,} agricultural records for anomaly diagnostics.")

# Extract core agricultural features
core_numeric = ["Area", "Annual_Rainfall", "Fertilizer", "Pesticide", "Yield"]
feature_cols = [c for c in core_numeric if c in df.columns]

# Compute agronomic intensity metrics
df["Fertilizer_per_Ha"] = df["Fertilizer"] / df["Area"].replace(0, np.nan)
df["Pesticide_per_Ha"] = df["Pesticide"] / df["Area"].replace(0, np.nan)

diagnostic_cols = feature_cols + ["Fertilizer_per_Ha", "Pesticide_per_Ha"]

imputer = SimpleImputer(strategy="median")
X_imp = imputer.fit_transform(df[diagnostic_cols])

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_imp)

# 2. Multi-Algorithm Anomaly Detection Suite
CONTAMINATION = 0.025  # Expected anomaly rate (2.5%)

print(f"\nFitting anomaly detection algorithms (target contamination = {CONTAMINATION*100:.1f}%)...")

# Algorithm 1: Isolation Forest
iso_forest = IsolationForest(
    n_estimators=150,
    contamination=CONTAMINATION,
    random_state=config.RANDOM_STATE,
    n_jobs=-1
)
iso_pred = iso_forest.fit_predict(X_scaled)  # -1 = anomaly, 1 = normal
iso_scores = iso_forest.decision_function(X_scaled)  # Lower is more abnormal
df["Anomaly_IForest"] = (iso_pred == -1).astype(int)
df["IForest_Score"] = iso_scores
print(f"  [1] Isolation Forest: Flagged {df['Anomaly_IForest'].sum():,} anomalies ({df['Anomaly_IForest'].mean()*100:.2f}%)")

# Algorithm 2: Local Outlier Factor (LOF)
lof = LocalOutlierFactor(
    n_neighbors=25,
    contamination=CONTAMINATION,
    n_jobs=-1
)
lof_pred = lof.fit_predict(X_scaled)
df["Anomaly_LOF"] = (lof_pred == -1).astype(int)
print(f"  [2] Local Outlier Factor (LOF): Flagged {df['Anomaly_LOF'].sum():,} anomalies ({df['Anomaly_LOF'].mean()*100:.2f}%)")

# Algorithm 3: Robust Elliptic Envelope (Mahalanobis covariance)
elliptic = EllipticEnvelope(
    contamination=CONTAMINATION,
    random_state=config.RANDOM_STATE,
    support_fraction=0.85
)
elliptic_pred = elliptic.fit_predict(X_scaled)
df["Anomaly_Elliptic"] = (elliptic_pred == -1).astype(int)
print(f"  [3] Elliptic Envelope: Flagged {df['Anomaly_Elliptic'].sum():,} anomalies ({df['Anomaly_Elliptic'].mean()*100:.2f}%)")

# Algorithm 4: One-Class SVM
oc_svm = OneClassSVM(
    kernel="rbf",
    nu=CONTAMINATION,
    gamma="scale"
)
oc_svm_pred = oc_svm.fit_predict(X_scaled)
df["Anomaly_OCSVM"] = (oc_svm_pred == -1).astype(int)
print(f"  [4] One-Class SVM: Flagged {df['Anomaly_OCSVM'].sum():,} anomalies ({df['Anomaly_OCSVM'].mean()*100:.2f}%)")

# 3. Consensus Ensemble Scoring
algo_cols = ["Anomaly_IForest", "Anomaly_LOF", "Anomaly_Elliptic", "Anomaly_OCSVM"]
df["Ensemble_Anomaly_Count"] = df[algo_cols].sum(axis=1)

# Categorize severity
def assign_severity(count):
    if count >= 3:
        return "High-Confidence Anomaly"
    elif count == 2:
        return "Moderate Anomaly"
    elif count == 1:
        return "Low-Confidence Outlier"
    return "Normal Farm Record"

df["Anomaly_Category"] = df["Ensemble_Anomaly_Count"].apply(assign_severity)
df["Is_Confirmed_Anomaly"] = (df["Ensemble_Anomaly_Count"] >= 2).astype(int)

high_conf = int((df["Anomaly_Category"] == "High-Confidence Anomaly").sum())
moderate = int((df["Anomaly_Category"] == "Moderate Anomaly").sum())
normal_cnt = int((df["Anomaly_Category"] == "Normal Farm Record").sum())
confirmed_cnt = int(df["Is_Confirmed_Anomaly"].sum())

print("\n--- Ensemble Anomaly Diagnostics Breakdown ---")
print(f"  Total Evaluated: {len(df):,}")
print(f"  Confirmed Anomalies (>= 2 algorithms): {confirmed_cnt:,} ({confirmed_cnt/len(df)*100:.2f}%)")
print(f"  - High Confidence (>= 3 algorithms): {high_conf:,}")
print(f"  - Moderate Anomaly (2 algorithms): {moderate:,}")
print(f"  - Normal Farm Records: {normal_cnt:,}")

# 4. Root Cause Classification for Anomalies
def diagnose_root_cause(row):
    reasons = []
    # Excessive fertilizer intensity (> 600 kg/ha or < 5 kg/ha)
    if row["Fertilizer_per_Ha"] > 600:
        reasons.append("Extreme Fertilizer Over-Application")
    elif row["Fertilizer_per_Ha"] < 5 and row["Area"] > 50:
        reasons.append("Severe Nutrient Deprivation")

    # Pesticide intensity (> 10 kg/ha)
    if row["Pesticide_per_Ha"] > 10:
        reasons.append("Pesticide Toxicity Hazard")

    # Extreme rainfall / drought
    if row["Annual_Rainfall"] < 400:
        reasons.append("Catastrophic Drought Stress")
    elif row["Annual_Rainfall"] > 3500:
        reasons.append("Extreme Flood / Waterlogging")

    # Abnormal yield
    if row["Yield"] > 150:
        reasons.append("Suspicious Yield Outlier (Reporting Error)")
    elif row["Yield"] <= 0.05:
        reasons.append("Total Crop Failure")

    if not reasons:
        reasons.append("Multivariate Latent Discrepancy")
    return " & ".join(reasons)

anomalies_df = df[df["Is_Confirmed_Anomaly"] == 1].copy()
anomalies_df["Root_Cause"] = anomalies_df.apply(diagnose_root_cause, axis=1)

# -------------------------------------------------------------
# PLOT 1: 2D PCA Space Anomaly Visualization
# -------------------------------------------------------------
pca_2d = PCA(n_components=2, random_state=config.RANDOM_STATE)
X_pca = pca_2d.fit_transform(X_scaled)
df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]

fig, ax = plt.subplots(figsize=(10, 6.5))

# Plot normal records
normal_mask = (df["Is_Confirmed_Anomaly"] == 0)
ax.scatter(
    df.loc[normal_mask, "PC1"],
    df.loc[normal_mask, "PC2"],
    c="#94a3b8",
    alpha=0.45,
    s=22,
    label=f"Normal Records ({normal_mask.sum():,})",
    edgecolors="none"
)

# Plot moderate anomalies
mod_mask = (df["Anomaly_Category"] == "Moderate Anomaly")
ax.scatter(
    df.loc[mod_mask, "PC1"],
    df.loc[mod_mask, "PC2"],
    c="#f59e0b",
    alpha=0.85,
    s=55,
    label=f"Moderate Anomalies ({mod_mask.sum():,})",
    edgecolors="k",
    linewidths=0.5
)

# Plot high-confidence anomalies
high_mask = (df["Anomaly_Category"] == "High-Confidence Anomaly")
ax.scatter(
    df.loc[high_mask, "PC1"],
    df.loc[high_mask, "PC2"],
    c="#ef4444",
    alpha=0.95,
    s=85,
    marker="X",
    label=f"High-Confidence Anomalies ({high_mask.sum():,})",
    edgecolors="darkred",
    linewidths=0.8
)

ax.set_title("Agricultural Anomaly Detection in Latent 2D PCA Space", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% Variance)", fontsize=11, fontweight="bold")
ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% Variance)", fontsize=11, fontweight="bold")
ax.legend(loc="upper right", framealpha=0.9, fontsize=10)
ax.grid(True, linestyle=":", alpha=0.5)
plt.tight_layout()

scatter_path = os.path.join(anomaly_dir, "anomaly_detection_scatter.png")
plt.savefig(scatter_path, dpi=300)
plt.close(fig)
print(f"Saved anomaly 2D projection: {scatter_path}")

# -------------------------------------------------------------
# PLOT 2: Isolation Forest Score Distribution & Threshold
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.8))
scores_norm = df.loc[df["Anomaly_IForest"] == 0, "IForest_Score"]
scores_anom = df.loc[df["Anomaly_IForest"] == 1, "IForest_Score"]

ax.hist(scores_norm, bins=45, color="#10b981", alpha=0.7, label=f"Normal Records (n={len(scores_norm):,})", density=True)
ax.hist(scores_anom, bins=25, color="#ef4444", alpha=0.8, label=f"Flagged Anomalies (n={len(scores_anom):,})", density=True)

threshold_val = iso_forest.offset_
ax.axvline(0, color="navy", linestyle="--", lw=2, label="Decision Threshold (Score = 0.0)")

ax.set_title("Isolation Forest Anomaly Score Distribution", fontsize=12, fontweight="bold", pad=10)
ax.set_xlabel("Anomaly Decision Score (Lower = More Anomalous)", fontsize=10, fontweight="bold")
ax.set_ylabel("Probability Density", fontsize=10, fontweight="bold")
ax.legend(framealpha=0.9)
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

dist_path = os.path.join(anomaly_dir, "anomaly_score_distribution.png")
plt.savefig(dist_path, dpi=300)
plt.close(fig)
print(f"Saved score distribution plot: {dist_path}")

# -------------------------------------------------------------
# PLOT 3: Algorithm Agreement Comparison Matrix
# -------------------------------------------------------------
algo_counts = {
    "Isolation Forest": int(df["Anomaly_IForest"].sum()),
    "Local Outlier Factor": int(df["Anomaly_LOF"].sum()),
    "Elliptic Envelope": int(df["Anomaly_Elliptic"].sum()),
    "One-Class SVM": int(df["Anomaly_OCSVM"].sum()),
    "Ensemble (>= 2 Algos)": confirmed_cnt
}

fig, ax = plt.subplots(figsize=(8.5, 4.8))
bar_colors = ["#3b82f6", "#06b6d4", "#8b5cf6", "#f97316", "#ef4444"]
bars = ax.bar(list(algo_counts.keys()), list(algo_counts.values()), color=bar_colors, width=0.55, edgecolor="k", linewidth=0.5)

ax.set_title("Multi-Algorithm Outlier Detection Count & Ensemble Consensus", fontsize=12, fontweight="bold", pad=12)
ax.set_ylabel("Flagged Anomalous Records", fontsize=10, fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.6)

for bar in bars:
    y = bar.get_height()
    pct = (y / len(df)) * 100
    ax.text(bar.get_x() + bar.get_width() / 2, y + 2, f"{y:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.xticks(rotation=15, ha="right", fontsize=9, fontweight="bold")
plt.ylim(0, max(algo_counts.values()) * 1.3)
plt.tight_layout()

agree_path = os.path.join(anomaly_dir, "algorithm_agreement_matrix.png")
plt.savefig(agree_path, dpi=300)
plt.close(fig)
print(f"Saved algorithm agreement plot: {agree_path}")

# -------------------------------------------------------------
# PLOT 4: Agronomic Profile (Normal vs Confirmed Anomalies)
# -------------------------------------------------------------
compare_features = ["Area", "Annual_Rainfall", "Fertilizer_per_Ha", "Pesticide_per_Ha", "Yield"]
compare_features = [f for f in compare_features if f in df.columns]

mean_normal = df[df["Is_Confirmed_Anomaly"] == 0][compare_features].mean()
mean_anom = df[df["Is_Confirmed_Anomaly"] == 1][compare_features].mean()

# Relative percentage difference
rel_diff = ((mean_anom - mean_normal) / mean_normal.replace(0, 1)) * 100

fig, ax = plt.subplots(figsize=(9, 5))
pos_colors = ["#ef4444" if val >= 0 else "#3b82f6" for val in rel_diff]
bars = ax.barh(compare_features, rel_diff, color=pos_colors, alpha=0.85, edgecolor="k", height=0.45)

ax.axvline(0, color="black", linestyle="-", lw=1)
ax.set_title("Agronomic Discrepancy Profile: Anomalies vs Normal Records (% Deviation)", fontsize=12, fontweight="bold", pad=10)
ax.set_xlabel("Percentage Difference Relative to Normal Means (%)", fontsize=10, fontweight="bold")
ax.grid(axis="x", linestyle="--", alpha=0.5)

for bar in bars:
    w = bar.get_width()
    offset = 4 if w >= 0 else -8
    ha = "left" if w >= 0 else "right"
    ax.text(w + offset, bar.get_y() + bar.get_height() / 2, f"{w:+.1f}%", va="center", ha=ha, fontsize=9, fontweight="bold")

plt.tight_layout()
prof_path = os.path.join(anomaly_dir, "anomalous_farms_profile.png")
plt.savefig(prof_path, dpi=300)
plt.close(fig)
print(f"Saved anomaly profile plot: {prof_path}")

# 5. Save Flagged Anomalies CSV
export_cols = [c for c in ["Crop", "State", "Season", "Crop_Year", "Area", "Annual_Rainfall", "Fertilizer", "Pesticide", "Yield", "Fertilizer_per_Ha", "Anomaly_Category", "Ensemble_Anomaly_Count", "Root_Cause"] if c in anomalies_df.columns]
anomalies_export_path = os.path.join(anomaly_dir, "anomalies_flagged.csv")
anomalies_df[export_cols].sort_values(by="Ensemble_Anomaly_Count", ascending=False).to_csv(anomalies_export_path, index=False)
print(f"Saved flagged anomalies table ({len(anomalies_df):,} records): {anomalies_export_path}")

# 6. Save Model Artifact
anomaly_artifact = {
    "isolation_forest": iso_forest,
    "scaler": scaler,
    "imputer": imputer,
    "diagnostic_cols": diagnostic_cols,
    "contamination": CONTAMINATION
}
joblib_path = os.path.join(anomaly_dir, "isolation_forest.joblib")
joblib.dump(anomaly_artifact, joblib_path)
print(f"Saved serialized anomaly model: {joblib_path}")

# 7. Save JSON Summary
summary_dict = {
    "total_evaluated": len(df),
    "contamination_rate": CONTAMINATION,
    "algorithm_flags": algo_counts,
    "breakdown": {
        "normal_records": normal_cnt,
        "high_confidence_anomalies": high_conf,
        "moderate_anomalies": moderate,
        "confirmed_anomalies_total": confirmed_cnt,
        "anomaly_percentage": round((confirmed_cnt / len(df)) * 100, 2)
    },
    "mean_metrics": {
        "normal": {k: round(float(v), 2) for k, v in mean_normal.items()},
        "anomalous": {k: round(float(v), 2) for k, v in mean_anom.items()},
        "deviation_pct": {k: round(float(v), 1) for k, v in rel_diff.items()}
    },
    "top_anomalous_records": anomalies_df[export_cols].head(10).to_dict(orient="records")
}
json_path = os.path.join(anomaly_dir, "anomaly_summary.json")
with open(json_path, "w") as f:
    json.dump(summary_dict, f, indent=2)
print(f"Saved anomaly summary JSON: {json_path}")

# Also copy summary to outputs/
shared_json = os.path.join(config.OUTPUTS_DIR, "outputs", "co4_anomaly_summary.json")
with open(shared_json, "w") as f:
    json.dump(summary_dict, f, indent=2)

# 8. Save Detailed Technical Report
report_path = os.path.join(anomaly_dir, "anomaly_detection_report.txt")
with open(report_path, "w") as f:
    f.write("=" * 65 + "\n")
    f.write("AGRICULTURAL ANOMALY DETECTION & OUTLIER REPORT\n")
    f.write("=" * 65 + "\n\n")
    f.write(f"Evaluated Records: {len(df):,}\n")
    f.write(f"Target Contamination Rate: {CONTAMINATION*100:.1f}%\n")
    f.write(f"Diagnostic Features: {', '.join(diagnostic_cols)}\n\n")
    f.write("1. MULTI-ALGORITHM DETECTION SUMMARY:\n")
    for algo, cnt in algo_counts.items():
        f.write(f"   - {algo:<24}: {cnt:>5} records ({cnt/len(df)*100:.2f}%)\n")
    f.write(f"\n2. ENSEMBLE CONSENSUS:\n")
    f.write(f"   - High-Confidence (>= 3 algorithms): {high_conf} records\n")
    f.write(f"   - Moderate (2 algorithms)          : {moderate} records\n")
    f.write(f"   - Total Confirmed Outliers         : {confirmed_cnt} ({confirmed_cnt/len(df)*100:.2f}%)\n")
    f.write(f"\n3. FEATURE DEVIATION COMPARISON (Anomalies vs Normal):\n")
    for f_name in compare_features:
        f.write(f"   - {f_name:<20}: Normal={mean_normal[f_name]:>10.2f} | Anomalous={mean_anom[f_name]:>10.2f} | Diff={rel_diff[f_name]:>+7.1f}%\n")
    f.write("\n4. SAMPLE OF DETECTED HIGH-SEVERITY ANOMALIES:\n")
    for i, (_, row) in enumerate(anomalies_df.head(8).iterrows(), start=1):
        f.write(f"   [{i}] Crop: {row.get('Crop', 'N/A')} | State: {row.get('State', 'N/A')} | Year: {row.get('Crop_Year', 'N/A')}\n")
        f.write(f"       Area: {row.get('Area', 0):,.1f} ha | Rainfall: {row.get('Annual_Rainfall', 0):.1f} mm | Yield: {row.get('Yield', 0):.2f} t/ha\n")
        f.write(f"       Fertilizer/Ha: {row.get('Fertilizer_per_Ha', 0):.1f} kg/ha | Algos Flagged: {row.get('Ensemble_Anomaly_Count', 0)}/4\n")
        f.write(f"       Diagnosis: {row.get('Root_Cause', 'Unknown')}\n\n")
    f.write("=" * 65 + "\n")

print(f"Saved anomaly report: {report_path}")
print("Anomaly detection execution complete.\n")
