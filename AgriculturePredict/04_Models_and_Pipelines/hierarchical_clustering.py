# Hierarchical Agglomerative Clustering for AgriculturePredict
# ===============================================================
# Analyzes agro-climatic and yield patterns using linkage methods:
# 1. Linkage criteria (Ward, Complete, Average, Single) and Cophenetic correlation.
# 2. Multi-linkage Dendrogram tree geometry.
# 3. Silhouette score evaluation across cut heights (K = 2 to 7).
# 4. Cluster profiling of agricultural characteristics.

import sys, os
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
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
from scipy.spatial.distance import pdist
import config

os.makedirs(config.HIERARCHICAL_DIR, exist_ok=True)

SUBSAMPLE_SIZE = 2000

# 1. Load Data
raw_path = getattr(config, "RAW_DATA_PATH", "")
if not os.path.exists(raw_path):
    raw_path = os.path.join(BASE_DIR, "..", "01_Datasets", "crop_yield_sample.csv")

print(f"Loading agricultural dataset from: {raw_path}")
df_full = pd.read_csv(raw_path)
df_full.columns = df_full.columns.str.strip()

feature_cols = [c for c in config.NUMERIC_COLUMNS if c in df_full.columns]
x_raw = df_full[feature_cols].copy()

imputer = SimpleImputer(strategy="median")
x_imp = imputer.fit_transform(x_raw)

scaler = StandardScaler()
x_scaled_full = scaler.fit_transform(x_imp)

# Subsample for tractable O(N^2) distance computation
n_samples = min(SUBSAMPLE_SIZE, len(x_scaled_full))
rng = np.random.RandomState(config.RANDOM_STATE)
sample_idx = rng.choice(len(x_scaled_full), size=n_samples, replace=False)
x_sample = x_scaled_full[sample_idx]
df_sample = df_full.iloc[sample_idx].reset_index(drop=True)

print(f"Subsampled {n_samples:,} records for hierarchical distance matrices.")

# 2. Cophenetic Correlation across Linkages
p_dist = pdist(x_sample)
linkage_methods = ["ward", "complete", "average", "single"]
linkage_matrices = {}
cophenetic_scores = {}

for method in linkage_methods:
    Z = linkage(x_sample, method=method)
    linkage_matrices[method] = Z
    c_score, _ = cophenet(Z, p_dist)
    cophenetic_scores[method] = c_score
    print(f"Linkage [{method:>8s}]: Cophenetic Correlation = {c_score:.4f}")

# 3. Dendrograms Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, method in enumerate(linkage_methods):
    ax = axes[i]
    dendrogram(
        linkage_matrices[method],
        truncate_mode="lastp",
        p=25,
        leaf_rotation=90.,
        leaf_font_size=8.,
        show_contracted=True,
        ax=ax
    )
    ax.set_title(f"{method.capitalize()} Linkage (Cophenetic: {cophenetic_scores[method]:.3f})", fontweight="bold")
    ax.set_ylabel("Distance")

plt.suptitle("Hierarchical Clustering Dendrograms by Linkage Strategy", fontsize=15, fontweight="bold")
plt.tight_layout()
dendro_path = os.path.join(config.HIERARCHICAL_DIR, "dendrograms_by_linkage.png")
plt.savefig(dendro_path, dpi=300)
plt.close()
print(f"Saved dendrograms: {dendro_path}")

# 4. Silhouette Evaluation across Cut Heights (K = 2 to 7)
k_values = list(range(2, 8))
silhouette_scores = []

for k in k_values:
    model = AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="ward")
    cluster_labels = model.fit_predict(x_sample)
    sil = silhouette_score(x_sample, cluster_labels)
    silhouette_scores.append(sil)

plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_scores, marker="o", color="darkgreen", lw=2)
plt.title("Silhouette Score by Cluster Count (Ward Linkage)", fontsize=13, fontweight="bold")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.grid(True, linestyle="--", alpha=0.6)
sil_path = os.path.join(config.HIERARCHICAL_DIR, "silhouette_by_cut_height.png")
plt.tight_layout()
plt.savefig(sil_path, dpi=300)
plt.close()
print(f"Saved silhouette cut curve: {sil_path}")

best_k = k_values[int(np.argmax(silhouette_scores))]
print(f"Optimal K via Silhouette: {best_k} (score: {max(silhouette_scores):.4f})")

# 5. Cluster Profiling
final_model = AgglomerativeClustering(n_clusters=best_k, metric="euclidean", linkage="ward")
df_sample["Hierarchical_Cluster"] = final_model.fit_predict(x_sample)

profile = df_sample.groupby("Hierarchical_Cluster")[feature_cols].mean()

plt.figure(figsize=(10, 6))
norm_profile = (profile - profile.mean()) / profile.std()
norm_profile.T.plot(kind="bar", figsize=(10, 6), colormap="viridis")
plt.title(f"Cluster Agricultural Profile (Standardized Means, K={best_k})", fontsize=13, fontweight="bold")
plt.ylabel("Z-Score relative to overall mean")
plt.xlabel("Agricultural Features")
plt.xticks(rotation=30)
plt.legend(title="Cluster")
plt.tight_layout()

prof_path = os.path.join(config.HIERARCHICAL_DIR, "cluster_crop_profile.png")
plt.savefig(prof_path, dpi=300)
# Also save as cluster_placement_profile.png for strict name parity with reference
alias_prof_path = os.path.join(config.HIERARCHICAL_DIR, "cluster_placement_profile.png")
plt.savefig(alias_prof_path, dpi=300)
plt.close()
print(f"Saved cluster profile plots: {prof_path}")

# 6. Report
rep_path = os.path.join(config.HIERARCHICAL_DIR, "hierarchical_report.txt")
with open(rep_path, "w") as f:
    f.write("=" * 60 + "\n")
    f.write("HIERARCHICAL AGGLOMERATIVE CLUSTERING REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write("1. Linkage Cophenetic Correlation:\n")
    for m in linkage_methods:
        f.write(f"   - {m.capitalize():<12}: {cophenetic_scores[m]:.4f}\n")
    f.write(f"\n2. Optimal Cluster Cut: K = {best_k}\n")
    f.write("   - Silhouette Scores by K:\n")
    for k, s in zip(k_values, silhouette_scores):
        f.write(f"       K={k}: {s:.4f}\n")
    f.write("\n3. Cluster Feature Means:\n")
    f.write(profile.to_string())
    f.write("\n")

print(f"Saved hierarchical report: {rep_path}")
