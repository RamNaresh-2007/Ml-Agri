# Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
# ======================================================================
# Adapted for AgriculturePredict Agro-Ecological Profiling.
# Partitions crop production regions and isolates anomalous farm records as NOISE (-1).

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
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import config

os.makedirs(config.DBSCAN_DIR, exist_ok=True)

SUBSAMPLE_SIZE = 5000

# 1. Load Data
train_csv_path = os.path.join(getattr(config, "SPLITS_DIR", ""), "train.csv")
if os.path.exists(train_csv_path):
    df_full = pd.read_csv(train_csv_path)
elif hasattr(config, "RAW_DATA_PATH") and os.path.exists(config.RAW_DATA_PATH):
    raw_df = pd.read_csv(config.RAW_DATA_PATH)
    from sklearn.model_selection import train_test_split
    df_full, _ = train_test_split(raw_df, test_size=0.3, random_state=config.RANDOM_STATE)
    df_full = df_full.reset_index(drop=True)
else:
    raise FileNotFoundError("Could not find train.csv or RAW_DATA_PATH.")

feature_cols = [c for c in config.NUMERIC_COLUMNS if c in df_full.columns]
x_raw_full = df_full[feature_cols].copy()

imputer = SimpleImputer(strategy="median")
x_imputed_full = pd.DataFrame(imputer.fit_transform(x_raw_full), columns=feature_cols, index=x_raw_full.index)

scaler = StandardScaler()
x_scaled_full = scaler.fit_transform(x_imputed_full)

n_samples = min(SUBSAMPLE_SIZE, len(x_scaled_full))
rng = np.random.RandomState(config.RANDOM_STATE)
sample_idx = rng.choice(len(x_scaled_full), size=n_samples, replace=False)
x_scaled = x_scaled_full[sample_idx]
df = df_full.iloc[sample_idx].reset_index(drop=True)

print(f"Loaded {len(df_full):,} agricultural records, subsampled {n_samples:,} for DBSCAN")

# 2. Choosing eps: The k-distance plot
min_samples = 2 * len(feature_cols)
k = min_samples
nbrs = NearestNeighbors(n_neighbors=k).fit(x_scaled)
distances, _ = nbrs.kneighbors(x_scaled)
k_distances = np.sort(distances[:, k - 1])

plt.figure(figsize=(8, 5))
plt.plot(k_distances, color="navy", lw=2)
plt.title(f"K-Distance Plot for DBSCAN (k = min_samples = {k})", fontsize=13, fontweight="bold")
plt.xlabel("Points Sorted by Distance")
plt.ylabel(f"{k}-NN Distance")
plt.grid(True, linestyle="--", alpha=0.6)
k_dist_path = os.path.join(config.DBSCAN_DIR, "k_distance_plot.png")
plt.tight_layout()
plt.savefig(k_dist_path, dpi=300)
plt.close()
print(f"Saved k-distance plot: {k_dist_path}")

# 3. Fitting DBSCAN
# Typical elbow distance on scaled features is around 0.8 - 1.2
eps = 1.0
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(x_scaled)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
noise_pct = (n_noise / len(labels)) * 100

print(f"Estimated Clusters: {n_clusters}")
print(f"Estimated Noise Points: {n_noise} ({noise_pct:.2f}%)")

# 4. 2D PCA Visualization
pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
x_pca = pca.fit_transform(x_scaled)

plt.figure(figsize=(9, 6))
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k_label, col in zip(unique_labels, colors):
    if k_label == -1:
        col = [0.6, 0.6, 0.6, 0.5]  # Gray for noise
        label_name = "Noise (-1)"
    else:
        label_name = f"Cluster {k_label}"

    class_member_mask = (labels == k_label)
    xy = x_pca[class_member_mask]
    plt.scatter(xy[:, 0], xy[:, 1], c=[col], label=label_name, edgecolors="k", s=30, alpha=0.7)

plt.title(f"DBSCAN Agricultural Clusters in 2D PCA Space (eps={eps}, min_samples={min_samples})", fontsize=13, fontweight="bold")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
pca_plot_path = os.path.join(config.DBSCAN_DIR, "dbscan_clusters_pca_2d.png")
plt.savefig(pca_plot_path, dpi=300)
plt.close()
print(f"Saved PCA clustering plot: {pca_plot_path}")

# 5. Summary Report
report_path = os.path.join(config.DBSCAN_DIR, "dbscan_report.txt")
with open(report_path, "w") as f:
    f.write("=" * 60 + "\n")
    f.write("DBSCAN AGRO-ECOLOGICAL CLUSTERING REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Parameters:\n")
    f.write(f"  - eps: {eps}\n")
    f.write(f"  - min_samples: {min_samples}\n")
    f.write(f"  - Subsample Size: {len(labels)}\n\n")
    f.write(f"Results:\n")
    f.write(f"  - Number of Clusters Formed: {n_clusters}\n")
    f.write(f"  - Noise Points (Outliers): {n_noise} ({noise_pct:.2f}%)\n")
    f.write(f"  - Core Features Used: {', '.join(feature_cols)}\n\n")
    f.write("Cluster Distribution:\n")
    for cl in sorted(unique_labels):
        cnt = np.sum(labels == cl)
        f.write(f"  Cluster {cl:>2}: {cnt:>5} records ({cnt/len(labels)*100:.1f}%)\n")

print(f"DBSCAN report written to: {report_path}")
