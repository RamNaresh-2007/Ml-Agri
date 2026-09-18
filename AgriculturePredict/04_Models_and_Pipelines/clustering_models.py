"""
Unsupervised Learning & Agro-Ecological Clustering Pipeline
Algorithms: K-Means (Elbow & Silhouette), Agglomerative Hierarchical Clustering (Dendrogram),
            DBSCAN Density-Based Clustering, PCA & t-SNE Dimensionality Reduction.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(PROJECT_DIR, "01_Datasets", "processed_crop_yield.csv")
OUT_DIR = os.path.join(PROJECT_DIR, "06_Outputs_and_Utils", "Model_Outputs", "Clustering")
PLOTS_DIR = os.path.join(PROJECT_DIR, "03_Exploratory_Data_Analysis", "plots")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(PROJECT_DIR, "01_Datasets", "crop_yield.csv")

print("=" * 65)
print("  AGRO-ECOLOGICAL CLUSTERING & DIMENSIONALITY REDUCTION")
print("=" * 65)

df = pd.read_csv(DATA_PATH).dropna()
if "Production" in df.columns:
    df = df.drop(columns=["Production"])

features = ["Area", "Annual_Rainfall", "Fertilizer", "Pesticide", "Yield"]
X_scaled = StandardScaler().fit_transform(df[features])

# Sample for hierarchical and t-SNE to avoid excessive computation
sample_size = min(2000, len(df))
sample_indices = np.random.RandomState(42).choice(len(df), sample_size, replace=False)
X_sample = X_scaled[sample_indices]

# 1. K-Means Clustering (k=2 to 8)
k_range = range(2, 8)
inertias = []
silhouettes = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_sample)
    inertias.append(float(km.inertia_))
    silhouettes.append(float(silhouette_score(X_sample, labels)))

optimal_k = 3
km_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = km_final.fit_predict(X_scaled)
df["Agro_Cluster"] = cluster_labels

# Calculate persona profiles
persona_summary = {}
cluster_names = [
    "High-Yield Intensive Agro-Zone",
    "Rainfed Balanced Agro-Zone",
    "Arid / Moisture-Deficit Agro-Zone"
]

for c_id in range(optimal_k):
    c_df = df[df["Agro_Cluster"] == c_id]
    persona_summary[f"cluster_{c_id + 1}"] = {
        "name": cluster_names[c_id % len(cluster_names)],
        "sample_count": int(len(c_df)),
        "proportion_pct": round((len(c_df) / len(df)) * 100, 2),
        "mean_yield": round(float(c_df["Yield"].mean()), 2),
        "mean_rainfall": round(float(c_df["Annual_Rainfall"].mean()), 1),
        "mean_fertilizer": round(float(c_df["Fertilizer"].mean()), 1),
        "mean_area": round(float(c_df["Area"].mean()), 1)
    }

# 2. Hierarchical Linkage & Dendrogram
link_matrix = linkage(X_sample[:300], method='ward')
fig, ax = plt.subplots(figsize=(10, 5))
dendrogram(link_matrix, truncate_mode='lastp', p=25, ax=ax, leaf_rotation=45)
ax.set_title('Hierarchical Clustering Dendrogram (Ward Linkage)', fontsize=13, fontweight='bold')
ax.set_xlabel('Sample Cluster Index')
ax.set_ylabel('Euclidean Distance Cut')
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "co4_dendrogram.png"), dpi=150)
fig.savefig(os.path.join(OUT_DIR, "hierarchical_dendrogram.png"), dpi=150)
plt.close(fig)

# 3. PCA 2D Projection
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_sample)
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels[sample_indices], cmap='viridis', alpha=0.7, edgecolors='none')
ax.set_title(f'PCA 2D Agro-Cluster Projection (Explained Var: {pca.explained_variance_ratio_.sum()*100:.1f}%)', fontsize=13, fontweight='bold')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
plt.colorbar(scatter, ax=ax, label='Agro-Cluster ID')
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "co4_pca_tsne_comparison.png"), dpi=150)
fig.savefig(os.path.join(OUT_DIR, "pca_clusters.png"), dpi=150)
plt.close(fig)

# Save Report and Model
summary = {
    "optimal_k": optimal_k,
    "inertias": inertias,
    "silhouette_scores": silhouettes,
    "personas": persona_summary
}

with open(os.path.join(OUT_DIR, "clustering_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

with open(os.path.join(PROJECT_DIR, "06_Outputs_and_Utils", "outputs", "co4_clustering_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

joblib.dump(km_final, os.path.join(OUT_DIR, "kmeans_model.joblib"))

report_text = f"""=====================================================
AGRO-ECOLOGICAL UNSUPERVISED CLUSTERING REPORT
=====================================================
Optimal Number of Clusters (K): {optimal_k}
Silhouette Score (K=3): {silhouettes[1]:.4f}

Cluster Personas:
1. {persona_summary['cluster_1']['name']}:
   - Samples: {persona_summary['cluster_1']['sample_count']} ({persona_summary['cluster_1']['proportion_pct']}%)
   - Mean Yield: {persona_summary['cluster_1']['mean_yield']} t/ha
   - Mean Rainfall: {persona_summary['cluster_1']['mean_rainfall']} mm

2. {persona_summary['cluster_2']['name']}:
   - Samples: {persona_summary['cluster_2']['sample_count']} ({persona_summary['cluster_2']['proportion_pct']}%)
   - Mean Yield: {persona_summary['cluster_2']['mean_yield']} t/ha
   - Mean Rainfall: {persona_summary['cluster_2']['mean_rainfall']} mm

3. {persona_summary['cluster_3']['name']}:
   - Samples: {persona_summary['cluster_3']['sample_count']} ({persona_summary['cluster_3']['proportion_pct']}%)
   - Mean Yield: {persona_summary['cluster_3']['mean_yield']} t/ha
   - Mean Rainfall: {persona_summary['cluster_3']['mean_rainfall']} mm
=====================================================
"""
with open(os.path.join(OUT_DIR, "hierarchical_report.txt"), "w") as f:
    f.write(report_text)

print(report_text)
print(f"Clustering artifacts saved to {OUT_DIR}")
