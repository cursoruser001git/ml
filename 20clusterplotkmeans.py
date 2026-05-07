# K-Means clustering visualization — Requirements: scikit-learn, numpy, pandas, matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 1) Create example 2D dataset (replace with your two-feature data)
X, true_labels = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)
df = pd.DataFrame(X, columns=["feat1", "feat2"])

# 2) Optional scaling
scaler = StandardScaler()
X_s = scaler.fit_transform(df)

# 3) Fit K-Means
k = 4
km = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = km.fit_predict(X_s)
centers = km.cluster_centers_

# 4) Evaluation
sil = silhouette_score(X_s, labels)
print(f"K-Means (k={k}) silhouette score: {sil:.4f}")

# 5) Prepare plot (color by cluster, show centers and optional true labels)
plt.figure(figsize=(8,6))
palette = plt.cm.get_cmap("tab10", k)
for i in range(k):
    pts = X_s[labels == i]
    plt.scatter(pts[:, 0], pts[:, 1], s=40, color=palette(i), label=f"Cluster {i}", alpha=0.7, edgecolor='k', linewidth=0.2)

# Plot cluster centers
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=150, marker='X', label='Centers')

# Optionally plot true labels' boundaries (if available) as small markers
if 'true_labels' in globals():
    plt.scatter(X_s[:,0], X_s[:,1], c=true_labels, cmap='coolwarm', s=8, alpha=0.25, label='True labels (overlay)')

plt.title(f"K-Means clustering (k={k}) — silhouette={sil:.3f}")
plt.xlabel("Scaled feat1")
plt.ylabel("Scaled feat2")
plt.legend(loc='best')
plt.grid(False)
plt.tight_layout()
plt.show()

# 6) Print cluster sizes and sample assignments
df['cluster'] = labels
print("Cluster sizes:\n", df['cluster'].value_counts().sort_index())
print("\nSample points with cluster assignment:\n", df.head())
