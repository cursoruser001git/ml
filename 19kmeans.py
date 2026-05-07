# Requirements: scikit-learn, pandas, numpy, matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Generate example 2D data (replace X with your two-feature dataset)
X, true_labels = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)
df = pd.DataFrame(X, columns=["feat1", "feat2"])

# Optional scaling
scaler = StandardScaler()
X_s = scaler.fit_transform(df)

# Fit K-Means
k = 4
km = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = km.fit_predict(X_s)
centers = km.cluster_centers_

# Evaluation
sil_score = silhouette_score(X_s, labels)
print(f"K-Means (k={k}) silhouette score: {sil_score:.4f}")

# If you have ground-truth labels, compute ARI (optional)
if 'true_labels' in globals():
    ari = adjusted_rand_score(true_labels, labels)
    print(f"Adjusted Rand Index (if ground truth available): {ari:.4f}")

# Add cluster labels to DataFrame
df["cluster"] = labels

# Simple 2D scatter plot
plt.figure(figsize=(6,5))
for i in range(k):
    pts = X_s[labels == i]
    plt.scatter(pts[:,0], pts[:,1], s=30, label=f"Cluster {i}")
# plot centers (scaled back to original feature space for readability)
centers_orig = scaler.inverse_transform(centers)
plt.scatter(centers[:,0], centers[:,1], c='black', s=100, marker='x', label='Centers')
plt.title(f"K-Means clustering (k={k})")
plt.xlabel("Scaled feat1")
plt.ylabel("Scaled feat2")
plt.legend()
plt.tight_layout()
plt.show()

# Print basic cluster sizes and first few points
print("Cluster sizes:")
print(df["cluster"].value_counts().sort_index())
print("\nSample points with cluster assignment:")
print(df.head())
