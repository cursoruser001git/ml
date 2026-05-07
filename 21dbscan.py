# Requirements: scikit-learn, pandas, numpy, matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

# 1) Example 2D data (replace with your two-feature dataset)
X, _ = make_moons(n_samples=400, noise=0.08, random_state=42)
df = pd.DataFrame(X, columns=["x1", "x2"])

# 2) Scale
scaler = StandardScaler()
X_s = scaler.fit_transform(df)

# 3) Fit DBSCAN (library only)
eps = 0.3
min_samples = 5
db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=-1)
db.fit(X_s)

# 4) Use DBSCAN attributes to label points (library outputs)
labels = db.labels_               # -1 = noise
core_indices = db.core_sample_indices_  # indices of core samples (provided by sklearn)

# 5) Build DataFrame showing label and type (core/border/noise)
df['label'] = labels
df['point_type'] = 'border'
df.loc[labels == -1, 'point_type'] = 'noise'
df.loc[core_indices, 'point_type'] = 'core'

# 6) Summaries
print("Unique labels:", np.unique(labels))
print("\nPoint type counts:\n", df['point_type'].value_counts())
print("\nCluster sizes (excluding noise):\n", df[df['label'] != -1]['label'].value_counts().sort_index())

# 7) Plot clusters (library results only)
unique_labels = sorted(set(labels))
palette = plt.cm.get_cmap('tab10', max(1, len(unique_labels)))
plt.figure(figsize=(8,6))
for lab in unique_labels:
    if lab == -1:
        pts = X_s[labels == lab]
        plt.scatter(pts[:,0], pts[:,1], c='red', marker='x', s=40, label='noise')
    else:
        pts = X_s[labels == lab]
        core_mask = np.isin(np.where(labels == lab)[0], core_indices)
        # plot core and border using indices from sklearn outputs
        pts_all_idx = np.where(labels == lab)[0]
        pts_core = X_s[pts_all_idx[core_mask]]
        pts_border = X_s[pts_all_idx[~core_mask]]
        plt.scatter(pts_core[:,0], pts_core[:,1], s=80, color=palette(lab), edgecolor='k', label=f'cluster {lab} core')
        if pts_border.size:
            plt.scatter(pts_border[:,0], pts_border[:,1], s=30, color=palette(lab), alpha=0.7, label=f'cluster {lab} border')

plt.title(f"DBSCAN (eps={eps}, min_samples={min_samples})")
plt.xlabel("Scaled x1")
plt.ylabel("Scaled x2")
plt.legend(loc='best', fontsize='small')
plt.tight_layout()
plt.show()

# 8) Show sample rows (library outputs)
print("\nFirst 10 points with label and type:\n", df.head(10))
