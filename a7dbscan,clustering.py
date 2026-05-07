from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import AgglomerativeClustering, BisectingKMeans, DBSCAN

# ==========================================
# 1. LOAD AND PREPARE DATASETS
# ==========================================
# Dataset 1: 3 normal circular clusters (150 points)
X_blobs, y_blobs = make_blobs(n_samples=150, centers=3, cluster_std=0.6)

# Dataset 2: 2 crescent moon shapes (150 points)
X_moons, y_moons = make_moons(n_samples=150, noise=0.05)


# ==========================================
# 2. AGGLOMERATIVE CLUSTERING (Bottom-Up Hierarchical)
# ==========================================
# Step 1: Initialize the model (tell it we want 3 clusters)
agg_model = AgglomerativeClustering(n_clusters=3)

# Step 2: Fit the model and get the cluster labels for each point
agg_labels = agg_model.fit_predict(X_blobs)


# ==========================================
# 3. DIVISIVE CLUSTERING (Top-Down Hierarchical)
# ==========================================
# Note: Sklearn uses 'BisectingKMeans' as its standard Divisive algorithm.
# It starts with 1 giant cluster and splits it until it reaches 3.
div_model = BisectingKMeans(n_clusters=3)

div_labels = div_model.fit_predict(X_blobs)


# ==========================================
# 4. DBSCAN (Spatial / Density-Based)
# ==========================================
# eps = Maximum distance between two points to be considered neighbors
# min_samples = Minimum number of points needed to form a "dense region" (cluster)
dbscan_model = DBSCAN(eps=0.3, min_samples=5)

# Notice we use the 'Moons' dataset here, because DBSCAN is great at weird shapes!
dbscan_labels = dbscan_model.fit_predict(X_moons)