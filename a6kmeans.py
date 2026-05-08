from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Create dataset (3 distinct groups)
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

# 2. Initialize and fit K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

# 3. Analyze the clusters
centroids = kmeans.cluster_centers_
score = silhouette_score(X, labels)

print("Cluster Centers:\n", centroids)
print(f"Silhouette Score: {score:.2f} (Close to 1.0 means excellent clustering)")
